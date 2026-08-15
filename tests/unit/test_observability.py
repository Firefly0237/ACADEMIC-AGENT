"""可观测：span 树、成本归因、召回明细、badcase 归因（M7 §11–§13）。

**归因矩阵的判定顺序是这份测试的核心**：先怀疑生成、去调 prompt，是最常见也最
浪费时间的排障方式——因为生成层的问题有一半根源在上游，而上游的问题在明细表里
是一眼可见的。所以顺序必须被断言，而不是被文档描述。

召回明细那部分守的是「同一份数据两用」：`cited` 与 `in_context` 一比，
**离线的 citation precision 不需要重跑检索**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from citequest.core.answer import CitationAnchor, Evidence
from citequest.core.enums import DegradeReason, Granularity, PipelineStage, RetrievalMode
from citequest.core.errors import InvalidParamError
from citequest.core.search import RetrievedChunk
from citequest.observability.badcase import BadcaseClass, classify_badcase, conflicting_evidence_present
from citequest.observability.cost import ModelPrice, PriceTable, attribute_cost
from citequest.observability.retrieval_log import RetrievalDetail, RetrievalDetailWriter, build_details
from citequest.observability.trace import SamplingPolicy, SpanKind, TraceTree


@pytest.fixture
def evidences(retrieved_fulltext_chunk: RetrievedChunk) -> list[Evidence]:
    return [
        Evidence(
            citation_id=1,
            anchor=CitationAnchor(
                paper_id=retrieved_fulltext_chunk.paper_id, chunk_id=retrieved_fulltext_chunk.chunk_id
            ),
            text=retrieved_fulltext_chunk.text,
            title="Paper 1",
            year=2021,
            granularity=Granularity.FULLTEXT,
        )
    ]


class TestSpanTree:
    """四层 `request → turn → tool_call → llm_call`。"""

    def test_children_carry_their_parent_id(self) -> None:
        tree = TraceTree(index_version="v1")
        with (
            tree.span(SpanKind.REQUEST, "POST /v1/answer") as request,
            tree.span(SpanKind.TURN, "turn-0"),
            tree.span(SpanKind.LLM_CALL, "generate", stage=PipelineStage.GENERATE) as call,
        ):
            call.completion_tokens = 128
        llm = tree.by_kind(SpanKind.LLM_CALL)[0]
        turn = tree.by_kind(SpanKind.TURN)[0]
        assert llm.parent_span_id == turn.span_id
        assert turn.parent_span_id == request.span_id
        assert tree.root is request

    def test_required_fields_are_populated(self) -> None:
        tree = TraceTree(index_version="v1")
        with tree.span(SpanKind.LLM_CALL, "hyde", stage=PipelineStage.HYDE, model_id="qwen-1.5b", prompt_version="p3"):
            pass
        node = tree.nodes[0]
        # prompt_version 与 index_version 是归因的两个支点：prompt 改了但没记版本，
        # 两次评测之间的差异就无法归因。
        assert (node.prompt_version, node.index_version, node.model_id) == ("p3", "v1", "qwen-1.5b")
        assert node.duration_ms >= 0.0

    def test_business_and_infrastructure_errors_are_separated(self) -> None:
        # 两类失败的处置完全不同，混在一个 error_rate 里，告警一响就得先人工分拣。
        tree = TraceTree()
        with pytest.raises(InvalidParamError), tree.span(SpanKind.REQUEST, "r"):
            raise InvalidParamError("bad query")
        assert tree.nodes[0].error_class == "business"

        tree2 = TraceTree()
        with pytest.raises(TimeoutError), tree2.span(SpanKind.REQUEST, "r"):
            raise TimeoutError("milvus timeout")
        assert tree2.nodes[0].error_class == "infrastructure"

    def test_errors_are_always_sampled(self) -> None:
        # 按固定比例采样所有请求的后果是：出问题的那 0.5% 恰好没被采到。
        tree = TraceTree(policy=SamplingPolicy(base_rate=0.0))
        with pytest.raises(TimeoutError), tree.span(SpanKind.REQUEST, "r"):
            raise TimeoutError("boom")
        assert tree.should_export() is True

    def test_degraded_requests_are_always_sampled(self) -> None:
        tree = TraceTree(policy=SamplingPolicy(base_rate=0.0))
        with tree.span(SpanKind.REQUEST, "r") as node:
            node.degraded = DegradeReason.RERANK_TIMEOUT
        assert tree.should_export() is True


class TestCostAttribution:
    """三个维度必须各自加得回总额——**加不回说明有 span 漏挂了父节点**。"""

    def test_three_dimensions_sum_to_the_total(self) -> None:
        tree = TraceTree()
        with tree.span(SpanKind.REQUEST, "req"), tree.span(SpanKind.TOOL_CALL, "search_papers"):
            with tree.span(SpanKind.LLM_CALL, "hyde", stage=PipelineStage.HYDE, model_id="small") as hyde:
                hyde.prompt_tokens, hyde.completion_tokens = 200, 128
            with tree.span(SpanKind.LLM_CALL, "generate", stage=PipelineStage.GENERATE, model_id="big") as gen:
                gen.prompt_tokens, gen.completion_tokens = 4000, 512

        table = PriceTable(
            prices={
                "small": ModelPrice(prompt_per_million=1.0, completion_per_million=2.0),
                "big": ModelPrice(prompt_per_million=10.0, completion_per_million=30.0),
            }
        )
        attribution = attribute_cost(tree.nodes, table)
        assert sum(attribution.by_step.values()) == pytest.approx(attribution.total)
        assert sum(attribution.by_model.values()) == pytest.approx(attribution.total)
        assert sum(attribution.by_tool.values()) == pytest.approx(attribution.total)

    def test_dominant_step_is_answerable(self) -> None:
        tree = TraceTree()
        with tree.span(SpanKind.LLM_CALL, "generate", stage=PipelineStage.GENERATE, model_id="big") as gen:
            gen.prompt_tokens, gen.completion_tokens = 4000, 512
        with tree.span(SpanKind.LLM_CALL, "hyde", stage=PipelineStage.HYDE, model_id="big") as hyde:
            hyde.prompt_tokens, hyde.completion_tokens = 100, 64
        table = PriceTable(prices={"big": ModelPrice(prompt_per_million=10.0, completion_per_million=30.0)})
        assert attribute_cost(tree.nodes, table).dominant_step() == "generate"

    def test_cost_is_written_back_onto_the_span(self) -> None:
        # 回填后 trace 后端里每一条自带成本，不必再关联一次。
        tree = TraceTree()
        with tree.span(SpanKind.LLM_CALL, "generate", model_id="big") as gen:
            gen.prompt_tokens = 1_000_000
        attribute_cost(
            tree.nodes, PriceTable(prices={"big": ModelPrice(prompt_per_million=3.0, completion_per_million=6.0)})
        )
        assert tree.nodes[0].cost_estimate == pytest.approx(3.0)

    def test_cached_tokens_are_not_counted_as_billable_total(self) -> None:
        # cached 单独计价，混进来会让成本估算系统性偏高。
        tree = TraceTree()
        with tree.span(SpanKind.LLM_CALL, "generate", model_id="big") as gen:
            gen.prompt_tokens, gen.cached_tokens = 100, 900
        assert tree.nodes[0].total_tokens == 100


class TestRetrievalDetail:
    """召回明细：一举两得的那份产物。"""

    def test_cited_comes_from_raw_citation_ids(
        self,
        retrieved_fulltext_chunk: RetrievedChunk,
        retrieved_abstract_chunk: RetrievedChunk,
        evidences: list[Evidence],
    ) -> None:
        details = build_details(
            [retrieved_fulltext_chunk, retrieved_abstract_chunk],
            request_id="01J0000000000000000000000A",
            index_version="v1",
            retrieval_mode=RetrievalMode.HYBRID,
            context_chunk_ids=[retrieved_fulltext_chunk.chunk_id],
            evidences=evidences,
            cited_citation_ids=[1],
        )
        assert [d.in_context for d in details] == [True, False]
        assert [d.cited for d in details] == [True, False]

    def test_citation_precision_is_computable_without_rerunning_retrieval(
        self, retrieved_fulltext_chunk: RetrievedChunk, evidences: list[Evidence]
    ) -> None:
        # 这就是 §12.2 那条「同一份数据两用」：cited ∩ in_context 直接可算。
        details = build_details(
            [retrieved_fulltext_chunk],
            request_id="r1",
            index_version="v1",
            retrieval_mode=RetrievalMode.HYBRID,
            context_chunk_ids=[retrieved_fulltext_chunk.chunk_id],
            evidences=evidences,
            cited_citation_ids=[1],
        )
        cited_in_context = [d for d in details if d.cited and d.in_context]
        assert len(cited_in_context) == 1

    def test_recalled_by_reports_the_channels(self, retrieved_abstract_chunk: RetrievedChunk) -> None:
        details = build_details(
            [retrieved_abstract_chunk], request_id="r1", index_version="v1", retrieval_mode=RetrievalMode.DENSE_ONLY
        )
        assert details[0].recalled_by == ["dense"]

    def test_writer_round_trips_by_request_id(self, tmp_path: Path, retrieved_fulltext_chunk: RetrievedChunk) -> None:
        writer = RetrievalDetailWriter(tmp_path)
        first = build_details(
            [retrieved_fulltext_chunk], request_id="r1", index_version="v1", retrieval_mode=RetrievalMode.HYBRID
        )
        second = build_details(
            [retrieved_fulltext_chunk], request_id="r2", index_version="v1", retrieval_mode=RetrievalMode.HYBRID
        )
        path = writer.write(first, date_key="20260813")
        writer.write(second, date_key="20260813")
        assert path is not None
        assert len(writer.read(path, request_id="r1")) == 1
        assert len(writer.read(path)) == 2

    def test_disabled_writer_is_a_no_op(self, tmp_path: Path, retrieved_fulltext_chunk: RetrievedChunk) -> None:
        writer = RetrievalDetailWriter(tmp_path, enabled=False)
        details = build_details(
            [retrieved_fulltext_chunk], request_id="r1", index_version="v1", retrieval_mode=RetrievalMode.HYBRID
        )
        assert writer.write(details, date_key="20260813") is None


def _detail(**overrides: object) -> RetrievalDetail:
    base = {
        "request_id": "r1",
        "index_version": "v1",
        "chunk_id": "W1:f:s001:p00001",
        "paper_id": "W1",
        "granularity": Granularity.FULLTEXT,
        "in_candidate": True,
        "in_context": True,
        "cited": True,
        "retrieval_mode": RetrievalMode.HYBRID,
    }
    return RetrievalDetail(**{**base, **overrides})  # type: ignore[arg-type]


class TestBadcaseMatrix:
    """**判定顺序不能颠倒**：先切层再修。"""

    def test_missing_from_candidates_but_in_corpus_is_a_retrieval_miss(self) -> None:
        verdict = classify_badcase([_detail(paper_id="W9")], gold_paper_id="W1", gold_in_corpus=True)
        assert verdict.verdict is BadcaseClass.RETRIEVAL
        assert "FLAT" in verdict.next_action

    def test_not_in_corpus_is_not_a_retrieval_bug(self) -> None:
        # 库里根本没有这篇——把它算进检索错会让「提 ef」被反复尝试在一个
        # ef 永远解决不了的问题上。
        verdict = classify_badcase([], gold_paper_id="W1", gold_in_corpus=False)
        assert verdict.verdict is BadcaseClass.NOT_A_BADCASE
        assert "语料覆盖" in verdict.next_action

    def test_high_single_channel_rank_dropped_by_fusion(self) -> None:
        detail = _detail(in_candidate=False, in_context=False, cited=False, rank_sparse=3, rank_fused=None)
        verdict = classify_badcase([detail], gold_paper_id="W1", gold_in_corpus=True)
        assert verdict.verdict is BadcaseClass.FUSION
        assert verdict.signals["channel"] == "sparse"

    def test_in_candidate_but_not_in_context_is_truncation(self) -> None:
        detail = _detail(in_context=False, cited=False, score_rerank=0.31)
        verdict = classify_badcase([detail], gold_paper_id="W1", gold_in_corpus=True)
        assert verdict.verdict is BadcaseClass.TRUNCATION

    def test_generation_error_only_after_the_upstream_layers_are_clean(self) -> None:
        verdict = classify_badcase([_detail()], gold_paper_id="W1", gold_in_corpus=True, answer_consistent=False)
        assert verdict.verdict is BadcaseClass.GENERATION

    def test_citation_error_is_the_last_layer(self) -> None:
        verdict = classify_badcase(
            [_detail()],
            gold_paper_id="W1",
            gold_in_corpus=True,
            answer_consistent=True,
            citation_points_to_supporting_chunk=False,
        )
        assert verdict.verdict is BadcaseClass.CITATION

    def test_unknown_generation_signals_stop_the_chain_instead_of_guessing(self) -> None:
        # answer_consistent 为 None 表示没跑生成侧判定，链路停在第 ③ 步——**不猜**。
        verdict = classify_badcase([_detail()], gold_paper_id="W1", gold_in_corpus=True)
        assert verdict.verdict is BadcaseClass.NOT_A_BADCASE

    def test_best_chunk_wins_over_a_truncated_sibling(self) -> None:
        # 同一篇论文一个块被截断、另一个块进了上下文时，不能判成截断错。
        details = [_detail(chunk_id="a", in_context=False, cited=False), _detail(chunk_id="b")]
        verdict = classify_badcase(details, gold_paper_id="W1", gold_in_corpus=True, answer_consistent=True)
        assert verdict.evidence_chunk_id == "b"


class TestConflictSignal:
    def test_mixed_granularity_across_papers_is_flagged(self) -> None:
        details = [
            _detail(chunk_id="a", paper_id="W1"),
            _detail(chunk_id="b", paper_id="W2", granularity=Granularity.ABSTRACT),
        ]
        assert conflicting_evidence_present(details) is True

    def test_single_paper_context_is_never_a_conflict(self) -> None:
        assert conflicting_evidence_present([_detail()]) is False
