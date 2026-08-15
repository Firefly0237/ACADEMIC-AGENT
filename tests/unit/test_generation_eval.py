"""生成侧评测与报告落盘（M7 §7、§8.3、§14.3）。

两条口径在这里被守住：

1. **citation precision 与 recall 分开报**——只报 precision 会掩盖「大量该引的
   句子根本没引」；分母是**引用边数**，不是标记出现次数。
2. **RAGAS faithfulness 在 statement 数为 0 时未定义**，必须特判——否则会静默
   产生 NaN 或被当成 0 分拉低均值。

报告部分守的是「口径与结果一起落盘」：缺任何一项配置，该次结果就不可比，
而排查掉点时**第一维永远是口径**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from citequest.config.settings import Settings
from citequest.core.answer import AnswerSentence, Citation, CitationAnchor, Evidence
from citequest.core.enums import Granularity
from citequest.evaluation.datasets.schemas import EvalDataset, EvalQuery, Provenance, SubsetStats
from citequest.evaluation.generation_eval import citation_precision_recall, ragas_faithfulness
from citequest.evaluation.judges import JudgeConfig
from citequest.evaluation.metrics import MetricReport
from citequest.evaluation.report import EvalConfigSnapshot, render_markdown, write_report


class KeywordJudge:
    """按关键词判蕴含的桩：premise 里出现 hypothesis 的第一个词就算支持。

    用一个**确定性**的假判官而不是真 MiniCheck：这里要测的是分子分母怎么数，
    而不是模型判得准不准——把一个 770M 的权重拉进单测只会让这条断言变慢且变脆。
    """

    def supports(self, premise: str, hypothesis: str) -> bool:
        head = hypothesis.split()[0].strip(".,")
        return head.lower() in premise.lower()


def _evidence(cid: int, text: str) -> Evidence:
    return Evidence(
        citation_id=cid,
        anchor=CitationAnchor(paper_id=f"W{cid}", chunk_id=f"W{cid}:f:s001:p00001"),
        text=text,
        title=f"Paper {cid}",
        year=2021,
        granularity=Granularity.FULLTEXT,
    )


def _sentence(seq: int, text: str, cids: list[int]) -> AnswerSentence:
    return AnswerSentence(
        seq=seq,
        text=text,
        citations=[
            Citation(citation_id=c, anchor=CitationAnchor(paper_id=f"W{c}", chunk_id=f"W{c}:f:s001:p00001"))
            for c in cids
        ],
    )


class TestCitationQuality:
    def test_precision_counts_edges_not_markers(self) -> None:
        evidences = [_evidence(1, "Rotary embeddings extrapolate well."), _evidence(2, "ALiBi is a bias method.")]
        sentences = [
            _sentence(0, "Rotary embeddings extrapolate well [1].", [1]),
            _sentence(1, "Rotary embeddings are cheap [2].", [2]),
        ]
        quality = citation_precision_recall(sentences, evidences, KeywordJudge())
        assert quality.total_edges == 2
        # 第二句引了 [2]，但 chunk_2 讲的是 ALiBi——**按引用边判定，它不成立**。
        # 若改成「对全部 chunk 取最大相似度」，它会被算成支持。
        assert quality.precision == pytest.approx(0.5)

    def test_recall_counts_sentences_that_carry_a_citation(self) -> None:
        evidences = [_evidence(1, "Rotary embeddings extrapolate well.")]
        sentences = [
            _sentence(0, "Rotary embeddings extrapolate well [1].", [1]),
            _sentence(1, "This is an unsupported closing remark.", []),
        ]
        quality = citation_precision_recall(sentences, evidences, KeywordJudge())
        assert quality.recall == pytest.approx(0.5)

    def test_out_of_range_citation_counts_in_the_denominator_only(self) -> None:
        # 它确实是一条模型声称的引用边，而且必然不成立——计入分母不计入分子。
        evidences = [_evidence(1, "Rotary embeddings extrapolate well.")]
        sentences = [_sentence(0, "Some claim [9].", [])]
        quality = citation_precision_recall(sentences, evidences, KeywordJudge())
        assert quality.total_edges == 1
        assert quality.precision == 0.0

    def test_duplicate_marker_is_one_edge(self) -> None:
        evidences = [_evidence(1, "Rotary embeddings extrapolate well.")]
        sentences = [_sentence(0, "Rotary embeddings extrapolate well [1] and again [1].", [1])]
        assert citation_precision_recall(sentences, evidences, KeywordJudge()).total_edges == 1


class TestFaithfulness:
    def test_zero_statements_is_undefined_not_zero(self) -> None:
        # 未定义与 0 分的来源完全不同：前者是「没有可判的陈述」，
        # 后者是「陈述全部不被支持」。
        result = ragas_faithfulness([], "context", KeywordJudge(), JudgeConfig(model_id="j", kind="llm"))
        assert result.score is None

    def test_score_is_supported_over_total(self) -> None:
        statements = ["Rotary works.", "Nonsense claim."]
        result = ragas_faithfulness(
            statements, "Rotary embeddings extrapolate well.", KeywordJudge(), JudgeConfig(model_id="j", kind="llm")
        )
        assert result.score == pytest.approx(0.5)

    def test_judge_identity_travels_with_the_score(self) -> None:
        # 判官身份（模型 / temperature / prompt 版本）必须跟着分数走，
        # 否则两次评测的差异无法归因。
        judge = JudgeConfig(model_id="gpt-x", kind="llm", prompt_version="v7")
        result = ragas_faithfulness(["Rotary works."], "Rotary works.", KeywordJudge(), judge)
        assert result.judge.prompt_version == "v7"


@pytest.fixture
def dataset() -> EvalDataset:
    return EvalDataset(
        name="unarXive_citrec@in_corpus",
        queries=[EvalQuery(query_id="q1", text="a passage", layer=Granularity.FULLTEXT)],
        qrels={"q1": {"W1": 1}},
        provenance=Provenance(
            dataset_id="saier/unarXive_citrec",
            split="test",
            license="CC-BY-SA-4.0",
            label_source="论文原作者的真实引用行为",
            notes=["每条 query 只有 1 个 label，Recall@100 等价于 Hit@100"],
        ),
        subset_stats=SubsetStats(total_records=1000, kept_records=253, corpus_size=270000),
    )


class TestReport:
    def test_markdown_header_states_every_contested_choice(self, dataset: EvalDataset) -> None:
        snapshot = EvalConfigSnapshot.from_settings(Settings(), dataset_id="saier/unarXive_citrec", split="test")
        report = MetricReport(config_name="hybrid", dataset_name=dataset.name, per_query={"recall@100": {"q1": 1.0}})
        text = render_markdown(report, snapshot, dataset)
        assert "macro-averaged over queries" in text  # 聚合口径
        assert "group_limit=100" in text  # 与线上 60 的唯一差异
        assert "25.30%" in text  # 过滤率必报
        assert "论文原作者" in text  # 标签来源
        assert "Hit@100" in text  # 局限与数字同时讲

    def test_missing_interval_renders_as_na_not_blank(self, dataset: EvalDataset) -> None:
        # 留空看起来像是忘了算，写 n/a 是一个明确的状态。
        snapshot = EvalConfigSnapshot.from_settings(Settings(), dataset_id="d", split="test")
        report = MetricReport(config_name="hybrid", dataset_name=dataset.name, per_query={"mrr@10": {"q1": 1.0}})
        assert "n/a" in render_markdown(report, snapshot, dataset)

    def test_write_report_keeps_per_query_values(self, dataset: EvalDataset, tmp_path: Path) -> None:
        # 只存均值等于让每一次对比都必须重跑一遍检索。
        snapshot = EvalConfigSnapshot.from_settings(Settings(), dataset_id="d", split="test")
        report = MetricReport(config_name="hybrid", dataset_name=dataset.name, per_query={"mrr@10": {"q1": 0.5}})
        json_path, md_path = write_report(report, snapshot, dataset, tmp_path)
        assert json_path.exists() and md_path.exists()
        assert '"per_query"' in json_path.read_text(encoding="utf-8")

    def test_filenames_are_scoped_by_index_version(self, dataset: EvalDataset, tmp_path: Path) -> None:
        # 覆盖掉的那一份正是「掉点之前长什么样」的唯一证据。
        snapshot = EvalConfigSnapshot.from_settings(Settings(), dataset_id="d", split="test")
        report = MetricReport(config_name="hybrid", dataset_name=dataset.name, per_query={"mrr@10": {"q1": 0.5}})
        json_path, _ = write_report(report, snapshot, dataset, tmp_path)
        assert snapshot.index_version in json_path.name
