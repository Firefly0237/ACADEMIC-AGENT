"""参数正本的回归测试：配置默认值必须等于决策记录 §4 表里的数。

这份测试存在的理由不是「测 pydantic 能不能存住数字」，而是**防止参数被顺手改掉**。
§4 里有四项建库前不可逆的参数，改了就是 6–10 小时的全量重灌；还有几项（RRF k、
rerank 输入输出比、HyDE N）是面试里要逐条辩护的数字。它们被谁在哪次提交里悄悄
改成别的值，只有这样一份断言查得出来。

改这里的任何一个期望值，必须同时改决策记录 §4——**两处不一致就是这份测试要拦的东西**。
"""

from __future__ import annotations

import re

import pytest

from citequest.config.settings import Settings


@pytest.fixture
def s() -> Settings:
    return Settings()


class TestIrreversibleIndexParams:
    """建库前不可逆的四项（§4.2 标 ⚠ 的三项 + §4.1 的占位符策略）。"""

    def test_analyzer_is_english(self, s: Settings) -> None:
        # 官方 decision guide 对 English-only 内容明确推荐 english 而非 standard。
        assert s.milvus.text_en_analyzer_params == {"type": "english"}

    def test_bm25_params_are_official_defaults(self, s: Settings) -> None:
        assert s.milvus.bm25_k1 == 1.2
        assert s.milvus.bm25_b == 0.75

    def test_text_exact_exists_and_does_not_carry_bm25(self, s: Settings) -> None:
        # text_exact 挂上 BM25 就变成第三条检索路，RRF 的两路结构就不成立了。
        assert s.milvus.text_exact_enable_match is True
        assert s.milvus.text_exact_attach_bm25 is False

    def test_placeholders_are_stripped(self, s: Settings) -> None:
        # 删而不是替换成被引标题：替换等于把 citrec 的 label 泄漏进索引文本。
        assert s.corpus.strip_placeholders is True


class TestFusionAndGrouping:
    """§4.4 检索与融合。"""

    def test_rrf_k_is_60(self, s: Settings) -> None:
        assert s.retrieval.rrf_k == 60

    def test_group_by_paper_id(self, s: Settings) -> None:
        assert s.retrieval.group_by_field == "paper_id"

    def test_strict_group_size_off(self, s: Settings) -> None:
        # 官方警告 True 在数据分布不均时降性能，而混合粒度语料就是典型的分布不均。
        assert s.retrieval.strict_group_size is False

    def test_search_and_rag_path_budgets(self, s: Settings) -> None:
        assert (s.retrieval.search_group_limit, s.retrieval.search_group_size) == (60, 1)
        assert (s.retrieval.rag_group_limit, s.retrieval.rag_group_size) == (16, 3)
        assert s.retrieval.rag_max_chunks == 48

    def test_eval_path_uses_larger_limit(self, s: Settings) -> None:
        # 评测口径与线上不同，且必须在报告里显式声明。
        assert s.retrieval.eval_group_limit == 100


class TestDerivedEf:
    """§4.2 的 ef 派生式：`ef = max(C × group_limit × group_size, 64)`。

    写死 ef 是真 bug——官方默认行为就是 ef=limit，写死 64 反而低于默认。
    """

    def test_search_path_ef(self, s: Settings) -> None:
        assert s.search_ef == 240  # 4 × 60 × 1

    def test_rag_path_ef(self, s: Settings) -> None:
        assert s.rag_ef == 192  # 4 × 16 × 3

    def test_floor_applies_to_tiny_budgets(self, s: Settings) -> None:
        # 4×1×1=4 低于下界 64，取下界。
        assert s.milvus.ef_for(1, 1) == 64

    def test_ef_never_below_entity_budget(self, s: Settings) -> None:
        for group_limit, group_size in [(60, 1), (16, 3), (100, 1), (24, 2), (12, 4)]:
            assert s.milvus.ef_for(group_limit, group_size) >= group_limit * group_size

    def test_rejects_nonpositive_budget(self, s: Settings) -> None:
        with pytest.raises(ValueError, match="必须为正"):
            s.milvus.ef_for(0, 3)


class TestRerankAndGeneration:
    """§4.5 重排与生成。"""

    def test_rerank_48_to_8(self, s: Settings) -> None:
        assert (s.rerank.input_size, s.rerank.output_size) == (48, 8)

    def test_rerank_normalize_is_on(self, s: Settings) -> None:
        # 对外统一 sigmoid → [0,1]；raw logit 只在内部流通。
        assert s.rerank.normalize is True

    def test_rerank_threshold_is_not_guessed(self, s: Settings) -> None:
        # 阈值必须按 Cohere 官方流程在 30–50 条标定集上标定，标定前就是 None。
        assert s.rerank.score_threshold is None

    def test_search_path_has_no_rerank(self, s: Settings) -> None:
        # 连带后果必须主动承认：RAG 路的指标不能给搜索质量背书。
        assert s.rerank.enabled_on_search_path is False

    def test_hyde_deviates_from_paper_default(self, s: Settings) -> None:
        # 论文默认 N=20，这里是 1，属于明确偏离，面试要主动说。
        assert s.rewrite.hyde_n == 1
        assert s.rewrite.hyde_temperature == 0.3
        assert s.rewrite.hyde_max_tokens == 128

    def test_decompose_does_not_iterate(self, s: Settings) -> None:
        assert s.rewrite.decompose_iterations == 1
        assert s.rewrite.decompose_max_subqueries == 4

    def test_context_is_eight_chunks_with_granularity_annotated(self, s: Settings) -> None:
        assert s.generation.context_max_chunks == 8
        # 不标粒度，模型会把摘要当方法细节来编——本层唯一「不做就静默幻觉」的开关。
        assert s.generation.annotate_granularity_in_prompt is True


class TestGroundingAndAgent:
    """§4.5 / §4.6：纠正循环与 grounding。"""

    def test_recursion_limit_is_explicit(self, s: Settings) -> None:
        assert s.agent.recursion_limit == 8
        assert s.agent.max_correction_rounds == 1

    def test_hhem_runs_once_on_first_sentence(self, s: Settings) -> None:
        assert s.grounding.online_first_sentence_only is True
        assert s.grounding.online_per_edge_enabled is False

    def test_grounding_thresholds_are_uncalibrated(self, s: Settings) -> None:
        # 不沿用余弦时代的 0.7/0.5。未标定时一律 unverified。
        assert s.grounding.supported_threshold is None
        assert s.grounding.weak_threshold is None

    def test_only_deterministic_check_can_block(self, s: Settings) -> None:
        assert s.grounding.hard_gate_on_citation_out_of_range is True

    def test_checkpointer_not_yet_chosen(self, s: Settings) -> None:
        # 后端选定前不得声称已具备持久化能力。
        assert s.agent.checkpointer_backend == "memory"


class TestEmbeddingRoute:
    """§4.3：没有自写推理服务，dense 编码走 TEI + Milvus Function。"""

    def test_provider_is_tei(self, s: Settings) -> None:
        assert s.embedding.provider == "TEI"

    def test_truncate_explicitly_enabled(self, s: Settings) -> None:
        # 官方默认 false，不显式开的话超长文本是报错而不是截断。
        assert s.embedding.truncate is True
        assert s.embedding.truncation_direction == "right"

    def test_batch_size_matches_official_default(self, s: Settings) -> None:
        assert s.embedding.max_client_batch_size == 32

    def test_dim_consistent_across_groups(self, s: Settings) -> None:
        assert s.embedding.dim == s.milvus.dense_dim == 1024

    def test_offline_bgem3_off_by_default(self, s: Settings) -> None:
        # pymilvus[model] 的进程内封装禁止进在线 QPS 路径。
        assert s.embedding.offline_bgem3_enabled is False


class TestLatencyBudgetHonesty:
    """§4.7：全表是预算不是性能，且无一行实测。"""

    def test_measured_flag_is_false(self, s: Settings) -> None:
        # 为 True 之前，禁止在任何对外材料里引用具体毫秒数。
        assert s.latency_budget.measured is False


class TestCrossGroupInvariants:
    """`Settings` 的跨组校验必须真的会拦人。"""

    def test_rejects_dim_mismatch(self) -> None:
        with pytest.raises(ValueError, match="不一致"):
            Settings(embedding={"dim": 768})  # type: ignore[arg-type]

    def test_rejects_rerank_input_exceeding_retrieval(self) -> None:
        with pytest.raises(ValueError, match="超过 RAG 路可带回的 chunk 数"):
            Settings(rerank={"input_size": 64})  # type: ignore[arg-type]

    def test_rejects_half_calibrated_grounding_thresholds(self) -> None:
        with pytest.raises(ValueError, match="同时标定或同时留空"):
            Settings(grounding={"supported_threshold": 0.7})  # type: ignore[arg-type]

    def test_rejects_context_larger_than_rerank_output(self) -> None:
        with pytest.raises(ValueError, match=re.escape("超过 rerank.output_size")):
            Settings(generation={"context_max_chunks": 16})  # type: ignore[arg-type]
