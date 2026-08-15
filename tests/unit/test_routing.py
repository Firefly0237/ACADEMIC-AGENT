"""两级路由（M3 §2、§4）：规则快筛的五条规则 + 检索形态 + 降级链。

被测的两条不变量：

1. **搜索结果始终返回**——路由只决定要不要在搜索结果之上再叠一层 AI Answer，
   判错的最坏后果是多了或少了一个 AI Answer；
2. **本层不输出任何权重**——`QueryPlan` 里没有 `dense_weight`/`sparse_weight`，
   需要通道偏置的场景一律表达成「只发一路，不做融合」。

Level 2 分类器通过注入给桩：**降级路径（模型不可用 → 走默认值）是这一层最容易
写错、又最难在集成环境里复现的分支**，它必须能在不起任何模型的情况下被断言。
"""

from __future__ import annotations

from typing import Any

import pytest

from citequest.config.settings import Settings
from citequest.core.enums import DegradeReason, RetrievalMode, RewriteStrategy, RouteType
from citequest.core.errors import InvalidParamError
from citequest.core.query_plan import QuerySignals
from citequest.query.intent import Level2Output, QueryRouter, level1_route
from citequest.query.normalize import extract_signals, normalize_query


def _signals(text: str, **overrides: Any) -> QuerySignals:
    """走**生产代码**的信号抽取，不在测试里另写一份规则。

    这条很重要：如果这里手工构造 `QuerySignals`，测的就只是 `level1_route` 的
    if 链，而词表匹配、token 计数、CJK 判定这些真正会漂的部分完全没被覆盖。
    """
    signals = extract_signals(normalize_query(text))
    return signals.model_copy(update=overrides) if overrides else signals


class TestLevel1Rules:
    """五条规则的优先级顺序——**顺序本身就是被测对象**（按成本排，贵的条件最严）。"""

    def test_quoted_phrase_goes_search_only(self) -> None:
        # 确切短语的语义是「找这个字符串」，不是「找相近的意思」。
        assert level1_route(_signals('"chain of thought" prompting')) is RouteType.SEARCH_ONLY

    def test_year_or_category_becomes_structured(self) -> None:
        # STRUCTURED 不是独立链路（= SEARCH_ONLY + 非空 filter），单列只为让监控
        # 能看见结构化条件的占比。
        assert level1_route(_signals("attention", year_expr=(2020, None))) is RouteType.STRUCTURED
        assert level1_route(_signals("attention", category_terms=["cs.CL"])) is RouteType.STRUCTURED

    def test_very_short_query_without_question_word_is_keyword_search(self) -> None:
        assert level1_route(_signals("rotary position embedding")) is RouteType.SEARCH_ONLY

    def test_comparison_plus_length_goes_multihop(self) -> None:
        # 规则 4 的条件比规则 5 更严：宁可把该多跳的判成单步 QA（回答略浅），
        # 也不要把单步问题判成多跳（多发几路检索 + 更长的合成，成本翻倍）。
        route = level1_route(_signals("how does RoPE compare with ALiBi on long context tasks"))
        assert route is RouteType.SEARCH_PLUS_MULTIHOP

    def test_question_word_plus_length_goes_qa(self) -> None:
        assert level1_route(_signals("what is rotary position embedding used for")) is RouteType.SEARCH_PLUS_QA

    def test_unmatched_query_falls_through_to_level2(self) -> None:
        # 返回 None 是「规则说不准」，不是「没有路由」——它是走 Level 2 的信号。
        assert level1_route(_signals("transformer scaling behaviour under limited data")) is None


class TestRetrievalMode:
    """检索形态：两条分支的结论形式相同——**只发一路，不做融合**。"""

    async def test_cjk_query_degrades_to_dense_only(self) -> None:
        # 中文 query 的 token 与英文文档的 token 不重叠，BM25 命中率结构性接近零。
        # 打点 dense_only 让这件事在监控里可见，而不是被「我们支持多语言」盖住。
        outcome = await QueryRouter(Settings()).route("旋转位置编码在长文本上的效果如何")
        assert outcome.plan.retrieval_mode is RetrievalMode.DENSE_ONLY
        assert outcome.plan.response_language == "ZH"

    async def test_quoted_phrase_goes_sparse_only_with_phrase_match(self) -> None:
        # sparse 单路必须携带 PHRASE_MATCH，否则就退化成「只用 BM25 排序但不做
        # 精确约束」，恰好丢掉了这条路由存在的唯一理由（契约层已有校验）。
        outcome = await QueryRouter(Settings()).route('"lost in the middle" long context')
        assert outcome.plan.retrieval_mode is RetrievalMode.SPARSE_ONLY
        assert outcome.plan.phrase_match is not None

    async def test_plain_english_query_stays_hybrid(self) -> None:
        outcome = await QueryRouter(Settings()).route("what is rotary position embedding used for")
        assert outcome.plan.retrieval_mode is RetrievalMode.HYBRID

    async def test_plan_carries_no_channel_weights(self) -> None:
        # 本层不输出任何权重——融合唯一 RRFRanker(k=60)，通道偏置这个概念不存在。
        outcome = await QueryRouter(Settings()).route("what is rotary position embedding used for")
        assert not {"dense_weight", "sparse_weight"} & set(outcome.plan.model_dump())


class TestLevel2Fallback:
    """Level 2 的降级链：**整体走默认值，不做部分采纳**。"""

    async def test_missing_classifier_degrades_and_is_not_cached(self) -> None:
        # 未注入分类器不是异常状态（SEARCH_ONLY 占多数的部署可以完全不接 Level 2），
        # 但它必须被打点，且**降级结果不进缓存**——缓存一次偶然的模型不可用，
        # 等于把它固化成永久判定。
        outcome = await QueryRouter(Settings()).route("transformer scaling behaviour under limited data")
        assert DegradeReason.ROUTER_FALLBACK in outcome.degraded
        assert outcome.cacheable is False
        assert outcome.plan.route is RouteType.SEARCH_PLUS_QA

    async def test_classifier_exception_falls_back_wholesale(self) -> None:
        class ExplodingClassifier:
            async def classify(self, normalized: str, signals: QuerySignals) -> Level2Output:
                raise RuntimeError("模型不可用")

        router = QueryRouter(Settings(), level2_classifier=ExplodingClassifier())
        outcome = await router.route("transformer scaling behaviour under limited data")
        assert DegradeReason.ROUTER_FALLBACK in outcome.degraded
        assert outcome.plan.classification_level == 2

    async def test_level2_fills_only_the_slots_regex_left_empty(self) -> None:
        # 合并语义：模型抽到的条件只填**正则没填过**的槽位。
        #
        # 顺带记一条链路事实：query 里只要出现年份或类目词面，Level 1 规则 2 就直接
        # 判 STRUCTURED，**根本走不到 Level 2**——所以「同一维度两边都有值」这种
        # 冲突在 `route()` 这条路径上不可能发生。`_merge_level2_filters` 里的
        # 正则优先分支因此是一层防御，不是热路径。
        class Classifier:
            async def classify(self, normalized: str, signals: QuerySignals) -> Level2Output:
                return Level2Output(
                    route=RouteType.SEARCH_PLUS_QA,
                    needs_fulltext=True,
                    year_from=2021,
                    primary_category="cs.CV",
                )

        router = QueryRouter(Settings(), level2_classifier=Classifier())
        outcome = await router.route("scaling behaviour of transformers with limited data")
        assert outcome.plan.classification_level == 2
        assert outcome.plan.filters.year_from == 2021
        assert outcome.plan.filters.primary_category == "cs.CV"
        assert outcome.degraded == []  # 正常出结果的 Level 2 不打降级点


class TestRewriteStrategy:
    """改写策略：`direct` 是所有改写必须打败的基线。"""

    async def test_hyde_is_not_attached_to_the_search_only_path(self) -> None:
        # 搜索路首字节预算 p50 ≤100ms，一次 400ms 的 LLM 调用会把这条路的定位打没。
        outcome = await QueryRouter(Settings()).route("rotary position embedding")
        assert outcome.plan.route is RouteType.SEARCH_ONLY
        assert outcome.plan.rewrite_strategy is RewriteStrategy.DIRECT

    async def test_multihop_route_uses_decompose(self) -> None:
        outcome = await QueryRouter(Settings()).route("how does RoPE compare with ALiBi on long context tasks")
        assert outcome.plan.rewrite_strategy is RewriteStrategy.DECOMPOSE


class TestGuards:
    async def test_empty_query_is_rejected_at_the_router(self) -> None:
        # 长度闸门在 API 层，本模块只拦空 query——两处阈值不一致时，
        # 靠后的那道永远是死代码。
        with pytest.raises(InvalidParamError):
            await QueryRouter(Settings()).route("   ​  ")
