"""两级路由与 `QueryPlan` 组装（M3 §2、§4、§8）。

```
query → 归一化/信号（normalize） → 条件抽取（extraction）
      → Level 1 规则快筛（确定性，命中即定）
      → Level 2 小 LLM 兜底（一次调用出四个字段）
      → granularity 路由（granularity.py）
      → QueryPlan
```

**两条不变量**（M3 §1.1）：

1. **搜索结果始终返回。** 路由从不做「搜索 or 问答」的二选一，只决定要不要在
   搜索结果之上再叠一层 AI Answer。判错的最坏后果是多了或少了一个 AI Answer。
2. **本模块不输出任何权重。** `QueryPlan` 里没有 `dense_weight`/`sparse_weight`——
   融合唯一 `RRFRanker(k=60)`，通道偏置这个概念在这套融合器下不存在。
   需要偏置的场景一律表达成「只发一路，不做融合」。

**为什么是两级而不是三级**（M3 §2.4）：中间那层 embedding 锚点分类器的全部辩护
建立在「query embedding 反正检索时也要算，路由复用它边际成本≈0」。这个前提在本
系统里**架构性地不存在**——dense 走 Milvus TEI Embedding Function（Data-in/Data-out），
客户端从头到尾拿不到 query embedding。它赖以成立的免费前提被上游的部署形态抽掉了。
代价是走到 Level 2 的比例升到 30–40%【设计，未实测】。
"""

from __future__ import annotations

from dataclasses import dataclass

from citequest.config.settings import Settings
from citequest.core.enums import DegradeReason, RetrievalMode, RewriteStrategy, RouteType
from citequest.core.errors import InvalidParamError
from citequest.core.query_plan import QueryFilters, QueryPlan, QuerySignals
from citequest.observability.logging import get_logger
from citequest.query.extraction import extract_filters
from citequest.query.granularity import GranularityDecision, decide_granularity
from citequest.query.lexicons import ROUTER_CONFIG_VERSION
from citequest.query.normalize import extract_signals, normalize_query

__all__ = ["Level2Output", "QueryRouter", "RoutingOutcome", "level1_route"]

_logger = get_logger(__name__)

#: Level 1 的三条长度阈值（M3 §2.2）。它们不是相似度魔法常数，是**词数**——
#: 词数是可直接观察的量，不随 query 语言、领域漂移，与 §9 ② 里被删掉的
#: 「> 0.75 判高置信」这类阈值不是一类东西。
_SHORT_QUERY_TOKENS = 3
_QA_MIN_TOKENS = 4
_MULTIHOP_MIN_TOKENS = 6


@dataclass(slots=True)
class Level2Output:
    """Level 2 小 LLM 的一次调用产出四个字段（M3 §2.3）。

    **四件事一起出，不拆成四次调用**：拆开只会让延迟乘以调用数，并引入四个判断
    互相矛盾的可能（判成 SEARCH_ONLY 却又标 needs_fulltext=true）。
    一次调用、一个 schema、一次校验，**输出不合 schema 就整体走默认值，不做部分采纳**。
    """

    route: RouteType
    needs_fulltext: bool
    year_from: int | None = None
    year_to: int | None = None
    primary_category: str | None = None
    rewrite_strategy: RewriteStrategy = RewriteStrategy.DIRECT


@dataclass(slots=True)
class RoutingOutcome:
    """路由结果 + 可观测信息。

    `degraded` 与 `granularity` 单独带出来而不是塞进 `QueryPlan`：前者是运行期事实
    （不进路由缓存，M3 §5.2 第 3 条），后者的 coverage_note 要一路带到生成层的
    prompt 里，而 `QueryPlan` 是给检索层看的参数包。
    """

    plan: QueryPlan
    granularity: GranularityDecision
    degraded: list[DegradeReason]
    cacheable: bool = True
    """降级结果不进缓存——缓存一次偶然的模型不可用，等于把它固化成永久判定。"""


def level1_route(signals: QuerySignals) -> RouteType | None:
    """Level 1 规则快筛：按优先级判定，命中即返回，未命中返回 None（M3 §2.2）。

    **级内规则的顺序同样按成本排**：规则 4（多跳）的条件比规则 5（单步 QA）更严，
    要求比较词**且**长度过线。宁可把一个该多跳的 query 判成单步 QA（回答略浅），
    也不要把单步问题判成多跳（decompose 多发几路检索 + 更长的合成，成本翻倍）。
    """
    # 1. 引号短语 → 纯检索。确切短语的语义是「找这个字符串」，不是「找相近的意思」。
    if signals.phrase_spans:
        return RouteType.SEARCH_ONLY

    # 2. 年份或类目词面命中 → STRUCTURED。它不是独立链路（= SEARCH_ONLY + 非空 filter），
    #    单列只是为了让监控能看见结构化条件的占比。
    if signals.year_expr is not None or signals.category_terms:
        return RouteType.STRUCTURED

    # 3. 极短且无疑问词 → 关键词式检索，不值得叠 AI Answer。
    if signals.token_count <= _SHORT_QUERY_TOKENS and not signals.question_words:
        return RouteType.SEARCH_ONLY

    # 4. 比较词 + 足够长 → 多跳（最贵的一条，所以条件最严）。
    if signals.comparison_words and signals.token_count > _MULTIHOP_MIN_TOKENS:
        return RouteType.SEARCH_PLUS_MULTIHOP

    # 5. 疑问词 + 足够长 → 单步 QA。
    if signals.question_words and signals.token_count > _QA_MIN_TOKENS:
        return RouteType.SEARCH_PLUS_QA

    return None


def _retrieval_mode(signals: QuerySignals) -> RetrievalMode:
    """检索形态（M3 §4）。两条分支的结论形式相同：**只发一路，不做融合**。

    - 引号短语 → sparse 单路：TEXT_MATCH/PHRASE_MATCH 只判命中不给分数，而 RRF
      只用名次不用分数，两者叠在一起就**没有任何机制让「压倒性的精确命中」在排序上
      体现强度**。解法不是把加权融合加回来，而是让这类 query 根本不进融合。
    - 中文 → dense 单路：sparse 是 Milvus 内置 BM25（服务端 Tantivy 分词），
      中文 query 的 token 与英文文档的 token 不重叠，**命中率结构性地接近零**。
      打点 `dense_only` 让「hybrid 在中文上退化成单路」在监控里可见、可度量，
      而不是被一句「我们支持多语言」盖住。
    """
    if signals.phrase_spans:
        return RetrievalMode.SPARSE_ONLY
    if signals.has_cjk:
        return RetrievalMode.DENSE_ONLY
    return RetrievalMode.HYBRID


def _rewrite_strategy(route: RouteType, *, hyde_enabled: bool) -> RewriteStrategy:
    """改写策略（M3 §6.1）。`direct` 是所有改写必须打败的基线，零延迟零成本。

    HyDE 只挂在 SEARCH_PLUS_QA 上，**不挂 SEARCH_ONLY**：搜索路首字节预算是
    p50 ≤100ms，一次 400ms 的 LLM 调用会直接把这条路的定位打没。
    """
    if route is RouteType.SEARCH_PLUS_MULTIHOP:
        return RewriteStrategy.DECOMPOSE
    if route is RouteType.SEARCH_PLUS_QA and hyde_enabled:
        return RewriteStrategy.HYDE
    return RewriteStrategy.DIRECT


class QueryRouter:
    """M3 的对外入口：`route(raw_query) -> RoutingOutcome`。

    Level 2 通过 `level2_classifier` 注入而不是在这里直接建一个 LLM 客户端，
    有两个具体理由：单元测试要能在不起模型服务的情况下断言全部规则分支；
    以及**降级路径必须可测**——「模型不可用时走默认值」是这一层最容易写错、
    又最难在集成环境里复现的分支。
    """

    def __init__(self, settings: Settings, *, level2_classifier: object | None = None) -> None:
        self._settings = settings
        # 注入对象需实现 `classify(normalized: str, signals: QuerySignals) -> Level2Output`。
        # 不用 Protocol 类型标注是为了避免 query 层反向依赖 llm 层（llm 依赖 config，
        # config 不依赖 query，保持这条单向）。
        self._level2 = level2_classifier

    async def route(self, raw_query: str) -> RoutingOutcome:
        """完整的四阶段：归一化 → 抽取 → 分级判定 → 组装。

        **整个方法是 async 的，即使 Level 1 命中时一次 await 都不会发生**：
        路由要么是纯字符串操作，要么带一次 LLM 调用，而调用方（API 与 MCP）
        无法预先知道走哪条。给它两套同步/异步入口只会让降级路径被复制两份。

        长度闸门**不在这里**：单一长度闸门放在 API 层（M3 §8.2），本模块不再设
        第二道——两处阈值不一致时靠后的那道永远是死代码。这里只拦空 query。
        """
        normalized = normalize_query(raw_query)
        if not normalized:
            raise InvalidParamError("query 归一化后为空")

        extraction = extract_filters(normalized, default_slop=self._settings.retrieval.phrase_match_default_slop)
        signals = extract_signals(
            normalized,
            year_expr=extraction.year_expr,
            category_terms=extraction.category_terms,
        )

        degraded: list[DegradeReason] = []
        cacheable = True
        route = level1_route(signals)
        level2: Level2Output | None = None

        if route is None:
            level2, level2_degraded = await self._run_level2(normalized, signals)
            route = level2.route
            if level2_degraded is not None:
                degraded.append(level2_degraded)
                cacheable = False  # 降级结果不进缓存（M3 §5.2 第 3 条）
            classification_level = 2
        else:
            classification_level = 1

        filters = extraction.filters
        if level2 is not None:
            filters = self._merge_level2_filters(filters, level2)

        response_language = "ZH" if signals.has_cjk else "EN"
        granularity = decide_granularity(
            signals,
            filters,
            needs_fulltext_hint=level2.needs_fulltext if level2 else None,
            response_language=response_language,
        )
        filters = filters.model_copy(update={"granularity": granularity.granularity})

        mode = _retrieval_mode(signals)
        strategy = (
            level2.rewrite_strategy
            if level2 is not None and level2.rewrite_strategy is not RewriteStrategy.DIRECT
            else _rewrite_strategy(route, hyde_enabled=self._settings.rewrite.hyde_enabled)
        )

        plan = QueryPlan(
            original_query=raw_query,
            search_query=extraction.search_query,
            route=route,
            retrieval_mode=mode,
            granularity_route=granularity.route,
            filters=filters,
            phrase_match=extraction.phrase_match,
            rewrite_strategy=strategy,
            response_language=response_language,
            classification_level=classification_level,
            router_config_version=ROUTER_CONFIG_VERSION,
            detected_language="zh" if signals.has_cjk else "en",
        )

        _logger.info(
            "query_routed",
            route=route.value,
            retrieval_mode=mode.value,
            granularity_route=granularity.route.value,
            classification_level=classification_level,
            rewrite=strategy.value,
            router_config_version=ROUTER_CONFIG_VERSION,
        )
        return RoutingOutcome(plan=plan, granularity=granularity, degraded=degraded, cacheable=cacheable)

    async def _run_level2(self, normalized: str, signals: QuerySignals) -> tuple[Level2Output, DegradeReason | None]:
        """Level 2 兜底，含失败降级（M3 §2.3 / §8.2）。

        默认值是 `SEARCH_PLUS_QA + needs_fulltext=false + 无 filter + direct`，
        **偏体验而不是偏成本**：宁可多生成一个 AI Answer。代价是这部分模糊 query
        会触发 LLM 成本；监控项「触发了 AI Answer 但用户没展开」用来反推该不该
        把默认值翻转成偏成本的一侧。
        """
        fallback = Level2Output(route=RouteType.SEARCH_PLUS_QA, needs_fulltext=False)
        if self._level2 is None:
            # 未注入分类器不是异常状态：SEARCH_ONLY 占多数的部署可以完全不接 Level 2。
            return fallback, DegradeReason.ROUTER_FALLBACK

        classify = getattr(self._level2, "classify", None)
        if classify is None:  # pragma: no cover - 注入对象契约错误应在装配期发现
            raise TypeError("level2_classifier 必须实现 classify(normalized, signals) -> Level2Output")
        try:
            return await classify(normalized, signals), None
        except Exception as exc:
            # 模型不可用或输出不合 schema → **整体走默认值，不做部分采纳**。
            _logger.warning("level2_router_fallback", error_type=type(exc).__name__)
            return fallback, DegradeReason.ROUTER_FALLBACK

    @staticmethod
    def _merge_level2_filters(filters: QueryFilters, level2: Level2Output) -> QueryFilters:
        """把 Level 2 抽到的条件并进正则抽到的条件。

        **正则的结果优先**：正则命中是确定性的，模型输出是概率性的。同一个维度
        两边都有值时保留正则那份，避免模型把一个它「觉得更合理」的年份覆盖掉
        用户明确写出的那个。
        """
        update: dict[str, object] = {}
        if filters.year_from is None and level2.year_from is not None:
            update["year_from"] = level2.year_from
        if filters.year_to is None and level2.year_to is not None:
            update["year_to"] = level2.year_to
        if filters.primary_category is None and level2.primary_category:
            update["primary_category"] = level2.primary_category
        return filters.model_copy(update=update) if update else filters
