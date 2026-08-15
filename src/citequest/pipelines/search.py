"""纯检索链路的 workflow 编排（M6 §1、M2）。

```
query → 路由(routing) → 过滤条件合并 → grouping search(60×1) → 撤稿硬排除
      → 论文级元数据补齐 → 排序模式 → 浅分页切片 → 两类分面 → SearchResult
```

**这条链路上没有 LLM、没有 rerank、没有 grounding 闸门**——它是唯一能讲
「亚百毫秒」的一条路（首字节 p50 ≤100ms / p95 ≤250ms【预算，非实测】）。
一旦往这里加任何一次模型调用，这个定位就没了。

编排形态是 **routing**：路由结果决定走哪一路检索（hybrid / dense_only /
sparse_only）与哪一档 granularity 处置，分支在写代码时就已画完，
运行时 LLM 做的是**分类**不是**规划**——它选一条已经存在的边，不创造新的边。

**Milvus 调用走 `asyncio.to_thread`**：pymilvus 是同步库，直接 await 不了，
而一个同步阻塞调用会让整个事件循环停摆，此时所有在途的 SSE 流一起卡住
（M6 §5.1 第 1 条前提）。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol

from citequest.config.settings import Settings
from citequest.core.corpus import Attribution
from citequest.core.enums import DegradeReason, Granularity, PipelineStage, SortMode
from citequest.core.query_plan import QueryFilters, QueryPlan
from citequest.core.search import ExactFacets, PaperHit, SearchMeta, SearchResult
from citequest.indexing.ann_requests import GroupingSpec
from citequest.observability.logging import get_logger
from citequest.observability.spans import span
from citequest.orchestration.conflicts import exclude_retracted
from citequest.query.intent import QueryRouter, RoutingOutcome
from citequest.retrieval.facets import ExactFacetSource, build_in_result_facets
from citequest.retrieval.grouping import to_paper_hits
from citequest.retrieval.hybrid import HybridRetriever

__all__ = ["NullPaperMetadataStore", "PaperMetadataStore", "SearchCommand", "SearchPipeline"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class SearchCommand:
    """service 层的检索入参。

    **它不是 HTTP 请求模型**：`api/schemas` 里那份跟着前端契约与向后兼容走，
    这一份跟着检索设计走。两者合成一套的代价是每次内部重构都变成一次对外破坏性变更；
    而且 MCP adapter 也构造这个对象，它根本没有 HTTP 概念。
    """

    q: str
    page: int = 1
    page_size: int = 20
    sort: SortMode = SortMode.RELEVANCE
    granularity: Granularity = Granularity.ANY
    year_from: int | None = None
    year_to: int | None = None
    primary_category: str | None = None
    request_id: str = ""
    #: 研究会话继承来的排除条件（`orchestration.research`），已并入过滤器后传进来。
    exclude_papers: frozenset[str] = field(default_factory=frozenset)


class PaperMetadataStore(Protocol):
    """论文级元数据取数口（标题 / 署名 / 引用量 / 撤稿名单）。

    这些字段**不在 chunk schema 里**，因为它们是论文级属性：放进每个 chunk 会在
    147 万行上重复存储，而标题改一次就要重写整篇论文的全部块。
    载体与全库精确分面同一个（DuckDB 读 parquet），不引入 Postgres。
    """

    def fetch_many(self, paper_ids: list[str]) -> dict[str, tuple[str, Attribution, int | None]]: ...

    def retracted(self) -> frozenset[str]: ...


class NullPaperMetadataStore:
    """未接元数据源时的实现：**返回空映射，让 paper_id 当标题显示**。

    `to_paper_hits` 对缺失标题的处理是用 paper_id 占位——**宁可显示 ID 也不显示
    一个编出来的标题**。这条降级路径必须能跑，否则元数据源一挂，整条检索路
    就跟着挂了，而检索是底线。
    """

    def fetch_many(self, paper_ids: list[str]) -> dict[str, tuple[str, Attribution, int | None]]:
        _logger.info("paper_metadata_unavailable", requested=len(paper_ids))
        return {}

    def retracted(self) -> frozenset[str]:
        # 空撤稿名单不是「没有撤稿论文」，是「不知道有哪些」。接真实实现后这里
        # 应当读一份随 index_version 发布的撤稿名单。
        return frozenset()


class SearchPipeline:
    """`/v1/search` 与 MCP `search_papers` 共用的同一份编排（K11）。

    **任何业务规则都不允许写在 route handler 里**，否则 MCP 那条路会静默地行为
    不一致——而不一致的表现是「同一个问题在 Claude 里和在网页里答案不同」，
    这种 bug 没人报得清楚。
    """

    def __init__(
        self,
        router: QueryRouter,
        retriever: HybridRetriever,
        settings: Settings,
        *,
        metadata: PaperMetadataStore | None = None,
        facets: ExactFacetSource | None = None,
    ) -> None:
        self._router = router
        self._retriever = retriever
        self._settings = settings
        self._metadata: PaperMetadataStore = metadata or NullPaperMetadataStore()
        self._facets = facets or ExactFacetSource(settings.milvus.index_version)

    async def run(self, command: SearchCommand, *, routing: RoutingOutcome | None = None) -> SearchResult:
        """一次完整的检索。返回体里**没有 relevance_score 字段**（M6 §1.5）。

        不是忘了给：RRF 只用名次不用分数，k=60 时上界约 2/61≈0.033，这个数没有
        任何物理含义。需要「相关/不相关」判断只能用 rerank 层的分数，
        而搜索路本来就不开 rerank。

        :param routing: 已经跑过的路由结果。RAG 路会先调 `plan_for` 拿 plan，
            再把同一个结果传回来——**同一条 query 不路由两次**：两次不仅是纯开销，
            结果还可能不同，那会让同一个 request_id 的 trace 自相矛盾。
        """
        started = time.perf_counter()
        outcome = routing or await self._route(command)
        plan = outcome.plan
        degraded = list(outcome.degraded)

        hits, group_count = await self._retrieve(plan, degraded)
        hits = self._sort(hits, command.sort)
        page_papers = self._paginate(hits, command)

        exact = await self._exact_facets(command, degraded)
        took_ms = int((time.perf_counter() - started) * 1000)

        return SearchResult(
            meta=SearchMeta(
                request_id=command.request_id,
                index_version=self._settings.milvus.index_version,
                retrieval_mode=plan.retrieval_mode,
                took_ms=took_ms,
                cache="miss",
                degraded=degraded,
            ),
            group_count=group_count,
            page=command.page,
            page_size=command.page_size,
            papers=page_papers,
            facets_in_results=build_in_result_facets(hits),
            facets_exact=exact,
        )

    async def plan_for(self, command: SearchCommand) -> RoutingOutcome:
        """只跑路由，供 RAG 链路复用——**同一条 query 不路由两次**。"""
        return await self._route(command)

    # ---- 各阶段 ---------------------------------------------------------

    async def _route(self, command: SearchCommand) -> RoutingOutcome:
        with span(PipelineStage.ROUTE, query_len=len(command.q)) as sp:
            outcome = await self._router.route(command.q)
            merged = self._merge_user_filters(outcome.plan, command)
            sp.attrs["route"] = merged.route.value
            sp.attrs["retrieval_mode"] = merged.retrieval_mode.value
            sp.attrs["classification_level"] = merged.classification_level
            return RoutingOutcome(
                plan=merged,
                granularity=outcome.granularity,
                degraded=outcome.degraded,
                cacheable=outcome.cacheable,
            )

    def _merge_user_filters(self, plan: QueryPlan, command: SearchCommand) -> QueryPlan:
        """用户显式传的过滤条件**覆盖**路由推断出来的。

        方向不能反：路由是推断，用户参数是明示。让推断覆盖明示，用户就会看到
        「我明明选了只看 2023 年之后」却仍然出现旧论文——而这类 bug 不报错。
        """
        filters = plan.filters
        granularity = command.granularity if command.granularity is not Granularity.ANY else filters.granularity
        merged = QueryFilters(
            granularity=granularity,
            year_from=command.year_from if command.year_from is not None else filters.year_from,
            year_to=command.year_to if command.year_to is not None else filters.year_to,
            primary_category=command.primary_category or filters.primary_category,
            exclude_retracted=filters.exclude_retracted,
        )
        return plan.model_copy(update={"filters": merged})

    async def _retrieve(self, plan: QueryPlan, degraded: list[DegradeReason]) -> tuple[list[PaperHit], int]:
        grouping = GroupingSpec.for_search(self._settings.retrieval)
        with span(PipelineStage.SEARCH, group_limit=grouping.group_limit, group_size=grouping.group_size) as sp:
            # pymilvus 是同步库：不丢进线程池就会阻塞事件循环，把所有在途 SSE 流一起卡住。
            outcome = await asyncio.to_thread(self._retriever.retrieve, plan, grouping)
            sp.attrs["group_count"] = outcome.parsed.group_count
            # grouping 实际返回组数 / 请求组数，是三个「静默失败」指标之一：
            # 凑不满组时不会报错，只会返回更少的组，指标悄悄下滑。
            sp.attrs["fill_ratio"] = round(outcome.parsed.fill_ratio, 3)

        paper_ids = [group[0].paper_id for group in outcome.parsed.groups]
        meta = self._metadata.fetch_many(paper_ids)
        hits = to_paper_hits(
            outcome.parsed,
            titles={pid: value[0] for pid, value in meta.items()},
            attributions={pid: value[1] for pid, value in meta.items()},
            cited_by_counts={pid: value[2] for pid, value in meta.items() if value[2] is not None},
        )
        kept, removed = exclude_retracted(hits, self._metadata.retracted())
        if removed:
            # 撤稿排除会让实际组数少于请求组数，但它不是降级——它是合规拦截。
            _logger.warning("retracted_in_search_result", removed=len(removed))
        return kept, outcome.parsed.group_count

    def _sort(self, hits: list[PaperHit], mode: SortMode) -> list[PaperHit]:
        """三档排序。**引用量与时效不进单一相关性分数**（决策记录 H1）。

        六信号加权公式已被剔除：`max_citations` 取候选集内最大值会让同一篇论文
        在不同 query 下拿到不同引用分，排序不可复现也不可缓存。
        `relevance` 档保持 Milvus 返回的顺序——**本层不重排**。
        """
        if mode is SortMode.CITATIONS:
            return sorted(hits, key=lambda h: h.cited_by_count or 0, reverse=True)
        if mode is SortMode.DATE:
            return sorted(hits, key=lambda h: h.year, reverse=True)
        return hits

    def _paginate(self, hits: list[PaperHit], command: SearchCommand) -> list[PaperHit]:
        """浅分页：**page 分的是论文不是 chunk**，且翻页不重新检索（M6 §1.2）。

        上限 3 页 × 20 篇 = 60 组，正好是一次 grouping search 的 `group_limit`。
        为什么只给 3 页：ANN 返回的是近邻候选而不是稳定全序，limit 一路放大会同时
        损失召回稳定性和延迟；而学术检索里翻不到第 60 篇还没找到，正确的产品动作
        是收窄 query 或加过滤。
        """
        offset = (command.page - 1) * command.page_size
        return hits[offset : offset + command.page_size]

    async def exact_facets(
        self,
        *,
        year_from: int | None = None,
        year_to: int | None = None,
        primary_category: str | None = None,
    ) -> ExactFacets | None:
        """全库精确分面。挂了就给 None，**不影响检索结果本身**。

        它与 `facets_in_results` 是两种东西，所以是两个字段：后者是结果集内的
        近似分布，把它渲染成筛选器计数就是把近似值当精确值展示。

        公开方法而不是私有：`/v1/facets` 端点直接用它——分面的更新节奏跟着
        `index_version` 走而不是跟着 query 走，所以它值得一个独立入口。
        """
        try:
            return await asyncio.to_thread(
                self._facets.fetch,
                year_from=year_from,
                year_to=year_to,
                primary_category=primary_category,
            )
        except NotImplementedError:
            # 数据源未接：这是骨架期的正常状态，不是故障。
            return None
        except Exception as exc:
            _logger.warning("exact_facets_failed", error_type=type(exc).__name__)
            return None

    async def _exact_facets(self, command: SearchCommand, degraded: list[DegradeReason]) -> ExactFacets | None:
        """检索链路里的分面取数：拿不到就打一次降级点。

        降级如果不可观测，线上会长期在降级状态下运行而没人知道——
        指标看起来一切正常，只是筛选器的计数一直是空的。
        """
        facets = await self.exact_facets(
            year_from=command.year_from,
            year_to=command.year_to,
            primary_category=command.primary_category,
        )
        if facets is None:
            degraded.append(DegradeReason.EXACT_FACETS_UNAVAILABLE)
        return facets
