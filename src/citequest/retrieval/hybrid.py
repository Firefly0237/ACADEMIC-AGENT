"""混合检索的执行：两路 ANN + RRF + grouping（决策记录 D6/D9、M2 §12）。

```
QueryPlan
   ├─ 过滤表达式（retrieval.filters）
   ├─ 1–2 路 AnnSearchRequest（indexing.ann_requests）
   │     dense : HNSW/COSINE, ef = max(4×gl×gs, 64)
   │     sparse: SPARSE_INVERTED_INDEX/BM25
   ├─ hybrid_search(ranker=RRFRanker(60), group_by_field="paper_id")
   └─ 解析成 PaperHit（retrieval.grouping）
```

**融合器唯一 `RRFRanker(k=60)`**。没有 WeightedRanker、没有权重字段——需要通道
偏置的场景改为**单路不融合**（中文 dense_only、引号短语 sparse_only）：单路下
量纲问题根本不存在，所以不需要一个加权器去解决一个可以避免制造的问题。

**HyDE 的两路在这一层合并**：原 query 与假设文档各跑一次完整检索，结果按
chunk_id 去重、保留最高名次。不把两段文本拼成一条 query 发出去——拼接会让 BM25
的词频统计被假设文档的长度主导，而**原 query 那一路存在的意义正是当锚点**。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from citequest.config.settings import Settings
from citequest.core.enums import RetrievalMode
from citequest.core.query_plan import QueryPlan
from citequest.core.search import RetrievedChunk
from citequest.indexing.ann_requests import GroupingSpec, build_ann_requests
from citequest.indexing.client import MilvusIndexClient
from citequest.observability.logging import get_logger
from citequest.retrieval.filters import build_plan_filter
from citequest.retrieval.grouping import ParsedGroups, parse_groups

__all__ = ["HybridRetriever", "RetrievalOutcome", "merge_by_best_rank"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievalOutcome:
    """一次检索的完整产物。

    `parsed` 与 `chunks` 都留着：前者带组结构（搜索路要按论文分页），
    后者是展平的候选（RAG 路要送 rerank）。**不让调用方自己展平**——
    展平顺序决定了 rerank 输入顺序，而 rerank 之后的 8 条要按分数重排，
    两处顺序语义不同，混起来最容易写出「按 RRF 顺序当成相关性顺序」的 bug。
    """

    parsed: ParsedGroups
    chunks: list[RetrievedChunk]
    mode: RetrievalMode
    filter_expr: str
    channels: int = 1
    subquery_of: dict[str, list[int]] = field(default_factory=dict)
    """chunk_id → 命中它的检索文本序号（0 = 原 query）。decompose 路按子问题
    均分 rerank 配额时要用它（M4 §8.3），单 query 路留空。"""


class HybridRetriever:
    """检索链路的执行体。

    它依赖 `MilvusIndexClient` 而不是自己建连接：客户端层负责连接、alias、
    startup probe；这一层负责「一次 query 该发几路、带什么过滤、结果怎么解析」。
    分层的判据是**能不能在没有 Milvus 的单元测试里被断言**——请求装配与结果解析能，
    连接不能。
    """

    def __init__(self, client: MilvusIndexClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def retrieve(
        self,
        plan: QueryPlan,
        grouping: GroupingSpec,
        *,
        query_texts: list[str] | None = None,
    ) -> RetrievalOutcome:
        """执行一次（或几路合并的）混合检索。

        :param query_texts: 要发出去的检索文本。默认只发 `plan.search_query`；
            HyDE 传 `[原 query, 假设文档]`，decompose 传子问题列表。
            **原 query 恒在首位**，合并时它的名次优先。
        """
        texts = query_texts or [plan.search_query]
        if not texts:  # pragma: no cover - 防御：空列表会静默返回空结果
            raise ValueError("query_texts 不能为空")

        filter_expr = build_plan_filter(plan, self._settings.milvus)
        text_field = self._settings.milvus.text_en_field

        merged: dict[str, RetrievedChunk] = {}
        origins: dict[str, list[int]] = {}
        group_lists: list[list[RetrievedChunk]] = []
        channels = 0

        for index, text in enumerate(texts):
            requests = build_ann_requests(
                text,
                self._settings.milvus,
                grouping,
                mode=plan.retrieval_mode,
                filter_expr=filter_expr,
            )
            channels = max(channels, len(requests))
            raw = self._client.hybrid_search(requests, grouping)
            # pymilvus 每条 query 文本返回一个结果列表；这里一次只发一条文本，
            # 所以取第一维。多文本靠**多次调用**而不是一次多 query——
            # 一次多 query 下 grouping 的语义官方未给，不赌未核实的行为。
            rows = list(raw[0]) if raw else []
            parsed = parse_groups(
                rows,
                text_field=text_field,
                group_by_field=self._settings.retrieval.group_by_field,
                requested_groups=grouping.group_limit,
            )
            if index == 0:
                group_lists = parsed.groups
            for chunk in parsed.chunks:
                origins.setdefault(chunk.chunk_id, []).append(index)
                merged[chunk.chunk_id] = merge_by_best_rank(merged.get(chunk.chunk_id), chunk)

        outcome = RetrievalOutcome(
            parsed=ParsedGroups(groups=group_lists, requested_groups=grouping.group_limit),
            chunks=list(merged.values()),
            mode=plan.retrieval_mode,
            filter_expr=filter_expr,
            channels=channels,
            subquery_of=origins if len(texts) > 1 else {},
        )
        _logger.info(
            "hybrid_retrieved",
            texts=len(texts),
            channels=channels,
            groups=outcome.parsed.group_count,
            chunks=len(outcome.chunks),
            fill_ratio=round(outcome.parsed.fill_ratio, 3),
            mode=plan.retrieval_mode.value,
        )
        return outcome


def merge_by_best_rank(existing: RetrievedChunk | None, incoming: RetrievedChunk) -> RetrievedChunk:
    """同一 chunk 被多路命中时保留「更好的那一条」。

    **判据是 RRF 分数更高**，而这里用它是合法的：M6 §1.5 禁止的是拿 RRF 分数做
    **相关性阈值**（它无物理含义、跨请求不可比），而同一次检索内部比较两条记录的
    名次是它唯一正当的用途——RRF 本来就是一个名次融合分。

    合并而不是保留先到的那条，是因为 HyDE 那一路可能把某个 chunk 顶到更前面，
    丢掉这个信息等于让 HyDE 的收益在去重时被抹掉。
    """
    if existing is None:
        return incoming
    if (incoming.rrf_score or 0.0) > (existing.rrf_score or 0.0):
        return incoming
    return existing
