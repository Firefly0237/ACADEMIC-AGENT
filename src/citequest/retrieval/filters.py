"""检索层的过滤表达式装配（M3 §7、M2 §8）。

**表达式的构造在 `indexing.expressions`，不在这里**：那套算子的正确性完全依赖
schema 里的字段名与 analyzer 配置（`PHRASE_MATCH` 作用在 `text_exact` 而不是
`text_en`，因为 stemmer 会让短语匹配与用户直觉不一致）。本模块只做三件检索层
才知道的事：

1. 把 `QueryPlan` 里的过滤条件与**运行期**条件（撤稿黑名单）合并；
2. 撤稿论文在**应用侧**过滤而不是下推——名单是小集合，下推成一个巨大的
   `not in [...]` 会让表达式本身成为瓶颈，而 chunk 上根本没有 `is_retracted` 字段
   （撤稿标记落在论文级元数据，因为它要按 paper_id 精确失效缓存）；
3. 记录过滤选择率，供 M7 那条「不同过滤选择率下的 P99」曲线用。

**加过滤不是免费的**（M3 §7.3）：过滤选择率一旦很低，HNSW 的图导航会走不通并
退化成暴力扫描；官方经验边界是 filter ratio 低于 85% 时图索引仍优于 IVF 系列。
"""

from __future__ import annotations

from citequest.config.milvus import MilvusSettings
from citequest.core.query_plan import QueryPlan
from citequest.core.search import RetrievedChunk
from citequest.indexing.expressions import build_filter_expression
from citequest.observability.logging import get_logger

__all__ = ["apply_retracted_blocklist", "build_plan_filter"]

_logger = get_logger(__name__)


def build_plan_filter(plan: QueryPlan, milvus: MilvusSettings, *, cited_open_alex_id: str | None = None) -> str:
    """`QueryPlan` → 一条 Milvus 过滤表达式。

    短语约束**跟着 plan 走而不是跟着 route 走**：`sparse_only` 与 `phrase_match`
    的一致性已由 `QueryPlan` 的模型校验保证（sparse 单路必须携带 phrase_match，
    否则就退化成「只用 BM25 排序但不做精确约束」，恰好丢掉这条路由存在的唯一理由）。

    :param cited_open_alex_id: 「别人在什么语境下引用 X」的入口（M1 §7.2）。
        它是一次标量等值过滤 + 一次正常混合检索，没有新机制。
    """
    expr = build_filter_expression(
        plan.filters,
        milvus,
        phrase=plan.phrase_match,
        cited_open_alex_id=cited_open_alex_id,
    )
    _logger.debug("filter_built", expr_len=len(expr), granularity=plan.filters.granularity.value)
    return expr


def apply_retracted_blocklist(chunks: list[RetrievedChunk], blocklist: frozenset[str]) -> list[RetrievedChunk]:
    """在应用侧剔除撤稿论文的 chunk。

    **撤稿是确定性信号，不是概率判断**（M4 §5.5）：它不进 prompt、不进证据、
    并且缓存必须按 paper_id 精确失效而不能等 TTL——这是缓存失效纪律里唯一的例外，
    理由是合规。

    这里返回新列表而不是原地删：调用方通常还持有过滤前的候选用于打点
    「本次过滤掉了几条」，原地改会让那个数字消失。
    """
    if not blocklist:
        return chunks
    kept = [c for c in chunks if c.paper_id not in blocklist]
    removed = len(chunks) - len(kept)
    if removed:
        _logger.info("retracted_filtered", removed=removed, remaining=len(kept))
    return kept
