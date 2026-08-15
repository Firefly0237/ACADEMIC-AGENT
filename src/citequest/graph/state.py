"""检索纠正循环的图状态（M5 §3）。

用 `TypedDict` 而不是 pydantic 模型：LangGraph 的状态在节点之间是**部分更新**的
——节点返回一个只含改动字段的 dict，框架负责合并。pydantic 模型在这个语义下
每次都要构造完整对象，反而更容易写出「忘了带上某个字段导致它被重置」的 bug。

状态里刻意**没有** `messages` 字段：这条循环不是对话式 agent，节点之间传的是
检索结果与判定，不是聊天记录。加一个 messages 会让人以为可以往里塞自由对话，
而那正是「窄循环」被撑成自由 ReAct 的第一步。
"""

from __future__ import annotations

from typing import TypedDict

from citequest.core.enums import DegradeReason, RetrievalQualityGrade
from citequest.core.query_plan import QueryPlan
from citequest.core.search import RetrievedChunk

__all__ = ["CorrectionState"]


class CorrectionState(TypedDict, total=False):
    """一次纠正循环的全部状态。

    `total=False` 是因为节点只返回改动字段——把它写成 `total=True` 会让每个节点
    的返回值都要凑齐全部字段，那是把「部分更新」这个机制手工退化掉。
    """

    #: 原始 query。**永远不被改写覆盖**——改写产物放 `rewritten_query`，
    #: 因为最终要回答的是原问题，rerank 与生成都必须拿原 query 当排序目标。
    query: str
    #: 检索参数包（M3 的唯一产物）。改写只换 `search_query`，路由判定不重跑：
    #: 同一条 query 路由两次是纯开销，而且两次结果可能不同，那会让 trace 对不上。
    plan: QueryPlan
    #: 当前轮次的候选 chunk。
    chunks: list[RetrievedChunk]
    #: 评估器给的三档（仅概念层引 CRAG，**已声明偏离**：CRAG 的 Incorrect 分支是
    #: web search，本项目不做——引外部网页会直接破坏引用契约）。
    grade: RetrievalQualityGrade
    #: 已经跑过的纠正轮数。**上限 1**，不是靠 prompt 请模型自觉，是代码级硬约束。
    rounds: int
    #: 改写后的检索文本。原 query 仍在 `query` 里，两路一起发起检索后 RRF 融合——
    #: 改写本身会幻觉，保留原 query 做锚点，改错时还有一路兜底。
    rewritten_query: str
    #: 连续多少轮没有新增有效 chunk。与调用指纹去重一起构成仅有的两层确定性止损判据。
    no_progress_rounds: int
    #: 已见过的 chunk_id，用来算「有没有新增」。**这是确定性判据，不依赖模型自评**。
    seen_chunk_ids: list[str]
    #: 循环内产生的降级。超时退回首轮结果时打 `CORRECTION_LOOP_TIMEOUT`。
    degraded: list[DegradeReason]
    #: 绝对截止时间戳（`time.monotonic()` 口径）。deadline 逐层传递，
    #: 每层发起下游调用前先看剩余预算——没有这条，「每层 2s」在四层链路上累积成 8s。
    deadline_at: float
