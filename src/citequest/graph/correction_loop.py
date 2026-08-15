"""图装配与执行：`recursion_limit=8`，纠正上限 1 次（M5 §3.2）。

```
START ──▶ grade ──(Correct | 达到上限 | 无进展 | 超时)──▶ END
            │
            └─(Ambiguous/Incorrect)──▶ rewrite ──▶ retrieve ──▶ grade（回环）
```

三条护栏全部是**代码级硬约束**，不依赖模型配合——这正是选 LangGraph 而不是
自由 ReAct 的核心理由：自由 ReAct 的轮数与 token 上限只能靠 prompt 请模型自觉，
而 `recursion_limit` 是框架强制的。

`recursion_limit=8` 为什么要**显式设**：官方默认是 1000（1.0.6 起），离安全边界
差两个数量级。无论官方默认到底是多少，「显式设 8」这个动作本身都成立。
8 的来源是循环体的步数上界：1 轮纠正 = grade + rewrite + retrieve + grade = 4 步，
留一倍余量到 8，多出来的余量用来吸收框架自身的记账步，而不是用来多跑一轮
——多跑一轮由 `max_correction_rounds` 挡住，两道闸各管各的。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from citequest.config.settings import Settings
from citequest.core.enums import DegradeReason
from citequest.core.query_plan import QueryPlan
from citequest.core.search import RetrievedChunk
from citequest.graph.nodes import CorrectionNodes, decide_after_grade
from citequest.graph.state import CorrectionState
from citequest.observability.logging import get_logger

__all__ = ["CorrectionOutcome", "build_correction_graph", "run_correction_loop"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class CorrectionOutcome:
    """循环产物。

    `triggered` 与 `rounds` 必须带出来：**止损判据是事先写死的**
    ——「纠正循环相对静态 pipeline 提升 < 5% → 只保留工具化与 MCP，把循环砍掉」
    （决策记录 K10）。要判这个 5%，就必须能统计循环的触发率与它带来的差异，
    而这两个字段是唯一的数据来源。
    """

    chunks: list[RetrievedChunk]
    rounds: int
    triggered: bool
    degraded: list[DegradeReason]


def build_correction_graph(nodes: CorrectionNodes, settings: Settings) -> Any:
    """装配状态图并编译。

    **不挂 checkpointer**（默认）：官方明确 InMemorySaver 只供开发测试、进程重启
    即丢，而持久化后端**尚未选型**（决策记录 U8）。挂一个内存 checkpointer 会让
    这一层看起来已经具备持久化能力，那是在骗自己——**在后端选定之前，只能说
    「设计上落在官方扩展点」**。要接后端时，改的只是这一行的 `checkpointer=` 参数。
    """
    # 返回值标 Any 而不是 `CompiledStateGraph`：该类的导入路径在 LangGraph 1.x
    # 期间调整过，把它写进签名等于把本模块绑死在一个内部路径上。
    graph = StateGraph(CorrectionState)
    graph.add_node("grade", nodes.grade)
    graph.add_node("rewrite", nodes.rewrite)
    graph.add_node("retrieve", nodes.retrieve)

    graph.add_edge(START, "grade")
    graph.add_conditional_edges(
        "grade",
        lambda state: decide_after_grade(state, settings),
        {"rewrite": "rewrite", "end": END},
    )
    graph.add_edge("rewrite", "retrieve")
    # 回环边：retrieve 之后重新评估。**这是整个系统里唯一一条回边**——
    # 它的存在与否，就是「workflow」与「有一处 agentic」的分界。
    graph.add_edge("retrieve", "grade")

    return graph.compile()


async def run_correction_loop(
    compiled: Any,
    *,
    query: str,
    plan: QueryPlan,
    chunks: list[RetrievedChunk],
    settings: Settings,
) -> CorrectionOutcome:
    """跑一次循环，**超时退回首轮检索结果而不是报错**。

    降级底线在这里体现得最直接：这条路径单独设 5s 超时，超时后返回的是**进来时
    那批 chunk**，不是一个异常。用户看不到任何变化，只是没享受到纠正的收益——
    「永远返回搜索结果，不返回失败」这条底线在图层的落地就是这一段。
    """
    if not settings.agent.correction_loop_enabled:
        return CorrectionOutcome(chunks=chunks, rounds=0, triggered=False, degraded=[])

    budget_ms = settings.agent.correction_loop_timeout_ms
    deadline_at = time.monotonic() + budget_ms / 1000
    initial: CorrectionState = {
        "query": query,
        "plan": plan,
        "chunks": chunks,
        "rounds": 0,
        "no_progress_rounds": 0,
        "seen_chunk_ids": sorted(c.chunk_id for c in chunks),
        "degraded": [],
        "deadline_at": deadline_at,
    }

    try:
        final: CorrectionState = await compiled.ainvoke(
            initial,
            # **显式设置**。官方默认 1000，离安全边界差两个数量级。
            config={"recursion_limit": settings.agent.recursion_limit},
        )
    except Exception as exc:
        # 循环失败不能影响主链路：退回首轮结果并打点。这里捕获宽泛异常是刻意的
        # ——图层的任何故障都不该把一个已经拿到结果的请求变成失败。
        _logger.warning("correction_loop_failed", error_type=type(exc).__name__)
        return CorrectionOutcome(
            chunks=chunks,
            rounds=0,
            triggered=False,
            degraded=[DegradeReason.CORRECTION_LOOP_TIMEOUT],
        )

    degraded = list(final.get("degraded", []))
    rounds = final.get("rounds", 0)
    if time.monotonic() >= deadline_at and DegradeReason.CORRECTION_LOOP_TIMEOUT not in degraded:
        degraded.append(DegradeReason.CORRECTION_LOOP_TIMEOUT)

    result = final.get("chunks") or chunks
    _logger.info(
        "correction_loop_done",
        rounds=rounds,
        triggered=rounds > 0,
        chunks_before=len(chunks),
        chunks_after=len(result),
        degraded=[d.value for d in degraded],
    )
    return CorrectionOutcome(chunks=result, rounds=rounds, triggered=rounds > 0, degraded=degraded)
