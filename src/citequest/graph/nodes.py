"""纠正循环的三个节点：评估 → 改写 → 重检索（M5 §3.1）。

```
检索完成
   ▼
[grade] 轻量评估器，三档 Correct / Ambiguous / Incorrect
   │ Correct            → 直接进生成
   │ Ambiguous/Incorrect
   ▼
[rewrite] 改写一次
   ▼
[retrieve] 重新检索**本库**
   │
   └─ 仍不足 → 「信息不足」回答（不硬凑）
```

机制层对齐 LangGraph 官方 agentic-rag 教程（grade documents → rewrite question →
retrieve again）；概念层只引 CRAG 的「轻量评估器给三档置信度、不同档触发不同动作」
这一个抽象。**必须声明的偏离**：CRAG 的 Incorrect 分支用大规模 web search 扩展
检索结果，我把论文里最关键的那个分支换掉了——回答必须落在库内可引用的文献上，
引外部网页会直接破坏引用契约。这条偏离声明本身是加分项，它证明读的是论文不是摘要。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from citequest.config.settings import Settings
from citequest.core.enums import PipelineStage, RetrievalQualityGrade
from citequest.core.search import RetrievedChunk
from citequest.graph.state import CorrectionState
from citequest.indexing.ann_requests import GroupingSpec
from citequest.observability.logging import get_logger
from citequest.observability.spans import span
from citequest.retrieval.hybrid import HybridRetriever

__all__ = ["CorrectionNodes", "decide_after_grade"]

_logger = get_logger(__name__)

#: 评估器签名：`(query, chunks) -> 三档`。注入而不是内建，理由与 Level 2 路由
#: 分类器相同——单测要能在不起模型服务的情况下断言全部分支，而**降级路径
#: （评估器不可用时不进循环）恰恰是最难在集成环境里复现的那条**。
GradeFn = Callable[[str, list[RetrievedChunk]], RetrievalQualityGrade]

#: 改写器签名：`(原 query) -> 改写后的检索文本`。
RewriteFn = Callable[[str], str]


class CorrectionNodes:
    """三个节点的宿主。

    做成类而不是三个自由函数，是因为它们共享同一组依赖（检索器、评估器、改写器、
    配置）。LangGraph 的节点只接受 state 一个参数，依赖只能靠闭包或绑定方法带进来
    ——绑定方法比闭包好调试，栈帧上能看见 `self`。
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        settings: Settings,
        *,
        grade_fn: GradeFn | None = None,
        rewrite_fn: RewriteFn | None = None,
    ) -> None:
        self._retriever = retriever
        self._settings = settings
        self._grade_fn = grade_fn
        self._rewrite_fn = rewrite_fn

    async def grade(self, state: CorrectionState) -> CorrectionState:
        """评估这次检索够不够（唯一需要运行时决策的地方）。

        判据只有一条：**这一步的输入，在上一步结束之前是否物理上不存在。**
        「这次检索够不够」只有跑完才知道，改写成什么样取决于跑出来的结果长什么样
        ——全系统只有这一行过关，所以 agentic 的部分只有这一处。

        两条**确定性前置判定**先走，它们不需要模型：
        1. 空结果 → Incorrect；
        2. 评估器不可用 → **判 Correct 直接出循环**。没有信号就不该进循环——
           盲目改写一次是纯延迟，而且它会把一个健康的检索结果换成一个未经验证的。
        """
        chunks = state.get("chunks", [])
        with span(PipelineStage.CORRECTION_LOOP, node="grade", chunks=len(chunks)) as sp:
            if not chunks:
                sp.attrs["grade"] = RetrievalQualityGrade.INCORRECT.value
                return {"grade": RetrievalQualityGrade.INCORRECT}
            if self._grade_fn is None:
                _logger.info("grade_skipped", reason="no_evaluator")
                sp.attrs["grade"] = RetrievalQualityGrade.CORRECT.value
                return {"grade": RetrievalQualityGrade.CORRECT}

            grade = self._grade_fn(state["query"], chunks)
            sp.attrs["grade"] = grade.value
            return {"grade": grade}

    async def rewrite(self, state: CorrectionState) -> CorrectionState:
        """改写一次。**原 query 不动**，改写产物只进 `rewritten_query`。

        改写本身会幻觉（把「它」解成错的论文），所以改写后的 query 与原 query
        **一起**发起检索、用 RRF(k=60) 融合：改错时还有一路兜底。

        改写器不可用时返回原 query——此时下一轮检索与上一轮完全相同，
        `no_progress` 判据会立刻止损，不会白跑第二轮。
        """
        query = state["query"]
        if self._rewrite_fn is None:
            _logger.info("rewrite_skipped", reason="no_rewriter")
            return {"rewritten_query": query}
        rewritten = self._rewrite_fn(query)
        _logger.info("query_rewritten", original_len=len(query), rewritten_len=len(rewritten))
        return {"rewritten_query": rewritten}

    async def retrieve(self, state: CorrectionState) -> CorrectionState:
        """重新检索**本库**（不是 web），并计算「这一轮有没有新增」。

        `no_progress_rounds` 是**确定性止损判据**：连续 N 轮新增有效 chunk 数为 0
        即止损（默认 N=2）。它与调用指纹去重一起构成仅有的两层确定性判据——
        止损不能靠模型自评「我觉得够了」，那是把边界交还给了它自己。
        """
        plan = state["plan"]
        texts = [state["query"]]
        rewritten = state.get("rewritten_query")
        if rewritten and rewritten != state["query"]:
            texts.append(rewritten)

        grouping = GroupingSpec.for_rag(self._settings.retrieval)
        with span(PipelineStage.CORRECTION_LOOP, node="retrieve", texts=len(texts)) as sp:
            outcome = await asyncio.to_thread(self._retriever.retrieve, plan, grouping, query_texts=texts)
            sp.attrs["chunks"] = len(outcome.chunks)

        seen = set(state.get("seen_chunk_ids", []))
        fresh = [c for c in outcome.chunks if c.chunk_id not in seen]
        no_progress = state.get("no_progress_rounds", 0) + (0 if fresh else 1)

        _logger.info(
            "correction_retrieved",
            round=state.get("rounds", 0) + 1,
            total=len(outcome.chunks),
            fresh=len(fresh),
            no_progress_rounds=no_progress,
        )
        return {
            "chunks": outcome.chunks,
            "rounds": state.get("rounds", 0) + 1,
            "seen_chunk_ids": sorted(seen | {c.chunk_id for c in outcome.chunks}),
            "no_progress_rounds": no_progress,
        }


def decide_after_grade(state: CorrectionState, settings: Settings) -> str:
    """条件边：评估之后要不要进纠正（返回 `"rewrite"` 或 `"end"`）。

    **四条止损，任一命中就出循环**，全部是代码级硬约束而不是 prompt 请求：

    1. `grade == Correct` —— 够用了，直接进生成；
    2. `rounds >= max_correction_rounds`（**默认 1**）—— 上界由延迟给死：
       触发一次的增量是 +580ms(p50)/+1,380ms(p95)【估算】，再加一次到 5.4s 量级，
       那已经不是「慢一点」而是换了一个产品形态。下界由收益给死：改写只有一个
       信号源（同一个评估器对同一个库），同一信号源第二次给出实质不同方向的概率
       很低，多跑一轮大概率是把第一次的错误再确认一遍。CRAG 的 Incorrect 分支
       之所以能反复迭代出收益，是因为它换了检索源（外部 web），而这里没换源。
       **所以上限 1 次 = 「给一次改正机会」而不是「让它自己收敛」**——
       收敛需要新信息，而这条循环里没有新信息进来。
    3. `no_progress_rounds >= 2` —— 连续两轮没有新增有效 chunk，确定性止损；
    4. deadline 已过 —— **超时退回首轮检索结果而不是报错**（降级底线：
       永远返回搜索结果，不返回失败）。
    """
    if state.get("grade") is RetrievalQualityGrade.CORRECT:
        return "end"
    if state.get("rounds", 0) >= settings.agent.max_correction_rounds:
        return "end"
    if state.get("no_progress_rounds", 0) >= settings.agent.no_progress_rounds_to_stop:
        _logger.info("correction_stopped", reason="no_progress")
        return "end"
    deadline_at = state.get("deadline_at")
    if deadline_at is not None and time.monotonic() >= deadline_at:
        # 这里**只判不改状态**：LangGraph 的条件边函数只决定走哪条边，
        # 它的返回值不参与状态合并，在这里改 state 是改一个不会被持久化的副本。
        # 降级打点由 `correction_loop.run_correction_loop` 在拿到终态后统一补。
        _logger.warning("correction_stopped", reason="deadline")
        return "end"
    return "rewrite"
