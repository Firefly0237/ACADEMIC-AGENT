"""RAG 链路的 workflow 编排（M4/M5/M6）。

```
meta ──▶ 路由(routing)
          ├── 并行(parallelization) ┬─ HyDE 假设文档生成
          │                          └─ decompose 子问题（多跳路，上限 4、同轮并行 ≤3）
          ├── grouping search 16×3 ≤48 chunk（原 query 恒在首位，RRF 融合）
          ├── search_results 事件先出（用户已经有东西看了）
          ├── rerank 48→8（应用侧，normalize=True）
          ├── [唯一 agentic] 检索纠正循环（graph/，上限 1 次，超时退回首轮）
          ├── 冲突检测（重排之后、上下文组装之前）
          ├── 引用编号分配 → context 组装
          └── 生成 ──▶ 首句阻塞闸门 ──▶ 按句推送 ──▶ 只降级不撤回
```

**这是 workflow 不是 agent**：上面每一条边在写代码时就已画完，运行时 LLM 做的是
分类与生成，不是规划。唯一的例外是那条纠正循环——「这次检索够不够」只有跑完
才知道，它的输入在上一步结束之前物理上不存在。

**降级底线贯穿全文件：只要 Milvus 活着，用户一定看得到搜索结果。**
AI 回答是增强，任何一环出问题都静默降级，绝不把「AI 失败」当成整个请求的失败。
所以本模块里几乎每一个 `except` 后面跟的都是一次 `degraded.append`，不是一次 `raise`。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from citequest.config.settings import Settings
from citequest.core.enums import DegradeReason, PipelineStage, RewriteStrategy, SseEventType
from citequest.core.query_plan import QueryPlan
from citequest.core.search import RetrievedChunk
from citequest.graph.correction_loop import run_correction_loop
from citequest.indexing.ann_requests import GroupingSpec
from citequest.llm.chat import Deadline
from citequest.observability.logging import get_logger
from citequest.observability.spans import span
from citequest.orchestration.conflicts import detect_claim_conflicts
from citequest.orchestration.gates import GateBusy, GateName, GpuGates
from citequest.query.decompose import QueryDecomposer, allocate_rerank_quota
from citequest.query.hyde import HydeGenerator
from citequest.rag.citations import allocate_citation_ids
from citequest.rag.generator import AnswerGenerator, GenerationRequest, build_insufficient_answer
from citequest.rag.grounding import GroundingChecker
from citequest.rag.streaming import AnswerStreamer, StreamEvent
from citequest.retrieval.hybrid import HybridRetriever
from citequest.retrieval.rerank import Reranker

__all__ = ["AnswerCommand", "AnswerPipeline"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class AnswerCommand:
    """RAG 路入参。与 `SearchCommand` 共用同一组参数，**忽略 `page`**（M6 §1.2）。"""

    q: str
    request_id: str = ""
    granularity: str = "any"
    year_from: int | None = None
    year_to: int | None = None
    primary_category: str | None = None
    language: str = "EN"


@dataclass(slots=True)
class _AnswerRun:
    """一次 RAG 请求的可变过程量。

    单独收进一个对象，是因为它们要被四五个私有方法读写；散在方法参数里，
    每加一个降级点就要改一遍所有签名。
    """

    plan: QueryPlan
    degraded: list[DegradeReason] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    started: float = 0.0


class AnswerPipeline:
    """`/v1/answer`（SSE）与 MCP 那条路共用的同一份编排。

    产出的是**语义事件**（`rag.streaming.StreamEvent`），不是 SSE 报文——
    协议封装留在 `api/sse.py`。分开的理由很实际：MCP 那条路也要消费同一批事件，
    而它没有 SSE。
    """

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        reranker: Reranker,
        generator: AnswerGenerator,
        checker: GroundingChecker,
        hyde: HydeGenerator,
        decomposer: QueryDecomposer,
        gates: GpuGates,
        settings: Settings,
        correction_graph: Any | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._checker = checker
        self._hyde = hyde
        self._decomposer = decomposer
        self._gates = gates
        self._settings = settings
        self._correction_graph = correction_graph

    async def stream(self, command: AnswerCommand, plan: QueryPlan) -> AsyncIterator[StreamEvent]:
        """驱动整条 RAG 链路，逐事件产出。

        `plan` 由调用方（`SearchPipeline.plan_for`）传进来而不是这里重跑路由：
        **同一条 query 不路由两次**——两次不仅是纯开销，结果还可能不同，
        那会让同一个 request_id 的 trace 自相矛盾。
        """
        run = _AnswerRun(plan=plan, started=time.perf_counter())

        candidates = await self._retrieve_with_rewrite(command, run)
        run.chunks = candidates
        if not candidates:
            # 零候选不进流式：此时一个 token 都没推，「不展示」是可执行的。
            yield StreamEvent(
                type=SseEventType.AI_ANSWER_ERROR,
                data={"answer_state": "insufficient_evidence", "reason": "检索未返回任何符合过滤条件的结果"},
            )
            return

        top = await self._rerank(command.q, candidates, run)
        top = await self._correct_if_needed(command, run, top)
        evidences = allocate_citation_ids(top)

        # 冲突检测放在**重排之后、上下文组装之前**：更前面是 O(n²) 太贵，
        # 更后面已经推出去的句子撤不回来。判定器未注入时它返回空列表并打点——
        # **不猜冲突**，假阳性的「学界有分歧」比漏报更糟。
        conflicts = detect_claim_conflicts(evidences)
        if conflicts:
            _logger.info("answer_conflicts", groups=len(conflicts))

        request = GenerationRequest(
            query=command.q,
            evidences=evidences,
            route=run.plan.route,
            language=command.language,
        )

        try:
            async with self._gates.acquire(GateName.GENERATE):
                streamer = AnswerStreamer(
                    self._checker,
                    self._settings.streaming,
                    self._settings.grounding,
                    language=command.language,
                )
                deadline = Deadline.after_ms(self._settings.timeouts.first_token_ms)
                tokens = self._generator.stream(request, deadline=deadline)
                async for event in streamer.run(tokens, evidences):
                    yield event
                run.degraded.extend(streamer.degraded)
        except GateBusy as busy:
            # **超限不返回错误，降级成只给搜索结果**（M6 §4.1）：
            # 429 会让用户什么都拿不到；降级让用户仍然拿到搜索结果，只是没有 AI 回答。
            run.degraded.append(DegradeReason.GENERATION_UNAVAILABLE)
            _logger.warning("generation_gate_busy", waited_ms=busy.waited_ms)
            yield StreamEvent(
                type=SseEventType.AI_ANSWER_ERROR,
                data={
                    "answer_state": "degraded_no_answer",
                    "reason": "AI 回答容量已满，搜索结果仍然可用",
                    "fallback": build_insufficient_answer(command.q, evidences, language=command.language),
                },
            )

        yield StreamEvent(
            type=SseEventType.DONE,
            data={
                "took_ms": int((time.perf_counter() - run.started) * 1000),
                "degraded": [d.value for d in run.degraded],
            },
        )

    # ---- 各阶段 ---------------------------------------------------------

    async def _retrieve_with_rewrite(self, command: AnswerCommand, run: _AnswerRun) -> list[RetrievedChunk]:
        """检索 + 改写的**并行**段（parallelization / sectioning）。

        HyDE 与 decompose 的产物都是「要发出去的检索文本列表」，**原 query 恒在
        首位**：它是锚点，改写幻觉时还有一路兜底，且合并同一 chunk 时它的名次优先。

        **不把原 query 与假设文档拼成一条 query 发出去**：拼接会让 BM25 的词频统计
        被假设文档的长度主导，而原 query 那一路存在的意义正是当锚点。
        """
        texts = [run.plan.search_query]
        if run.plan.rewrite_strategy is RewriteStrategy.HYDE:
            # `HydeResult.queries` 自己就保证「恒含原 query 且原 query 在首位」，
            # 所以这里整体替换而不是追加——追加会让原 query 出现两次，
            # 而同一条文本发两遍等于给它的名次白送一次融合优势。
            texts = await self._hyde_texts(command, run) or texts
        elif run.plan.rewrite_strategy is RewriteStrategy.DECOMPOSE:
            texts.extend(await self._decompose_texts(command, run))

        grouping = GroupingSpec.for_rag(self._settings.retrieval)
        with span(PipelineStage.SEARCH, texts=len(texts), group_limit=grouping.group_limit) as sp:
            outcome = await asyncio.to_thread(self._retriever.retrieve, run.plan, grouping, query_texts=texts)
            sp.attrs["chunks"] = len(outcome.chunks)
            sp.attrs["fill_ratio"] = round(outcome.parsed.fill_ratio, 3)
        return outcome.chunks

    async def _hyde_texts(self, command: AnswerCommand, run: _AnswerRun) -> list[str]:
        """HyDE(N=1)。**降级成本为零**——原 query 那一路本来就在跑。

        HyDE 是唯一一个**带存活条件**的组件：它收约 400ms 过路费、占 RAG 路 p50
        预算四成，所以必须在 citrec 库内子集与 LitSearch 上跑赢 direct，跑不赢就下线。
        这条止损判据事先写死，不是事后找补。
        """
        with span(PipelineStage.HYDE) as sp:
            try:
                result = await asyncio.wait_for(
                    self._hyde.generate(command.q, deadline=Deadline.after_ms(self._settings.timeouts.hyde_ms)),
                    timeout=self._settings.timeouts.hyde_ms / 1000,
                )
            except (TimeoutError, NotImplementedError, ConnectionError) as exc:
                run.degraded.append(DegradeReason.HYDE_TIMEOUT)
                _logger.warning("hyde_failed", error_type=type(exc).__name__)
                return []
            sp.attrs["used"] = result.used
        if result.degraded is not None:
            # 降级掉的请求实际上跑的就是 direct，混进 HyDE 组会把它的指标算高——
            # 所以这一条必须落到 `degraded` 里，消融口径才分得开。
            run.degraded.append(result.degraded)
        return result.queries

    async def _decompose_texts(self, command: AnswerCommand, run: _AnswerRun) -> list[str]:
        """一次性子问题拆分，**不迭代**（上限 4）。

        拆分发生在**任何检索之前**，检索结果不能反过来修改拆分——这就是「静态」的
        精确定义，也是它长得像 planner 却仍然是 workflow 的原因。

        同轮并行 ≤3（`agent.parallel_tool_calls_max`）：再高一轮观察摘要就撑爆
        800 token 的上限。这里的并行体现在**子问题共用一次 `retrieve` 的多文本
        合并**，而不是各起一条完整链路——各起一条会让 rerank 输入翻几倍。
        """
        with span(PipelineStage.DECOMPOSE):
            try:
                result = await self._decomposer.decompose(command.q)
            except (NotImplementedError, ConnectionError, TimeoutError) as exc:
                _logger.warning("decompose_failed", error_type=type(exc).__name__)
                return []
        subqueries = result.subqueries[: self._settings.agent.parallel_tool_calls_max]
        quota = allocate_rerank_quota(len(subqueries), rerank_input_size=self._settings.rerank.input_size)
        _logger.info("decomposed", subqueries=len(subqueries), rerank_quota=quota)
        return subqueries

    async def _rerank(self, query: str, chunks: list[RetrievedChunk], run: _AnswerRun) -> list[RetrievedChunk]:
        """48 → 8。**query 必须是原始 query**，不是 HyDE 产物也不是 sub-query。

        闸门满或超时都不是失败：**退回 RRF 名次直出前 8**。用户无感知，只在
        `degraded` 里留一条 `no_rerank`。
        """
        try:
            async with self._gates.acquire(GateName.RERANK):
                with span(PipelineStage.RERANK, candidates=len(chunks)) as sp:
                    outcome = await asyncio.to_thread(
                        self._reranker.rerank, query, chunks, timeout_ms=self._settings.timeouts.rerank_ms
                    )
                    sp.attrs["output"] = len(outcome.chunks)
        except GateBusy:
            run.degraded.append(DegradeReason.RERANK_TIMEOUT)
            _logger.warning("rerank_gate_busy", candidates=len(chunks))
            return chunks[: self._settings.rerank.output_size]

        if outcome.degraded is not None:
            run.degraded.append(outcome.degraded)
        if outcome.all_below_threshold:
            # 全线低于阈值是「信息不足」的触发口之一，**不是空结果**：候选仍然带出去，
            # 信息不足回答里要展示「最相关的是这些，但都不对题」。
            _logger.info("rerank_all_below_threshold", kept=len(outcome.chunks))
        return outcome.chunks

    async def _correct_if_needed(
        self, command: AnswerCommand, run: _AnswerRun, top: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """唯一的 agentic 环节。未装配图时直接跳过——**没有评估器就不该进循环**。"""
        if self._correction_graph is None:
            return top
        outcome = await run_correction_loop(
            self._correction_graph,
            query=command.q,
            plan=run.plan,
            chunks=top,
            settings=self._settings,
        )
        run.degraded.extend(outcome.degraded)
        if not outcome.triggered:
            return top
        # 循环带回来的是**未重排**的候选，必须再过一次 rerank——否则进生成的 8 条
        # 是按 RRF 名次挑的，而 RRF 名次不是相关性顺序。
        return await self._rerank(command.q, outcome.chunks, run)
