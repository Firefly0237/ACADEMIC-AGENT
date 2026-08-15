"""`GET /v1/answer`：带引用的 RAG 回答，SSE 流式（M6 §1.1 / §2）。

用 SSE 而不是 WebSocket：这里是纯单向的服务端→客户端推送，不需要双向通道。
SSE 基于普通 HTTP、自带重连语义、经得起任何反向代理，而 WebSocket 要为一个用不到
的双向能力付出协议升级、心跳、代理兼容三份成本。

**事件时序**（三条不变量见 `api.sse`）：

```
t=0     event: meta               ← 立刻，先于一切下游调用
t≈70ms  event: search_results     ← 检索路完成，用户已经有东西看了
        （此时 AI 路在生成首句，前端显示骨架屏，没有任何 AI token 出网）
t≈?     ├─ 过闸 → ai_answer_start → ai_answer_delta(seq=1) …
        └─ 不过闸 → ai_answer_error{INSUFFICIENT_GROUNDING}（一个 token 都没推）
t=end   event: ai_answer_done → event: done
```

**AI 回答超限不返回 429，而是静默降级成纯搜索**（M6 §4.1）：429 会让用户什么都
拿不到；降级让用户仍然拿到搜索结果，只是没有 AI 回答。这与全局降级底线一致——
**搜索结果是底线，AI 回答是增强。**
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from citequest.api.deps import ServicesDep
from citequest.api.ratelimit import client_key
from citequest.api.schemas.answer import AnswerParams
from citequest.api.schemas.common import PaperCard
from citequest.api.sse import error_frame, meta_event, render
from citequest.core.enums import DegradeReason, SseEventType
from citequest.core.errors import CiteQuestError
from citequest.observability.context import current_request_id
from citequest.observability.logging import get_logger
from citequest.pipelines.answer import AnswerCommand
from citequest.pipelines.search import SearchCommand
from citequest.rag.streaming import StreamEvent

__all__ = ["router"]

_logger = get_logger(__name__)

router = APIRouter(tags=["answer"])


@router.get("/v1/answer")
async def answer(
    request: Request,
    services: ServicesDep,
    params: Annotated[AnswerParams, Depends()],
) -> EventSourceResponse:
    """RAG 问答。**整条流永远是 HTTP 200**，失败以 SSE 事件下发。

    这不是风格选择：SSE 一旦发出首帧，状态码已经写进响应行，物理上改不了。
    所以「检索成功但 AI 失败」只能是 200 + `ai_answer_error`。
    """
    request_id = current_request_id() or ""
    ai_allowed = request.app.state.answer_limiter.check(client_key(request)).allowed

    return EventSourceResponse(
        _event_stream(services, params, request_id=request_id, ai_allowed=ai_allowed),
        # 心跳防止中间代理按空闲超时断连。它与 `X-Accel-Buffering: no`
        # （由中间件下发）是两件事：一个防断连，一个防攒批。
        ping=services.settings.streaming.heartbeat_interval_ms // 1000,
    )


async def _event_stream(
    services: ServicesDep,
    params: AnswerParams,
    *,
    request_id: str,
    ai_allowed: bool,
) -> AsyncIterator[dict[str, str]]:
    """事件生成器。

    **不捕获 `asyncio.CancelledError`**：用户关掉页面时，取消信号必须一路传到
    下游、取消整棵任务树并释放它持有的信号量。在中间层用裸 `except Exception`
    吞掉它，会让取消信号在半路消失、生成任务变成僵尸继续占着 GPU——
    这是高并发下最隐蔽的资源泄漏，而且它专挑最贵的资源泄漏。
    """
    settings = services.settings

    # ---- 1. meta：立刻发，先于任何下游调用 ----
    # 它同时是一次「连接确实建立了」的确认：如果中间有代理在缓冲，
    # 问题会在这里立刻暴露，而不是等 3 秒后才发现整条流是哑的。
    yield meta_event(
        request_id=request_id,
        index_version=settings.milvus.index_version,
        route="pending",
        server_time=datetime.now(UTC).isoformat(),
    )

    command = SearchCommand(
        q=params.q,
        page=1,
        page_size=params.page_size,
        granularity=params.granularity,
        year_from=params.year_from,
        year_to=params.year_to,
        primary_category=params.primary_category,
        request_id=request_id,
    )

    # ---- 2. search_results：检索路先出，用户已经有东西看了 ----
    try:
        routing = await services.search.plan_for(command)
        # 把已经跑出来的路由结果传回去：**同一条 query 不路由两次**。
        result = await services.search.run(command, routing=routing)
    except CiteQuestError as exc:
        # **检索挂了没有降级对象**——检索是底线。此时整条流以 error 收尾。
        _logger.warning("answer_search_failed", code=exc.code.value)
        yield error_frame({"error": exc.to_payload(request_id)})
        yield {"event": SseEventType.DONE.value, "data": "{}"}
        return

    yield render(
        StreamEvent(
            type=SseEventType.SEARCH_RESULTS,
            data={
                "group_count": result.group_count,
                "papers": [PaperCard.from_domain(p).model_dump(mode="json") for p in result.papers],
                "retrieval_mode": result.meta.retrieval_mode.value,
                "degraded": [d.value for d in result.meta.degraded],
            },
        )
    )

    # ---- 3. AI 路。超限时静默降级：搜索结果已经发出去了，AI 区域不渲染 ----
    if not ai_allowed:
        _logger.info("answer_rate_limited_degraded", request_id=request_id)
        yield render(
            StreamEvent(
                type=SseEventType.AI_ANSWER_ERROR,
                data={
                    "answer_state": "degraded_no_answer",
                    "reason": "AI answer quota exceeded, search results are still available.",
                    "degraded": [DegradeReason.GENERATION_UNAVAILABLE.value],
                },
            )
        )
        yield {"event": SseEventType.DONE.value, "data": "{}"}
        return

    async for event in services.answer.stream(
        AnswerCommand(
            q=params.q,
            request_id=request_id,
            granularity=params.granularity.value,
            year_from=params.year_from,
            year_to=params.year_to,
            primary_category=params.primary_category,
            language=params.language,
        ),
        routing.plan,
    ):
        yield render(event)
