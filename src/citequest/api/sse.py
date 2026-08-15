"""SSE 协议封装：语义事件 → `text/event-stream` 帧（M6 §2）。

`pipelines` 产出的是**语义事件**（`rag.streaming.StreamEvent`），协议封装留在这里。
分开的理由很实际：MCP 那条路也要消费同一批事件，而它没有 SSE。

事件全表（有序、有状态，前端可以按它写一个状态机；**任何一条流都必须以
`done` 或 `error` 收尾**）：

| # | event | 何时发 |
|---|---|---|
| 1 | `meta` | **连接建立后立刻发，先于任何计算** |
| 2 | `search_results` | 检索路完成 |
| 3 | `ai_answer_start` | **首句通过 grounding 闸门之后** |
| 4 | `ai_answer_delta` | 每完成一个**句子** |
| 5 | `ai_answer_flagged` | 某句校验为弱支持时，**在该句已推送之后** |
| 6 | `ai_answer_done` | 生成结束 |
| 7 | `ai_answer_error` | AI 路失败但检索路成功 |
| 8 | `error` | 检索路都失败 |
| 9 | `done` | 流结束（成功或失败都发） |

三条不变量：
- **`meta` 永远是第一个事件，且在任何下游调用之前就发。** 它不只是给 ID：
  它同时是一次「连接确实建立了」的确认。如果中间有代理在缓冲，问题会在这里
  立刻暴露，而不是等 3 秒后才发现整条流是哑的。
- **`ai_answer_start` 出现，就意味着首句已经过闸。**
- **`ai_answer_flagged` 只能降级、不能撤回。** 它永远指向一个已推送的 `seq`。

**明确不做 `Last-Event-ID` 断点续传**（M6 §2.5）：重放要求服务端保存每条流的
已生成内容并能从中间续，而生成是带采样温度的不确定过程——续出来的下半段和断掉的
上半段未必自洽，「续传」会制造一个**从未真实生成过的回答**。所以断连即结束，
前端重连等于重新发起一次请求，靠 L1 缓存兜住重复成本。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from citequest.core.enums import SseEventType
from citequest.observability.logging import get_logger
from citequest.rag.streaming import StreamEvent

__all__ = ["error_frame", "meta_event", "render", "to_sse_stream"]

_logger = get_logger(__name__)


def render(event: StreamEvent) -> dict[str, str]:
    """一个语义事件 → `sse-starlette` 的帧字典。

    `data` 用紧凑分隔符序列化：SSE 的每一帧都要过网络，而回答动辄几十帧，
    每帧省下的空白在长回答上是可观的。
    """
    return {
        "event": event.type.value,
        "data": json.dumps(event.data, ensure_ascii=False, separators=(",", ":")),
    }


def meta_event(*, request_id: str, index_version: str, route: str, server_time: str) -> dict[str, str]:
    """构造首个 `meta` 帧。

    它必须在**任何下游调用之前**发出——包括路由、检索、生成。这一点是可验证的：
    构造它需要的四个字段全部来自请求本身与配置，不依赖任何下游返回值。
    如果哪天有人往这里加一个需要等待的字段，这条不变量就破了。
    """
    return {
        "event": SseEventType.META.value,
        "data": json.dumps(
            {
                "request_id": request_id,
                "index_version": index_version,
                "route": route,
                "server_time": server_time,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


async def to_sse_stream(
    events: AsyncIterator[StreamEvent],
    *,
    first_frames: list[dict[str, str]],
) -> AsyncIterator[dict[str, str]]:
    """把语义事件流包成 SSE 帧流，并保证**一定以 `done` 收尾**。

    `first_frames` 是在任何下游调用之前就已经能构造出来的帧（`meta`，
    以及检索先完成时的 `search_results`）。它们先发，是整条流的建连确认。

    **客户端断连必须让取消信号向上传播**（M6 §2.4 第 3 条）：用户关掉页面，
    SSE 连接断了，但生成任务还在 GPU 上跑到底——这是高并发下最隐蔽的资源泄漏，
    而且它专挑最贵的资源泄漏。所以这里**不捕获 `asyncio.CancelledError`**：
    在中间层用裸 `except Exception` 吞掉它，会让取消信号在半路消失、任务变成僵尸。
    """
    for frame in first_frames:
        yield frame

    saw_terminal = False
    async for event in events:
        if event.type in (SseEventType.DONE, SseEventType.ERROR):
            saw_terminal = True
        yield render(event)

    if not saw_terminal:
        # 兜底的 `done`：前端的状态机以它为终点，缺了它连接会一直挂着直到超时。
        # 这不是防御性编程，是契约的一部分——「任何一条流都必须以 done 或 error 收尾」。
        yield {"event": SseEventType.DONE.value, "data": "{}"}


def error_frame(payload: dict[str, Any]) -> dict[str, str]:
    """检索路都失败时的 `error` 帧，内容与 HTTP 错误信封同构。

    **部分失败不用 HTTP 状态码表达**：SSE 一旦发出首帧，状态码已经写进响应行，
    物理上改不了。所以事件表必须自带错误事件，这个函数就是它的落地。
    """
    return {
        "event": SseEventType.ERROR.value,
        "data": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    }
