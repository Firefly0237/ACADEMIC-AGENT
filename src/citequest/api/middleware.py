"""中间件：request_id 下发、结构化访问日志、SSE 反缓冲头（M6 §1.3 / §2.4）。

**`request_id` 是不可省的一条。** 前端的 👍/👎 反馈按钮回传的就是它；线上一条
trace、一行结构化日志、一棵 span 树用的也是同一个 ID。不下发 request_id，
用户反馈就只剩一个孤立的布尔值，无法回溯到那一次具体的检索与生成——
**反馈闭环在第一步就断了。**

用 ULID 而不是 UUIDv4：同样 128 位、同样可离线生成，但前 48 位是毫秒时间戳，
所以字典序即时间序，日志检索与分区归档都受益。

**响应侧三处同时下发**：HTTP 响应头 `X-Request-Id`、JSON 的 `meta.request_id`、
SSE 的首个 `meta` 事件。三处不是冗余——它们分别服务于代理层排查、前端反馈、
以及流式连接的建连确认。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from citequest.config.api import ApiSettings
from citequest.observability.context import (
    bind_request_id,
    new_request_id,
    reset_request_id,
    sanitize_client_request_id,
)
from citequest.observability.logging import get_logger

__all__ = ["RequestContextMiddleware"]

_logger = get_logger(__name__)

#: 需要关闭代理缓冲的路径前缀。Nginx 默认 `proxy_buffering on`，它会攒够一个
#: buffer 才下发，表现是「前 2 秒什么都没有，然后一次性全出来」——功能看起来是对的，
#: 流式效果完全没有。**应用侧下发这个头比改 Nginx 配置更稳**，因为它跟着代码走，
#: 不依赖某台机器的配置对不对。
_STREAMING_PATHS = ("/v1/answer",)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """绑定 request_id、记访问日志、给流式端点加反缓冲头。"""

    def __init__(self, app: Callable[..., object], settings: ApiSettings) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._settings = settings

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = self._resolve_request_id(request)
        token = bind_request_id(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            # **必须在 finally 里回滚**：ContextVar 不回滚会泄漏到后续请求，
            # 而泄漏的表现是日志里两次请求共用一个 ID——排查时最误导的一种现象。
            reset_request_id(token)

        took_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Request-Id"] = request_id
        if request.url.path.startswith(_STREAMING_PATHS):
            response.headers["X-Accel-Buffering"] = "no"
            # 流式端点上必须关 gzip：压缩器同样会攒缓冲，效果与代理缓冲一模一样。
            response.headers["Cache-Control"] = "no-cache, no-transform"
        _logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            took_ms=took_ms,
        )
        return response

    def _resolve_request_id(self, request: Request) -> str:
        """沿用客户端的 `X-Request-Id`，但**必须校验字符集与长度**。

        这个字段会原样进结构化日志与 trace，**不校验就是一个日志注入面**：
        一个带换行或控制字符的值可以伪造出整行假日志。校验失败不报错——
        沿用失败不是错误，自己生成一个就行。
        """
        if not self._settings.accept_client_request_id:
            return new_request_id()
        return sanitize_client_request_id(request.headers.get("X-Request-Id")) or new_request_id()
