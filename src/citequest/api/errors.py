"""领域异常 → HTTP 的映射。**只有这一层知道状态码**（M6 §1.4）。

错误码本身定义在 `core.errors`，因为 MCP adapter 与 FastAPI 共用同一套 service 层：
错误码定义在 api 层就意味着 MCP 那条路要么复制一份，要么静默地行为不一致。
留在本层的只有协议细节——状态码、响应头、信封格式。

三条实现级约定：

1. **`INTERNAL` 绝不回显堆栈或内部服务名**，只回 request_id。堆栈进日志，
   不进响应体——响应体是给公网的。
2. **429 必须同时给出配额头**（`Retry-After` / `X-RateLimit-*`）。只给状态码不给
   恢复时间，客户端只能盲目重试，等于把限流器变成了自我攻击的放大器。
3. **部分失败不用 HTTP 状态码表达**。SSE 一旦发出首帧，状态码已经写进响应行，
   物理上改不了。所以「检索成功但 AI 失败」仍是 200，失败以 SSE 事件下发——
   这也是为什么 SSE 事件表必须自带错误事件，不能指望 HTTP 层。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from citequest.core.errors import CiteQuestError, ErrorCode, InvalidParamError, RateLimitedError
from citequest.observability.context import current_request_id
from citequest.observability.logging import get_logger

__all__ = ["ERROR_HTTP_STATUS", "install_exception_handlers"]

_logger = get_logger(__name__)

#: 错误码 → HTTP 状态码。**这张表是这一层存在的全部理由。**
#: 新增错误码必须同时更新这里和前端状态机，否则会落到 500 这个默认档上。
ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_PARAM: 400,
    ErrorCode.QUERY_EMPTY: 400,
    ErrorCode.QUERY_TOO_LONG: 400,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.SEARCH_UNAVAILABLE: 503,
    ErrorCode.UPSTREAM_TIMEOUT: 504,
    ErrorCode.INTERNAL: 500,
}


def _first_error_message(errors: Sequence[Any]) -> str:
    """只回**第一条**校验失败的位置与原因。

    全量回显会把内部字段名与结构一起摊给公网，那是一份免费的接口内部结构说明书；
    而对客户端来说，一次改一个参数本来就是正确的修法。
    """
    first: dict[str, Any] = dict(errors[0]) if errors else {}
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "query")
    if location:
        return f"参数 {location} 非法：{first.get('msg', 'invalid')}"
    return str(first.get("msg", "请求参数非法"))


def _envelope(error: CiteQuestError, request_id: str) -> JSONResponse:
    """渲染统一错误信封。**信封里一定有 request_id。**"""
    status = ERROR_HTTP_STATUS.get(error.code, 500)
    headers = {"X-Request-Id": request_id}
    if isinstance(error, RateLimitedError) and error.retry_after_ms is not None:
        # 秒级向上取整：`Retry-After` 的单位是秒，向下取整会让客户端在
        # 额度还没恢复时就重试一次，等于把退避变成了一次白跑。
        headers["Retry-After"] = str(max(1, math.ceil(error.retry_after_ms / 1000)))
    return JSONResponse(status_code=status, content={"error": error.to_payload(request_id)}, headers=headers)


def install_exception_handlers(app: FastAPI) -> None:
    """挂四个处理器：领域异常、请求校验异常、模型级校验异常、兜底异常。"""

    @app.exception_handler(CiteQuestError)
    async def _domain_error(request: Request, exc: CiteQuestError) -> JSONResponse:
        request_id = current_request_id() or ""
        _logger.warning("domain_error", code=exc.code.value, path=request.url.path, retryable=exc.retryable)
        return _envelope(exc, request_id)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """FastAPI 的 422 换成本项目的 400 `INVALID_PARAM`。

        统一成一种信封，是因为前端只写一个错误状态机；两种格式意味着前端要先
        判断「这是哪一层的错误」，而那个判断迟早会漏一个分支。
        """
        return _envelope(InvalidParamError(_first_error_message(exc.errors())), current_request_id() or "")

    @app.exception_handler(PydanticValidationError)
    async def _model_validation_error(request: Request, exc: PydanticValidationError) -> JSONResponse:
        """模型级校验（`model_validator`）失败同样翻译成 400。

        **这个处理器不是重复的**：字段级约束（`page` 的 `le=3`）由 FastAPI 包成
        `RequestValidationError`，而参数模型自己的 `model_validator`（年份倒挂）
        是在依赖构造期抛出的**裸 pydantic `ValidationError`**，走不到上面那个处理器。
        少了这一条，「year_from > year_to」会变成 500——一个纯粹的用户参数错误
        被报成服务端故障，客户端还会照着 `retryable` 去重试。
        """
        return _envelope(InvalidParamError(_first_error_message(exc.errors())), current_request_id() or "")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """兜底。**堆栈进日志，不进响应体。**"""
        request_id = current_request_id() or ""
        _logger.error(
            "unhandled_error",
            error_type=type(exc).__name__,
            path=request.url.path,
            # 异常消息本身也不进响应：它经常带着内部主机名、SQL 片段或文件路径。
            error=str(exc),
        )
        internal = CiteQuestError("Internal error. Please report the request_id.")
        return _envelope(internal, request_id)
