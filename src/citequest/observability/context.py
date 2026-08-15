"""请求上下文：request_id 的生成、注入与读取。

**一套 ID，贯穿全链路**（M6 §6.5）：`request_id` 同时是 HTTP 响应头、结构化
日志字段、Langfuse trace id、以及用户反馈的关联键。**不引第二套 ID**——两套 ID
意味着每次排查都要先做一次关联，而关联表本身会成为新的故障点。

用 ULID 而不是 UUIDv4：同样 128 位、同样可离线生成，但前 48 位是毫秒时间戳，
所以字典序即时间序，日志检索与分区归档都受益。
"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token

__all__ = [
    "REQUEST_ID_PATTERN",
    "bind_request_id",
    "current_request_id",
    "new_request_id",
    "reset_request_id",
    "sanitize_client_request_id",
]

#: ULID 规范字符集（Crockford Base32，去掉 I/L/O/U），长度固定 26。
REQUEST_ID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")

_request_id: ContextVar[str | None] = ContextVar("citequest_request_id", default=None)


def new_request_id() -> str:
    """生成一个新的 ULID。

    这里用 `ulid` 包而不是自己拼时间戳 + 随机数：单调性、字符集、编码长度
    三件事自己实现都容易出微妙的错，而这个库是标准做法。
    """
    from ulid import ULID

    return str(ULID())


def sanitize_client_request_id(raw: str | None) -> str | None:
    """校验客户端传来的 X-Request-Id。

    **不校验就是一个日志注入面**（M6 §1.3）：这个字段会原样进结构化日志与 trace，
    一个带换行或控制字符的值可以伪造出整行假日志。所以只接受严格的 ULID 形态，
    不合法就返回 None，由调用方自己生成一个——沿用失败不是错误，不必报错。
    """
    if raw is None:
        return None
    candidate = raw.strip()
    if REQUEST_ID_PATTERN.match(candidate):
        return candidate
    return None


def bind_request_id(request_id: str) -> Token[str | None]:
    """把 request_id 绑到当前上下文，返回可用于回滚的 Token。

    用 ContextVar 而不是线程局部变量：FastAPI 的路径函数是 async 的，
    同一个线程会交错执行多个请求，threading.local 会串号。
    """
    return _request_id.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """回滚绑定。中间件必须在 finally 里调用它，否则上下文会泄漏到后续请求。"""
    _request_id.reset(token)


def current_request_id() -> str | None:
    """读取当前 request_id。未绑定时返回 None（例如离线脚本、建库任务）。"""
    return _request_id.get()
