"""领域异常与错误码。

错误码放在 core 而不是 api，是因为 MCP adapter 与 FastAPI 共用同一套 service 层
（K11）：错误码定义在 api 层就意味着 MCP 那条路要么复制一份，要么静默地行为不一致。
HTTP 状态码映射留在 `citequest.api.errors`——那是协议细节，不是领域概念。

错误契约见 M6 §1.4。三条实现级约定：
1. `INTERNAL` 绝不回显堆栈或内部服务名，只回 request_id（响应体是给公网的）；
2. 429 必须同时给出配额头，只给状态码不给恢复时间会把限流器变成自我攻击的放大器；
3. **部分失败不用 HTTP 状态码表达**——SSE 一旦发出首帧，状态码已经写进响应行，
   物理上改不了，所以检索成功但 AI 失败仍是 200，失败以 SSE 事件下发。
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "CiteQuestError",
    "ErrorCode",
    "InvalidParamError",
    "RateLimitedError",
    "SearchUnavailableError",
    "UpstreamTimeoutError",
]


class ErrorCode(StrEnum):
    """错误码全集（M6 §1.4）。新增错误码必须同时更新前端状态机。"""

    INVALID_PARAM = "INVALID_PARAM"
    QUERY_EMPTY = "QUERY_EMPTY"
    QUERY_TOO_LONG = "QUERY_TOO_LONG"
    RATE_LIMITED = "RATE_LIMITED"
    SEARCH_UNAVAILABLE = "SEARCH_UNAVAILABLE"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    INTERNAL = "INTERNAL"


class CiteQuestError(Exception):
    """所有领域异常的基类。

    `retryable` 是契约的一部分而不是提示：客户端据它决定退避重试还是直接改参数。
    把它放在异常上（而不是让 api 层按 code 查表）是为了让 MCP adapter 不必
    再维护一份同样的表。
    """

    code: ErrorCode = ErrorCode.INTERNAL
    retryable: bool = False

    def __init__(self, message: str, *, retry_after_ms: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after_ms = retry_after_ms

    def to_payload(self, request_id: str) -> dict[str, object]:
        """渲染成 M6 §1.4 的错误信封。**信封里一定有 request_id。**"""
        payload: dict[str, object] = {
            "code": self.code.value,
            "message": self.message,
            "request_id": request_id,
            "retryable": self.retryable,
        }
        if self.retry_after_ms is not None:
            payload["retry_after_ms"] = self.retry_after_ms
        return payload


class InvalidParamError(CiteQuestError):
    """参数越界。`page>3`、`page_size>20`、枚举非法都归这里。

    `q` 超长返回 400 而不是截断：截断会让用户看到的输入与实际执行的 query 不一致，
    而且截断后的 query 会以截断形态进缓存 key，同一段长文本的不同前缀可能命中同一条缓存。
    """

    code = ErrorCode.INVALID_PARAM
    retryable = False


class RateLimitedError(CiteQuestError):
    """令牌桶或并发闸门拒绝。必须携带 retry_after_ms。"""

    code = ErrorCode.RATE_LIMITED
    retryable = True


class SearchUnavailableError(CiteQuestError):
    """Milvus 不可用或超时。**这条没有降级对象**——检索是底线，检索挂了就没有兜底（K9）。"""

    code = ErrorCode.SEARCH_UNAVAILABLE
    retryable = True


class UpstreamTimeoutError(CiteQuestError):
    """下游超出本层 deadline。

    超时语义全系统只有一种（决策记录 §4.9）：已产生内容一律保留、不足部分标注；
    不因超时丢弃已生成内容，不用断连表达超时。两种语义会让前端无法写出一致的状态机。
    """

    code = ErrorCode.UPSTREAM_TIMEOUT
    retryable = True
