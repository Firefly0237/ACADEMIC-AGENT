"""`POST /v1/feedback`：👍 / 👎 的反馈闭环（M6 §7.5）。

**body 里必须带 request_id。** 服务端把它写进离线数据管道，与那次请求的 span 树、
检索候选、`degraded` 状态关联。**没有 request_id 的反馈是废数据**——只知道
「有人不满意」，不知道他不满意的是哪一次检索、走的哪条路由、有没有降级。

这个端点是整条可观测链路的**闭合点**：`request_id` 同时是 HTTP 响应头、
结构化日志字段、trace id 和这里的关联键。**不引第二套 ID**——两套 ID 意味着
每次排查都要先做一次关联，而关联表本身会成为新的故障点。
"""

from __future__ import annotations

from fastapi import APIRouter

from citequest.api.schemas.papers import FeedbackRequest, FeedbackResponse
from citequest.core.errors import InvalidParamError
from citequest.observability.context import REQUEST_ID_PATTERN
from citequest.observability.logging import get_logger

__all__ = ["router"]

_logger = get_logger(__name__)

router = APIRouter(tags=["feedback"])


@router.post("/v1/feedback", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    """收下一条反馈。

    **必须校验 request_id 的形态再落日志**：这个字段会原样进结构化日志与离线管道，
    一个带换行或控制字符的值可以伪造出整行假日志。这与中间件对客户端
    `X-Request-Id` 的处理是同一条纪律，只是这里的值来自 body 而不是 header。

    校验失败抛 400 而不是静默丢弃：静默丢弃会让前端以为反馈提交成功了，
    而实际上什么都没记下——反馈闭环最怕的就是这种「看起来在工作」。
    """
    if not REQUEST_ID_PATTERN.match(payload.request_id):
        raise InvalidParamError("request_id 必须是响应里返回的 ULID（26 位 Crockford Base32）")

    _logger.info(
        "user_feedback",
        target_request_id=payload.request_id,
        helpful=payload.helpful,
        has_comment=payload.comment is not None,
    )
    # TODO(接真实实现): 写入离线数据管道（一行 JSONL 落盘 / 消息队列），
    #   与该 request_id 的 span 树、检索候选明细、degraded 状态关联。
    #   这里刻意不写库——反馈是写入侧，读侧完全在离线，在线路径上不该多一个数据库。
    return FeedbackResponse(accepted=True, request_id=payload.request_id)
