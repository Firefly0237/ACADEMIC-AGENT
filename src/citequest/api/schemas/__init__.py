"""API 请求与响应模型。

与 `core/` 的 domain 模型**刻意解耦**，理由是这两者的变化原因不同：
domain 模型跟着检索与生成的设计走，API 模型跟着前端契约与向后兼容走。
合成一套的代价是每次内部重构都变成一次对外破坏性变更。

解耦的第一个实际用处就在 `RetrievedChunk` 上：它的 `rrf_score` 是内部字段，
序列化到 API 响应时**必须剔除**。RRF 分数无物理含义（k=60 时上界约 0.033），
暴露出去必然有人拿它做阈值。响应模型里干脆没有这个字段，比写文档劝阻可靠。
"""

from citequest.api.schemas.answer import (
    AiAnswerDelta,
    AiAnswerDone,
    AiAnswerError,
    AiAnswerFlagged,
    AiAnswerStart,
    AnswerParams,
    SearchResultsEvent,
    SourceRef,
)
from citequest.api.schemas.common import ChunkCard, ErrorBody, ErrorEnvelope, MetaModel, PaperCard
from citequest.api.schemas.papers import (
    FacetsResponse,
    FeedbackRequest,
    FeedbackResponse,
    PaperDetailResponse,
)
from citequest.api.schemas.search import SearchParams, SearchResponse, SearchResultsBlock

__all__ = [
    "AiAnswerDelta",
    "AiAnswerDone",
    "AiAnswerError",
    "AiAnswerFlagged",
    "AiAnswerStart",
    "AnswerParams",
    "ChunkCard",
    "ErrorBody",
    "ErrorEnvelope",
    "FacetsResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "MetaModel",
    "PaperCard",
    "PaperDetailResponse",
    "SearchParams",
    "SearchResponse",
    "SearchResultsBlock",
    "SearchResultsEvent",
    "SourceRef",
]
