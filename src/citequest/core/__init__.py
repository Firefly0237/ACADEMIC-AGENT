"""领域层：跨模块共享的枚举、数据契约与异常。

这一层**不依赖任何外部服务**（不 import pymilvus / fastapi / torch），
因此可以被 ingestion / indexing / retrieval / rag / api / mcp_server / evaluation
全部直接引用而不产生循环依赖，单元测试也不需要起任何容器。
"""

from citequest.core.answer import (
    AnswerChunk,
    AnswerSentence,
    Citation,
    CitationAnchor,
    Evidence,
    GroundingVerdict,
)
from citequest.core.corpus import Attribution, Chunk, CiteEdge, Paper
from citequest.core.enums import (
    CitationSupport,
    DegradeReason,
    Granularity,
    GranularityRoute,
    LicenseId,
    PipelineStage,
    RetrievalMode,
    RetrievalQualityGrade,
    RewriteStrategy,
    RouteType,
    SecType,
    SortMode,
    SourceDataset,
    SseEventType,
)
from citequest.core.errors import (
    CiteQuestError,
    ErrorCode,
    InvalidParamError,
    RateLimitedError,
    SearchUnavailableError,
    UpstreamTimeoutError,
)
from citequest.core.query_plan import PhraseMatchSpec, QueryFilters, QueryPlan, QuerySignals
from citequest.core.search import (
    ExactFacets,
    InResultFacets,
    PaperHit,
    RetrievedChunk,
    SearchMeta,
    SearchResult,
)

__all__ = [
    "AnswerChunk",
    "AnswerSentence",
    "Attribution",
    "Chunk",
    "Citation",
    "CitationAnchor",
    "CitationSupport",
    "CiteEdge",
    "CiteQuestError",
    "DegradeReason",
    "ErrorCode",
    "Evidence",
    "ExactFacets",
    "Granularity",
    "GranularityRoute",
    "GroundingVerdict",
    "InResultFacets",
    "InvalidParamError",
    "LicenseId",
    "Paper",
    "PaperHit",
    "PhraseMatchSpec",
    "PipelineStage",
    "QueryFilters",
    "QueryPlan",
    "QuerySignals",
    "RateLimitedError",
    "RetrievalMode",
    "RetrievalQualityGrade",
    "RetrievedChunk",
    "RewriteStrategy",
    "RouteType",
    "SearchMeta",
    "SearchResult",
    "SearchUnavailableError",
    "SecType",
    "SortMode",
    "SourceDataset",
    "SseEventType",
    "UpstreamTimeoutError",
]
