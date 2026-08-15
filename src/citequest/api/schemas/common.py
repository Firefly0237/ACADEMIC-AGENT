"""跨端点共用的响应模型（M6 §1.3）。

与 `core/` 的 domain 模型**刻意解耦**：domain 模型跟着检索与生成的设计走，
API 模型跟着前端契约与向后兼容走。合成一套的代价是每次内部重构都变成一次对外
破坏性变更。

解耦的第一个实际用处就在 `RetrievedChunk` 上：它的 `rrf_score` 是内部字段，
序列化到 API 响应时**必须剔除**——RRF 分数无物理含义（k=60 时上界约 0.033），
暴露出去必然有人拿它做阈值。**响应模型里干脆没有这个字段，比写文档劝阻可靠。**
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from citequest.core.corpus import Attribution
from citequest.core.enums import DegradeReason, Granularity, LicenseId, RetrievalMode, SourceDataset
from citequest.core.search import PaperHit, RetrievedChunk

__all__ = ["ChunkCard", "ErrorBody", "ErrorEnvelope", "MetaModel", "PaperCard"]


class ErrorBody(BaseModel):
    """错误体。**一定有 request_id**——没有它，用户报障时只能说「刚才搜不出来」。"""

    code: str
    message: str
    request_id: str
    retryable: bool
    retry_after_ms: int | None = None


class ErrorEnvelope(BaseModel):
    """所有错误共用一个信封。SSE 的 `error` 事件用的是同一个结构。"""

    error: ErrorBody


class MetaModel(BaseModel):
    """响应元信息。

    `retrieval_mode` 的作用是**把降级度量出来而不是掩盖掉**：线上可以直接统计
    `dense_only` 的占比，而不是等到有人抱怨中文搜不准才去查。
    `degraded` 非空的请求占比同样是「静默失败」监控项之一——降级如果不可观测，
    线上会长期在降级状态下运行而没人知道。
    """

    request_id: str
    index_version: str
    retrieval_mode: RetrievalMode
    took_ms: int
    cache: str = "miss"
    degraded: list[DegradeReason] = Field(default_factory=list)


class ChunkCard(BaseModel):
    """一条命中片段。**没有 rrf_score，也没有 relevance_score。**

    引用锚点是四元组 `(paper_id, section, sec_number, paragraph_idx)`；
    摘要层只有 paper_id，其余三项为 None——**不伪造段落级定位**。
    显示一个不存在的「3.2 节」比不显示更糟：用户拿着它去原文里找不到，
    就会认定整套引用都不可信。
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    section: str | None = None
    sec_number: str | None = None
    paragraph_idx: int | None = None
    text: str

    @classmethod
    def from_domain(cls, chunk: RetrievedChunk) -> ChunkCard:
        return cls(
            chunk_id=chunk.chunk_id,
            section=chunk.section,
            sec_number=chunk.sec_number,
            paragraph_idx=chunk.paragraph_idx,
            text=chunk.text,
        )


class PaperCard(BaseModel):
    """一条论文级结果。

    `license` / `source_dataset` / `attribution` **逐条下发，不做全局声明**：
    全文层是 CC-BY-SA-4.0、摘要层是 CC0，两层混在同一个响应里；在线返回检索片段
    属于许可意义上的分发，需要署名；而汇编（Collection）不是改编（Adaptation），
    所以不能用一句全局混合声明糊过去。这里采取的是保守做法。

    `has_fulltext` 与 `granularity` 冗余是**刻意的**：API 响应直接下发，
    前端不做推导——推导逻辑一旦进前端，它就会在下一次改版里被改错。
    """

    paper_id: str
    title: str
    year: int
    primary_category: str | None = None
    cited_by_count: int | None = None
    granularity: Granularity
    has_fulltext: bool
    best_chunk: ChunkCard
    license: LicenseId
    source_dataset: SourceDataset
    attribution: Attribution

    @classmethod
    def from_domain(cls, hit: PaperHit) -> PaperCard:
        return cls(
            paper_id=hit.paper_id,
            title=hit.title,
            year=hit.year,
            primary_category=hit.primary_category,
            cited_by_count=hit.cited_by_count,
            granularity=hit.granularity,
            has_fulltext=hit.has_fulltext,
            best_chunk=ChunkCard.from_domain(hit.best_chunk),
            license=hit.license,
            source_dataset=hit.source_dataset,
            attribution=hit.attribution,
        )
