"""`/v1/papers/{paper_id}`、`/v1/facets`、`/v1/feedback` 的模型（M6 §1.1 / §7.5）。

三个端点放在一个文件里，是因为它们各自只有一两个模型，且都属于「检索链路之外的
辅助契约」。拆成三个文件只会让读者多点三次。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from citequest.api.schemas.common import ChunkCard
from citequest.core.corpus import Attribution
from citequest.core.enums import Granularity, LicenseId, SourceDataset
from citequest.core.search import ExactFacets
from citequest.pipelines.papers import PaperDetail

__all__ = ["FacetsResponse", "FeedbackRequest", "FeedbackResponse", "PaperDetailResponse"]


class PaperDetailResponse(BaseModel):
    """单篇详情：元数据 + 该篇在库内的 chunk 列表 + 许可署名。"""

    paper_id: str
    title: str
    year: int
    primary_category: str | None = None
    cited_by_count: int | None = None
    granularity: Granularity
    has_fulltext: bool
    license: LicenseId
    source_dataset: SourceDataset
    attribution: Attribution
    chunks: list[ChunkCard] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, detail: PaperDetail) -> PaperDetailResponse:
        return cls(
            paper_id=detail.paper_id,
            title=detail.title,
            year=detail.year,
            primary_category=detail.primary_category,
            cited_by_count=detail.cited_by_count,
            granularity=detail.granularity,
            has_fulltext=detail.has_fulltext,
            license=detail.license,
            source_dataset=detail.source_dataset,
            attribution=detail.attribution,
            chunks=[
                ChunkCard(
                    chunk_id=ref.chunk_id,
                    section=ref.section,
                    sec_number=ref.sec_number,
                    paragraph_idx=ref.paragraph_idx,
                    text=ref.text,
                )
                for ref in detail.chunks
            ],
        )


class FacetsResponse(BaseModel):
    """全库精确分面，**与检索解耦**。

    它单列一个端点而不是塞进 `/v1/search`，是因为它的更新节奏跟着 `index_version`
    走，而不是跟着 query 走——同一个 index_version 下它是常量，前端可以缓存整天。
    """

    facets_exact: ExactFacets | None = None


class FeedbackRequest(BaseModel):
    """👍 / 👎。**body 里必须带 request_id。**

    **没有 request_id 的反馈是废数据**——只知道「有人不满意」，不知道他不满意的是
    哪一次检索、走的哪条路由、有没有降级。服务端把它写进离线数据管道，
    与那次请求的 span 树、检索候选、`degraded` 状态关联，反馈闭环才成立。
    """

    request_id: str = Field(min_length=26, max_length=26, description="ULID，来自响应的 meta.request_id")
    helpful: bool
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    """回执。只回 accepted，不回任何统计——反馈是写入侧，读侧是离线管道。"""

    accepted: bool = True
    request_id: str
