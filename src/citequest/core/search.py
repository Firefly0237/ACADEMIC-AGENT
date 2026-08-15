"""检索侧数据契约：RetrievedChunk / PaperHit / SearchResult / Facets。

一条硬约束写进类型里（M6 §1.5）：**RRF 分数不对外暴露，也不允许任何上游拿它
做相关性阈值。** RRF 只用名次不用分数，k=60 时上界约 2/61≈0.033，这个数没有
任何物理含义。所以 `PaperHit` 上没有 relevance_score 字段——不是忘了给，是不能给。
需要「相关/不相关」判断只能用 rerank 层的分数（对外统一 normalize=True）。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from citequest.core.corpus import Attribution
from citequest.core.enums import DegradeReason, Granularity, LicenseId, RetrievalMode, SecType, SourceDataset

__all__ = [
    "ExactFacets",
    "InResultFacets",
    "PaperHit",
    "RetrievedChunk",
    "SearchMeta",
    "SearchResult",
]


class RetrievedChunk(BaseModel):
    """一条召回的 chunk。

    `rrf_score` 只允许出现在**内部**链路上（用于调试与评测复现），序列化到
    API 响应时必须剔除——见 M6 §1.5。`rerank_score` 才是唯一可以做阈值的分数。
    """

    chunk_id: str
    paper_id: str
    text: str
    granularity: Granularity
    section: str | None = None
    sec_number: str | None = None
    sec_type: SecType | None = None
    paragraph_idx: int | None = None
    year: int
    primary_category: str | None = None
    license: LicenseId
    source_dataset: SourceDataset
    rrf_score: float | None = Field(
        default=None,
        description="内部字段。RRF 分数无物理含义，禁止做阈值，禁止出现在对外响应里（M6 §1.5）",
    )
    rerank_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="bge-reranker-v2-m3 的 sigmoid 归一化分（normalize=True）。这是全系统唯一可做阈值的相关性分数",
    )
    dense_rank: int | None = Field(default=None, ge=1, description="dense 单路名次，用于复现 RRF 与做通道归因")
    sparse_rank: int | None = Field(default=None, ge=1, description="BM25 单路名次")


class PaperHit(BaseModel):
    """论文级检索结果。`page` 分的是论文不是 chunk（M6 §1.2）。

    grouping search 用 `group_by_field=paper_id` 在服务端完成 chunk→paper 聚合，
    所以这里的 `chunks` 长度上限就是 `group_size`（搜索路 1 / RAG 路 3）。
    """

    paper_id: str
    title: str
    year: int
    primary_category: str | None = None
    cited_by_count: int | None = None
    granularity: Granularity
    has_fulltext: bool
    chunks: list[RetrievedChunk] = Field(min_length=1, description="该论文在本次检索中命中的块，按组内序")
    license: LicenseId
    source_dataset: SourceDataset
    attribution: Attribution

    @property
    def best_chunk(self) -> RetrievedChunk:
        """组内第一条即最佳块——Milvus grouping search 保证组内有序，本层不重排。"""
        return self.chunks[0]


class InResultFacets(BaseModel):
    """当前结果集内的分布。**天然是近似值**，所以必须带 based_on 与 sample_size。

    把结果内分布渲染成筛选器计数，是在把近似值当精确值展示——这不是性能问题，
    是正确性问题（决策记录 D11）。契约层拆成两个字段，让前端没有办法混用。
    """

    model_config = ConfigDict(frozen=True)

    based_on: str = Field(default="current_result_set", frozen=True)
    sample_size: int = Field(ge=0, description="算这份分布用了多少组")
    primary_category: dict[str, int] = Field(default_factory=dict)
    granularity: dict[str, int] = Field(default_factory=dict)


class ExactFacets(BaseModel):
    """全库精确聚合。由 SQLite/DuckDB 读 parquet 跑 GROUP BY，或建库时离线算成
    随 index_version 发布的静态 JSON——**不引入 Postgres**（决策记录 D11 / O16）。

    只做年份与领域两个维度；venue 的全库精确 top-10 已剔除（O24：对论文检索没有
    真实决策价值）。
    """

    model_config = ConfigDict(frozen=True)

    based_on: str = Field(default="whole_index", frozen=True)
    index_version: str
    years: dict[str, int] = Field(default_factory=dict)
    primary_category: dict[str, int] = Field(default_factory=dict)


class SearchMeta(BaseModel):
    """响应元信息。`request_id` 是不可省的一条（M6 §1.3）。

    它同时是 HTTP 响应头、结构化日志字段、Langfuse trace id 和用户反馈的关联键。
    **不引第二套 ID**——两套 ID 意味着每次排查都要先做一次关联，而关联表本身
    会成为新的故障点。
    """

    request_id: str = Field(description="ULID。前 48 位是毫秒时间戳，故字典序即时间序，日志检索与分区归档都受益")
    index_version: str
    retrieval_mode: RetrievalMode
    took_ms: int = Field(ge=0)
    cache: str = Field(default="miss", pattern="^(hit|miss|bypass)$")
    degraded: list[DegradeReason] = Field(default_factory=list, description="非空占比是三个静默失败监控项之一")


class SearchResult(BaseModel):
    """`/v1/search` 的完整响应体骨架（M6 §1.3）。"""

    meta: SearchMeta
    group_count: int = Field(ge=0, description="Milvus 实际返回的组数。它与请求组数的比值是第三个静默失败指标")
    page: int = Field(
        ge=1, le=3, description="上限 3 页——ANN 返回的是近邻候选而非稳定全序，limit 放大同时损失召回稳定性与延迟"
    )
    page_size: int = Field(ge=1, le=20)
    papers: list[PaperHit]
    facets_in_results: InResultFacets | None = None
    facets_exact: ExactFacets | None = None

    @model_validator(mode="after")
    def _page_slice_within_groups(self) -> SearchResult:
        # 翻页不重新检索，是从缓存的 60 组里切片（M6 §1.2）。
        # 页内条数超过总组数说明切片逻辑或缓存版本对不上，属于会静默返回重复结果的 bug。
        if len(self.papers) > self.group_count:
            raise ValueError(f"papers({len(self.papers)}) 不能多于 group_count({self.group_count})")
        return self
