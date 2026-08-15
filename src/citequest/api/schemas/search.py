"""`/v1/search` 的请求与响应模型（M6 §1.2 / §1.3）。

请求参数的三条约束都是**在契约层就把错误变成显式失败**：

1. `q` 长度上限 500，**超出返回 400 而不是静默截断**。截断会让「用户看到的输入」
   与「实际执行的 query」不一致，用户无法解释自己拿到的结果；而且截断后的 query
   会以截断形态进缓存 key，同一段长文本的不同前缀可能命中同一条缓存。
2. `page` 上限 3。ANN 返回的是近邻候选而不是稳定全序，limit 一路放大会同时损失
   召回稳定性和延迟；而学术检索里翻不到第 60 篇还没找到，正确的产品动作是收窄
   query 或加过滤，不是继续翻。
3. `granularity` 是一个**刻意暴露给调用方的能力边界**：全文层覆盖 2019–2022，
   摘要层覆盖 2023–2025。方法级、实验级问题只有全文层答得了。把这件事做成一个
   接口字段，比在文档里写一句免责声明强得多，因为前者是可执行的。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from citequest.api.schemas.common import MetaModel, PaperCard
from citequest.core.enums import Granularity, SortMode
from citequest.core.search import ExactFacets, InResultFacets, SearchResult

__all__ = ["SearchParams", "SearchResponse", "SearchResultsBlock"]


class SearchParams(BaseModel):
    """查询参数。校验失败统一翻译成 400 `INVALID_PARAM`（见 `api.errors`）。"""

    q: str = Field(min_length=1, max_length=500)
    page: int = Field(default=1, ge=1, le=3)
    page_size: int = Field(default=20, ge=1, le=20)
    sort: SortMode = SortMode.RELEVANCE
    granularity: Granularity = Granularity.ANY
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    primary_category: str | None = Field(default=None, max_length=32)

    @field_validator("q")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # 去空白后为空是 `QUERY_EMPTY` 而不是 `INVALID_PARAM`：前端对这两者的
        # 提示不同（一个提示用户输入，一个提示改参数）。
        if not value.strip():
            raise ValueError("q 去空白后为空")
        return value

    @model_validator(mode="after")
    def _year_range_ordered(self) -> SearchParams:
        # 年份倒挂会生成恒假的过滤表达式，Milvus 返回空结果**且不报错**——
        # 典型的「查不到东西但没人知道为什么」，所以在入口拦掉。
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError(f"year_from({self.year_from}) 不能大于 year_to({self.year_to})")
        return self


class SearchResultsBlock(BaseModel):
    """结果块。`group_count` 是 Milvus 实际返回的组数。

    它与请求组数的比值是三个「静默失败」监控项之一：**凑不满组时不会报错**，
    只会返回更少的组，指标悄悄下滑。
    """

    group_count: int
    page: int
    page_size: int
    papers: list[PaperCard]


class SearchResponse(BaseModel):
    """完整响应。

    **`facets_in_results` 与 `facets_exact` 是两种东西，所以是两个字段。**
    前者是当前结果集内的分布，天然近似，必须带 `based_on` 与 `sample_size`；
    后者是全库精确聚合，才能当筛选器计数用。把结果内分布渲染成筛选器计数，
    是在把近似值当精确值展示——这不是性能问题，是正确性问题。
    契约层就拆开，让前端**没有办法**混用。
    """

    meta: MetaModel
    results: SearchResultsBlock
    facets_in_results: InResultFacets | None = None
    facets_exact: ExactFacets | None = None

    @classmethod
    def from_domain(cls, result: SearchResult) -> SearchResponse:
        return cls(
            meta=MetaModel(
                request_id=result.meta.request_id,
                index_version=result.meta.index_version,
                retrieval_mode=result.meta.retrieval_mode,
                took_ms=result.meta.took_ms,
                cache=result.meta.cache,
                degraded=result.meta.degraded,
            ),
            results=SearchResultsBlock(
                group_count=result.group_count,
                page=result.page,
                page_size=result.page_size,
                papers=[PaperCard.from_domain(hit) for hit in result.papers],
            ),
            facets_in_results=result.facets_in_results,
            facets_exact=result.facets_exact,
        )
