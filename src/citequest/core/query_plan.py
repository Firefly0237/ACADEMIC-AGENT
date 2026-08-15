"""查询理解层（M3）的输出契约。

`QueryPlan` 是 M3 唯一的对外产物，下游 M4/M5 只读它、不再回头看原始 query。

**这张表里没有的字段同样是设计的一部分**（M3 §8.1）：
- 没有 `dense_weight` / `sparse_weight`——融合唯一 RRF，无权重概念（D6/H13）；
- 没有 `spelling_suggestion`——拼写纠错已剔除（O5）；
- 没有 `top_k`——下游用的是 `group_limit` + `group_size`，语义是组数与每组条数。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from citequest.core.enums import (
    Granularity,
    GranularityRoute,
    RetrievalMode,
    RewriteStrategy,
    RouteType,
)

__all__ = ["PhraseMatchSpec", "QueryFilters", "QueryPlan", "QuerySignals"]


class QuerySignals(BaseModel):
    """Level 1 规则快筛抽取的确定性信号（M3 §1）。

    全部是纯字符串操作，预算 p50 1ms —— 这层的存在理由就是让绝大多数 query
    不必付 Level 2 小 LLM 的 120–300ms。
    """

    model_config = ConfigDict(frozen=True)

    has_cjk: bool = Field(description="CJK 字符检测。命中 → dense 单路（BM25 打英文文档命中率≈0）")
    phrase_spans: list[str] = Field(
        default_factory=list, description="引号包裹的确切短语。非空 → sparse 单路 + PHRASE_MATCH 过滤"
    )
    year_expr: tuple[int | None, int | None] | None = Field(default=None, description="自然语言年份表达式抽取结果")
    category_terms: list[str] = Field(default_factory=list, description="词面命中的 arXiv 类目，如 cs.CL")
    question_words: list[str] = Field(default_factory=list)
    comparison_words: list[str] = Field(default_factory=list, description="compare / vs / 对比 / 区别，多跳候选信号")
    method_words: list[str] = Field(
        default_factory=list,
        description=(
            "方法/实验级词面（M3 §1.2、§3.3 第 ① 类）。命中 → granularity 硬过滤到 fulltext。"
            "**放在信号对象里而不是让 granularity 路由自己再解析一遍原文**，"
            "因为 M3 §1.2 只允许同一段文本解析一次"
        ),
    )
    recency_words: list[str] = Field(
        default_factory=list,
        description="时效词面（latest / SOTA / 最新）。与 method_words 同时命中即进入 both_with_coverage_note",
    )
    token_count: int = Field(ge=0, description="空白 + 标点切分后的粗计数，不用分词器（这一层不能付模型的钱）")


class QueryFilters(BaseModel):
    """结构化过滤条件。最终下推为 Milvus 标量过滤表达式。

    注意「加过滤不是免费的」：过滤选择率一旦很低，HNSW 的图导航会走不通并退化成
    暴力扫描；官方经验边界是 filter ratio 低于 85% 时图索引仍优于 IVF 系列（M3 §7）。
    """

    model_config = ConfigDict(frozen=True)

    granularity: Granularity = Granularity.ANY
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    primary_category: str | None = None
    exclude_retracted: bool = True

    @model_validator(mode="after")
    def _year_range_ordered(self) -> QueryFilters:
        # 年份倒挂会生成一个恒假的过滤表达式，Milvus 返回空结果且不报错——
        # 典型的「查不到东西但没人知道为什么」。
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError(f"year_from({self.year_from}) 不能大于 year_to({self.year_to})")
        return self


class PhraseMatchSpec(BaseModel):
    """PHRASE_MATCH 的短语与 slop（决策记录 §4.2，需 Milvus 2.5.17+）。

    slop 默认 0（严格相邻同序），放宽档 1 / 2；slop=2 时容许词序反转。
    作用在 `text_exact` 字段上——该字段不挂 BM25 Function、不产生分数、不进融合，
    只做硬过滤，所以它不会变成第三条检索路。
    """

    model_config = ConfigDict(frozen=True)

    phrase: str
    slop: int = Field(default=0, ge=0, le=2)


class QueryPlan(BaseModel):
    """M3 的完整输出（M3 §8.1）。"""

    original_query: str
    search_query: str = Field(description="归一化并剥离结构化条件后的检索文本")
    route: RouteType
    retrieval_mode: RetrievalMode
    granularity_route: GranularityRoute
    filters: QueryFilters = Field(default_factory=QueryFilters)
    phrase_match: PhraseMatchSpec | None = None
    rewrite_strategy: RewriteStrategy = RewriteStrategy.DIRECT
    response_language: str = Field(default="EN", pattern="^(EN|ZH)$", description="回答语言与 query 语言一致")
    classification_level: int = Field(ge=1, le=2, description="1=规则快筛命中，2=小 LLM 兜底。仅用于监控分布突变")
    router_config_version: str = Field(description="路由配置版本号，进日志与 span；改锚点必须触发路由回归测试")
    detected_language: str = Field(default="en")

    @model_validator(mode="after")
    def _mode_matches_route(self) -> QueryPlan:
        # 引号短语路由（sparse 单路）必须带上 PHRASE_MATCH，否则就退化成
        # 「只用 BM25 排序但不做精确约束」，恰好丢掉了这条路由存在的唯一理由（D6 恢复项）。
        if self.retrieval_mode is RetrievalMode.SPARSE_ONLY and self.phrase_match is None:
            raise ValueError("sparse_only 路由必须携带 phrase_match，否则精确短语约束丢失")
        return self
