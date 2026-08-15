"""检索、融合与重排参数（决策记录 §4.4 / §4.5）。

两条不可协商的约束：
1. **融合器唯一 RRFRanker(k=60)**。没有 WeightedRanker，没有权重字段，没有
   `dense_weight` 参数——它们已被整体删除（H13）。需要通道偏置的场景改为单路不融合。
2. **RRF 分数无物理含义**（k=60 时上界约 2/61≈0.033），禁止任何上游拿它做相关性
   阈值。阈值统一由 rerank 层给。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["RerankSettings", "RetrievalSettings"]


class RetrievalSettings(BaseModel):
    """混合检索与 grouping 参数。"""

    rrf_k: int = Field(
        default=60,
        gt=0,
        lt=16384,
        description=(
            "官方默认 60、范围 (0,16384)、推荐 [10,100]；pymilvus 源码 RRFRanker(k: int = 60)。"
            "Cormack SIGIR'09 Table 1 的敏感性数据显示 [10,100] 区间几乎平坦（k=60→MAP .2145），"
            "取 60 既是官方默认也是论文原值，是零辩护成本的选择"
        ),
    )
    group_by_field: str = Field(default="paper_id", description="grouping search 的分组键，chunk→paper 聚合靠它")
    strict_group_size: bool = Field(
        default=False,
        description="官方警告 True 在数据分布不均时降性能，而混合粒度语料就是典型的分布不均",
    )

    # ---- 搜索路：60 组 × 1 ----
    search_group_limit: int = Field(default=60, ge=1, description="对应浅分页 3 页 × 20 篇")
    search_group_size: int = Field(default=1)

    # ---- RAG 路：16 组 × 3 ≤ 48 chunk ----
    rag_group_limit: int = Field(default=16, ge=1, description="16×3=48 ≤ rerank 输入上限 50")
    rag_group_size: int = Field(
        default=3, ge=1, description="M7 必做 group_size ∈ {2,3,4} 消融（对应 group_limit 24/16/12）"
    )

    # ---- 评测路：加大到 100 ----
    eval_group_limit: int = Field(
        default=100,
        ge=1,
        description="覆盖 Recall@100。报告里必须**显式声明**评测用的是加大 limit 的同一条管线，否则口径与线上 60 不一致",
    )

    # ---- 分页（M6 §1.2）----
    max_page: int = Field(default=3, ge=1, description="上限 3 页。翻页不重新检索，从缓存的 60 组里切片")
    page_size: int = Field(default=20, ge=1, le=20)

    # ---- 精确匹配 ----
    phrase_match_default_slop: int = Field(default=0, ge=0, le=2, description="0=严格相邻同序；2 时容许词序反转")
    text_match_default_logic: str = Field(
        default="OR",
        pattern="^(OR|AND)$",
        description="Milvus 默认 OR；要 AND 得写成 TEXT_MATCH(t,'a') and TEXT_MATCH(t,'b')",
    )

    # ---- 排序模式（引用量与时效不进单一相关性分数）----
    default_sort: str = Field(default="relevance", pattern="^(relevance|citations|date)$")

    @property
    def rag_max_chunks(self) -> int:
        """RAG 路一次检索最多带回多少 chunk。它同时是 rerank 的输入上限。"""
        return self.rag_group_limit * self.rag_group_size


class RerankSettings(BaseModel):
    """应用侧 cross-encoder 精排（决策记录 §4.5）。

    **部署在应用侧而不是用 Milvus 内置 Model Ranker Function**，理由是官方硬限制：
    「Model rankers cannot be used with grouping searches」。两个官方能力互斥时
    选 grouping——chunk→paper 聚合是产品必需，rerank 放哪层是实现细节（O18）。
    """

    model_id: str = Field(default="BAAI/bge-reranker-v2-m3", description="约 568M，FP16 约 1.1 GB")
    normalize: bool = Field(
        default=True,
        description=(
            "对外统一 normalize=True（sigmoid → [0,1]）。内部是 raw logit。"
            "统一口径的理由很具体：0.3 这个阈值按 sigmoid 反推 ≈ logit −0.85，"
            "等于把模型判为倾向不相关的 chunk 计为有效证据（H2）"
        ),
    )
    input_size: int = Field(
        default=48, ge=1, description="48 → 8 约 6x。公开区间：Anthropic 150→20=7.5x；LlamaIndex 10→3≈3.3x"
    )
    output_size: int = Field(default=8, ge=1)
    batch_size: int = Field(default=32, ge=1, description="48 pair 分 2 批前向")
    use_fp16: bool = Field(default=True)
    enabled_on_search_path: bool = Field(
        default=False,
        description="搜索路不开 rerank。连带后果必须主动承认：**RAG 路的指标不能给搜索质量背书**（K5）",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "**不拍数，默认 None（不过滤）**。标定流程照 Cohere 官方 reranking best practices："
            "30–50 条代表性 query → 标临界相关文档 → 过 reranker 取分 → 取均值。"
            "阈值绑模型版本，换模型必须重标并记版本号"
        ),
    )
    calibrated_for_model_version: str | None = Field(
        default=None, description="score_threshold 是在哪个模型版本上标定的"
    )

    def validate_pipeline(self, retrieval: RetrievalSettings) -> None:
        """校验 rerank 输入不超过检索路实际能带回的 chunk 数。

        这条校验存在的理由：`input_size` 与 `rag_group_limit × rag_group_size`
        是两个独立配置项，任何一个被单独改动都会让 rerank 静默地少收候选——
        不报错，只是指标悄悄下滑。
        """
        available = retrieval.rag_max_chunks
        if self.input_size > available:
            raise ValueError(
                f"rerank input_size({self.input_size}) 超过 RAG 路可带回的 chunk 数"
                f"({retrieval.rag_group_limit}×{retrieval.rag_group_size}={available})"
            )
        if self.output_size > self.input_size:
            raise ValueError(f"rerank output_size({self.output_size}) 不能大于 input_size({self.input_size})")
