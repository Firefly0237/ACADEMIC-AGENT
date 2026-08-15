"""评测参数（决策记录 §4.8）。

两个基准各自对应一层语料，这是刻意的：头条基准 unarXive_citrec 是段落级的
（只测得到全文层），第二基准 LitSearch 是 title+abstract 的（只测得到摘要层）。
**分粒度分别报，禁止合并成一个数字**——22 万条摘要会把 5 万篇全文的表现稀释掉。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

__all__ = ["EvalSettings"]


class EvalSettings(BaseModel):
    """评测数据集、指标与显著性检验配置。"""

    # ---- 头条基准 ----
    citrec_dataset_id: str = Field(
        default="saier/unarXive_citrec", description="test 225,348 条，CC-BY-SA-4.0，label = 被引论文 OpenAlex ID"
    )
    citrec_split: str = Field(default="test")
    citrec_min_subset_size: int = Field(
        default=3_000,
        ge=1,
        description=(
            "**头条基准退场阈值**：过滤到库内后低于此规模 → 头条退回 LitSearch，citrec 降级为补充证据。"
            "低于几千条时 bootstrap 95% CI 会宽到任何消融结论都不显著（U3，本轮最大单点风险）"
        ),
    )
    citrec_report_filter_rate: bool = Field(
        default=True,
        description="**必报数字**：label ∈ 本库 OpenAlex ID 集合的比例，即库内引用边覆盖率的重测（接此前实测的 25.3%）。这是证据链不是遮掩",
    )
    citrec_strip_marker_neighborhood: bool = Field(
        default=True,
        description="防泄漏第一条：删除 marker 及 marker_offsets 邻域。第二条在索引侧（已删除 {{cite:uuid}} 而非替换成标题）",
    )

    # ---- 第二基准 ----
    litsearch_dataset_id: str = Field(
        default="princeton-nlp/LitSearch", description="query 597 / corpus 64,183 title+abstract"
    )
    litsearch_retrieval_field: str = Field(default="title_abstract", description="官方检索字段")
    litsearch_bm25_ndcg10_reference: float = Field(
        default=40.2,
        description="**部分核实：单一来源（CASPER Table 2）**，引用时必须说明",
    )
    litsearch_bm25_recall100_reference: float = Field(default=74.4)
    litsearch_ndcg10_tolerance: float = Field(
        default=3.0, description="超窗先查索引与分词器，**不许直接开始报提升数字**"
    )
    litsearch_recall100_tolerance: float = Field(default=4.0)
    reproduction_wording: str = Field(
        default="sanity_check",
        description="只做量级一致性检查，**叫 sanity check 不叫 reproduction**（Milvus analyzer / rank-bm25 / Anserini 分词各不相同）",
    )

    # ---- 结构校验（不产出检索指标）----
    imrad_dataset_id: str = Field(
        default="saier/unarXive_imrad_clf",
        description="530k。**不作检索指标**，只校验 chunk 的 sec_type 分布与原始分布是否一致",
    )

    # ---- 标定集（一等交付物）----
    calibration_set_size: int = Field(
        default=40,
        ge=30,
        le=50,
        description="30–50 条「临界相关」query，Cohere 官方流程。**与被降级的 200 条回归集分开命名**——载体被降级，依赖它的阈值结论就同时悬空",
    )
    calibration_set_path: Path = Field(default=Path("data/external/calibration_set.jsonl"))

    # ---- 指标与统计 ----
    metrics: list[str] = Field(default_factory=lambda: ["recall@100", "mrr@10", "ndcg@10", "recall@5", "recall@20"])
    aggregation: str = Field(default="macro", description="按 query 平均，不是全局池化")
    bootstrap_samples: int = Field(default=1_000, ge=1)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    significance_alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Tukey HSD 校正后")
    significance_tests: list[str] = Field(
        default_factory=lambda: ["paired_t", "fisher_randomization"],
        description="**双跑**，不一致以 randomization 为准并标注。不跑 Wilcoxon / sign（Smucker CIKM'07：检出能力差且会误判显著）",
    )
    multiple_comparison_correction: str = Field(default="tukey_hsd")
    report_per_query_win_loss_tie: bool = Field(
        default=True,
        description="**少了 win/loss/tie 三元计数这个消融就不可信**——均值相同可以是「全面小赢」也可以是「大赢大输相抵」",
    )
    report_split_by_granularity: bool = Field(default=True, description="fulltext 与 abstract 必须分开报，禁止合并")

    # ---- 必做消融与 A/B ----
    ablation_configs: list[str] = Field(
        default_factory=lambda: ["bm25_only", "dense_only", "hybrid", "hybrid_rerank"],
        description="共用同一份 query、qrels、代码，一次只变一项",
    )
    ablation_group_sizes: list[int] = Field(default_factory=lambda: [2, 3, 4], description="对应 group_limit 24/16/12")
    ab_learned_sparse_vs_bm25: bool = Field(
        default=True,
        description="**必做**。这是「为什么敢放弃 learned sparse」唯一的数字来源——没有它，L2 层永远只能说「我推测增益小」",
    )
    ab_hyde_vs_direct: bool = Field(default=True, description="HyDE 的存活条件：跑不赢 direct 就砍")
    ab_placeholder_delete_vs_keep_latex: bool = Field(
        default=True, description="小样本 A/B，代价是放弃公式内容可检索性"
    )

    reports_dir: Path = Field(default=Path("reports"))
