"""评测作为一等模块：数据集适配、指标、跑分入口（M7）。

**唯一的设计原则**（M7 §1.1）：

> 每一个对外报出的数字，都必须能回答「样本是什么 / 谁给的标签 / 怎么聚合 /
> 和谁比」这四个问题。**回答不了的数字，不报。**

这条原则直接决定了两个看起来「反常」的选择：主口径用公开基准而不是自建标注集
（自建流程下 Recall@20 在构造上恒等于 1，pooling bias）；生成侧主回归用一个 770M
的判别式模型而不是 LLM 判官（可复现优先于绝对准确，**跑不起的评测等于没有评测**）。

模块划分：

- `datasets/`         —— citrec 库内子集 与 LitSearch 的加载、防泄漏与对齐
- `metrics.py`        —— nDCG / MRR / Recall / MAP，逐 query 值 + macro 聚合
- `retrieval_eval.py` —— L1 检索层跑分：四档消融共用同一份 query/qrels/代码
- `generation_eval.py`—— L2 生成层：MiniCheck 主回归 + ALCE 引用质量 + RAGAS 校验
- `judges.py`         —— 判官配置与 Cohen's Kappa 准入（**< 0.6 不进指标表**）
- `ablation.py`       —— 显著性检验：配对 t + Fisher randomization 双跑 + win/loss/tie
- `calibration.py`    —— 临界相关标定集（30–50 条，**一等交付物，不可降级**）
- `report.py`         —— 配置快照 + 报告落盘（缺任何一项配置则该次结果不可比）
- `runners/`          —— 必须连真实 Milvus 才有意义的曲线扫描

四条报告纪律，缺一条这份评测就不可信：

1. **必报过滤率**——citrec 的 label ∈ 本库 OpenAlex ID 集合的比例，即库内引用边
   覆盖率（接此前实测的 25.3%）。这是证据链不是遮掩。
2. **分粒度报**——fulltext 与 abstract 分开，禁止合并成一个数字。22 万条摘要会
   把 5 万篇全文的表现稀释掉。
3. **报 per-query win/loss/tie 三元计数**——均值相同可以是「全面小赢」，也可以是
   「大赢大输相抵」，这两件事的结论完全相反。
4. **显式声明评测用了加大的 group_limit=100**（线上 60）——口径不一致不写出来
   就是造假。

显著性检验配对 t + Fisher randomization **双跑**，不一致以 randomization 为准并
标注；多重比较用 Tukey HSD。不跑 Wilcoxon / sign（Smucker CIKM'07：检出能力差
且会误判显著）。
"""

from citequest.evaluation.ablation import (
    ComparisonResult,
    WinLossTie,
    bootstrap_ci,
    compare_reports,
    fisher_randomization_p,
    paired_t_p,
    tukey_hsd_p,
    win_loss_tie,
)
from citequest.evaluation.calibration import (
    CalibrationItem,
    ThresholdCalibration,
    calibrate_rerank_threshold,
    distribution_crossover,
    load_calibration_set,
    retest_agreement,
)
from citequest.evaluation.datasets import (
    EvalDataset,
    EvalQuery,
    Provenance,
    SubsetStats,
    build_in_corpus_subset,
    build_litsearch_dataset,
    headline_survives,
)
from citequest.evaluation.generation_eval import (
    CitationQuality,
    FaithfulnessResult,
    GenerationEvalReport,
    citation_precision_recall,
    ragas_faithfulness,
)
from citequest.evaluation.judges import (
    KAPPA_ADMISSION_THRESHOLD,
    ConfusionMatrix,
    JudgeCalibration,
    JudgeConfig,
    cohens_kappa,
    confusion_matrix,
)
from citequest.evaluation.metrics import (
    MetricReport,
    macro_average,
    ndcg_at_k,
    per_query_metric,
    recall_at_k,
    reciprocal_rank_at_k,
)
from citequest.evaluation.report import EvalConfigSnapshot, render_markdown, write_report
from citequest.evaluation.retrieval_eval import (
    ABLATION_PRESETS,
    AblationPreset,
    evaluate_dataset,
    split_by_layer,
)

__all__ = [
    "ABLATION_PRESETS",
    "KAPPA_ADMISSION_THRESHOLD",
    "AblationPreset",
    "CalibrationItem",
    "CitationQuality",
    "ComparisonResult",
    "ConfusionMatrix",
    "EvalConfigSnapshot",
    "EvalDataset",
    "EvalQuery",
    "FaithfulnessResult",
    "GenerationEvalReport",
    "JudgeCalibration",
    "JudgeConfig",
    "MetricReport",
    "Provenance",
    "SubsetStats",
    "ThresholdCalibration",
    "WinLossTie",
    "bootstrap_ci",
    "build_in_corpus_subset",
    "build_litsearch_dataset",
    "calibrate_rerank_threshold",
    "citation_precision_recall",
    "cohens_kappa",
    "compare_reports",
    "confusion_matrix",
    "distribution_crossover",
    "evaluate_dataset",
    "fisher_randomization_p",
    "headline_survives",
    "load_calibration_set",
    "macro_average",
    "ndcg_at_k",
    "paired_t_p",
    "per_query_metric",
    "ragas_faithfulness",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "render_markdown",
    "retest_agreement",
    "split_by_layer",
    "tukey_hsd_p",
    "win_loss_tie",
    "write_report",
]
