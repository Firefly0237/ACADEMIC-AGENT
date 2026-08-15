"""消融编排与显著性检验（M7 §8）。

**检验方法不自选，照公开结论做**：

| 项 | 取值 | 依据 |
|---|---|---|
| 主检验 | 配对 Student t（ranx 默认）+ **Fisher randomization，两个都跑** | Smucker/Allan/Carterette CIKM 2007 在 TREC 3、5–8 全部 ad-hoc run 上比较五种检验：randomization / bootstrap / t 三者实践差别很小 |
| 不一致时 | **以 randomization 为准，并标注两者不一致** | 同上 |
| **不跑** | **Wilcoxon 与 sign 检验** | 同一篇：检出能力差，且有**误判显著**的风险 |
| 多重比较 | Tukey HSD（四配置两两 6 组，不校正会虚增显著性） | — |
| 置信区间 | bootstrap 95%，1000 次重采样 | — |

**「不跑 Wilcoxon」这件事要主动说**：它是 IR 论文里最常见的检验之一，不跑它是
一个有出处的选择，而不是漏了。

**win/loss/tie 三元计数是本模块最重要的产物**（M7 §8.3 第 4 条）：均值涨了 2 点
可以是「全面小赢」，也可以是「30% query 大涨、45% query 小跌」——这两种情况的
工程含义完全相反。**少了这个计数，消融就不可信**，所以它不是可选输出。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from citequest.config.eval import EvalSettings
from citequest.evaluation.metrics import MetricReport, macro_average
from citequest.observability.logging import get_logger

__all__ = [
    "ComparisonResult",
    "WinLossTie",
    "bootstrap_ci",
    "compare_reports",
    "fisher_randomization_p",
    "paired_t_p",
    "tukey_hsd_p",
    "win_loss_tie",
]

_logger = get_logger(__name__)

#: 判定「打平」的容差。浮点相等在指标上几乎不会精确成立，用 == 会把大量真实的
#: 平局算成微小的输赢，让 win/loss 计数变成噪声。1e-9 远小于任何有意义的指标差异。
_TIE_EPS = 1e-9


class WinLossTie(BaseModel):
    """逐 query 的三元计数。**报告纪律第 4 条，缺它消融不可信。**"""

    model_config = ConfigDict(frozen=True)

    win: int = Field(ge=0, description="treatment 优于 baseline 的 query 数")
    loss: int = Field(ge=0)
    tie: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.win + self.loss + self.tie


def win_loss_tie(baseline: dict[str, float], treatment: dict[str, float]) -> WinLossTie:
    """逐 query 比一遍。**只统计两边都有的 query**。

    交集而不是并集：一边缺失的 query 说明两次跑分的样本集不同，那本身就是一个
    口径不一致的信号（M7 §14.1 第一维）。把它算成 win 或 loss 等于用一个 bug
    去支持一个结论。
    """
    shared = baseline.keys() & treatment.keys()
    if len(shared) != len(baseline):
        _logger.warning("wlt_sample_mismatch", baseline=len(baseline), treatment=len(treatment), shared=len(shared))
    win = loss = tie = 0
    for qid in shared:
        delta = treatment[qid] - baseline[qid]
        if delta > _TIE_EPS:
            win += 1
        elif delta < -_TIE_EPS:
            loss += 1
        else:
            tie += 1
    return WinLossTie(win=win, loss=loss, tie=tie)


def bootstrap_ci(
    values: dict[str, float],
    *,
    samples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 20240101,
) -> tuple[float, float]:
    """对 per-query 值做 bootstrap，返回均值的置信区间。

    **重采样的单位是 query 而不是判定**：指标本身就是按 query 平均的，
    对判定重采样会得到一个与报出来的均值不同分布的量。

    `seed` 固定是刻意的：同一份 run 两次跑出不同的 CI，会让「这次改动是否显著」
    变成一个每次刷新都可能翻转的结论。要看 CI 的稳定性就换 seed 显式跑两次，
    而不是让它默认随机。
    """
    if not values:
        return (0.0, 0.0)
    population = list(values.values())
    rng = random.Random(seed)
    size = len(population)
    means: list[float] = []
    for _ in range(samples):
        resample = [population[rng.randrange(size)] for _ in range(size)]
        means.append(sum(resample) / size)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low = means[int(tail * samples)]
    high = means[min(samples - 1, int((1.0 - tail) * samples))]
    return (low, high)


def paired_t_p(baseline: dict[str, float], treatment: dict[str, float]) -> float | None:
    """配对 Student t 的 p 值（双尾）。ranx 的默认检验。

    **接真实实现**：走 `scipy.stats.ttest_rel(treatment, baseline).pvalue`。
    scipy 在 eval 依赖组里；缺失时返回 None 而不是自己实现一个 t 分布的 CDF——
    **自己实现统计检验是这类项目最容易出现、也最难被发现的一类错误**，
    而 randomization 检验（纯重排，不需要任何分布假设）本来就在双跑清单里。
    """
    shared = sorted(baseline.keys() & treatment.keys())
    if len(shared) < 2:
        return None
    try:
        from scipy import stats
    except ImportError:  # pragma: no cover - 依赖缺失路径
        _logger.warning("scipy_missing", fallback="只报 fisher randomization 的 p 值")
        return None
    result = stats.ttest_rel([treatment[q] for q in shared], [baseline[q] for q in shared])
    return float(result.pvalue)


def fisher_randomization_p(
    baseline: dict[str, float],
    treatment: dict[str, float],
    *,
    permutations: int = 1_000,
    seed: int = 20240101,
) -> float:
    """Fisher randomization（置换）检验的双尾 p 值。

    零假设是「两个系统在每条 query 上的得分可以互换」，所以置换的方式就是**逐
    query 以 0.5 概率交换两个系统的得分**，看观测到的均值差在置换分布里有多极端。

    它是双跑里的**仲裁者**：不需要正态性假设，而 IR 的 per-query 指标（大量 0、
    大量 1）显然不正态。两个检验结论不一致时以它为准，并把不一致本身报出来。
    """
    shared = sorted(baseline.keys() & treatment.keys())
    if not shared:
        return 1.0
    diffs = [treatment[q] - baseline[q] for q in shared]
    observed = abs(sum(diffs) / len(diffs))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(permutations):
        shuffled = sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)
        if abs(shuffled) >= observed:
            extreme += 1
    # +1 平滑（Monte Carlo 置换检验的标准做法）：0 次极端不代表 p=0，
    # 只代表「小于 1/(permutations+1)」，报 0 是在夸大证据强度。
    return (extreme + 1) / (permutations + 1)


def tukey_hsd_p(reports: dict[str, MetricReport], metric: str) -> dict[tuple[str, str], float] | None:
    """四配置两两比较的 Tukey HSD 校正 p 值（M7 §8.2）。

    **不校正会虚增显著性**：四档两两共 6 组比较，各自按 α=0.05 判定时，
    「至少一组假阳性」的概率远高于 5%。

    **接真实实现**：走 `scipy.stats.tukey_hsd(*样本组)`，返回 `pvalue` 矩阵。
    scipy 缺失时返回 None，调用方据此在报告里写「未做多重比较校正」——
    **不做校正可以，假装做了不行**。

    这里传的是每组的 per-query 值数组。Tukey HSD 假设各组独立同方差，
    而配对设计下各组共用同一批 query，属于已知的保守偏差（p 值偏大），
    这条也要写进报告：它让结论更难显著，不会制造假阳性。
    """
    try:
        from scipy import stats
    except ImportError:  # pragma: no cover - 依赖缺失路径
        _logger.warning("scipy_missing", fallback="报告里写明未做 Tukey HSD 校正")
        return None

    names = sorted(reports)
    samples = [list(reports[name].per_query[metric].values()) for name in names]
    matrix = stats.tukey_hsd(*samples).pvalue
    return {(names[i], names[j]): float(matrix[i][j]) for i in range(len(names)) for j in range(len(names)) if i < j}


@dataclass(slots=True)
class ComparisonResult:
    """两档配置在一个指标上的完整对比。**五条报告纪律的载体。**"""

    metric: str
    baseline_name: str
    treatment_name: str
    baseline_mean: float
    treatment_mean: float
    delta: float
    ci_baseline: tuple[float, float]
    ci_treatment: tuple[float, float]
    p_paired_t: float | None
    p_randomization: float
    wlt: WinLossTie
    correction: str
    tests_disagree: bool
    """两个检验在 α 下结论不同。**必须原样印进报告**——以 randomization 为准，
    但把分歧藏起来等于替读者做了一个他不知道自己被做了的判断。"""


def compare_reports(
    baseline: MetricReport,
    treatment: MetricReport,
    settings: EvalSettings,
    *,
    metric: str,
) -> ComparisonResult:
    """跑完一次两档对比：均值 + CI + 双检验 + win/loss/tie。

    **一次只变一项**由调用方保证（`AblationPreset.differs_from`），这里只负责统计。
    分开的理由是统计部分要能被单测直接断言，而配置差异检查需要 preset 上下文。
    """
    base_values = baseline.per_query[metric]
    treat_values = treatment.per_query[metric]

    p_t = paired_t_p(base_values, treat_values)
    p_r = fisher_randomization_p(base_values, treat_values, permutations=settings.bootstrap_samples)
    alpha = settings.significance_alpha
    disagree = p_t is not None and ((p_t < alpha) != (p_r < alpha))

    result = ComparisonResult(
        metric=metric,
        baseline_name=baseline.config_name,
        treatment_name=treatment.config_name,
        baseline_mean=macro_average(base_values),
        treatment_mean=macro_average(treat_values),
        delta=macro_average(treat_values) - macro_average(base_values),
        ci_baseline=bootstrap_ci(base_values, samples=settings.bootstrap_samples, confidence=settings.confidence_level),
        ci_treatment=bootstrap_ci(
            treat_values, samples=settings.bootstrap_samples, confidence=settings.confidence_level
        ),
        p_paired_t=p_t,
        p_randomization=p_r,
        wlt=win_loss_tie(base_values, treat_values),
        correction=settings.multiple_comparison_correction,
        tests_disagree=disagree,
    )
    if disagree:
        _logger.warning(
            "significance_tests_disagree",
            metric=metric,
            p_paired_t=p_t,
            p_randomization=p_r,
            resolution="以 randomization 为准，并在报告里标注两者不一致",
        )
    _logger.info(
        "ablation_compared",
        metric=metric,
        baseline=baseline.config_name,
        treatment=treatment.config_name,
        delta=round(result.delta, 4),
        p_randomization=round(p_r, 4),
        win=result.wlt.win,
        loss=result.wlt.loss,
        tie=result.wlt.tie,
    )
    return result
