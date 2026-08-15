"""消融与显著性（M7 §8）。

**win/loss/tie 是这份测试的重点**：均值涨了 2 点可以是「全面小赢」，也可以是
「30% query 大涨、45% query 小跌」——这两种情况的工程含义完全相反。
少了这个计数，消融就不可信。

显著性检验只测 Fisher randomization（纯重排，不需要任何分布假设，不依赖 scipy）；
配对 t 走 scipy，缺失时返回 None——**这条降级也要被断言**，否则 CI 上会因为一个
可选依赖直接红掉，而它其实并不影响仲裁结论（randomization 才是仲裁者）。
"""

from __future__ import annotations

import pytest

from citequest.config.eval import EvalSettings
from citequest.evaluation.ablation import (
    bootstrap_ci,
    compare_reports,
    fisher_randomization_p,
    win_loss_tie,
)
from citequest.evaluation.metrics import MetricReport
from citequest.evaluation.retrieval_eval import ABLATION_PRESETS


class TestWinLossTie:
    def test_counts_are_three_way(self) -> None:
        baseline = {"q1": 0.5, "q2": 0.8, "q3": 0.4}
        treatment = {"q1": 0.9, "q2": 0.2, "q3": 0.4}
        wlt = win_loss_tie(baseline, treatment)
        assert (wlt.win, wlt.loss, wlt.tie) == (1, 1, 1)
        assert wlt.total == 3

    def test_mean_gain_can_hide_a_broad_regression(self) -> None:
        # 这条用例就是这个指标存在的理由：均值涨了，但 4/5 的 query 变差了。
        baseline = dict.fromkeys(["q1", "q2", "q3", "q4"], 0.5) | {"q5": 0.0}
        treatment = dict.fromkeys(["q1", "q2", "q3", "q4"], 0.4) | {"q5": 1.0}
        assert sum(treatment.values()) > sum(baseline.values())
        wlt = win_loss_tie(baseline, treatment)
        assert (wlt.win, wlt.loss) == (1, 4)

    def test_only_shared_queries_are_compared(self) -> None:
        # 一边缺失的 query 说明两次跑分的样本集不同，那本身就是口径不一致的信号。
        wlt = win_loss_tie({"q1": 0.5, "q2": 0.5}, {"q1": 0.9})
        assert wlt.total == 1


class TestBootstrap:
    def test_interval_brackets_the_mean(self) -> None:
        values = {f"q{i}": float(i % 2) for i in range(200)}
        low, high = bootstrap_ci(values, samples=200)
        assert low <= 0.5 <= high

    def test_is_deterministic_under_a_fixed_seed(self) -> None:
        # 同一份 run 两次跑出不同的 CI，会让「这次改动是否显著」变成一个
        # 每次刷新都可能翻转的结论。
        values = {f"q{i}": float(i % 3 == 0) for i in range(100)}
        assert bootstrap_ci(values, samples=200) == bootstrap_ci(values, samples=200)

    def test_constant_values_give_a_degenerate_interval(self) -> None:
        low, high = bootstrap_ci(dict.fromkeys(["a", "b", "c"], 1.0), samples=50)
        assert (low, high) == (1.0, 1.0)


class TestRandomization:
    def test_identical_systems_are_not_significant(self) -> None:
        values = {f"q{i}": float(i % 2) for i in range(50)}
        assert fisher_randomization_p(values, dict(values), permutations=200) > 0.05

    def test_uniform_improvement_is_significant(self) -> None:
        baseline = dict.fromkeys([f"q{i}" for i in range(60)], 0.2)
        treatment = dict.fromkeys([f"q{i}" for i in range(60)], 0.9)
        assert fisher_randomization_p(baseline, treatment, permutations=500) < 0.05

    def test_p_is_never_reported_as_zero(self) -> None:
        # +1 平滑：0 次极端不代表 p=0，只代表「小于 1/(permutations+1)」。
        # 报 0 是在夸大证据强度。
        baseline = dict.fromkeys([f"q{i}" for i in range(60)], 0.0)
        treatment = dict.fromkeys([f"q{i}" for i in range(60)], 1.0)
        assert fisher_randomization_p(baseline, treatment, permutations=100) > 0.0

    def test_empty_overlap_is_not_significant(self) -> None:
        assert fisher_randomization_p({}, {}, permutations=10) == 1.0


class TestAblationPresets:
    """四档共用同一份 query、qrels、代码，**一次只变一项**。"""

    def test_presets_cover_the_four_configs(self) -> None:
        assert set(ABLATION_PRESETS) == set(EvalSettings().ablation_configs)

    def test_adjacent_configs_differ_in_exactly_one_dimension(self) -> None:
        # C3 → C4 只加 rerank；任何一次「同时改了两处」的对比都不进报告。
        changed = ABLATION_PRESETS["hybrid"].differs_from(ABLATION_PRESETS["hybrid_rerank"])
        assert changed == ["rerank"]

    def test_bm25_only_is_the_published_baseline(self) -> None:
        assert "基线" in ABLATION_PRESETS["bm25_only"].role


class TestCompareReports:
    def test_comparison_carries_all_five_reporting_disciplines(self) -> None:
        baseline = MetricReport(
            config_name="hybrid",
            dataset_name="citrec",
            # 逐条略有差异而不是全表同一个数：常数差会让配对 t 落进
            # 「方差为零」的退化情形，那不是真实的评测数据形态。
            per_query={"recall@100": {f"q{i}": 0.2 + (i % 5) * 0.01 for i in range(40)}},
        )
        treatment = MetricReport(
            config_name="hybrid_rerank",
            dataset_name="citrec",
            per_query={"recall@100": {f"q{i}": 0.8 + (i % 3) * 0.01 for i in range(40)}},
        )
        result = compare_reports(baseline, treatment, EvalSettings(bootstrap_samples=200), metric="recall@100")
        assert result.delta == pytest.approx(0.59, abs=0.02)
        assert result.wlt.win == 40  # 纪律 4
        assert result.ci_baseline[0] <= result.baseline_mean <= result.ci_baseline[1]  # 纪律 1
        assert result.p_randomization < 0.05  # 纪律 3
        assert result.correction == "tukey_hsd"  # 校正方法必须标出来
