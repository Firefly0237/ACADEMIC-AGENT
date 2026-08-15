"""检索指标的口径（M7 §6）。**这份测试守的是「怎么算的」而不是「算得快不快」。**

三条被写死的口径：

- **聚合是 macro**（先按 query 算再对 query 平均），不是全局池化；
- **MRR@k 截断记 0**，不是 1/(k+1)；
- **Recall 的分母是「已判定」相关文档数**——citrec 单 label 时它恒为 1，
  于是 Recall@100 等价于 Hit@100，报告里必须这么写。

指标值全部是手算的：二值 rel 下 `DCG = Σ 1/log2(i+1)`，第 1 位贡献 1、
第 2 位 0.6309、第 3 位 0.5。
"""

from __future__ import annotations

import math

import pytest

from citequest.evaluation.metrics import (
    MetricReport,
    average_precision,
    macro_average,
    ndcg_at_k,
    parse_metric,
    per_query_metric,
    recall_at_k,
    reciprocal_rank_at_k,
)

_SINGLE = {"W1": 1}
_MULTI = {"W1": 1, "W2": 1}


class TestNdcg:
    def test_perfect_ranking_is_one(self) -> None:
        assert ndcg_at_k(["W1", "W9"], _SINGLE, 10) == pytest.approx(1.0)

    def test_single_relevant_degenerates_to_log_discount(self) -> None:
        # IDCG=1 时 nDCG@k = 1/log2(rank+1)。这条退化形态必须能讲，
        # 否则「你 nDCG 和 MRR 为什么走势一样」会问住。
        assert ndcg_at_k(["W9", "W1"], _SINGLE, 10) == pytest.approx(1 / math.log2(3))

    def test_outside_k_scores_zero(self) -> None:
        assert ndcg_at_k(["W9", "W8", "W1"], _SINGLE, 2) == 0.0

    def test_ideal_ordering_uses_all_judged_relevants(self) -> None:
        # 两个相关文档排在第 1、3 位：DCG = 1 + 0.5，IDCG = 1 + 0.6309。
        value = ndcg_at_k(["W1", "W9", "W2"], _MULTI, 10)
        assert value == pytest.approx((1 + 0.5) / (1 + 1 / math.log2(3)), rel=1e-6)

    def test_no_judgements_yields_zero_not_nan(self) -> None:
        # 除零静默产生 NaN，而 NaN 会污染整个 macro 平均且不报错。
        assert ndcg_at_k(["W1"], {}, 10) == 0.0


class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert reciprocal_rank_at_k(["W1"], _SINGLE, 10) == 1.0

    def test_truncation_records_zero_not_one_over_k_plus_one(self) -> None:
        # 本项目用 MRR@10：第一个相关文档在第 11 位时该 query 记 **0**。
        ranking = [f"W{i}" for i in range(90, 100)] + ["W1"]
        assert reciprocal_rank_at_k(ranking, _SINGLE, 10) == 0.0

    def test_only_the_first_relevant_counts(self) -> None:
        assert reciprocal_rank_at_k(["W9", "W1", "W2"], _MULTI, 10) == pytest.approx(0.5)


class TestRecall:
    def test_single_relevant_is_a_hit_indicator(self) -> None:
        # citrec 上分母恒为 1 → Recall@100 等价于 Hit@100 / Success@100。
        assert recall_at_k(["W9", "W1"], _SINGLE, 100) == 1.0
        assert recall_at_k(["W9", "W8"], _SINGLE, 100) == 0.0

    def test_denominator_is_the_judged_set(self) -> None:
        assert recall_at_k(["W1"], _MULTI, 10) == pytest.approx(0.5)


class TestAveragePrecision:
    def test_ap_shares_the_recall_denominator(self) -> None:
        # AP 的分母与 Recall 同病，这正是 MAP 不作为对外主指标的理由。
        # 相关文档在第 1、3 位：AP = (1/1 + 2/3) / 2。
        assert average_precision(["W1", "W9", "W2"], _MULTI, 10) == pytest.approx((1 + 2 / 3) / 2)


class TestAggregation:
    def test_missing_run_counts_as_zero_not_as_a_dropped_query(self) -> None:
        # 以 qrels 的 query 集合为准：从 run 侧遍历会让一条结果都没返回的 query
        # 凭空消失——**分母缩小、指标上升，而且没有任何报错**。
        qrels = {"q1": _SINGLE, "q2": _SINGLE}
        values = per_query_metric({"q1": ["W1"]}, qrels, "recall@10")
        assert values == {"q1": 1.0, "q2": 0.0}
        assert macro_average(values) == pytest.approx(0.5)

    def test_macro_weights_every_query_equally(self) -> None:
        # 全局池化会让「相关文档多的 query」主导整个指标。
        assert macro_average({"q1": 1.0, "q2": 0.0, "q3": 0.0}) == pytest.approx(1 / 3)

    def test_empty_input_is_zero(self) -> None:
        assert macro_average({}) == 0.0


class TestMetricNameParsing:
    def test_valid_names(self) -> None:
        assert parse_metric("ndcg@10") == ("ndcg", 10)
        assert parse_metric("Recall@100") == ("recall", 100)

    @pytest.mark.parametrize("bad", ["ndcg10", "ndcg@", "precision@10", "@10"])
    def test_invalid_names_raise(self, bad: str) -> None:
        # 宽容解析在这里是有害的：把 `ndcg10` 静默当成 `ndcg@10` 意味着某天有人
        # 写错指标名，报告照样跑出一个数——而那个数的 k 是猜出来的。
        with pytest.raises(ValueError, match="指标名"):
            parse_metric(bad)


class TestMetricReport:
    def test_slice_keeps_per_query_values_untouched(self) -> None:
        # 分粒度切片**不重算**：逐 query 值与样本集无关，重算只会引入两条
        # 不同的代码路径。
        report = MetricReport(
            config_name="hybrid",
            dataset_name="citrec",
            per_query={"recall@100": {"q1": 1.0, "q2": 0.0, "q3": 1.0}},
        )
        sliced = report.slice_by({"q1", "q2"})
        assert sliced.mean("recall@100") == pytest.approx(0.5)
        assert sliced.query_count("recall@100") == 2
        # 原报告不受影响。
        assert report.query_count("recall@100") == 3

    def test_aggregation_label_is_part_of_the_report(self) -> None:
        # 报告表头必须原样打出 macro——这一句挡掉「是每个 query 算占比再求平均吗」。
        report = MetricReport(config_name="hybrid", dataset_name="citrec", per_query={"mrr@10": {"q1": 1.0}})
        assert report.aggregation == "macro"
