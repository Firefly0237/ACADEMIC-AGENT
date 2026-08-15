"""阈值标定与判官准入（M7 §4、§7.3、§7.4）。

两件事在这里被守住：

1. **阈值有明确语义**——「等于我判定为临界相关的那些文档的平均得分」。
   它不是一个魔法数字，所以它必须由标定流程产出，而不是由「0.5 看起来挺合理」定。
2. **Kappa < 0.6 的判官不进指标表**——报 Kappa 而不是准确率，因为准确率在类别
   不平衡时会骗人：90% 样本本来就是「支持」时，一个永远输出「支持」的 judge
   也有 90% 准确率。
"""

from __future__ import annotations

import pytest

from citequest.config.eval import EvalSettings
from citequest.evaluation.calibration import (
    CalibrationItem,
    calibrate_rerank_threshold,
    distribution_crossover,
    load_calibration_set,
    retest_agreement,
)
from citequest.evaluation.judges import (
    KAPPA_ADMISSION_THRESHOLD,
    JudgeConfig,
    cohens_kappa,
    confusion_matrix,
    judge_bias_checklist,
)


class ScriptedReranker:
    """按文本前缀给分的桩。**打分口径与线上一致（normalize=True → [0,1]）。**"""

    def __init__(self, mapping: dict[str, float]) -> None:
        self._mapping = mapping

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [self._mapping[doc] for doc in documents]


def _item(doc_id: str, label: str, *, query: str = "q1", round_index: int = 1) -> CalibrationItem:
    return CalibrationItem(query=query, doc_id=doc_id, text=doc_id, label=label, round_index=round_index)


class TestThresholdCalibration:
    def test_threshold_is_the_mean_of_borderline_scores(self) -> None:
        items = [_item("b1", "borderline"), _item("b2", "borderline"), _item("r1", "relevant")]
        client = ScriptedReranker({"b1": 0.30, "b2": 0.50, "r1": 0.95})
        result = calibrate_rerank_threshold(items, client, EvalSettings(), model_version="bge-reranker-v2-m3@2024-06")
        assert result.threshold == pytest.approx(0.40)
        assert result.sample_size == 2
        # 阈值与模型版本绑死——没有版本号的阈值等于没有标定过。
        assert result.model_version.endswith("2024-06")

    def test_missing_borderline_items_is_an_error_not_a_default(self) -> None:
        # 阈值的语义正是「临界相关文档的平均得分」，缺它就无从标定——
        # 这时给一个默认值等于凭空造一个魔法数字。
        with pytest.raises(ValueError, match="borderline"):
            calibrate_rerank_threshold(
                [_item("r1", "relevant")], ScriptedReranker({"r1": 0.9}), EvalSettings(), model_version="v1"
            )

    def test_second_round_annotations_do_not_pollute_the_threshold(self) -> None:
        # 重标的那 20 条是用来测自一致性的，把它们算进均值等于给部分样本双倍权重。
        items = [_item("b1", "borderline"), _item("b1", "borderline", round_index=2)]
        client = ScriptedReranker({"b1": 0.4})
        assert calibrate_rerank_threshold(items, client, EvalSettings(), model_version="v1").sample_size == 1

    def test_boundaries_ride_along_with_the_threshold(self) -> None:
        result = calibrate_rerank_threshold(
            [_item("b1", "borderline")], ScriptedReranker({"b1": 0.4}), EvalSettings(), model_version="v1"
        )
        # 三条边界原样进报告：单人标注、确认偏误、绑模型版本。
        assert any("IAA" in b for b in result.boundaries)
        assert any("概率" in b for b in result.boundaries)


class TestDistributionCrossover:
    def test_crossover_lands_between_the_two_modes(self) -> None:
        positive = [0.8, 0.85, 0.9, 0.95]
        negative = [0.05, 0.1, 0.15, 0.2]
        crossing = distribution_crossover(positive, negative, bins=10)
        assert 0.2 <= crossing <= 0.85

    def test_one_sided_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="正负两侧"):
            distribution_crossover([0.9], [])


class TestTestRetest:
    def test_agreement_is_computed_over_the_reannotated_subset(self) -> None:
        items = [
            _item("d1", "borderline"),
            _item("d2", "relevant"),
            _item("d1", "borderline", round_index=2),
            _item("d2", "borderline", round_index=2),
        ]
        assert retest_agreement(items) == pytest.approx(0.5)

    def test_without_a_second_round_it_is_none_not_one(self) -> None:
        # 返回 1.0 会被误读成「完全自一致」，而事实是「根本没测」。
        assert retest_agreement([_item("d1", "borderline")]) is None


class TestLoadCalibrationSet:
    def test_missing_file_fails_loudly(self, tmp_path) -> None:
        # 标定集是一等交付物：缺它则 rerank 阈值与 grounding 阈值同时悬空。
        with pytest.raises(FileNotFoundError, match="一等交付物"):
            load_calibration_set(tmp_path / "nope.jsonl")

    def test_jsonl_round_trip(self, tmp_path) -> None:
        path = tmp_path / "calibration_set.jsonl"
        path.write_text(_item("d1", "borderline").model_dump_json() + "\n", encoding="utf-8")
        assert load_calibration_set(path)[0].doc_id == "d1"


class TestJudgeAdmission:
    def test_accuracy_can_look_great_while_kappa_says_useless(self) -> None:
        # 90% 样本是「支持」，judge 永远输出「支持」：准确率 0.9，Kappa 0。
        human = [True] * 90 + [False] * 10
        judge = [True] * 100
        matrix = confusion_matrix(human, judge)
        assert matrix.accuracy == pytest.approx(0.9)
        assert cohens_kappa(matrix) == pytest.approx(0.0, abs=1e-9)

    def test_perfect_agreement_is_one(self) -> None:
        human = [True, False, True, False]
        assert cohens_kappa(confusion_matrix(human, human)) == pytest.approx(1.0)

    def test_degenerate_single_class_does_not_divide_by_zero(self) -> None:
        # 一批全是「支持」的校准样本会让 p_e = 1，公式分母为 0——这恰恰是
        # 类别不平衡时最可能出现的情况。
        assert cohens_kappa(confusion_matrix([True, True], [True, True])) == 1.0

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="条数不一致"):
            confusion_matrix([True, False], [True])

    def test_uncalibrated_judge_is_never_admitted(self) -> None:
        judge = JudgeConfig(model_id="gpt-x", kind="llm")
        assert judge.admissible(None) is False
        assert judge.admissible(KAPPA_ADMISSION_THRESHOLD - 0.01) is False
        assert judge.admissible(KAPPA_ADMISSION_THRESHOLD) is True

    def test_same_family_judge_is_rejected_regardless_of_kappa(self) -> None:
        # self-preference bias：让被告当法官。
        judge = JudgeConfig(model_id="qwen-judge", kind="llm", same_family_as_generator=True)
        assert judge.admissible(0.95) is False

    def test_judge_params_are_pinned(self) -> None:
        # R3：temperature / seed / prompt_version 固定并记录。
        judge = JudgeConfig(model_id="minicheck", kind="discriminative")
        assert (judge.temperature, judge.prompt_version) == (0.01, "v0")

    def test_bias_checklist_covers_all_four(self) -> None:
        assert set(judge_bias_checklist()) == {"self_preference", "position", "length", "score_collapse"}
