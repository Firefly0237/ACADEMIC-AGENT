"""判官模型的配置与校准协议（M7 §7.3、§7.4）。

> **没有校准的 judge，产出的不是指标，是气氛。**
> 「LLM 打了高分，你怎么知道它是对的」是一个只要答不上来就足以否定全部指标的问题。

**四条判官规则**（M7 §7.3）：

| 规则 | 内容 | 理由 |
|---|---|---|
| R1 | 判官**不与被评生成模型同源** | self-preference bias：让被告当法官 |
| R2 | 主回归优先用**判别式模型**（MiniCheck 770M） | 确定性、可复现、无 API 成本、无版本漂移 |
| R3 | 判官参数固定并记录（temperature 0.01、seed、**prompt 版本号进 span**） | prompt 改一个字都可能移动分数 |
| R4 | 判官结果**只用于相对比较** | 绝对分数依赖 rubric 与判官版本，跨项目不可比 |

**准入判定是硬的**：Cohen's Kappa < 0.6 → 该 judge 的结论**不进指标表**，
只能作为定性观察。报 Kappa 而不是准确率，是因为准确率在类别不平衡时会骗人——
90% 样本本来就是「支持」时，一个永远输出「支持」的 judge 也有 90% 准确率。

**单人项目的诚实边界**：人工标签仍然是单人给的，所以校准结果的上限是
「judge 与我一致」而不是「judge 正确」。这句话必须写进报告，它比任何一个漂亮的
Kappa 值都更能说明评测被认真设计过。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from citequest.observability.logging import get_logger

__all__ = [
    "KAPPA_ADMISSION_THRESHOLD",
    "ConfusionMatrix",
    "JudgeCalibration",
    "JudgeConfig",
    "cohens_kappa",
    "confusion_matrix",
    "judge_bias_checklist",
]

_logger = get_logger(__name__)

#: 准入线。低于它，judge 的结论只能作为定性观察，不进指标表。
KAPPA_ADMISSION_THRESHOLD = 0.6


class JudgeConfig(BaseModel):
    """一个判官的完整身份。**这五个字段一起构成「哪次评测用的哪个判官」。**

    少任何一项，两次评测之间的分数差异就无法归因——尤其是 `prompt_version`：
    prompt 改了但指标没记版本，差异只能靠猜。
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    kind: str = Field(pattern="^(discriminative|llm)$", description="判别式模型走主回归，LLM 只做交叉校验")
    temperature: float = Field(default=0.01, ge=0.0, description="R3：单次采样，不做多次投票")
    seed: int = Field(default=20240101)
    prompt_version: str = Field(default="v0", description="R3：进日志与 span。改 prompt 必须换版本号")
    same_family_as_generator: bool = Field(
        default=False, description="R1：与被评生成模型同源时为 True，此时该判官的结论不可用"
    )

    def admissible(self, kappa: float | None) -> bool:
        """这个判官的结论能不能进指标表。

        两道门都得过：与生成模型不同源（R1），且 Kappa ≥ 0.6（§7.4 第 ④ 步）。
        `kappa is None` 表示**还没校准**——未校准的 judge 一律不准入，
        这比「先用着，回头补校准」诚实，而后者从来不会被补上。
        """
        if self.same_family_as_generator:
            _logger.warning("judge_rejected_same_family", model_id=self.model_id)
            return False
        return kappa is not None and kappa >= KAPPA_ADMISSION_THRESHOLD


class ConfusionMatrix(BaseModel):
    """二值判定的混淆矩阵。**与 Kappa 一起报，不只报一个数**。

    只报 Kappa 看不出错误的方向：一个系统性地把「不支持」判成「支持」的 judge
    与一个反过来的 judge，可以有相同的 Kappa，但前者会让幻觉溜过去，后者只是
    过于保守。这两件事的处置完全不同。
    """

    model_config = ConfigDict(frozen=True)

    tp: int = Field(ge=0, description="人工=支持，judge=支持")
    fp: int = Field(ge=0, description="人工=不支持，judge=支持（**幻觉溜过去的那一格**）")
    fn: int = Field(ge=0, description="人工=支持，judge=不支持")
    tn: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        """**不要单独报它**（见类文档）。留在这里是为了与 Kappa 并排展示两者的差距。"""
        return (self.tp + self.tn) / self.total if self.total else 0.0


def confusion_matrix(human: Sequence[bool], judge: Sequence[bool]) -> ConfusionMatrix:
    """人工标签 vs judge 标签。两个序列必须等长且一一对应。"""
    if len(human) != len(judge):
        raise ValueError(f"人工标签({len(human)})与 judge 标签({len(judge)})条数不一致，无法配对")
    tp = sum(1 for h, j in zip(human, judge, strict=True) if h and j)
    fp = sum(1 for h, j in zip(human, judge, strict=True) if not h and j)
    fn = sum(1 for h, j in zip(human, judge, strict=True) if h and not j)
    tn = sum(1 for h, j in zip(human, judge, strict=True) if not h and not j)
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def cohens_kappa(matrix: ConfusionMatrix) -> float:
    """Cohen's Kappa：`(p_o − p_e) / (1 − p_e)`。

    `p_o` 是观测一致率，`p_e` 是**随机一致率**——两个标注者各按自己的边缘分布
    随机打标时的期望一致率。Kappa 扣掉了随机一致的部分，才反映真实判别能力。

    完全一致（`p_e = 1`）时返回 1.0：此时两者都只输出同一个类别，公式分母为 0。
    这个特判必须有，否则一批全是「支持」的校准样本会让整个校准崩在除零上，
    而这恰恰是类别不平衡时最可能出现的情况。
    """
    n = matrix.total
    if n == 0:
        return 0.0
    p_o = matrix.accuracy
    judge_yes = (matrix.tp + matrix.fp) / n
    human_yes = (matrix.tp + matrix.fn) / n
    p_e = judge_yes * human_yes + (1 - judge_yes) * (1 - human_yes)
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


class JudgeCalibration(BaseModel):
    """一次判官校准的完整产物（M7 §7.4 五步的产出）。"""

    model_config = ConfigDict(frozen=True)

    judge: JudgeConfig
    sample_size: int = Field(ge=0, description="分层随机抽样的条数，设计值 100")
    matrix: ConfusionMatrix
    kappa: float
    test_retest_agreement: float | None = Field(
        default=None, description="同一人间隔重标 20 条的自一致性。**是自一致性不是客观性**"
    )
    judge_rerun_drift: float | None = Field(default=None, description="同一批样本 judge 重跑两次的分数漂移")
    admitted: bool = Field(description="Kappa ≥ 0.6 且判官与生成模型不同源")
    honest_boundary: str = Field(
        default="人工标签由单人给出，校准结果的上限是「judge 与我一致」而不是「judge 正确」",
        description="这句话原样进报告，不允许删",
    )


def judge_bias_checklist() -> dict[str, str]:
    """四个必须主动声明的 judge 偏置及其处置（M7 §7.4）。

    做成函数返回而不是散在文档里，是因为**它要被渲染进每一份用到 LLM 判官的
    报告**——写在设计文档里的免责声明没人会在读指标时翻出来看。
    """
    return {
        "self_preference": "判官与生成模型不同源（R1）",
        "position": "成对比较时**交换顺序双跑**，只采纳两次结论一致的样本；不一致计为 tie 并单独报比例",
        "length": "同时给出回答长度分布；两个配置长度分布显著不同时，judge 的比较结论需打折说明",
        "score_collapse": "每次评测检查判官分数的分布与方差；方差过小视为 rubric 失效，回去改 rubric 而不是接受结果",
    }
