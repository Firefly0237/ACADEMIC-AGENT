"""HHEM-2.1-Open 客户端（决策记录 D13 / §4.6）。**只做协议适配。**

`vectara/hallucination_evaluation_model`：flan-t5-base 底座、0.1B、F32 下 <600MB、
Apache-2.0、**无上下文长度上限**。

**必须写在最前面的能力上限**（模型卡自报 balanced accuracy / F1）：

| 数据集 | HHEM-2.1-Open | GPT-4 |
|---|---|---|
| AggreFact-SOTA | 76.55 / **66.77** | 73.78 / 63.87 |
| RAGTruth-Summ | 64.42 / **44.83** | 62.62 / 40.59 |
| RAGTruth-QA | 74.28 / **60.00** | 74.11 / 57.78 |

**F1 只有 44.8–66.8**，意味着大约三分之一到一半的判定不可靠。拿这样一个信号去
拒绝展示用户的回答，等于随机地把正确回答打成幻觉。所以本系统的裁定是：
**HHEM 永远只当软信号，只降级不拦截**。硬拦截权只属于 [N] 越界这一条确定性校验。

它也不是没有价值：三项 balanced accuracy 全部优于 GPT-3.5-Turbo、RAGTruth 两项优于
GPT-4，**而它只有 0.1B**。作为一个 25ms 量级、能跑在共卡上的软信号，性价比合理。
社区对可靠性有公开质疑（仓库 issue #128 "more noise than signal"，maintainer 未回应）
——这条主动说，它**加强**而不是削弱「只当软信号」这个定位。

**语言边界**：仅英文。中文回答只做确定性引用校验，`citation_support` 标 `unverified`
而不是伪造一个分数；跨语言需商用版 HHEM-2.3，不在 v1 范围内。
"""

from __future__ import annotations

from citequest.config.rag import GroundingSettings
from citequest.observability.logging import get_logger

__all__ = ["HhemClient"]

_logger = get_logger(__name__)


class HhemClient:
    """一致性判定：`(premise, hypothesis) -> [0,1]`。

    接口形态刻意与 grounding 的判定对象对齐（决策记录 D13）：
    premise 是**句子所引的那一条 chunk**，hypothesis 是句子本身。
    不提供「一句 vs 全部 chunk」的批量接口——那正是被判定为**度量错了对象**的做法
    （余弦最大相似度问的是「这句话像不像我检索到的东西」，而该问的是
    「这句话是否被它自己引的那一篇支持」）。
    """

    def __init__(self, settings: GroundingSettings) -> None:
        self._settings = settings
        self._model: object | None = None

    def available(self) -> bool:
        """HHEM 是否可用。不可用不是错误——降级链的第一档就是跳过软信号，
        只跑确定性引用校验，`citation_support` 全标 `unverified`（M4 §9.2）。"""
        return self._model is not None

    def _ensure_model(self) -> object:
        """惰性加载。

        **接真实实现**：模型卡给的形态是
        ``AutoModelForSequenceClassification.from_pretrained(
            'vectara/hallucination_evaluation_model', trust_remote_code=True)``
        然后 ``model.predict([(premise, hypothesis)])``。
        `trust_remote_code=True` 这一条要在部署文档里显式写出来——它会执行仓库里的
        自定义代码，属于必须被知情同意的依赖。
        """
        if self._model is None:
            raise NotImplementedError(
                "接真实实现：加载 vectara/hallucination_evaluation_model（HHEM-2.1-Open）。"
                "骨架阶段返回不可用，grounding 层据此走「全标 unverified」的降级档"
            )
        return self._model

    def score(self, premise: str, hypothesis: str, *, language: str = "EN") -> float | None:
        """一致性分。返回 None 表示**未判定**，而不是判定为 0。

        这个区分是契约的一部分：`GroundingVerdict` 里 `unverified` 不允许携带
        `hhem_score`，因为「判过但弱」和「根本没判」对用户的含义完全不同——
        前者是「这条引用支持得不好」，后者是「我没检查」。

        :param language: 非英文直接返回 None（模型仅英文），不做任何折算。
        """
        if self._settings.english_only and language.upper() != "EN":
            return None
        if not premise.strip() or not hypothesis.strip():
            return None
        model = self._ensure_model()
        scores = getattr(model, "predict")([(premise, hypothesis)])  # noqa: B009
        value = float(scores[0])
        _logger.debug("hhem_scored", score=round(value, 4))
        return value
