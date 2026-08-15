"""L2 生成层跑分：MiniCheck 主回归 + ALCE 引用质量 + RAGAS 交叉校验（M7 §7）。

**三个工具，三种角色，排序是有意的**：

| 工具 | 规格 | 角色 | 判官类型 |
|---|---|---|---|
| **MiniCheck-Flan-T5-Large** | 770M，Apache-2.0，EMNLP 2024（arXiv:2404.10774） | **离线主回归** | 判别式模型 |
| **ALCE citation precision / recall** | EMNLP 2023（arXiv:2305.14627） | 引用质量专项 | 判别逻辑 + 蕴含模型 |
| **RAGAS faithfulness** | `= Yes 判定数 / statement 总数` | 第三方口径交叉校验 | **LLM 判官** |

**为什么主回归押在 770M 而不是更强的 LLM**：评测的第一属性是**可复现**，不是绝对
准确。MiniCheck 有公开评估、体积小、单卡半小时能跑完全量——意味着每次改动都跑得起。
**跑不起的评测等于没有评测。**

**最终口径**：RAGAS 与 MiniCheck 结论一致 → 互相印证；方向相反 → **以 MiniCheck
为准并把分歧本身报出来**，因为判别式模型可复现而 LLM 判官不可复现。对外声称的
生成侧质量结论只建立在 MiniCheck + ALCE 这两个可复现口径上。

**判定沿引用边**（与在线 `rag.grounding` 同一个定义）：句子标了 `[3]` 就判
chunk_3 是否蕴含这句话。不对全部 chunk 取最大相似度——后者问的是「这句话像不像
我检索到的任何东西」，前者问的才是「这句话是否被它**自己引的**那篇支持」。

**citation precision 不需要重跑检索**：`cited` 与 `in_context` 两个字段来自
`observability.retrieval_log` 的召回明细，评测复用的就是线上每次请求都会产生的
产物（M7 §12.2）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from citequest.core.answer import AnswerSentence, Evidence
from citequest.evaluation.judges import JudgeConfig
from citequest.observability.logging import get_logger
from citequest.rag.citations import citation_ids_in

__all__ = [
    "CitationQuality",
    "EntailmentJudge",
    "FaithfulnessResult",
    "GenerationEvalReport",
    "MiniCheckJudge",
    "citation_precision_recall",
    "ragas_faithfulness",
]

_logger = get_logger(__name__)


class EntailmentJudge(Protocol):
    """`(premise, hypothesis) -> 是否蕴含`。MiniCheck 与 HHEM 都符合这个形状。

    定成 Protocol 是为了让 `citation_precision_recall` 能在**不加载任何模型**的
    情况下被单测：引用质量的分子分母怎么算，是一个纯逻辑问题，不该被一个 770M
    的权重下载卡住。
    """

    def supports(self, premise: str, hypothesis: str) -> bool: ...


class MiniCheckJudge:
    """MiniCheck-Flan-T5-Large 的封装。**离线主回归判官。**

    **接真实实现**：官方形态是
    ``from minicheck.minicheck import MiniCheck; MiniCheck(model_name='flan-t5-large')``
    然后 ``scorer.score(docs=[premise], claims=[hypothesis])`` 返回 `(labels, probs, ...)`。
    官方 API 就是**句子级**的，这与本项目「按句推送、按句校验」的粒度天然对齐
    （`rag.streaming`），不需要在中间再做一次切分。

    模型在首次判定时才加载：评测脚本可能只想跑检索侧基线，启动就吃显存会让
    「跑一条不需要判官的基线」变成必须有 GPU。
    """

    def __init__(self, config: JudgeConfig) -> None:
        self._config = config
        self._scorer: object | None = None

    @property
    def config(self) -> JudgeConfig:
        return self._config

    def supports(self, premise: str, hypothesis: str) -> bool:
        if not premise.strip() or not hypothesis.strip():
            # 空 premise 判成「支持」会静默抬高 faithfulness；判成「不支持」会
            # 冤枉一条其实没有证据可判的边。两者都不对，所以在入口拦掉。
            raise ValueError("MiniCheck 的 premise 与 hypothesis 都不能为空")
        if self._scorer is None:
            raise NotImplementedError(
                "接真实实现：加载 MiniCheck-Flan-T5-Large（bespokelabs/bespoke-minicheck 或官方 minicheck 包），"
                "调 scorer.score(docs=[premise], claims=[hypothesis])"
            )
        raise NotImplementedError("接真实实现：把 scorer.score 的 label 转成 bool")


class CitationQuality(BaseModel):
    """ALCE 式引用质量。**precision 与 recall 必须分开报**（M7 §7.5）。

    只报 precision 会掩盖「大量该引的句子根本没引」——而沿引用边判定这个定义的
    已知代价，恰恰就是看不见不标引用的句子。recall 是那个缺口的度量。
    """

    model_config = ConfigDict(frozen=True)

    cited_sentences: int = Field(ge=0, description="至少标了一条 [N] 的句子数")
    supported_edges: int = Field(ge=0, description="标了 [N] 且确实被 chunk_N 支持的引用边数")
    total_edges: int = Field(ge=0, description="引用边总数。分母是**边数**不是标记出现次数")
    sentences_needing_citation: int = Field(ge=0, description="内容性句子数（本项目取全部句子，口径见下）")

    @property
    def precision(self) -> float:
        """标了 `[N]` 的边里，确实被支持的比例。"""
        return self.supported_edges / self.total_edges if self.total_edges else 0.0

    @property
    def recall(self) -> float:
        """该有引用且确实标了引用的句子占比。

        **口径声明**：本项目把「该有引用的句子」取为**全部句子**，因为判断一句话
        是不是关键论断需要一个没有标注数据去标定的判别器。按全部句子算的占比
        是一个稳定、可比较的口径，代价是过渡句会把 recall 系统性压低——
        **这个偏差在配置之间是恒定的，所以不影响相对比较**。
        """
        return self.cited_sentences / self.sentences_needing_citation if self.sentences_needing_citation else 0.0


def citation_precision_recall(
    sentences: Sequence[AnswerSentence],
    evidences: Sequence[Evidence],
    judge: EntailmentJudge,
) -> CitationQuality:
    """沿引用边算 ALCE 式的 citation precision / recall。

    复杂度是 `O(引用边数)` 而不是 `O(句数 × chunk 数)`——这是「沿引用边判定」
    相对「对全部 chunk 取最大相似度」的连带收益，不是优化。

    **跨句合成型幻觉在这里仍然会漏**（单看每句都被支持，合起来的结论没被任何
    一段支持）。这条边界必须与数字同时讲：它是判定单元的选择带来的，不是实现缺陷。
    """
    by_id = {e.citation_id: e for e in evidences}
    supported = 0
    total_edges = 0
    cited_sentences = 0

    for sentence in sentences:
        ids = citation_ids_in(sentence.text)
        if ids:
            cited_sentences += 1
        for citation_id in ids:
            evidence = by_id.get(citation_id)
            if evidence is None:
                # 越界引用在 `rag.grounding` 的硬 gate 上已经被拦过一次；能走到
                # 离线评测说明它是首句之后出现的（只降级不撤回）。**计入分母不计
                # 入分子**——它确实是一条模型声称的引用边，而且它必然不成立。
                total_edges += 1
                continue
            total_edges += 1
            if judge.supports(evidence.text, sentence.text):
                supported += 1

    quality = CitationQuality(
        cited_sentences=cited_sentences,
        supported_edges=supported,
        total_edges=total_edges,
        sentences_needing_citation=len(sentences),
    )
    _logger.info(
        "citation_quality",
        precision=round(quality.precision, 4),
        recall=round(quality.recall, 4),
        edges=total_edges,
    )
    return quality


class FaithfulnessResult(BaseModel):
    """RAGAS faithfulness = `Yes 判定数 / statement 总数`。

    **statement 数为 0 时分数未定义，必须特判**（M7 §7.5）——否则会静默产生 NaN
    或被当成 0 分拉低均值，而这个 0 的来源是「这条样本没有可判的陈述」，
    与「陈述全部不被支持」完全是两回事。
    """

    model_config = ConfigDict(frozen=True)

    supported_statements: int = Field(ge=0)
    total_statements: int = Field(ge=0)
    judge: JudgeConfig

    @property
    def score(self) -> float | None:
        """None 表示**未定义**，不是 0 分。调用方必须把它排除在均值之外。"""
        return self.supported_statements / self.total_statements if self.total_statements else None


def ragas_faithfulness(
    statements: Sequence[str],
    context: str,
    judge: EntailmentJudge,
    config: JudgeConfig,
) -> FaithfulnessResult:
    """第三方口径的交叉校验。**它不是主回归。**

    **接真实实现**：正式实现走 `ragas.metrics.faithfulness`，判官是一个 LLM
    （`config` 里记着 model_id / temperature / seed / prompt_version）。这里保留
    一个可注入 judge 的形态，是为了让「RAGAS 与 MiniCheck 结论是否一致」这件事
    能在同一份代码路径上被比较——两边各写一套判定循环，分歧就可能来自实现差异
    而不是判官差异。

    **faithfulness ≠ 事实正确性**：上下文本身错了，忠实度仍可以满分。本系统
    **不声称测量 factuality**——语料就是权威来源，超出语料的事实核查不在能力范围。
    分不清这两者是生成侧评测最常见的失分点。
    """
    supported = sum(1 for s in statements if s.strip() and judge.supports(context, s))
    result = FaithfulnessResult(
        supported_statements=supported,
        total_statements=sum(1 for s in statements if s.strip()),
        judge=config,
    )
    if result.score is None:
        _logger.warning("faithfulness_undefined", reason="statement 数为 0，该样本必须排除在均值之外")
    return result


class GenerationEvalReport(BaseModel):
    """一次生成侧评测的汇总。**MiniCheck 与 RAGAS 的分歧要原样带出来。**"""

    model_config = ConfigDict(frozen=True)

    sample_size: int = Field(ge=0)
    minicheck_supported_rate: float = Field(ge=0.0, le=1.0, description="主回归：可复现口径")
    citation_precision: float = Field(ge=0.0, le=1.0)
    citation_recall: float = Field(ge=0.0, le=1.0)
    ragas_faithfulness: float | None = Field(default=None, description="交叉校验；None 表示未跑或全部未定义")
    undefined_faithfulness_samples: int = Field(default=0, ge=0, description="statement 数为 0 被排除的样本数")
    judges: list[JudgeConfig] = Field(default_factory=list)

    @property
    def judges_disagree(self) -> bool:
        """两个口径方向相反（差 >0.1）时为 True。

        **处置是「以 MiniCheck 为准并把分歧报出来」**，不是取平均、也不是挑一个
        好看的。取平均会造出一个两边都不认的数字。
        """
        return (
            self.ragas_faithfulness is not None and abs(self.ragas_faithfulness - self.minicheck_supported_rate) > 0.1
        )
