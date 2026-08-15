"""引用校验与 grounding（M4 §5、决策记录 D13 / §4.6）。**本模块是这一层的核心。**

**三层防线，只有第一层敢用来拦截：**

| 层 | 手段 | 能力 | 敢不敢拦 |
|---|---|---|---|
| 确定性硬校验 | `[N]` 越界 / 全部非法（纯字符串 + 集合成员判定） | 误判率**结构性为零** | **敢。唯一敢硬拦的条件** |
| 在线软信号 | HHEM-2.1-Open，**只在首句跑一次** | 自报 F1 仅 44.8–66.8 | **不敢**。只做 UI 降级与弱支持标记 |
| 离线评测 | MiniCheck 主回归 + ALCE citation precision/recall + RAGAS faithfulness | 有公开基准背书 | 不进在线路径 |

**判定的定义**：句子 S 标了 `[3]`，就判 `(chunk_3 作为 premise, S 作为 hypothesis)`
是否成立蕴含。判定对象是**引用边**，不是「2 个句子 × 8 个 chunk = 16 次比较」。

**为什么不是「对全部 chunk 取最大相似度」**——它同时错在两个层次（M4 §5.2）：

1. **度量的东西不对**：余弦度量语义接近，不度量逻辑蕴含。「A 优于 B」与
   「B 优于 A」余弦极高但语义相反；带否定、数值、条件的句子会被系统性漏报，
   **而这三类恰恰是学术回答里信息密度最高的句子**。
2. **问的问题不对**：最大相似度问的是「这句话像不像我检索到的东西」，
   而带引用的回答该问的是「**这句话是否被它自己引的那一篇支持**」。
   模型写了一句话标 `[3]`，内容其实来自 chunk_7——按最大相似度它是被支持的，
   按引用边它是错误引用。而用户点击 `[3]` 跳过去看到一段对不上的原文，
   **这正是「引用可点击核对」这个承诺被击穿的方式**。

代价（诚实交代）：沿引用边判定**看不见模型不标引用的那些句子**。
补法是 prompt 规则 2 + 后校验统计「无引用句占比」并作为可观测指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from citequest.config.rag import GroundingSettings
from citequest.core.answer import Evidence, GroundingVerdict
from citequest.core.enums import CitationSupport, DegradeReason
from citequest.llm.hhem import HhemClient
from citequest.observability.logging import get_logger
from citequest.rag.citations import citation_ids_in

__all__ = ["GateResult", "GroundingChecker", "SentenceGrounding"]

_logger = get_logger(__name__)


@dataclass(slots=True)
class GateResult:
    """首句阻塞闸门的判定结果（M4 §6.3）。

    **不过闸时 `reason` 必须能落到用户可读的解释上**：信息不足回答要说清
    「为什么不足」，而不只是「不足」——「只有摘要粒度」「引用全部非法」是可行动的
    信息，「抱歉我不知道」不是。
    """

    passed: bool
    reason: str | None = None
    verdicts: list[GroundingVerdict] = field(default_factory=list)
    degraded: DegradeReason | None = None


@dataclass(slots=True)
class SentenceGrounding:
    """一句话的引用边判定汇总。"""

    seq: int
    citation_ids: list[int]
    out_of_range: list[int]
    verdicts: list[GroundingVerdict] = field(default_factory=list)

    @property
    def has_illegal_citation(self) -> bool:
        return bool(self.out_of_range)

    @property
    def has_no_citation(self) -> bool:
        return not self.citation_ids and not self.out_of_range


class GroundingChecker:
    """确定性校验 + 首句一次 HHEM。

    HHEM 通过构造注入而不是内部实例化：**「HHEM 不可用」是一条正常降级链**
    （跳过软信号、只跑确定性校验、`citation_support` 全标 `unverified`），
    它必须能在没有模型的环境下被单测覆盖。
    """

    def __init__(self, settings: GroundingSettings, *, hhem: HhemClient | None = None) -> None:
        self._settings = settings
        self._hhem = hhem

    # ---- 第一层：确定性硬校验 ------------------------------------------

    def check_sentence(self, seq: int, sentence: str, evidences: list[Evidence]) -> SentenceGrounding:
        """扫出这句话的引用边并做集合成员判定。

        **这里没有任何模型参与**，所以它的误判率是结构性为零的——
        整套引用可信体系建立在这个性质上。
        """
        valid = {e.citation_id for e in evidences}
        ids = citation_ids_in(sentence)
        return SentenceGrounding(
            seq=seq,
            citation_ids=[i for i in ids if i in valid],
            out_of_range=[i for i in ids if i not in valid],
        )

    # ---- 首句闸门：确定性校验 + 一次 HHEM ------------------------------

    def gate_first_sentence(
        self,
        sentence: str,
        evidences: list[Evidence],
        *,
        language: str = "EN",
    ) -> GateResult:
        """首句阻塞闸门（M4 §6.3）。

        **为什么首句是那个时刻**：第一个 token 出网之前，是「不展示」这个能力
        唯一还能执行的时刻。代价只有一句的生成时间，不是整篇。所以「不展示」这个
        语义被保留下来了，被删掉的只是「在流式中途撤回」这个**不可执行的形态**。

        **硬 gate 只有一条**：`[N]` 越界或引用全部非法 → 不过闸 → 完全不进流式。
        HHEM 分数**只降级不拦截**——拿一个自报 F1 只有 44.8–66.8 的模型去拒绝展示
        用户的回答，等于随机地把正确回答打成幻觉。
        """
        grounding = self.check_sentence(0, sentence, evidences)

        if self._settings.hard_gate_on_citation_out_of_range and grounding.has_illegal_citation:
            reason = f"首句引用越界：{grounding.out_of_range}，Evidence 区只有 {[e.citation_id for e in evidences]}"
            _logger.warning("first_sentence_gate_blocked", out_of_range=grounding.out_of_range)
            return GateResult(passed=False, reason=reason)

        if not evidences:
            return GateResult(passed=False, reason="没有可用证据，无法产生可核对的回答")

        verdicts = self.score_edges(grounding, sentence, evidences, language=language)
        degraded = None if self._hhem_usable(language) else DegradeReason.HHEM_UNAVAILABLE
        # **HHEM 低分不拦截**：它只让 `citation_support` 落到 weak，前端据此打灰、
        # 必要时追加 disclaimer。这条边界写在代码里，是为了让「以后想用它拦截」
        # 必须先改这一行，而不是悄悄加一个阈值。
        return GateResult(passed=True, verdicts=verdicts, degraded=degraded)

    # ---- 第二层：在线软信号 --------------------------------------------

    def _hhem_usable(self, language: str) -> bool:
        """HHEM 是否真的能用于这次判定。

        两道门：模型可用，且语言是英文。**中文回答只做确定性引用校验**，
        `citation_support` 标 `unverified` 而不是伪造一个分数——
        跨语言需要商用版 HHEM-2.3，不在 v1 范围内。
        """
        if self._hhem is None or not self._hhem.available():
            return False
        return not (self._settings.english_only and language.upper() != "EN")

    def score_edges(
        self,
        grounding: SentenceGrounding,
        sentence: str,
        evidences: list[Evidence],
        *,
        language: str = "EN",
        previous_sentence: str | None = None,
    ) -> list[GroundingVerdict]:
        """逐引用边打分。**在线只在首句调用一次**（`online_first_sentence_only`）。

        逐句在线判定是一个**默认关闭的开关**：一篇十几句的回答有几十条引用边，
        逐条在线判定意味着每句阻塞一次模型前向，叠在按句放行的流式上，
        流式体验被自己吃掉——而买到的还是一个不敢用来拦截的软信号。
        **必要性判据：花在线延迟买一个我不敢用的信号 → 不过关。**

        :param previous_sentence: 判定上下文保留前一句（对应 NeMo 的 `context_size`
            滑窗作用，官方示例 chunk_size 200 / context_size 50）。跨句合成型幻觉
            只能部分缓解——NeMo 文档也承认小块有 "missing cross-chunk violations" 的风险。
        """
        by_id = {e.citation_id: e for e in evidences}
        usable = self._hhem_usable(language)
        hypothesis = (
            f"{previous_sentence} {sentence}".strip()
            if (previous_sentence and self._settings.keep_previous_sentence_as_context)
            else sentence
        )

        verdicts: list[GroundingVerdict] = []
        for citation_id in grounding.citation_ids:
            evidence = by_id.get(citation_id)
            if evidence is None:  # pragma: no cover - citation_ids 已过集合成员判定
                continue
            if not usable:
                # 未判定就是 unverified，**且不带分数**——「判过但弱」和「根本没判」
                # 对用户的含义完全不同，混起来就分不清了。
                verdicts.append(
                    GroundingVerdict(
                        sentence_seq=grounding.seq, citation_id=citation_id, support=CitationSupport.UNVERIFIED
                    )
                )
                continue
            assert self._hhem is not None
            score = self._hhem.score(evidence.text, hypothesis, language=language)
            verdicts.append(self._verdict_from_score(grounding.seq, citation_id, score))
        return verdicts

    def _verdict_from_score(self, seq: int, citation_id: int, score: float | None) -> GroundingVerdict:
        """分数 → 三档支持强度。

        **阈值未标定时一律 `unverified`**：三档阈值待 M7 在 30–50 条标定集上按
        balanced accuracy 调，**不沿用余弦时代的 0.7/0.5**。拿一对没有出处的数
        去给用户的引用打「弱支持」标记，比不标更糟——它看起来像一个判断。
        """
        supported = self._settings.supported_threshold
        weak = self._settings.weak_threshold
        if score is None or supported is None or weak is None:
            return GroundingVerdict(sentence_seq=seq, citation_id=citation_id, support=CitationSupport.UNVERIFIED)
        if score >= supported:
            support = CitationSupport.SUPPORTED
            reason = None
        elif score >= weak:
            support = CitationSupport.WEAK
            reason = "引用与陈述的一致性低于标定阈值，建议点开原文核对"
        else:
            support = CitationSupport.WEAK
            reason = "引用与陈述的一致性明显偏低，建议点开原文核对"
        return GroundingVerdict(
            sentence_seq=seq,
            citation_id=citation_id,
            support=support,
            hhem_score=score,
            reason=reason,
        )

    # ---- 证据冲突：不裁决，并列呈现 -------------------------------------

    @staticmethod
    def conflict_dimensions(evidences: list[Evidence]) -> list[str]:
        """列出这批证据里存在分歧的**维度**，不判断谁对（M4 §5.6）。

        学术检索里证据冲突**不是异常，是常态**——分歧本身就是用户要的信息。
        系统不判断哪一方对，而是并列呈现 + 各自标引用 + 标注分歧维度。
        理由是 factuality 的直接推论：**裁决冲突需要 factuality 能力，
        而这个系统结构上没有**（语料就是论文本身，论文是一手证据不是权威事实）。
        强行裁决只有两种结果——按 rerank 分数选一方（把排序分数误当成真理度），
        或让模型糅一句听起来圆满的话（那是编造，不是综合）。

        **不做在线冲突检测器**：它要么是一次额外的模型前向（在线延迟），
        要么是一个自写启发式（无出处）。而既然结论是「不裁决」，
        **检测出冲突之后也不会做任何不同的事——检测一个不影响任何动作的信号是纯成本**。
        这个函数只汇报元数据层面已有的差异（年份跨度、粒度混合），零额外调用。
        """
        dimensions: list[str] = []
        years = {e.year for e in evidences}
        if len(years) > 1 and max(years) - min(years) >= 3:
            # 年份跨度大 → 很可能是「新旧更替」而不是「互相矛盾」。
            # 让模型和用户都能看见这一点，靠的是元数据头里本来就有的年份。
            dimensions.append("year")
        if len({e.granularity for e in evidences}) > 1:
            dimensions.append("granularity")
        if len({e.sec_type for e in evidences if e.sec_type}) > 1:
            dimensions.append("section_type")
        return dimensions
