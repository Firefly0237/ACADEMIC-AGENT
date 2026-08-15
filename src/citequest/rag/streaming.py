"""流式状态机：首句阻塞闸门 → 按句推送 → 只降级不撤回（M4 §6、决策记录 D14）。

**冲突是物理性的，不是参数没调好**（M4 §6.1）：

- 需求 A（体验）：首 token 预算 p50 ≤1.0s，所以必须流式，token 边生成边推。
- 需求 B（可信）：faithfulness 不达标 / 引用全部非法 → **不展示这个回答**。

**这两条不能同时成立。token 一旦出网就收不回来。** 「不展示」在逐 token 推送的
架构下**根本没有可执行的时刻**——等你算出 faithfulness，回答已经在用户屏幕上了。

工业界公认这个约束：NVIDIA NeMo Guardrails 官方文档主动承认 rail 触发时违规文本
"might have already been sent to the user. It's the responsibility of the caller to
manage this situation"。**引用它比自己编一套说辞可信得多。**

**解法是三段式：**

```
生成第一句 ─► 首句阻塞闸门（确定性引用校验 + HHEM 一次前向）
                 │过闸                          │不过闸
                 ▼                              ▼
          发 ai_answer_start              完全不进流式
          进入流式                        直接返回「信息不足」
                 │                        （此时还没推任何 token，
                 ▼                          「不展示」才是可执行的）
        首句之后：按句边界推送
        逐句确定性校验，保留前一句作判定上下文
        语义 = 先推后标，只降级不撤回
                 ▼
        ai_answer_done：下发逐引用支持强度
```

**四条代价全部写出来**（M4 §6.4）：
1. 首 token +375ms(p50) / +860ms(p95)【设计预算，未实测】；
2. 彻底放弃「整篇 faithfulness 低就撤回整个回答」——首句之后只能降级标注；
3. **跨句合成型幻觉仍会漏**：单看每句都被支持，合起来的结论没被任何一段支持；
4. 推送逻辑复杂度上升：从「攒 N token 就推」变成「按句边界推」。

从根上解这个问题的方向是 attribute-first-then-generate（ACL 2024, arXiv:2403.17104）
——让 grounding 变成生成时的构造物而不是事后的过滤器。**它是重构级改动，明确标 v2。**
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from citequest.config.rag import GroundingSettings, StreamingSettings
from citequest.core.answer import AnswerSentence, Evidence, GroundingVerdict
from citequest.core.enums import CitationSupport, DegradeReason, SseEventType
from citequest.observability.logging import get_logger
from citequest.rag.citations import bind_anchors, parse_citations
from citequest.rag.grounding import GateResult, GroundingChecker
from citequest.rag.sentences import Sentence, iter_complete_sentences, split_sentences

__all__ = ["AnswerStreamer", "StreamEvent", "StreamState"]

_logger = get_logger(__name__)


class StreamState(StrEnum):
    """状态机的五个状态。

    **状态是单向的**：`GATING → STREAMING → DONE`，或 `GATING → BLOCKED`。
    没有任何一条边从 `STREAMING` 回到 `GATING`——那条边就是「撤回」，
    而撤回在这个架构下不可执行。把它从状态图里删掉，是让这条语义**结构上**成立，
    而不是靠调用方自觉。
    """

    GATING = "gating"
    STREAMING = "streaming"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class StreamEvent:
    """一个待下发的 SSE 事件。

    这里只产出**语义事件**，不拼 SSE 报文——协议封装是 api 层的事（`sse-starlette`）。
    分开的理由是 MCP 那条路也要消费同一批事件，而它没有 SSE。
    """

    type: SseEventType
    data: dict[str, Any] = field(default_factory=dict)


class AnswerStreamer:
    """把 token 流变成「过闸的首句 + 按句推送 + 只降级不撤回」的事件流。

    **推送粒度必须与校验粒度一致**（M4 §6.3）：按 token 或按固定字节推送，
    会出现「半句已推、校验结果无处落地」的状态。句子这个粒度有两个独立来源——
    MiniCheck 官方 API 就是句子级，Anthropic Citations API 默认按句切块且
    明确不引用小于一句的片段。
    """

    def __init__(
        self,
        checker: GroundingChecker,
        streaming: StreamingSettings,
        grounding: GroundingSettings,
        *,
        language: str = "EN",
    ) -> None:
        self._checker = checker
        self._streaming = streaming
        self._grounding = grounding
        self._language = language

        self.state = StreamState.GATING
        self.sentences: list[AnswerSentence] = []
        self.verdicts: list[GroundingVerdict] = []
        self.degraded: list[DegradeReason] = []
        self.block_reason: str | None = None
        self._seq = 0
        self._previous_text: str | None = None

    async def run(self, tokens: AsyncIterator[str], evidences: list[Evidence]) -> AsyncIterator[StreamEvent]:
        """驱动整个状态机，产出事件流。

        `evidences` 在进入本方法前就已经定死——**控制流不可被 context 改变**
        （M4 §3.4 第 2 道防线，也是最强的一道）：路由、检索参数、工具调用、
        granularity 过滤全部在生成之前就已确定，生成阶段没有任何一个下游动作
        读取模型输出去做决策。即使注入成功地让模型输出一句怪话，它也改不了任何控制流
        ——因为这条链路上生成是终点，不是中间节点。
        """
        buffer = ""
        gated = False

        try:
            async for token in tokens:
                buffer += token
                complete, buffer = iter_complete_sentences(buffer)
                if not complete:
                    continue

                if not gated:
                    # ---- 第一段：首句阻塞闸门 ----
                    first = complete[0]
                    gate = self._gate(first, evidences)
                    if not gate.passed:
                        # **完全不进流式**：此时尚未推送任何 token，「不展示」是可执行的。
                        self.state = StreamState.BLOCKED
                        yield StreamEvent(
                            type=SseEventType.AI_ANSWER_ERROR,
                            data={"answer_state": "insufficient_evidence", "reason": gate.reason},
                        )
                        return
                    gated = True
                    self.state = StreamState.STREAMING
                    # `ai_answer_start` 出现即意味着首句已过 grounding 闸门——
                    # 这是 SSE 事件表的三条不变量之一（core.enums.SseEventType）。
                    yield StreamEvent(type=SseEventType.AI_ANSWER_START, data={"answer_state": "normal"})
                    yield self._emit(first, evidences, is_first=True)
                    for sentence in complete[1:]:
                        yield self._emit(sentence, evidences)
                    continue

                # ---- 第二段：按句推送 ----
                for sentence in complete:
                    yield self._emit(sentence, evidences)

        except Exception as exc:
            # 超时语义全系统只有一种（决策记录 §4.9）：**已产生内容一律保留、
            # 不足部分标注**；不因超时丢弃已生成内容，也不用断连表达超时。
            self.degraded.append(DegradeReason.GENERATION_TIMEOUT)
            _logger.warning("stream_interrupted", error_type=type(exc).__name__, pushed=len(self.sentences))
            if not gated:
                self.state = StreamState.FAILED
                yield StreamEvent(
                    type=SseEventType.AI_ANSWER_ERROR,
                    data={"answer_state": "degraded_no_answer", "reason": "生成中断且首句未过闸"},
                )
                return

        # ---- 收尾：残余缓冲里的最后一句 ----
        for sentence in split_sentences(buffer):
            if not gated:
                gate = self._gate(sentence, evidences)
                if not gate.passed:
                    self.state = StreamState.BLOCKED
                    yield StreamEvent(
                        type=SseEventType.AI_ANSWER_ERROR,
                        data={"answer_state": "insufficient_evidence", "reason": gate.reason},
                    )
                    return
                gated = True
                self.state = StreamState.STREAMING
                yield StreamEvent(type=SseEventType.AI_ANSWER_START, data={"answer_state": "normal"})
                yield self._emit(sentence, evidences, is_first=True)
                continue
            yield self._emit(sentence, evidences)

        self.state = StreamState.DONE if gated else StreamState.BLOCKED
        yield StreamEvent(
            type=SseEventType.AI_ANSWER_DONE,
            data={
                "citation_support": [v.model_dump() for v in self.verdicts],
                "degraded": [d.value for d in self.degraded],
                "sentence_count": len(self.sentences),
            },
        )

    # ---- 内部步骤 -------------------------------------------------------

    def _gate(self, sentence: Sentence, evidences: list[Evidence]) -> GateResult:
        """跑首句闸门并把降级与判定收进状态。"""
        gate = self._checker.gate_first_sentence(sentence.text, evidences, language=self._language)
        if gate.degraded is not None:
            self.degraded.append(gate.degraded)
        self.verdicts.extend(gate.verdicts)
        if not gate.passed:
            self.block_reason = gate.reason
        return gate

    def _emit(self, sentence: Sentence, evidences: list[Evidence], *, is_first: bool = False) -> StreamEvent:
        """推一句，并在推之后做逐句确定性校验（**先推后标**）。

        顺序是有意的：先构造事件、再记状态，因为「已推送」这个事实一旦成立就不可逆。
        把校验放在推送之后，正是「只降级不撤回」在代码里的形状——
        如果这里写成「先校验、不合格就不推」，那就是在流式中途做拦截，
        而中途拦截会让用户看到一个**卡住的半截回答**，比标一个弱支持更糟。
        """
        valid_ids = {e.citation_id for e in evidences}
        citations, out_of_range, comma_form = parse_citations(sentence.text, valid_ids)
        citations = bind_anchors(citations, evidences)

        if out_of_range and not is_first:
            # 首句之后出现越界引用**不撤回**，只降级：把这句的引用标记成不可核对，
            # 并通过 ai_answer_flagged 指回它的 seq（由调用方消费 verdicts 下发）。
            _logger.warning("citation_out_of_range_after_first", seq=self._seq, ids=out_of_range)
            for citation_id in out_of_range:
                self.verdicts.append(
                    GroundingVerdict(
                        sentence_seq=self._seq,
                        citation_id=citation_id,
                        support=CitationSupport.UNVERIFIED,
                        reason="引用编号越界，无法核对",
                    )
                )
        if comma_form:
            # `[1,3]` 是被 prompt 明确禁止的写法。它不影响解析（正则两种都收），
            # 但**必须打点**——它是 prompt 漂移的早期信号，而 prompt 漂移最终会
            # 表现为「某天引用边突然扫不出来了」。
            _logger.info("citation_comma_form", seq=self._seq)

        answer_sentence = AnswerSentence(
            seq=self._seq,
            text=sentence.text,
            citations=citations,
            is_first_sentence=is_first,
        )
        self.sentences.append(answer_sentence)

        if not is_first and self._grounding.online_per_edge_enabled:
            # 逐句在线判定是**默认关闭的开关**：花在线延迟买一个不敢用来拦截的软信号，
            # 必要性判据不过关。等 M7 实测出 HHEM 单次前向的真实毫秒数再决定是否打开。
            grounding = self._checker.check_sentence(self._seq, sentence.text, evidences)
            self.verdicts.extend(
                self._checker.score_edges(
                    grounding,
                    sentence.text,
                    evidences,
                    language=self._language,
                    previous_sentence=self._previous_text,
                )
            )

        self._previous_text = sentence.text
        self._seq += 1
        return StreamEvent(
            type=SseEventType.AI_ANSWER_DELTA,
            data={
                "seq": answer_sentence.seq,
                "text": answer_sentence.text,
                "citations": [c.citation_id for c in citations],
                "is_first_sentence": is_first,
            },
        )

    def flag(self, seq: int, reason: str) -> StreamEvent:
        """事后降级一个**已推送**的句子（`ai_answer_flagged`）。

        它永远指向一个已推送的 seq——这是 SSE 事件表的第三条不变量：
        **只能降级不能撤回**。这个方法刻意不修改 `self.sentences` 里的文本：
        文本已经在用户屏幕上了，改后端的副本只会让日志与用户所见不一致。
        """
        if seq >= self._seq:
            raise ValueError(f"ai_answer_flagged 只能指向已推送的句子，seq={seq} 尚未推送")
        return StreamEvent(type=SseEventType.AI_ANSWER_FLAGGED, data={"seq": seq, "reason": reason})
