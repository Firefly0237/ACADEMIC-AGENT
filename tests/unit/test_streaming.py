"""流式状态机：首句阻塞闸门 → 按句推送 → 只降级不撤回（M4 §6、决策记录 D14）。

被测的是那条**物理性冲突**的解法本身：

- 需求 A（体验）：必须流式，token 边生成边推；
- 需求 B（可信）：引用全部非法 → **不展示这个回答**。

两条不能同时成立，**token 一旦出网就收不回来**。解法是把「不展示」这个能力收缩到
唯一还能执行的时刻——第一个 token 出网之前。所以这份测试最重要的三条是：

1. 不过闸时**一个 delta 都不能推**（此时「不展示」才是可执行的）；
2. 过闸后**只降级不撤回**：首句之后的越界引用只标记，不撤回、不中断；
3. `ai_answer_flagged` **只能指向已推送的 seq**。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from citequest.config.rag import GroundingSettings, StreamingSettings
from citequest.core.answer import CitationAnchor, Evidence
from citequest.core.enums import Granularity, SseEventType
from citequest.rag.grounding import GroundingChecker
from citequest.rag.streaming import AnswerStreamer, StreamEvent, StreamState


async def _tokens(text: str, *, size: int = 7) -> AsyncIterator[str]:
    """把一段文本切成小块喂进去。**切点刻意落在句子中间**——按 token 推送时
    半句状态是常态，用整句喂会让缓冲区逻辑完全测不到。"""
    for start in range(0, len(text), size):
        yield text[start : start + size]


@pytest.fixture
def evidences() -> list[Evidence]:
    return [
        Evidence(
            citation_id=n,
            anchor=CitationAnchor(paper_id=f"W{n}", chunk_id=f"W{n}:f:s001:p00001"),
            text=f"Evidence {n}.",
            title=f"Paper {n}",
            year=2021,
            granularity=Granularity.FULLTEXT,
        )
        for n in (1, 2)
    ]


def _streamer() -> AnswerStreamer:
    # 不注入 HHEM：**「HHEM 不可用」是默认状态**（骨架阶段模型未加载），
    # 流式必须在这个状态下完整可用，否则整条链路就绑死在一个可选组件上。
    return AnswerStreamer(GroundingChecker(GroundingSettings()), StreamingSettings(), GroundingSettings())


async def _collect(streamer: AnswerStreamer, text: str, evidences: list[Evidence]) -> list[StreamEvent]:
    return [event async for event in streamer.run(_tokens(text), evidences)]


class TestFirstSentenceGate:
    """闸门不过 → **完全不进流式**。"""

    async def test_blocked_stream_emits_no_delta_at_all(self, evidences: list[Evidence]) -> None:
        streamer = _streamer()
        events = await _collect(streamer, "RoPE beats everything [9]. Second sentence [1].", evidences)
        types = [e.type for e in events]
        assert SseEventType.AI_ANSWER_DELTA not in types
        assert SseEventType.AI_ANSWER_START not in types
        assert types[0] is SseEventType.AI_ANSWER_ERROR
        assert events[0].data["answer_state"] == "insufficient_evidence"
        assert streamer.state is StreamState.BLOCKED

    async def test_start_event_means_the_gate_was_passed(self, evidences: list[Evidence]) -> None:
        # `ai_answer_start` 出现即意味着首句已过 grounding 闸门——SSE 事件表的
        # 三条不变量之一。
        streamer = _streamer()
        events = await _collect(streamer, "RoPE improves extrapolation [1]. It also helps [2].", evidences)
        types = [e.type for e in events]
        assert types[0] is SseEventType.AI_ANSWER_START
        assert types[1] is SseEventType.AI_ANSWER_DELTA
        assert events[1].data["is_first_sentence"] is True

    async def test_state_machine_is_one_way(self, evidences: list[Evidence]) -> None:
        # GATING → STREAMING → DONE，没有任何一条边回到 GATING——那条边就是
        # 「撤回」，而撤回在这个架构下不可执行。
        streamer = _streamer()
        await _collect(streamer, "First [1]. Second [2]. Third [1].", evidences)
        assert streamer.state is StreamState.DONE


class TestPushThenMark:
    """首句之后：**先推后标，只降级不撤回**。"""

    async def test_out_of_range_after_first_sentence_does_not_stop_the_stream(self, evidences: list[Evidence]) -> None:
        streamer = _streamer()
        events = await _collect(streamer, "First claim [1]. Second claim [9]. Third claim [2].", evidences)
        deltas = [e for e in events if e.type is SseEventType.AI_ANSWER_DELTA]
        # 三句全部推出去了——中途拦截会让用户看到一个卡住的半截回答，比标弱支持更糟。
        assert len(deltas) == 3
        # 越界那句被标成不可核对，通过 verdicts 下发。
        flagged = [v for v in streamer.verdicts if v.reason == "引用编号越界，无法核对"]
        assert [v.citation_id for v in flagged] == [9]

    async def test_pushed_sentences_are_never_rewritten(self, evidences: list[Evidence]) -> None:
        streamer = _streamer()
        await _collect(streamer, "First claim [1]. Second claim [9].", evidences)
        # 文本已经在用户屏幕上了，改后端的副本只会让日志与用户所见不一致。
        assert streamer.sentences[1].text.strip().startswith("Second claim [9]")

    async def test_flag_must_point_at_an_already_pushed_sentence(self, evidences: list[Evidence]) -> None:
        streamer = _streamer()
        await _collect(streamer, "Only one sentence [1].", evidences)
        event = streamer.flag(0, "HHEM 判定一致性偏低")
        assert event.type is SseEventType.AI_ANSWER_FLAGGED
        assert event.data["seq"] == 0
        with pytest.raises(ValueError, match="尚未推送"):
            streamer.flag(5, "指向一个不存在的句子")


class TestDoneEvent:
    """收尾事件必须带齐三样：逐引用支持强度、降级列表、句数。"""

    async def test_done_carries_verdicts_and_degrades(self, evidences: list[Evidence]) -> None:
        streamer = _streamer()
        events = await _collect(streamer, "First [1]. Second [2].", evidences)
        done = events[-1]
        assert done.type is SseEventType.AI_ANSWER_DONE
        assert done.data["sentence_count"] == 2
        # HHEM 未加载 → 首句那条边标 unverified，且必须能在 done 里看到。
        assert done.data["citation_support"][0]["support"] == "unverified"

    async def test_trailing_buffer_is_flushed(self, evidences: list[Evidence]) -> None:
        # 最后一句没有句末标点时，残余缓冲必须被冲出来——丢掉它的表现是
        # 「回答少了最后半句」，而且不报错。
        streamer = _streamer()
        events = await _collect(streamer, "First [1]. Trailing without period [2]", evidences)
        deltas = [e for e in events if e.type is SseEventType.AI_ANSWER_DELTA]
        assert len(deltas) == 2
        assert "Trailing" in deltas[1].data["text"]
