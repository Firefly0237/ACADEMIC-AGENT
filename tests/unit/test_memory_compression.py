"""会话状态与上下文压缩（M5 §4、§5）。

守的是三层容器那条**成本收益最悬殊**的设计：

| 层 | 进 planner 上下文吗 |
|---|---|
| 最近 N 轮原文（N=2） | **进** |
| 更早轮次的结构化摘要 | **进**（只有摘要） |
| 原始 chunk 全文（`evidence_pool`） | **不进** |

以及压缩的两条不变量：

1. **前三个字段是集合运算，永远不漂移**；只有 `narrative` 交给 LLM；
2. **增量叠加 O(n) 而不是全量重算 O(n²)**——这个收益只有在 `input_tokens`
   被记下来时才是可验证的，否则它只是一段推导。
"""

from __future__ import annotations

from citequest.config.agent import MemorySettings
from citequest.core.enums import RetrievalMode
from citequest.core.search import RetrievedChunk
from citequest.orchestration.memory import (
    ConfirmedClaim,
    ExclusionSet,
    SessionState,
    SessionSummary,
    Turn,
    compress,
)


def _turn(index: int, *, words: int = 20, papers: list[str] | None = None) -> Turn:
    return Turn(
        index=index,
        user_query=f"query number {index}",
        answer_digest=" ".join(f"w{i}" for i in range(words)),
        cited_papers=papers or [f"W{index}"],
        retrieval_mode=RetrievalMode.HYBRID,
    )


class TestThreeContainers:
    """原始 chunk **永远不进 planner 上下文**——这条是结构上的，不靠调用方自觉。"""

    def test_evidence_pool_never_leaks_into_planner_context(self, retrieved_fulltext_chunk: RetrievedChunk) -> None:
        state = SessionState(thread_id="t1")
        state.record_turn(_turn(0), [retrieved_fulltext_chunk])
        context = state.planner_context(keep_raw_turns=2)
        assert "evidence_pool" not in context
        # 但证据本身没丢：生成节点仍能按 chunk_id 精确取原文做引用锚点。
        assert retrieved_fulltext_chunk.chunk_id in state.evidence_pool

    def test_only_recent_turns_keep_their_raw_text(self) -> None:
        state = SessionState(thread_id="t1")
        for i in range(5):
            state.record_turn(_turn(i))
        context = state.planner_context(keep_raw_turns=2)
        recent = context["recent_turns"]
        assert isinstance(recent, list)
        assert [t["index"] for t in recent] == [3, 4]

    def test_focus_papers_are_most_recent_first(self) -> None:
        # 指代消解要的是「最近讨论过的」，所以重复出现的论文要被顶到最前，
        # 而不是留在它第一次出现的位置。
        state = SessionState(thread_id="t1")
        state.record_turn(_turn(0, papers=["W1"]))
        state.record_turn(_turn(1, papers=["W2"]))
        state.record_turn(_turn(2, papers=["W1"]))
        assert state.focus_papers[:2] == ["W1", "W2"]


class TestCompressionGates:
    """**轮数闸门 + token 水位闸门，任一命中即压**——两个都要，缺一个都会漏。"""

    def test_turn_gate_fires(self) -> None:
        settings = MemorySettings(compress_after_turns=3, compress_after_tokens=10**6)
        state = SessionState(thread_id="t1")
        for i in range(3):
            state.record_turn(_turn(i))
        assert state.should_compress(settings) is True

    def test_token_gate_fires_even_when_turn_count_is_low(self) -> None:
        # 只有轮数闸门时：一次粘贴 3,000 token 的摘要，三轮就撑爆上下文，
        # 而计数器才走到 3。
        settings = MemorySettings(compress_after_turns=100, compress_after_tokens=200)
        state = SessionState(thread_id="t1")
        state.record_turn(_turn(0, words=2000))
        assert state.should_compress(settings) is True

    def test_neither_gate_fires_on_a_short_session(self) -> None:
        settings = MemorySettings(compress_after_turns=6, compress_after_tokens=8_000)
        state = SessionState(thread_id="t1")
        state.record_turn(_turn(0))
        assert state.should_compress(settings) is False


class TestCompressionIsMostlyDeterministic:
    """三个结构化字段做集合运算——**即使 narrative 漂了，它们仍然精确**。"""

    def test_read_papers_is_a_union(self) -> None:
        state = SessionState(thread_id="t1", summary=SessionSummary(read_papers={"W0"}))
        state.record_turn(_turn(1, papers=["W1"]))
        state.record_turn(_turn(2, papers=["W2", "W1"]))
        result = compress(state, MemorySettings())
        assert result.summary.read_papers == {"W0", "W1", "W2"}

    def test_narrative_is_kept_when_no_writer_is_injected(self) -> None:
        # 传 None 是**降级不是等价实现**：它让压缩在没有 LLM 的环境下仍然把三个
        # 确定性字段合并掉，但明确不重写 narrative。
        state = SessionState(thread_id="t1", summary=SessionSummary(narrative="旧摘要"))
        state.record_turn(_turn(1))
        result = compress(state, MemorySettings())
        assert result.narrative_rewritten is False
        assert result.summary.narrative == "旧摘要"

    def test_narrative_is_truncated_to_the_budget(self) -> None:
        state = SessionState(thread_id="t1")
        state.record_turn(_turn(1))
        settings = MemorySettings(summary_max_tokens=10)  # 10 × 4 字符 = 40 字符上限
        result = compress(state, settings, narrative_writer=lambda old, turns: "x" * 500)
        assert len(result.summary.narrative) == 40

    def test_counter_resets_after_compression(self) -> None:
        state = SessionState(thread_id="t1")
        for i in range(3):
            state.record_turn(_turn(i))
        result = compress(state, MemorySettings())
        assert result.compressed_turns == 3
        assert state.turns_since_compress == 0

    def test_incremental_reads_less_than_full_recompute(self) -> None:
        # **选增量不是省了 5 倍，是把一个二次增长的成本换成了线性。**
        # 这条断言就是那句话的可验证形态：同一个会话状态，两种模式的输入 token 不同。
        def build() -> SessionState:
            state = SessionState(thread_id="t1")
            for i in range(10):
                state.record_turn(_turn(i, words=200))
            state.turns_since_compress = 2  # 只有最近 2 轮是待压缩的
            return state

        incremental = compress(build(), MemorySettings(incremental_compression=True))
        full = compress(build(), MemorySettings(incremental_compression=False))
        assert incremental.incremental is True
        assert incremental.input_tokens < full.input_tokens


class TestExclusionSet:
    """排除条件：**区间取交、集合取并，确定性合并，不经过 LLM**。"""

    def test_year_range_intersects(self) -> None:
        merged = ExclusionSet(year_min=2019, year_max=2024).merged_with(ExclusionSet(year_min=2021, year_max=2023))
        assert (merged.year_min, merged.year_max) == (2021, 2023)

    def test_excluded_sets_union(self) -> None:
        merged = ExclusionSet(exclude_papers={"W1"}).merged_with(ExclusionSet(exclude_papers={"W2"}))
        assert merged.exclude_papers == {"W1", "W2"}

    def test_one_sided_bound_survives(self) -> None:
        # 一侧为 None 时不能把已有的界抹掉——那会让「只看 2023 年之后的」
        # 在下一轮悄悄失效。
        merged = ExclusionSet(year_min=2021).merged_with(ExclusionSet())
        assert merged.year_min == 2021


class TestClaimsAreTraceable:
    """已确认结论必须带引用与轮次——**不能追溯来源的结论不如不存**。"""

    def test_claim_requires_provenance_fields(self) -> None:
        claim = ConfirmedClaim(claim="RoPE 外推更好", cited=["W1"], turn=2)
        assert claim.cited == ["W1"]
        assert claim.confirmed_at is not None
