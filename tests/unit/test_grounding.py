"""引用校验与 grounding（M4 §5、决策记录 D13/§4.6）。

三层防线里**只有第一层敢用来拦截**，这份测试守的就是这条边界：

| 层 | 手段 | 敢不敢拦 |
|---|---|---|
| 确定性硬校验 | `[N]` 越界 / 全部非法（纯字符串 + 集合成员判定） | **敢。唯一敢硬拦的条件** |
| 在线软信号 | HHEM-2.1-Open，只在首句跑一次，自报 F1 仅 44.8–66.8 | **不敢**，只做 UI 降级 |
| 离线评测 | MiniCheck / ALCE / RAGAS | 不进在线路径 |

HHEM 通过构造注入桩：**「HHEM 不可用」是一条正常降级链**（跳过软信号、只跑确定性
校验、`citation_support` 全标 `unverified`），它必须能在没有模型的环境下被断言。
"""

from __future__ import annotations

import pytest

from citequest.config.rag import GroundingSettings
from citequest.core.answer import CitationAnchor, Evidence
from citequest.core.enums import CitationSupport, DegradeReason, Granularity, SecType
from citequest.rag.citations import allocate_citation_ids, citation_ids_in, parse_citations, unsupported_sentences
from citequest.rag.grounding import GroundingChecker


class StubHhem:
    """可控的 HHEM 桩。`score_value=None` 模拟「模型返回未判定」。"""

    def __init__(self, *, usable: bool = True, score_value: float | None = 0.9) -> None:
        self._usable = usable
        self._score = score_value

    def available(self) -> bool:
        return self._usable

    def score(self, premise: str, hypothesis: str, *, language: str = "EN") -> float | None:
        return self._score


@pytest.fixture
def evidences() -> list[Evidence]:
    return [
        Evidence(
            citation_id=n,
            anchor=CitationAnchor(paper_id=f"W{n}", chunk_id=f"W{n}:f:s001:p00001", section="Method", sec_number="3"),
            text=f"Evidence text number {n}.",
            title=f"Paper {n}",
            year=2020 + n,
            granularity=Granularity.FULLTEXT,
            sec_type=SecType.METHOD,
        )
        for n in (1, 2)
    ]


class TestDeterministicHardGate:
    """唯一敢硬拦的条件：`[N]` 越界。**误判率结构性为零。**"""

    def test_out_of_range_citation_blocks_the_first_sentence(self, evidences: list[Evidence]) -> None:
        checker = GroundingChecker(GroundingSettings())
        gate = checker.gate_first_sentence("RoPE outperforms ALiBi [7].", evidences)
        assert gate.passed is False
        # 不过闸时 reason 必须落到用户可读的解释上——「引用全部非法」是可行动的
        # 信息，「抱歉我不知道」不是。
        assert "7" in (gate.reason or "")

    def test_legal_citation_passes(self, evidences: list[Evidence]) -> None:
        checker = GroundingChecker(GroundingSettings())
        assert checker.gate_first_sentence("RoPE improves extrapolation [1].", evidences).passed is True

    def test_empty_evidence_never_passes(self) -> None:
        # 没有证据就产不出可核对的回答，这与 [N] 越界是同一类硬条件。
        checker = GroundingChecker(GroundingSettings())
        assert checker.gate_first_sentence("Some claim without evidence.", []).passed is False

    def test_hard_gate_can_be_turned_off_only_explicitly(self, evidences: list[Evidence]) -> None:
        # 关掉它必须改配置这一行，而不是靠某处悄悄加一个条件。
        checker = GroundingChecker(GroundingSettings(hard_gate_on_citation_out_of_range=False))
        assert checker.gate_first_sentence("Claim [7].", evidences).passed is True

    def test_check_sentence_separates_legal_from_out_of_range(self, evidences: list[Evidence]) -> None:
        checker = GroundingChecker(GroundingSettings())
        grounding = checker.check_sentence(0, "A [1] and B [9].", evidences)
        assert grounding.citation_ids == [1]
        assert grounding.out_of_range == [9]
        assert grounding.has_illegal_citation is True


class TestHhemIsOnlyASoftSignal:
    """HHEM 低分**不拦截**——它只让 `citation_support` 落到 weak。"""

    def test_low_score_does_not_block(self, evidences: list[Evidence]) -> None:
        settings = GroundingSettings(supported_threshold=0.8, weak_threshold=0.5)
        checker = GroundingChecker(settings, hhem=StubHhem(score_value=0.01))
        gate = checker.gate_first_sentence("A weakly supported claim [1].", evidences)
        assert gate.passed is True
        assert gate.verdicts[0].support is CitationSupport.WEAK

    def test_unavailable_hhem_degrades_instead_of_failing(self, evidences: list[Evidence]) -> None:
        checker = GroundingChecker(GroundingSettings(), hhem=StubHhem(usable=False))
        gate = checker.gate_first_sentence("A claim [1].", evidences)
        assert gate.passed is True
        assert gate.degraded is DegradeReason.HHEM_UNAVAILABLE
        assert gate.verdicts[0].support is CitationSupport.UNVERIFIED

    def test_uncalibrated_thresholds_yield_unverified(self, evidences: list[Evidence]) -> None:
        # 阈值未标定时一律 unverified：拿一对没有出处的数去给用户的引用打
        # 「弱支持」标记，比不标更糟——它看起来像一个判断。
        checker = GroundingChecker(GroundingSettings(), hhem=StubHhem(score_value=0.95))
        verdicts = checker.gate_first_sentence("A claim [1].", evidences).verdicts
        assert verdicts[0].support is CitationSupport.UNVERIFIED
        # 未判定不允许携带分数——「判过但弱」和「根本没判」对用户的含义完全不同。
        assert verdicts[0].hhem_score is None

    def test_non_english_answer_only_gets_deterministic_check(self, evidences: list[Evidence]) -> None:
        # HHEM 仅英文。中文回答标 unverified 而不是伪造一个分数。
        settings = GroundingSettings(supported_threshold=0.8, weak_threshold=0.5)
        checker = GroundingChecker(settings, hhem=StubHhem(score_value=0.99))
        gate = checker.gate_first_sentence("旋转位置编码更优 [1]。", evidences, language="ZH")
        assert gate.passed is True
        assert gate.verdicts[0].support is CitationSupport.UNVERIFIED
        assert gate.degraded is DegradeReason.HHEM_UNAVAILABLE

    def test_edges_are_scored_per_citation_not_per_chunk(self, evidences: list[Evidence]) -> None:
        # 判定单元是**引用边**，不是「句数 × chunk 数」。一句标了两个引用就是两条边。
        settings = GroundingSettings(supported_threshold=0.8, weak_threshold=0.5)
        checker = GroundingChecker(settings, hhem=StubHhem(score_value=0.9))
        grounding = checker.check_sentence(0, "Both agree [1][2].", evidences)
        verdicts = checker.score_edges(grounding, "Both agree [1][2].", evidences)
        assert [v.citation_id for v in verdicts] == [1, 2]
        assert all(v.support is CitationSupport.SUPPORTED for v in verdicts)


class TestCitationParsing:
    """`[N]` 是系统分配的有限集合——**硬 gate 成立的全部前提**。"""

    def test_duplicate_markers_count_as_one_edge(self) -> None:
        # citation precision 的分母定义是**边数**不是标记出现次数。
        assert citation_ids_in("...[3] 并且 ...[3]") == [3]

    def test_comma_form_is_parsed_but_flagged(self) -> None:
        # `[1,3]` 被 prompt 明确禁止。校验必须能看见违规写法，
        # 看不见就等于默认它没发生。
        citations, out_of_range, comma_form = parse_citations("A [1,2].", {1, 2})
        assert [c.citation_id for c in citations] == [1, 2]
        assert out_of_range == []
        assert comma_form is True

    def test_ids_are_allocated_by_the_system_from_one(self, retrieved_fulltext_chunk, retrieved_abstract_chunk) -> None:
        # 模型无法凭空造出一个合法引用边——这是注入防护的第 3 道。
        allocated = allocate_citation_ids([retrieved_fulltext_chunk, retrieved_abstract_chunk])
        assert [e.citation_id for e in allocated] == [1, 2]
        # granularity 逐条带上：全链路唯一「不写就会静默产幻」的字段。
        assert allocated[1].granularity is Granularity.ABSTRACT
        # 摘要块的 section/para 直接为 None 而不是填 "N/A"。
        assert allocated[1].anchor.section is None

    def test_sentences_without_citation_are_counted_not_hidden(self) -> None:
        # 沿引用边判定的已知代价：完全不标引用的编造句没有边可判。
        # 补法不是把它藏起来，而是把它变成一个数字。
        assert unsupported_sentences(["Claim with [1].", "Claim without."]) == [1]


class TestConflictDimensions:
    """证据冲突：**不裁决，并列呈现**（M4 §5.6）。"""

    def test_year_gap_and_granularity_are_reported(self, evidences: list[Evidence]) -> None:
        wide = [*evidences, evidences[0].model_copy(update={"citation_id": 3, "year": 2015})]
        dimensions = GroundingChecker.conflict_dimensions(wide)
        assert "year" in dimensions

    def test_no_dimension_when_evidence_is_homogeneous(self, evidences: list[Evidence]) -> None:
        # 零额外模型调用：只汇报元数据层面已有的差异。检测一个不影响任何动作的
        # 信号是纯成本，所以这里刻意什么都不做。
        assert GroundingChecker.conflict_dimensions(evidences[:1]) == []
