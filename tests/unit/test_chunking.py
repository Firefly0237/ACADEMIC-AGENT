"""切块的三条硬规则（M1 §6.2）——**改切块 = 重建索引 + 全量重编码，6–10 小时**。

这份测试守的是那三条规则本身，不是实现细节：

- **R1** 聚合单位是完整段落，永不在段落中间切；
- **R2** 按 `section` 边界硬停，不跨 section 聚合；
- **R3** 目标 450 token（上界），最小 80 token（低于则并入同 section 相邻段）。

token 计数用 `WhitespaceTokenCounter`：**不是为了省事，是为了让输入长度可手算**。
测试里每个段落有几个「token」是数出来的，断言因此能写死具体数字而不是「大概」。
真实建库路径不允许用它（构造时必须显式承认是近似值）。
"""

from __future__ import annotations

import pytest

from citequest.config.corpus import CorpusSettings
from citequest.core.corpus import Paper
from citequest.core.enums import Granularity, LicenseId, SecType, SourceDataset
from citequest.ingestion.chunk_models import SourceParagraph
from citequest.ingestion.chunking import SectionAwareChunker
from citequest.ingestion.tokenization import WhitespaceTokenCounter


@pytest.fixture
def counter() -> WhitespaceTokenCounter:
    return WhitespaceTokenCounter(acknowledge_approximation=True)


@pytest.fixture
def settings() -> CorpusSettings:
    # 把 450/80 缩小到 20/5，只为让测试输入能手写。**比例关系保持不变**——
    # 被测的是规则，不是那两个具体数字（那两个数由 test_settings_defaults 守）。
    return CorpusSettings(chunk_target_tokens=20, chunk_min_tokens=5)


@pytest.fixture
def chunker(counter: WhitespaceTokenCounter, settings: CorpusSettings) -> SectionAwareChunker:
    return SectionAwareChunker(counter, settings)


@pytest.fixture
def paper() -> Paper:
    return Paper(
        paper_id="W1000000001",
        title="A Study of Chunking",
        year=2021,
        primary_category="cs.CL",
        granularity=Granularity.FULLTEXT,
        has_fulltext=True,
        license=LicenseId.CC_BY_SA_4_0,
        source_dataset=SourceDataset.UNARXIVE_OPEN_SUBSET,
        attribution={"text": "unarXive, CC BY-SA 4.0", "source_url": "https://doi.org/10.5281/zenodo.7752615"},  # type: ignore[arg-type]
    )


def _para(idx: int, words: int, section: str, sec_number: str, **kwargs: object) -> SourceParagraph:
    """造一个恰好 `words` 个空白 token 的段落。长度可手算是这份测试的前提。"""
    return SourceParagraph(
        paragraph_idx=idx,
        text=" ".join(f"w{idx}-{i}" for i in range(words)),
        section=section,
        sec_number=sec_number,
        sec_type=SecType.METHOD,
        **kwargs,  # type: ignore[arg-type]
    )


class TestSectionBoundary:
    """R2：section 边界硬停。"""

    def test_never_merges_across_sections(self, chunker: SectionAwareChunker, paper: Paper) -> None:
        # 两个 section 各 6 token，合起来 12 远小于 20——**能装下也不许合**。
        paragraphs = [_para(0, 6, "Method", "3"), _para(1, 6, "Results", "4")]
        chunks, _ = chunker.chunk_paper(paper, paragraphs)
        assert len(chunks) == 2
        assert [c.section for c in chunks] == ["Method", "Results"]

    def test_same_named_sections_that_are_not_adjacent_stay_separate(
        self, chunker: SectionAwareChunker, paper: Paper
    ) -> None:
        # 附录里又出现一个 "Discussion"：按名分组会把物理上不相邻的段落粘在一起，
        # 引用锚点随之错位。切分依据必须是「相邻段的 section 标识发生变化」。
        paragraphs = [
            _para(0, 6, "Discussion", "5"),
            _para(1, 6, "Appendix", "A"),
            _para(2, 6, "Discussion", "B"),
        ]
        chunks, stats = chunker.chunk_paper(paper, paragraphs)
        assert stats.section_count == 3
        assert len(chunks) == 3


class TestTargetAndMinimum:
    """R1 + R3：段落完整、450 是上界、80 以下并入前一块。"""

    def test_target_is_an_upper_bound_not_a_lower_bound(self, chunker: SectionAwareChunker, paper: Paper) -> None:
        # 三段各 8 token：8+8=16 ≤ 20 可以合，再加一段就 24 > 20 必须先封口。
        # 若实现写成「吸完再判断」，就会得到一个 24 token 的块——450 会变成下界。
        paragraphs = [_para(i, 8, "Method", "3") for i in range(3)]
        chunks, _ = chunker.chunk_paper(paper, paragraphs)
        assert [c.token_count for c in chunks] == [16, 8]

    def test_paragraph_is_never_split_in_the_middle(self, chunker: SectionAwareChunker, paper: Paper) -> None:
        # 每块的文本都必须是若干个完整段落拼起来的。
        paragraphs = [_para(i, 8, "Method", "3") for i in range(3)]
        chunks, _ = chunker.chunk_paper(paper, paragraphs)
        rendered = "\n\n".join(c.text for c in chunks)
        for para in paragraphs:
            assert para.text in rendered

    def test_short_tail_merges_into_previous_block(self, chunker: SectionAwareChunker, paper: Paper) -> None:
        # 18 + 3：第二段单独成块只有 3 token（< 5），必须并入前一块。
        paragraphs = [_para(0, 18, "Method", "3"), _para(1, 3, "Method", "3")]
        chunks, stats = chunker.chunk_paper(paper, paragraphs)
        assert len(chunks) == 1
        assert stats.short_tail_merges == 1
        # 并完重新计一次而不是把两个数相加——阈值必须落在最终文本上。
        assert chunks[0].token_count == 21

    def test_section_shorter_than_minimum_is_dropped(self, chunker: SectionAwareChunker, paper: Paper) -> None:
        # 整个 section 都不足下限且没有前一块可并 → 丢弃并计数。
        paragraphs = [_para(0, 3, "Acknowledgements", "6")]
        chunks, stats = chunker.chunk_paper(paper, paragraphs)
        assert chunks == []
        assert stats.dropped_short_sections == 1

    def test_ceiling_is_derived_not_configured(self, chunker: SectionAwareChunker) -> None:
        # 单块理论上界 = target + min，是「短尾并入前一块」的推论而不是一个配置项。
        assert chunker.max_possible_tokens == 25

    def test_rejects_minimum_not_below_target(self, counter: WhitespaceTokenCounter) -> None:
        with pytest.raises(ValueError, match="必须小于"):
            SectionAwareChunker(counter, CorpusSettings(chunk_target_tokens=20, chunk_min_tokens=20))


class TestChunkIdentity:
    """chunk_id 的确定性与唯一性——**重复主键在 upsert 下是静默覆盖**。"""

    def test_rechunking_is_deterministic(self, chunker: SectionAwareChunker, paper: Paper) -> None:
        paragraphs = [_para(i, 8, "Method", "3") for i in range(3)]
        first, _ = chunker.chunk_paper(paper, paragraphs)
        second, _ = chunker.chunk_paper(paper, paragraphs)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_ids_are_unique_within_a_paper(self, chunker: SectionAwareChunker, paper: Paper) -> None:
        paragraphs = [_para(i, 8, "Method", "3") for i in range(6)]
        chunks, _ = chunker.chunk_paper(paper, paragraphs)
        assert len({c.chunk_id for c in chunks}) == len(chunks)


class TestNonTextAndCitations:
    """C 类字段与非正文段落。"""

    def test_non_text_paragraphs_are_skipped_and_counted(self, chunker: SectionAwareChunker, paper: Paper) -> None:
        paragraphs = [
            _para(0, 8, "Method", "3"),
            _para(1, 8, "Method", "3", content_type="table"),
        ]
        chunks, stats = chunker.chunk_paper(paper, paragraphs)
        assert stats.skipped_non_text_paragraphs == 1
        assert len(chunks) == 1

    def test_cited_ids_are_deduplicated_but_ordered(self, chunker: SectionAwareChunker, paper: Paper) -> None:
        # 同一段里同一篇被引多次是常态；边表按 (chunk_id, cited_id) 唯一，
        # 重复只会让写入端做无用功。
        paragraphs = [
            _para(0, 8, "Method", "3", cited_open_alex_ids=["W9", "W8", "W9"]),
        ]
        chunks, _ = chunker.chunk_paper(paper, paragraphs)
        assert chunks[0].cited_open_alex_ids == ["W9", "W8"]


class TestAbstractLayer:
    """摘要层：**恒为 1 个 chunk，不切分**（M1 §6.9）。"""

    def test_abstract_is_exactly_one_chunk(self, chunker: SectionAwareChunker, paper: Paper) -> None:
        abstract = " ".join(f"a{i}" for i in range(60))  # 远超 target=20，仍然不切
        chunk = chunker.chunk_abstract(paper, "Title Words Here", abstract)
        assert chunk is not None
        assert chunk.granularity is Granularity.ABSTRACT
        assert chunk.sec_type is SecType.ABSTRACT
        # 四元锚点退化成 paper_id 一元——这不是缺陷，是粒度本身的诚实表达。
        assert (chunk.section, chunk.sec_number) == (None, None)

    def test_paper_without_abstract_is_dropped(self, chunker: SectionAwareChunker, paper: Paper) -> None:
        # 只剩 title 达不到 80 token 下限（这里是 5），返回 None 表示不进摘要层。
        assert chunker.chunk_abstract(paper, "Short", "") is None
