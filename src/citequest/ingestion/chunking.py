"""切块（M1 §6）——数据层最重的一节。

切块决定召回单位、重排输入尺寸、生成上下文预算、引用最小锚点。**改切块 = 重建
索引 + 一次全量重编码，6–10 小时量级**，所以它必须在建库前一次定死。

三条硬规则（M1 §6.2）：

- **R1** 聚合单位是完整段落，永不在段落中间切；
- **R2** 按 `section` 边界**硬停**，不跨 section 聚合；
- **R3** 目标 450 token（上界），最小 80 token（低于则并入同 section 相邻段）。

一句话算法：在同一个 section 内从第一段起依次吸入完整段落，累计 token 一旦要超过
450 就封口开新块；section 结束时无条件封口；封口后不足 80 token 的块并入同 section
的前一个块。

**overlap = 0，而且它不是一个独立选择**（§6.7）：overlap 的作用是用冗余补救一个
落在错误位置的切点，而 R1 让切点永远落在段落边界——段落边界是比句子边界更强的
语义边界，要补救的那个错误在结构上不会发生。为一个不存在的问题付 10–20% 的存储、
编码与索引成本没有理由。代价是跨段指代会丢，处理方式在检索层（`group_size=3`
把同一篇的多段一起带回），不在存储层。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from citequest.config.corpus import CorpusSettings
from citequest.core.corpus import Chunk, Paper
from citequest.core.enums import Granularity, SecType
from citequest.ingestion.chunk_models import ChunkingStats, SourceParagraph, build_chunk_id
from citequest.ingestion.sentence_split import TextPiece, split_long_text
from citequest.ingestion.tokenization import TokenCounter
from citequest.observability.logging import get_logger

__all__ = ["SectionAwareChunker"]

_logger = get_logger(__name__)

#: 只有正文类段落进检索。表格/图注碎片不特殊处理——为它们写识别规则的误伤率
#: 高于收益，交给 80 token 最小长度规则吸收（M1 §5.2）。
_TEXT_CONTENT_TYPES = frozenset({"paragraph", "list", "listing", "text"})


@dataclass
class _Block:
    """封口前的累加块。用可变 dataclass 而不是 pydantic，是因为它每段都要改一次。"""

    section: str | None
    sec_number: str | None
    sec_type: SecType | None
    section_ordinal: int
    start_paragraph_idx: int
    end_paragraph_idx: int
    texts: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)
    token_count: int = 0

    def render(self) -> str:
        # 段落之间保留空行：块内的段落边界对生成层有用（引用要指到具体段落），
        # 压成一行会让 `paragraph_idx` 这个锚点在文本上再也定位不回去。
        return "\n\n".join(t for t in self.texts if t)


@dataclass
class _Segment:
    """长段回退档切出来的片段。它冒充一个段落参与后续聚合。"""

    text: str
    cited_ids: list[str]
    formulas: list[str]
    token_count: int


class SectionAwareChunker:
    """按 section 边界硬停的段落聚合器。

    一个实例对应一套切块参数。参数全部来自 `CorpusSettings`（决策记录 §4.1），
    本类**不写死任何数值**——写死就等于在参数正本之外又造了一份。
    """

    def __init__(self, token_counter: TokenCounter, settings: CorpusSettings | None = None) -> None:
        self._counter = token_counter
        self._settings = settings or CorpusSettings()
        if self._settings.chunk_min_tokens >= self._settings.chunk_target_tokens:
            raise ValueError(
                f"chunk_min_tokens({self._settings.chunk_min_tokens}) 必须小于 "
                f"chunk_target_tokens({self._settings.chunk_target_tokens})"
            )

    @property
    def max_possible_tokens(self) -> int:
        """单块的理论上界 = 450 + 80 = 530。

        它不是配置项，是「短尾并入前一块」这条规则的推论：前一块最多 450，尾块
        最多 79。这个数是下游两处的输入——VARCHAR 长度是否够、TEI 的 `truncate`
        会不会真的截断——所以它必须被算出来而不是被感觉。
        """
        return self._settings.chunk_target_tokens + self._settings.chunk_min_tokens

    # ---- 全文层 ----------------------------------------------------------

    def chunk_paper(self, paper: Paper, paragraphs: Sequence[SourceParagraph]) -> tuple[list[Chunk], ChunkingStats]:
        """把一篇论文的段落序列切成 chunk 列表。

        `paragraphs` 必须已经过 `placeholders` 与 `cleaning`——本方法不做清洗，
        它假定拿到的就是要进索引的文本。这条分工是有意的：清洗失败要按篇隔离计数，
        切块失败是 bug，两者的处理方式完全不同。
        """
        stats = ChunkingStats(paper_count=1)
        chunks: list[Chunk] = []

        for section_ordinal, section_paragraphs in enumerate(self._iter_sections(paragraphs)):
            stats.section_count += 1
            blocks = self._build_blocks(section_ordinal, section_paragraphs, stats)
            blocks = self._merge_short_tails(blocks, stats)
            # 同一 section 内起止段号相同的块要编号：长段回退档会把一个段落切成多块，
            # 它们的 (start, end) 完全一样，不编号就是重复主键 → upsert 静默覆盖。
            seen_spans: dict[tuple[int, int], int] = {}
            for block in blocks:
                span = (block.start_paragraph_idx, block.end_paragraph_idx)
                part = seen_spans.get(span, 0)
                seen_spans[span] = part + 1
                chunks.append(self._to_chunk(paper, block, Granularity.FULLTEXT, part_ordinal=part))
                stats.chunk_count += 1
                stats.total_tokens += block.token_count
                if block.token_count > self.max_possible_tokens:
                    # 非 0 即 bug：要么长段回退档没生效，要么合并规则被改过。
                    stats.oversize_chunks += 1
                    _logger.warning(
                        "chunk_oversize",
                        paper_id=paper.paper_id,
                        token_count=block.token_count,
                        ceiling=self.max_possible_tokens,
                    )
        return chunks, stats

    def _iter_sections(self, paragraphs: Sequence[SourceParagraph]) -> Iterator[list[SourceParagraph]]:
        """按 `(section, sec_number)` 的变化切出 section 组。

        用「相邻段的 section 标识发生变化」而不是「按 section 名分组」，是因为
        同名 section 可能在文中出现两次（附录里又一个 "Discussion"），按名分组会把
        物理上不相邻的段落粘在一起，引用锚点随之错位。
        """
        current_key: tuple[str | None, str | None] | None = None
        bucket: list[SourceParagraph] = []
        for para in paragraphs:
            key = (para.section, para.sec_number)
            if current_key is not None and key != current_key:
                if bucket:
                    yield bucket
                bucket = []
            current_key = key
            bucket.append(para)
        if bucket:
            yield bucket

    def _build_blocks(
        self,
        section_ordinal: int,
        paragraphs: Sequence[SourceParagraph],
        stats: ChunkingStats,
    ) -> list[_Block]:
        """在一个 section 内做段落聚合（R1 + R3），section 结束无条件封口（R2）。"""
        target = self._settings.chunk_target_tokens
        blocks: list[_Block] = []
        current: _Block | None = None

        for para in paragraphs:
            stats.paragraph_count += 1
            if para.content_type not in _TEXT_CONTENT_TYPES:
                stats.skipped_non_text_paragraphs += 1
                continue
            if not para.text.strip():
                continue

            for segment in self._as_segments(para, stats):
                # 累计一旦要超过 450 就先封口——**先封口再吸入**，不是吸完再判断。
                # 反过来写会让每个块都恰好越界一个段落，450 就成了下界而不是上界。
                if current is not None and current.token_count + segment.token_count > target:
                    blocks.append(current)
                    current = None
                if current is None:
                    current = _Block(
                        section=para.section,
                        sec_number=para.sec_number,
                        sec_type=para.sec_type,
                        section_ordinal=section_ordinal,
                        start_paragraph_idx=para.paragraph_idx,
                        end_paragraph_idx=para.paragraph_idx,
                    )
                current.texts.append(segment.text)
                current.cited_ids.extend(segment.cited_ids)
                current.formulas.extend(segment.formulas)
                current.token_count += segment.token_count
                current.end_paragraph_idx = para.paragraph_idx

        if current is not None:
            blocks.append(current)
        return blocks

    def _as_segments(self, para: SourceParagraph, stats: ChunkingStats) -> list[_Segment]:
        """把一个段落变成一到多个可聚合片段。

        绝大多数段落原样返回（R1）。只有单段本身就超过 450 token 时才触发回退档：
        R1 与 R3 直接冲突，必须有一个确定的处置，否则会产出一个远超上界的块。
        """
        token_count = self._counter.count(para.text)
        if token_count <= self._settings.chunk_target_tokens:
            return [
                _Segment(
                    text=para.text,
                    cited_ids=list(para.cited_open_alex_ids),
                    formulas=list(para.formula_latex),
                    token_count=token_count,
                )
            ]

        stats.long_paragraph_fallbacks += 1
        pieces = split_long_text(para.text, self._settings.chunk_target_tokens, self._counter)
        return self._attach_citations(para, pieces)

    def _attach_citations(self, para: SourceParagraph, pieces: list[TextPiece]) -> list[_Segment]:
        """把段落的引用边与公式按字符偏移分配到各片段。

        没有偏移信息时全部归第一个片段并告警：这是**已知的精度损失**（后续片段
        会丢掉本该属于它们的引用边），但比平摊到所有片段好——平摊制造的是假阳性，
        而「别人在什么语境下引用 X」这个能力的价值全在于返回的段落确实引用了 X。
        """
        segments = [
            _Segment(text=t, cited_ids=[], formulas=[], token_count=self._counter.count(t)) for t, _, _ in pieces
        ]
        if not segments:
            return segments

        if len(para.cite_offsets) != len(para.cited_open_alex_ids):
            if para.cited_open_alex_ids:
                _logger.debug("cite_offsets_missing", paragraph_idx=para.paragraph_idx)
            segments[0].cited_ids.extend(para.cited_open_alex_ids)
        else:
            for offset, cited_id in zip(para.cite_offsets, para.cited_open_alex_ids, strict=True):
                index = next((i for i, (_, s, e) in enumerate(pieces) if s <= offset < e), 0)
                segments[index].cited_ids.append(cited_id)

        # 公式 LaTeX 只做展示与离线 A/B，不参与任何检索或判定，所以不值得为它
        # 再维护一套偏移；全部挂在第一个片段上，语义损失可接受。
        segments[0].formulas.extend(para.formula_latex)
        return segments

    def _merge_short_tails(self, blocks: list[_Block], stats: ChunkingStats) -> list[_Block]:
        """不足 80 token 的块并入同 section 的前一个块（R3 后半句）。

        没有前一个块（整个 section 都不足 80 token）就丢弃并计数——这类 section
        多半是致谢、一行小标题或解析残片，单独成块只会污染召回。
        """
        minimum = self._settings.chunk_min_tokens
        merged: list[_Block] = []
        for block in blocks:
            if block.token_count >= minimum:
                merged.append(block)
                continue
            if not merged:
                stats.dropped_short_sections += 1
                continue
            prev = merged[-1]
            prev.texts.extend(block.texts)
            prev.cited_ids.extend(block.cited_ids)
            prev.formulas.extend(block.formulas)
            prev.end_paragraph_idx = block.end_paragraph_idx
            # 重新计一次而不是把两个数加起来：80 与 450 这两个阈值必须落在**最终
            # 文本**上，加法会因为拼接符与子词切分产生几个 token 的偏差，而 §6.12
            # 的触边率统计要求 0 误差。
            prev.token_count = self._counter.count(prev.render())
            stats.short_tail_merges += 1
        return merged

    def _to_chunk(self, paper: Paper, block: _Block, granularity: Granularity, *, part_ordinal: int = 0) -> Chunk:
        """`_Block` → 领域模型 `Chunk`。字段分类见 M1 §6.11。

        一条纪律在这里落地：C 类字段（`cited_open_alex_ids` / `formula_latex`）
        **绝不进 analyzed 文本字段**。它们进去的那一刻，占位符造成的倒排污染就
        原封不动地回来了——删占位符不是为了删信息，是为了把信息从被索引的文本里
        挪到不被索引的字段里。
        """
        return Chunk(
            chunk_id=build_chunk_id(
                paper.paper_id,
                granularity,
                block.section_ordinal,
                block.start_paragraph_idx,
                block.end_paragraph_idx,
                part_ordinal,
            ),
            paper_id=paper.paper_id,
            granularity=granularity,
            text=block.render(),
            section=block.section,
            sec_number=block.sec_number,
            sec_type=block.sec_type,
            paragraph_idx=block.start_paragraph_idx,
            year=paper.year,
            primary_category=paper.primary_category,
            license=paper.license,
            source_dataset=paper.source_dataset,
            token_count=block.token_count,
            # 去重但保序：同一段里同一篇被引多次是常态，边表按 (chunk_id, cited_id)
            # 唯一，重复只会让写入端多做无用功。
            cited_open_alex_ids=list(dict.fromkeys(block.cited_ids)),
            formula_latex=block.formulas,
        )

    # ---- 摘要层 ----------------------------------------------------------

    def chunk_abstract(self, paper: Paper, title: str, abstract: str) -> Chunk | None:
        """摘要层：`title + abstract` 拼成一个 chunk，**恒为 1，不切分**（M1 §6.9）。

        决定性的理由是第一条：OpenAlex 摘要典型 150–250 词 ≈ 200–350 token，本来
        就在 450 的上界之内——**切它需要理由，不切它才是默认**。其次摘要是「背景 →
        问题 → 方法 → 结果」高度压缩的不可分割语义单位，切成两半会得到两个都不完整
        的块（前半没有结论，后半没有上下文），这与正文段落（本身就是独立论证单元）
        完全不同。

        返回 None 表示这篇不进摘要层：无摘要的论文只剩 title，达不到 80 token 下限。
        """
        text = f"{title.strip()}\n\n{abstract.strip()}".strip()
        if not text:
            return None
        token_count = self._counter.count(text)
        if token_count < self._settings.chunk_min_tokens:
            _logger.debug("abstract_chunk_dropped", paper_id=paper.paper_id, token_count=token_count)
            return None
        return Chunk(
            chunk_id=build_chunk_id(paper.paper_id, Granularity.ABSTRACT, 0, 0, 0),
            paper_id=paper.paper_id,
            granularity=Granularity.ABSTRACT,
            text=text,
            # 摘要层没有 section 结构，四元锚点退化成 paper_id 一元——这正是
            # `granularity` 必须暴露给调用方的原因：能力边界做成接口字段，
            # 比在文档里写一句免责声明强（M2 §5.3）。
            section=None,
            sec_number=None,
            sec_type=SecType.ABSTRACT,
            paragraph_idx=0,
            year=paper.year,
            primary_category=paper.primary_category,
            license=paper.license,
            source_dataset=paper.source_dataset,
            token_count=token_count,
        )
