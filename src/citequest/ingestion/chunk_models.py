"""切块的输入契约、统计契约与主键构造（M1 §6.1 / §6.12 / §8.1）。

与 `chunking.py` 分开是因为它们的**变化原因不同**：算法会随实测调整（450/80 的
取值、回退档的触发条件），而这三样一旦定死就不能动——`chunk_id` 的构造方式变了
等于全库主键变了，`SourceParagraph` 的字段变了等于上游解析要跟着改。
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import blake2s

from pydantic import BaseModel, Field

from citequest.core.corpus import Chunk, CiteEdge
from citequest.core.enums import Granularity, SecType

__all__ = [
    "ChunkingStats",
    "SourceParagraph",
    "build_chunk_id",
    "build_cite_edges",
]

#: chunk_id 的 VARCHAR 上限（M2 §5.1 字段表）。超了会被 Milvus 静默截断成重复主键。
_CHUNK_ID_MAX_LEN = 64


class SourceParagraph(BaseModel):
    """切块的输入单元：一个**已清洗**的段落 + 它的结构位置。

    结构（`section` / `sec_number` / `sec_type` / `content_type`）是上游数据集给的，
    本项目**不做标题正则识别**——真实论文里有人写 `IV. EXPERIMENTS`、有人根本不编号，
    正则的错误率很高（M1 §6.1）。
    """

    paragraph_idx: int = Field(ge=0, description="段落在全文中的序号，引用锚点第四元")
    text: str = Field(description="已过 placeholders + cleaning 的正文")
    section: str | None = None
    sec_number: str | None = None
    sec_type: SecType | None = None
    content_type: str = Field(default="paragraph")
    cited_open_alex_ids: list[str] = Field(default_factory=list, description="本段 {{cite:uuid}} 解析出的被引 ID")
    cite_offsets: list[int] = Field(
        default_factory=list,
        description=(
            "各被引 ID 在**清洗后文本**中的字符偏移，与 cited_open_alex_ids 一一对应。"
            "只有长段回退档（§6.8）用得到它：一段被切成多个块时，靠偏移把引用边分到正确的块上。"
            "留空则回退到「全部归第一个块」，那会让引用上下文检索在这些块上产生假阳性"
        ),
    )
    formula_latex: list[str] = Field(default_factory=list, description="被删除的公式 LaTeX，按出现顺序")


class ChunkingStats(BaseModel):
    """切块过程的统计（M1 §6.12 必跑统计脚本的输入）。

    这些计数不是日志装饰，它们各自卡住一个【待实测】：
    `oversize_chunks` 非 0 就是 bug 不是调参问题；`long_paragraph_fallbacks` 触发率
    高于个位数百分比意味着 450 这个上界要重估；`dropped_short_sections` 直接解释了
    「为什么 chunk 数比段落数少这么多」。
    """

    paper_count: int = Field(default=0, ge=0)
    section_count: int = Field(default=0, ge=0)
    paragraph_count: int = Field(default=0, ge=0)
    skipped_non_text_paragraphs: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    long_paragraph_fallbacks: int = Field(default=0, ge=0, description="单段超过 450 token 而退化到句子切分的次数")
    short_tail_merges: int = Field(default=0, ge=0, description="不足 80 token 并入前一块的次数")
    dropped_short_sections: int = Field(default=0, ge=0, description="整个 section 都不足 80 token 而被丢弃")
    oversize_chunks: int = Field(
        default=0, ge=0, description="最终 token_count > 450+80 的块数。**应恒为 0**，非 0 即 bug"
    )
    total_tokens: int = Field(default=0, ge=0)

    def merge(self, other: ChunkingStats) -> ChunkingStats:
        """按篇累加。全量跑完一次即得 §6.12 要求的分布报告。"""
        return ChunkingStats(
            paper_count=self.paper_count + other.paper_count,
            section_count=self.section_count + other.section_count,
            paragraph_count=self.paragraph_count + other.paragraph_count,
            skipped_non_text_paragraphs=self.skipped_non_text_paragraphs + other.skipped_non_text_paragraphs,
            chunk_count=self.chunk_count + other.chunk_count,
            long_paragraph_fallbacks=self.long_paragraph_fallbacks + other.long_paragraph_fallbacks,
            short_tail_merges=self.short_tail_merges + other.short_tail_merges,
            dropped_short_sections=self.dropped_short_sections + other.dropped_short_sections,
            oversize_chunks=self.oversize_chunks + other.oversize_chunks,
            total_tokens=self.total_tokens + other.total_tokens,
        )


def build_chunk_id(
    paper_id: str,
    granularity: Granularity,
    section_ordinal: int,
    start_paragraph_idx: int,
    end_paragraph_idx: int,
    part_ordinal: int = 0,
) -> str:
    """确定性构造 chunk_id（M1 §8.1）。

    `chunk_id = f(paper_id, granularity, section 序号, 块内起止 paragraph_idx)`。
    同一篇论文重跑一次切块得到完全相同的 id，于是三件事同时成立：重复投递不产生
    重复行（按主键 upsert 就是覆盖）、断点续跑安全（重叠部分被幂等吸收）、可比对
    （重跑后逐 id 比文本，能区分是清洗规则变了还是源数据变了）。

    :param part_ordinal: 段内片段序号，**只在长段回退档（§6.8）下非 0**。
        它是必须存在的第五元：单段超过 450 token 时会被切成多块，这些块的起止
        `paragraph_idx` 完全相同，四元组构造出的 id 会重复——而重复主键在 upsert
        语义下是**静默覆盖**，表现为「这篇论文的正文少了一段，但没有任何报错」。

    可读形态超过 VARCHAR(64) 时退化为 blake2s 摘要——**仍然是确定性的**，幂等性
    不受影响；只是牺牲了肉眼可读。不做截断，截断会造出重复主键，那是静默的数据覆盖。
    """
    marker = "f" if granularity is Granularity.FULLTEXT else "a"
    suffix = f"#{part_ordinal:02d}" if part_ordinal else ""
    readable = f"{paper_id}:{marker}:s{section_ordinal:03d}:p{start_paragraph_idx:05d}-{end_paragraph_idx:05d}{suffix}"
    if len(readable) <= _CHUNK_ID_MAX_LEN:
        return readable
    digest = blake2s(readable.encode("utf-8"), digest_size=16).hexdigest()
    return f"{marker}_{digest}"


def build_cite_edges(chunks: Sequence[Chunk]) -> list[CiteEdge]:
    """从 chunk 的 C 类字段摊平出引用边表（M1 §7.3 方案 B）。

    边表落进单机关系库，供「别人在什么语境下引用 X」做候选过滤。**不用 OpenAlex
    的 `referenced_works`**——实测 arXiv-primary CS 2023 抽样 200 篇里 94.0% 的
    `referenced_works_count` 为 0，那不是覆盖率低，是数据不存在。
    """
    return [
        CiteEdge(chunk_id=chunk.chunk_id, paper_id=chunk.paper_id, cited_open_alex_id=cited_id)
        for chunk in chunks
        for cited_id in chunk.cited_open_alex_ids
    ]
