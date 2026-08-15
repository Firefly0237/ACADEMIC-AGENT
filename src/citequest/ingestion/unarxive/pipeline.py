"""全文层装配：`UnarxiveRecord` → `Paper` + `list[Chunk]` + `list[CiteEdge]`。

这里把 M1 §5 的五段清洗与 §6 的切块串成一条链，顺序是固定的且**不可交换**：

```
原始段落 ─▶ S1 占位符（抽 key 再删）─▶ S2 LaTeX 噪声 ─▶ S3 NFKC 与空白 ─▶ 切块
```

S1 必须在 S3 之前：NFKC 会把全角花括号折叠成半角，先折叠会让占位符正则漏配
（详见 `placeholders.normalize_unicode_and_whitespace` 的 docstring）。

**清洗失败按篇隔离、计数、不阻断**——单篇失败是数据问题，整批失败才是代码问题，
两者的处理方式不同，混在一起就永远分不清今天这批数据到底是坏了还是代码坏了。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from pydantic import BaseModel, Field

from citequest.config.corpus import CorpusSettings
from citequest.core.corpus import Chunk, CiteEdge, Paper
from citequest.core.enums import Granularity, SecType
from citequest.ingestion.chunk_models import ChunkingStats, SourceParagraph, build_cite_edges
from citequest.ingestion.chunking import SectionAwareChunker
from citequest.ingestion.cleaning import clean_latex_noise
from citequest.ingestion.normalize import (
    LAYER_DEFAULTS,
    UNARXIVE_ATTRIBUTION,
    normalize_openalex_id,
    normalize_primary_category,
    resolve_paper_id,
    validate_year,
)
from citequest.ingestion.placeholders import normalize_unicode_and_whitespace, strip_placeholders
from citequest.ingestion.tokenization import TokenCounter
from citequest.ingestion.unarxive.reader import UnarxiveRecord
from citequest.observability.logging import get_logger

__all__ = ["FulltextBuildResult", "UnarxivePipeline"]

_logger = get_logger(__name__)

#: unarXive 的 `sec_type` 字面值 → 本项目枚举。**不做标题正则识别**（M1 §6.1）：
#: 真实论文里有人写 `IV. EXPERIMENTS`、有人不编号，正则错误率很高。
_SEC_TYPE_ALIASES: dict[str, SecType] = {
    "introduction": SecType.INTRODUCTION,
    "background": SecType.BACKGROUND,
    "related work": SecType.RELATED_WORK,
    "related_work": SecType.RELATED_WORK,
    "method": SecType.METHOD,
    "methods": SecType.METHOD,
    "methodology": SecType.METHOD,
    "experiment": SecType.EXPERIMENT,
    "experiments": SecType.EXPERIMENT,
    "evaluation": SecType.EXPERIMENT,
    "result": SecType.RESULT,
    "results": SecType.RESULT,
    "discussion": SecType.DISCUSSION,
    "conclusion": SecType.CONCLUSION,
    "conclusions": SecType.CONCLUSION,
    "abstract": SecType.ABSTRACT,
}


class FulltextBuildResult(BaseModel):
    """一篇论文的装配产物 + 该篇的切块统计。"""

    paper: Paper
    chunks: list[Chunk] = Field(default_factory=list)
    cite_edges: list[CiteEdge] = Field(default_factory=list)
    stats: ChunkingStats = Field(default_factory=ChunkingStats)
    placeholder_residual: int = Field(default=0, ge=0, description="清洗后残留的 `{{...}}`，汇总即占位符残留率")
    unresolved_cite_keys: int = Field(default=0, ge=0, description="在 bib_entries 里查不到 open_alex_id 的引用数")


class UnarxivePipeline:
    """把一条原始记录装配成可灌库的产物。

    有状态的只有 chunker（它持有 tokenizer），所以一个实例可以复用整轮建库；
    但它**不是线程安全的**——HF tokenizer 的 fast 实现有自己的并发约束，多进程
    并行时每个进程各建一个实例即可。
    """

    def __init__(self, token_counter: TokenCounter, settings: CorpusSettings | None = None) -> None:
        self._settings = settings or CorpusSettings()
        self._chunker = SectionAwareChunker(token_counter, self._settings)

    def build(self, record: UnarxiveRecord) -> FulltextBuildResult | None:
        """装配一篇。失败返回 None（调用方计数），不抛异常打断整批。"""
        try:
            paper = self._build_paper(record)
        except ValueError as exc:
            _logger.warning("fulltext_paper_rejected", paper_id=record.paper_id, reason=str(exc))
            return None

        cite_lookup = self._build_cite_lookup(record.bib_entries)
        paragraphs, residual, unresolved = self._build_paragraphs(record, cite_lookup)
        if not paragraphs:
            _logger.debug("fulltext_no_paragraph", paper_id=paper.paper_id)
            return None

        chunks, stats = self._chunker.chunk_paper(paper, paragraphs)
        if not chunks:
            return None
        return FulltextBuildResult(
            paper=paper,
            chunks=chunks,
            cite_edges=build_cite_edges(chunks),
            stats=stats,
            placeholder_residual=residual,
            unresolved_cite_keys=unresolved,
        )

    def build_many(self, records: Iterable[UnarxiveRecord]) -> Iterator[FulltextBuildResult]:
        """批量装配。跳过的记录只记日志，计数交给调用方的 checkpoint。"""
        for record in records:
            result = self.build(record)
            if result is not None:
                yield result

    # ---- 内部 ------------------------------------------------------------

    def _build_paper(self, record: UnarxiveRecord) -> Paper:
        """元数据 → `Paper`。

        `paper_id` 优先取 OpenAlex ID，取不到就退化为 arXiv ID **并保留**（M1 §5.5）。
        丢弃这类记录会让全文层规模在没人注意的情况下缩水，而缩水的量正好等于
        OpenAlex 对齐失败率——那是一个需要被度量而不是被隐藏的数字。
        """
        metadata = record.metadata
        openalex_id = normalize_openalex_id(_first_str(metadata, "open_alex_id", "openalex_id"))
        arxiv_id = _first_str(metadata, "arxiv_id", "id") or record.paper_id
        paper_id, degraded = resolve_paper_id(openalex_id, arxiv_id)
        if degraded:
            _logger.debug("paper_id_degraded_to_arxiv", arxiv_id=arxiv_id)

        year = validate_year(record.submitted_year)
        if year is None:
            raise ValueError("年份缺失或越界——它同时是 partition key，不能放行")

        license_id, source_dataset = LAYER_DEFAULTS[Granularity.FULLTEXT]
        title = str(metadata.get("title") or "").strip() or f"[untitled] {paper_id}"
        return Paper(
            paper_id=paper_id,
            title=title,
            year=year,
            primary_category=normalize_primary_category(record.primary_category),
            authors=_split_authors(metadata.get("authors")),
            venue=_first_str(metadata, "journal-ref", "venue"),
            doi=_first_str(metadata, "doi"),
            arxiv_id=arxiv_id,
            granularity=Granularity.FULLTEXT,
            has_fulltext=True,
            license=license_id,
            source_dataset=source_dataset,
            attribution=UNARXIVE_ATTRIBUTION.model_copy(
                update={"original_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None}
            ),
        )

    def _build_cite_lookup(self, bib_entries: dict[str, Any]) -> dict[str, str]:
        """`bib_entries[key].ids.open_alex_id` → key→OpenAlex ID 映射。

        **引用边一律来自这里，不用 OpenAlex 的 `referenced_works`**：实测
        arXiv-primary CS 2023 抽样 200 篇里 94.0% 的 `referenced_works_count` 为 0，
        那不是覆盖率低，是数据不存在（决策记录 §2.2）。
        """
        lookup: dict[str, str] = {}
        for key, entry in bib_entries.items():
            if not isinstance(entry, dict):
                continue
            ids = entry.get("ids")
            if not isinstance(ids, dict):
                continue
            normalized = normalize_openalex_id(ids.get("open_alex_id"))
            if normalized:
                lookup[key] = normalized
        return lookup

    def _build_paragraphs(
        self, record: UnarxiveRecord, cite_lookup: dict[str, str]
    ) -> tuple[list[SourceParagraph], int, int]:
        """逐段跑 S1→S2→S3，产出切块的输入。

        返回值里带上 `residual` 与 `unresolved`：前者是占位符残留率的分子，后者是
        「`bib_entries` 的 `open_alex_id` 匹配率」的分子——这个匹配率直接决定
        「别人在什么语境下引用 X」这个能力的覆盖面（M1 §7.3 的【待实测】项）。
        """
        paragraphs: list[SourceParagraph] = []
        residual_total = 0
        unresolved_total = 0

        for idx, raw in enumerate(record.body_text):
            text = str(raw.get("text") or "")
            if not text.strip():
                continue

            stripped = strip_placeholders(text)
            residual_total += stripped.residual_count
            cleaned = clean_latex_noise(stripped.text)
            final_text = normalize_unicode_and_whitespace(cleaned.text)
            if not final_text:
                continue

            cited_ids: list[str] = []
            for key in stripped.cite_keys:
                resolved = cite_lookup.get(key)
                if resolved is None:
                    unresolved_total += 1
                    continue
                cited_ids.append(resolved)

            paragraphs.append(
                SourceParagraph(
                    paragraph_idx=int(raw.get("paragraph_idx") or idx),
                    text=final_text,
                    section=_clean_optional_str(raw.get("section")),
                    sec_number=_clean_optional_str(raw.get("sec_number")),
                    sec_type=_map_sec_type(raw.get("sec_type")),
                    content_type=str(raw.get("content_type") or "paragraph"),
                    cited_open_alex_ids=cited_ids,
                    # 偏移在清洗过程中已经失真（删占位符会改变后续字符位置），
                    # 所以这里不给。后果是长段回退档下引用边全归第一个片段——
                    # 已在 chunking 里显式告警。**接真实实现：让 strip_placeholders
                    # 一并返回删除后各引用点的新偏移，再透传到这里。**
                    cite_offsets=[],
                    formula_latex=[],
                )
            )
        return paragraphs, residual_total, unresolved_total


def _map_sec_type(raw: object) -> SecType | None:
    if not raw:
        return None
    return _SEC_TYPE_ALIASES.get(str(raw).strip().lower(), SecType.OTHER)


def _clean_optional_str(raw: object) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _first_str(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _split_authors(raw: object) -> list[str]:
    """作者串 → 列表。**保留原始 display_name，不做缩写统一**（M1 §5.5）。

    缩写统一是有损的且没有权威映射表，自己造一张就是自建——而这个字段只用于
    展示与模糊去重的 Jaccard，两者都不要求规范化姓名。
    """
    if isinstance(raw, list):
        return [str(a).strip() for a in raw if str(a).strip()]
    if isinstance(raw, str):
        return [part.strip() for part in raw.replace(" and ", ",").split(",") if part.strip()]
    return []
