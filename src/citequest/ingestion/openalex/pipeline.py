"""摘要层装配：OpenAlex work → `Paper` + 恰好一个 `Chunk`（M1 §6.9）。

摘要层的规则简单到几乎没有决策：`title + abstract` 拼成一个 chunk，**恒为 1，
不切分**。三条理由里第一条是决定性的——OpenAlex 摘要典型 150–250 词 ≈ 200–350
token，本来就在 450 的上界之内，**切它需要理由，不切它才是默认**。

这一层的价值是**时效**（全文层只到 2022），代价同样明确且必须主动讲：摘要没有
section，方法级问题它答不了。所以 `granularity` 是刻意暴露给调用方的能力边界，
不是内部实现细节。

**必须写进接口契约的一条语义**：RAG 路的 `group_size=3` 对摘要层论文会静默返回
1 条，所以下游的上下文预算必须按「实际条数 ≤48」写，不能按 48 写死。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from pydantic import BaseModel, Field

from citequest.config.corpus import CorpusSettings
from citequest.core.corpus import Chunk, Paper
from citequest.core.enums import Granularity
from citequest.ingestion.chunking import SectionAwareChunker
from citequest.ingestion.cleaning import clean_latex_noise
from citequest.ingestion.normalize import (
    LAYER_DEFAULTS,
    build_openalex_attribution,
    normalize_doi,
    normalize_openalex_id,
    normalize_primary_category,
    resolve_paper_id,
    validate_year,
)
from citequest.ingestion.openalex.abstract import rebuild_abstract
from citequest.ingestion.placeholders import normalize_unicode_and_whitespace
from citequest.ingestion.tokenization import TokenCounter
from citequest.observability.logging import get_logger

__all__ = ["AbstractBuildResult", "OpenAlexPipeline"]

_logger = get_logger(__name__)

#: OpenAlex `primary_topic` → arXiv 类目的最小映射。**只覆盖 CS 一级子领域**，
#: 映射不到就留 None——留 None 只是少一个可选过滤维度，硬猜一个类目会让
#: `primary_category` 过滤返回错误的结果集，而那不报错。
_TOPIC_TO_ARXIV_CATEGORY: dict[str, str] = {
    "artificial intelligence": "cs.AI",
    "machine learning": "cs.LG",
    "computer vision": "cs.CV",
    "natural language processing": "cs.CL",
    "computational linguistics": "cs.CL",
    "information retrieval": "cs.IR",
    "computer networks": "cs.NI",
    "software engineering": "cs.SE",
    "computer security": "cs.CR",
    "distributed computing": "cs.DC",
    "human-computer interaction": "cs.HC",
    "robotics": "cs.RO",
    "database": "cs.DB",
    "theoretical computer science": "cs.CC",
}


class AbstractBuildResult(BaseModel):
    """一篇摘要层论文的装配产物。`chunk` 为 None 表示这篇不进库。"""

    paper: Paper
    chunk: Chunk | None = None
    abstract_gap_count: int = Field(default=0, ge=0)
    abstract_conflict_count: int = Field(default=0, ge=0)


class OpenAlexPipeline:
    """work JSON → 领域模型。

    复用的是**同一个** `SectionAwareChunker`（走它的 `chunk_abstract` 分支）而不是
    另写一套摘要切块：分层的复杂度被吸收进一个字段（`granularity`），而不是两条
    代码路径——两条路径意味着两套 bug，且不一致会在几个月后以「新灌进去的块跟老块
    长得不一样」的形式浮出来（M1 §8.1）。
    """

    def __init__(self, token_counter: TokenCounter, settings: CorpusSettings | None = None) -> None:
        self._settings = settings or CorpusSettings()
        self._chunker = SectionAwareChunker(token_counter, self._settings)

    def build(self, work: dict[str, Any]) -> AbstractBuildResult | None:
        """装配一篇。返回 None 表示这条记录不可用（无主键 / 年份越界）。"""
        try:
            paper = self._build_paper(work)
        except ValueError as exc:
            _logger.debug("abstract_paper_rejected", reason=str(exc))
            return None

        rebuilt = rebuild_abstract(work.get("abstract_inverted_index"), paper_id=paper.paper_id)
        if not rebuilt.is_usable:
            # 无摘要的论文不进摘要层——只有 title 的 chunk 达不到 80 token 下限。
            # 但**仍然返回 Paper**：论文级元数据（分面统计、引用边的被引方）还有用。
            return AbstractBuildResult(paper=paper, chunk=None)

        cleaned = clean_latex_noise(rebuilt.text)
        abstract_text = normalize_unicode_and_whitespace(cleaned.text)
        chunk = self._chunker.chunk_abstract(paper, paper.title, abstract_text)
        return AbstractBuildResult(
            paper=paper,
            chunk=chunk,
            abstract_gap_count=rebuilt.gap_count,
            abstract_conflict_count=rebuilt.conflict_count,
        )

    def build_many(self, works: Iterable[dict[str, Any]]) -> Iterator[AbstractBuildResult]:
        for work in works:
            result = self.build(work)
            if result is not None:
                yield result

    # ---- 内部 ------------------------------------------------------------

    def _build_paper(self, work: dict[str, Any]) -> Paper:
        openalex_id = normalize_openalex_id(work.get("id"))
        arxiv_id = _extract_arxiv_id(work)
        paper_id, _degraded = resolve_paper_id(openalex_id, arxiv_id)

        year = validate_year(work.get("publication_year"))
        if year is None:
            raise ValueError("publication_year 缺失或越界（它同时是 partition key）")

        license_id, source_dataset = LAYER_DEFAULTS[Granularity.ABSTRACT]
        return Paper(
            paper_id=paper_id,
            title=str(work.get("title") or work.get("display_name") or "").strip() or f"[untitled] {paper_id}",
            year=year,
            primary_category=_map_primary_category(work),
            authors=_extract_authors(work),
            venue=_extract_venue(work),
            doi=normalize_doi(work.get("doi")),
            arxiv_id=arxiv_id,
            cited_by_count=_safe_int(work.get("cited_by_count")),
            granularity=Granularity.ABSTRACT,
            has_fulltext=False,
            license=license_id,
            source_dataset=source_dataset,
            attribution=build_openalex_attribution(arxiv_id),
            # 撤稿**不物理删除**，只打标记按需过滤（M1 §5.4 / §8.3）。
            # 缓存必须按 paper_id 精确失效——这是失效纪律唯一的例外，因为它是合规要求。
            is_retracted=bool(work.get("is_retracted", False)),
        )


def _map_primary_category(work: dict[str, Any]) -> str | None:
    """`primary_topic.display_name` → arXiv 类目。映射不到返回 None。"""
    topic = work.get("primary_topic")
    if not isinstance(topic, dict):
        return None
    name = str(topic.get("display_name") or "").strip().lower()
    mapped = _TOPIC_TO_ARXIV_CATEGORY.get(name)
    return normalize_primary_category(mapped) if mapped else None


def _extract_arxiv_id(work: dict[str, Any]) -> str | None:
    """从 `locations[*].landing_page_url` / `ids.arxiv` 里抠 arXiv ID。

    arXiv ID 是两层语料对齐的唯一钥匙（G2 的真正用途），所以要多路探测：
    OpenAlex 在不同记录上把它放在不同位置，只看一处会让对齐率无声地掉一截。
    """
    ids = work.get("ids")
    if isinstance(ids, dict):
        raw = ids.get("arxiv")
        if isinstance(raw, str) and raw.strip():
            return raw.rsplit("/", 1)[-1].removeprefix("arXiv:").strip()

    locations = work.get("locations")
    if isinstance(locations, list):
        for loc in locations:
            if not isinstance(loc, dict):
                continue
            url = str(loc.get("landing_page_url") or "")
            if "arxiv.org/abs/" in url:
                return url.split("arxiv.org/abs/", 1)[1].strip("/") or None
    return None


def _extract_authors(work: dict[str, Any]) -> list[str]:
    """保留原始 `display_name`，不做缩写统一（M1 §5.5）。"""
    authorships = work.get("authorships")
    if not isinstance(authorships, list):
        return []
    names: list[str] = []
    for item in authorships:
        if not isinstance(item, dict):
            continue
        author = item.get("author")
        if isinstance(author, dict):
            name = str(author.get("display_name") or "").strip()
            if name:
                names.append(name)
    return names


def _extract_venue(work: dict[str, Any]) -> str | None:
    location = work.get("primary_location")
    if not isinstance(location, dict):
        return None
    source = location.get("source")
    if not isinstance(source, dict):
        return None
    name = str(source.get("display_name") or "").strip()
    return name or None


def _safe_int(raw: object) -> int | None:
    """负数与非数值一律归 None。

    `cited_by_count` 的领域约束是 `ge=0`，脏值直接构造 `Paper` 会抛校验错并让整篇
    记录被丢弃——而引用数缺失完全不影响这篇论文的检索价值，不该为它丢一篇。
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None
