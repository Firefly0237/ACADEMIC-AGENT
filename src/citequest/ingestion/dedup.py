"""S4 · 去重（M1 §5.4）。

**去重在这套语料上被显著降级了，但不能不做。** 摘要层的 filter（arXiv 为 primary
source + 发表日期 2023–2025）已经保证两层在时间上不重叠、同一 arXiv ID 只可能落进
一层，所以「全文层与摘要层撞车」这个难题在设计上就不存在。剩下的真实问题只有一个：
摘要层内部存在同一工作的多条 OpenAlex 记录（preprint 与正式发表版）。

两级：

- **Level 1 · 精确**：同 OpenAlex ID 取 `updated_date` 最新；同 DOI（归一化后）取
  非空字段更多的一条。
- **Level 2 · 模糊**：归一化 title 相同即判重；不同但相似度 > 0.9 时再比作者集合的
  Jaccard，> 0.7 判重；保留 `cited_by_count` 最高的一条。

**不做全量两两比较**：27 万篇是 360 亿次比较。用 MinHash LSH 对归一化 title 的
character n-gram 建近似索引，先筛候选对再精确比——这是标准做法，不是自造。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from citequest.core.corpus import Paper
from citequest.observability.logging import get_logger

__all__ = [
    "DedupReport",
    "dedup_exact",
    "dedup_fuzzy",
    "normalize_title",
]

_logger = get_logger(__name__)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")

#: title 归一化时去掉的英文停用词。刻意只留最高频的一小撮——去太多会让
#: "Attention Is All You Need" 与 "Attention for All" 归一化后撞在一起。
_TITLE_STOPWORDS = frozenset({"a", "an", "the", "of", "for", "on", "in", "and", "with", "to"})

#: title 相似度阈值与作者 Jaccard 阈值（M1 §5.4）。
_TITLE_SIMILARITY_THRESHOLD = 0.9
_AUTHOR_JACCARD_THRESHOLD = 0.7
#: 短 title 跳过模糊去重：< 5 词时误判率高（同上，三个特例之二）。
_MIN_TITLE_WORDS_FOR_FUZZY = 5


class DedupReport(BaseModel):
    """去重命中数（M1 §5.4 的【待实测】项）。

    它的用途不是炫耀，是确认 219,676 这个数在去重后还剩多少：**掉超过 1% 就说明
    摘要层的 arXiv primary source 过滤没有按预期收敛，要回头查 filter**，而不是
    庆祝去重有效。
    """

    input_count: int = Field(default=0, ge=0)
    exact_by_openalex_id: int = Field(default=0, ge=0)
    exact_by_doi: int = Field(default=0, ge=0)
    fuzzy_by_title: int = Field(default=0, ge=0)
    skipped_short_title: int = Field(default=0, ge=0)
    output_count: int = Field(default=0, ge=0)

    @property
    def drop_ratio(self) -> float:
        """掉了多少比例。超过 0.01 要回头查 filter，不是接受它。"""
        if self.input_count == 0:
            return 0.0
        return (self.input_count - self.output_count) / self.input_count


def normalize_title(title: str) -> str:
    """title 归一化：小写 → 去标点 → 去停用词 → 合并空白（M1 §5.4 Level 2 第 1 步）。"""
    lowered = _PUNCT_RE.sub(" ", title.lower())
    words = [w for w in _SPACE_RE.split(lowered) if w and w not in _TITLE_STOPWORDS]
    return " ".join(words)


def _non_empty_field_count(paper: Paper) -> int:
    """非空字段计数。同 DOI 撞车时用它挑「信息更全」的那条。"""
    return sum(1 for value in paper.model_dump().values() if value not in (None, "", [], {}))


def dedup_exact(papers: Iterable[Paper], report: DedupReport | None = None) -> list[Paper]:
    """Level 1 · 精确去重（可并行，这里按流式单机实现）。

    两条规则的取舍依据不同，所以不能合并成一条：
    - 同 OpenAlex ID → 取 `updated_date` 最新（同一实体的不同快照）；
    - 同 DOI → 取非空字段更多的一条（可能是两个 ID 指向同一篇，谁更新不确定）。

    `Paper` 上没有 `updated_date` 字段——它是源记录的字段，不进领域模型。所以这里
    的实现约定是**输入顺序即新鲜度顺序**（调用方按 `updated_date` 降序喂进来），
    先到者胜。把排序责任放在调用方是有意的：OpenAlex 的游标分页本来就能按
    `updated_date` 排序，在这里再排一次是白排一遍 27 万条。
    """
    rep = report or DedupReport()
    by_id: dict[str, Paper] = {}
    by_doi: dict[str, str] = {}

    for paper in papers:
        rep.input_count += 1
        if paper.paper_id in by_id:
            rep.exact_by_openalex_id += 1
            continue

        doi = paper.doi
        if doi:
            existing_id = by_doi.get(doi)
            if existing_id is not None:
                rep.exact_by_doi += 1
                incumbent = by_id[existing_id]
                if _non_empty_field_count(paper) > _non_empty_field_count(incumbent):
                    del by_id[existing_id]
                    by_id[paper.paper_id] = paper
                    by_doi[doi] = paper.paper_id
                continue
            by_doi[doi] = paper.paper_id

        by_id[paper.paper_id] = paper

    rep.output_count = len(by_id)
    return list(by_id.values())


def _title_similarity(left: str, right: str) -> float:
    """归一化 title 的相似度。

    用标准库的 `SequenceMatcher` 而不是引第三方编辑距离库：这一步只在 LSH 候选对
    上跑（候选对数量是线性的），没有性能压力，少一个依赖更划算。
    """
    return SequenceMatcher(None, left, right).ratio()


def _author_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    """作者集合的 Jaccard。两边都为空时返回 0——「都没有作者」不构成同一性证据。"""
    left_set = {a.strip().lower() for a in left if a.strip()}
    right_set = {a.strip().lower() for a in right if a.strip()}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _candidate_pairs(papers: Sequence[Paper]) -> Iterable[tuple[int, int]]:
    """候选对生成。

    **这里是全量两两比较的替代点，也是本模块唯一需要接真实实现的地方。**
    正式实现用 `datasketch.MinHashLSH` 对归一化 title 的 character 3-gram 建索引，
    `threshold≈0.85`、`num_perm=128`，`query()` 出候选对。当前骨架退化为「归一化
    title 的前 12 个字符相同」这个廉价分桶——它在 27 万条上可跑、行为可预测，但
    召回明显低于 LSH：首字不同的同一篇（如 "The Xxx" vs "Xxx"）会漏掉。
    **接真实实现：把下面的分桶换成 MinHashLSH。**
    """
    buckets: dict[str, list[int]] = {}
    for idx, paper in enumerate(papers):
        key = normalize_title(paper.title)[:12]
        buckets.setdefault(key, []).append(idx)
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                yield members[i], members[j]


def dedup_fuzzy(papers: Sequence[Paper], report: DedupReport | None = None) -> list[Paper]:
    """Level 2 · 模糊去重（捕获 preprint 与正式发表版）。

    三个特例（M1 §5.4）在这里落地：
    1. 同一工作的不同语言版本**不判重**——语言不同，检索价值不同。本实现靠
       归一化 title 天然不相同来达成，不做语言识别。
    2. **撤稿论文不物理删除**，标记 `is_retracted=True` 由检索层按需过滤。所以
       撤稿记录在这里与普通记录同等参与去重，不做特殊剔除。
    3. 无 DOI 且 title < 5 词 → 跳过模糊去重（短 title 误判率高），仅做精确去重。
    """
    rep = report or DedupReport()
    if rep.input_count == 0:
        rep.input_count = len(papers)

    dropped: set[int] = set()
    for i, j in _candidate_pairs(papers):
        if i in dropped or j in dropped:
            continue
        left, right = papers[i], papers[j]

        left_norm, right_norm = normalize_title(left.title), normalize_title(right.title)
        if left.doi is None and len(left_norm.split()) < _MIN_TITLE_WORDS_FOR_FUZZY:
            rep.skipped_short_title += 1
            continue

        if left_norm != right_norm:
            if _title_similarity(left_norm, right_norm) <= _TITLE_SIMILARITY_THRESHOLD:
                continue
            if _author_jaccard(left.authors, right.authors) <= _AUTHOR_JACCARD_THRESHOLD:
                continue

        # 保留 cited_by_count 最高的一条——通常是正式发表版，因为引用会累积到它上面。
        loser = j if (right.cited_by_count or 0) <= (left.cited_by_count or 0) else i
        dropped.add(loser)
        rep.fuzzy_by_title += 1

    kept = [p for idx, p in enumerate(papers) if idx not in dropped]
    rep.output_count = len(kept)
    if rep.drop_ratio > 0.01:
        _logger.warning("dedup_drop_ratio_high", drop_ratio=rep.drop_ratio, hint="回头查摘要层 filter 是否按预期收敛")
    return kept
