"""S5 · 字段标准化（M1 §5.5）。

规则一共六条，其中三条有明确的「不做什么」，比「做什么」更重要：

- `paper_id` 统一为去前缀的 OpenAlex ID（`W` + 数字）。全文层通过 arXiv ID 反查对齐；
  **对齐不上的记录保留 arXiv ID 作主键并标记，不丢弃**——丢弃会让全文层规模在
  没人注意的情况下缩水。
- 作者与 venue **保留原始 `display_name`，不做缩写统一**：缩写统一是有损的，而且
  没有权威映射表，自己造一张就是自建。
- `license` / `source_dataset` **逐条写入，不做全局声明**：两层许可不同，汇编
  （Collection）不是改编（Adaptation），一句全局混合声明糊不过去。
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from citequest.core.corpus import Attribution
from citequest.core.enums import Granularity, LicenseId, SourceDataset

__all__ = [
    "LAYER_DEFAULTS",
    "UNARXIVE_ATTRIBUTION",
    "build_openalex_attribution",
    "normalize_doi",
    "normalize_openalex_id",
    "normalize_primary_category",
    "resolve_paper_id",
    "validate_year",
]

_OPENALEX_ID_RE = re.compile(r"^(?:https?://openalex\.org/)?(W\d+)$", re.IGNORECASE)
_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/)", re.IGNORECASE)
_ARXIV_CATEGORY_RE = re.compile(r"^[a-z\-]+(?:\.[A-Z]{2})?$")

#: 全文层的署名串。**在线返回检索片段属于许可意义上的 Share，必须逐条带署名**
#: （决策记录 D2 第 5 条，标注为「法律上非确定」，这里取保守做法）。
UNARXIVE_ATTRIBUTION = Attribution(
    text="unarXive open subset (Saier et al., JCDL 2023), CC BY-SA 4.0",
    source_url="https://doi.org/10.5281/zenodo.7752615",
    original_url=None,
)


def normalize_openalex_id(raw: str | None) -> str | None:
    """`https://openalex.org/W2963341956` → `W2963341956`。

    保留 `W` 前缀而不是只留数字：`W` 区分 Work 与 Author/Source 等其它实体类型，
    去掉它之后 ID 就不再自解释，而这个字段会出现在引用边表与 API 响应里。
    """
    if not raw:
        return None
    match = _OPENALEX_ID_RE.match(raw.strip())
    if match is None:
        return None
    return match.group(1).upper()


def normalize_doi(raw: str | None) -> str | None:
    """去 `https://doi.org/` 前缀并全小写。

    全小写是因为 DOI 规范本身大小写不敏感，而去重要靠字符串相等——不归一化就会
    出现「同一篇论文两条记录」这类只在特定数据源组合下才暴露的重复。
    """
    if not raw:
        return None
    stripped = _DOI_PREFIX_RE.sub("", raw.strip()).strip().lower()
    return stripped or None


def validate_year(raw: int | str | None) -> int | None:
    """范围校验 `[1900, 当前年 + 1]`。

    上界给到「当前年 + 1」是因为预印本的 publication_date 可能落在次年（会议
    proceedings 提前挂出）。越界返回 None 而不是抛异常：年份异常的记录仍然有
    检索价值，但它不能进 `year` 这个**同时是 partition key** 的字段——一个 3021
    年的分区会永远只有一条数据。
    """
    if raw is None:
        return None
    try:
        year = int(raw)
    except (TypeError, ValueError):
        return None
    upper = datetime.now(tz=UTC).year + 1
    if year < 1900 or year > upper:
        return None
    return year


def normalize_primary_category(raw: str | None) -> str | None:
    """arXiv 一级类目形态校验（`cs.CL`）。

    不做映射表补全：全文层直接取 `metadata.categories` 的 primary，摘要层由
    OpenAlex 的 `primary_topic` 映射而来，两边都在各自的 source 里处理，
    这里只负责拒绝明显不是 arXiv 类目的字符串。
    """
    if not raw:
        return None
    candidate = raw.strip()
    if not _ARXIV_CATEGORY_RE.match(candidate):
        return None
    return candidate


def resolve_paper_id(openalex_id: str | None, arxiv_id: str | None) -> tuple[str, bool]:
    """确定主键，并返回「是否退化为 arXiv ID」。

    返回二元组而不是只返回 ID，是因为退化率本身是数据质量指标：它等于「全文层有
    多少篇没能对齐到 OpenAlex」，直接决定引用边表的覆盖面（M1 §7.3 的【待实测】项）。

    :raises ValueError: 两个 ID 都没有。这类记录无法被引用、无法被分组、无法被
        撤稿精确失效，进库只会变成一条永远查不到出处的数据。
    """
    normalized = normalize_openalex_id(openalex_id)
    if normalized:
        return normalized, False
    if arxiv_id and arxiv_id.strip():
        return arxiv_id.strip(), True
    raise ValueError("openalex_id 与 arxiv_id 不能同时为空——没有主键的记录无法进库")


def build_openalex_attribution(arxiv_id: str | None) -> Attribution:
    """摘要层署名。CC0 不强制署名，但**仍然逐条给**。

    理由不是许可要求，是口径一致：响应体里两层内容并置时，只有一层有 attribution
    会让前端不得不写一个「有没有署名」的分支，而那个分支迟早会被写成
    「没有署名 = 没有来源」。
    """
    return Attribution(
        text="OpenAlex (CC0)",
        source_url="https://openalex.org",
        original_url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
    )


#: 两层的固定取值。写成常量而不是每处现填，是为了让「新增一层语料」变成一处改动。
LAYER_DEFAULTS: dict[Granularity, tuple[LicenseId, SourceDataset]] = {
    Granularity.FULLTEXT: (LicenseId.CC_BY_SA_4_0, SourceDataset.UNARXIVE_OPEN_SUBSET),
    Granularity.ABSTRACT: (LicenseId.CC0, SourceDataset.OPENALEX),
}
