"""grouping search 结果的解析（决策记录 D9、M2 §12）。

Milvus 用 `group_by_field=paper_id` 在**服务端**完成 chunk→paper 聚合，
所以这一层不做任何排序或去重——它只把 pymilvus 返回的 `list[list[dict]]`
翻译成本项目的数据契约（`RetrievedChunk` / `PaperHit`）。

两条必须写进解析逻辑的语义（`indexing.ann_requests.GroupingSpec` 的 docstring 亦然）：

1. **`limit` 是组数不是条数**——所以对外参数叫 `group_limit`，不叫 `top_k`。
2. **`group_size` 是上界不是保证**——摘要层每篇只有 1 个 chunk，对这些论文要 3 段
   会**静默返回 1 条**；加上 `strict_group_size=False`，实际返回 ≤ `limit × size`。
   **下游的上下文预算必须按「可能少于 48」写**，按 48 写死就是一个偶发的下标越界。

还有一条硬约束在类型层就体现了（M6 §1.5）：**RRF 分数不对外暴露**。
k=60 时上界约 2/61≈0.033，这个数没有任何物理含义，禁止任何上游拿它做相关性阈值。
它只允许留在内部字段 `rrf_score` 上，用于调试与评测复现。
"""

from __future__ import annotations

from typing import Any

from citequest.core.corpus import Attribution
from citequest.core.enums import Granularity, LicenseId, SecType, SourceDataset
from citequest.core.search import PaperHit, RetrievedChunk
from citequest.observability.logging import get_logger

__all__ = ["ParsedGroups", "parse_groups", "to_paper_hits"]

_logger = get_logger(__name__)

#: 解析失败时的兜底署名。**不留空**：响应体里两层内容并置时，只有一层有
#: attribution 会让前端写出「没有署名 = 没有来源」这种分支（见 ingestion.normalize）。
_UNKNOWN_ATTRIBUTION = Attribution(text="unknown source", source_url="about:blank", original_url=None)


class ParsedGroups:
    """一次 grouping search 的解析产物。

    `requested_groups` 与 `len(groups)` 一起保留，是因为**两者之比是三个「静默失败」
    监控项之一**（M6 §6.5）：Milvus 实际返回的组数长期低于请求组数，说明过滤太窄
    或索引有问题，而这件事在响应里看不出来——用户只会觉得「结果有点少」。
    """

    __slots__ = ("groups", "requested_groups")

    def __init__(self, groups: list[list[RetrievedChunk]], requested_groups: int) -> None:
        self.groups = groups
        self.requested_groups = requested_groups

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def chunks(self) -> list[RetrievedChunk]:
        """按组序展平。rerank 的输入就是这个列表——**它可能少于 48 条**。"""
        return [chunk for group in self.groups for chunk in group]

    @property
    def fill_ratio(self) -> float:
        """实际组数 / 请求组数。低于 1 即 M3 §3.5 档 1 降级的触发信号，
        **不需要再拍一个「少于 8 组」之类的阈值**——「凑不满预算」本身就是事实。"""
        return self.group_count / self.requested_groups if self.requested_groups else 0.0


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _row_to_chunk(row: dict[str, Any], text_field: str) -> RetrievedChunk | None:
    """一行 Milvus 结果 → `RetrievedChunk`。字段缺失返回 None 并打日志。

    **单条解析失败不炸整次检索**：一条 chunk 的 `granularity` 写错只是少一条候选，
    而抛异常会让整个请求 500——检索是底线，它没有降级对象。缺失必须打日志，
    因为它指向的是灌库期的数据腐化，不是查询期的问题。
    """
    entity = row.get("entity", row)
    chunk_id = entity.get("chunk_id")
    paper_id = entity.get("paper_id")
    text = entity.get(text_field)
    year = _as_int(entity.get("year"))
    if not chunk_id or not paper_id or not text or year is None:
        _logger.warning("chunk_row_incomplete", chunk_id=chunk_id, paper_id=paper_id, has_text=bool(text))
        return None

    try:
        granularity = Granularity(entity.get("granularity", ""))
        license_id = LicenseId(entity.get("license", ""))
        source = SourceDataset(entity.get("source_dataset", ""))
    except ValueError as exc:
        _logger.warning("chunk_enum_invalid", chunk_id=chunk_id, error=str(exc))
        return None

    sec_type_raw = entity.get("sec_type")
    sec_type: SecType | None = None
    if sec_type_raw:
        try:
            sec_type = SecType(sec_type_raw)
        except ValueError:
            # 未知 sec_type 降级为 None 而不是丢弃整块：IMRaD 类型是可选的展示维度，
            # 而这块文本本身仍然是合法证据。
            _logger.debug("chunk_sec_type_unknown", value=sec_type_raw)

    return RetrievedChunk(
        chunk_id=str(chunk_id),
        paper_id=str(paper_id),
        text=str(text),
        granularity=granularity,
        section=entity.get("section"),
        sec_number=entity.get("sec_number"),
        sec_type=sec_type,
        paragraph_idx=_as_int(entity.get("paragraph_idx")),
        year=year,
        primary_category=entity.get("primary_category"),
        license=license_id,
        source_dataset=source,
        # distance 在 hybrid_search 下就是 RRF 分数。**只进内部字段**，
        # 序列化到 API 响应前必须剔除（M6 §1.5）。
        rrf_score=float(row["distance"]) if row.get("distance") is not None else None,
    )


def parse_groups(
    rows: list[dict[str, Any]],
    *,
    text_field: str,
    group_by_field: str = "paper_id",
    requested_groups: int,
) -> ParsedGroups:
    """把 pymilvus 的一次检索结果翻译成内部契约，并按分组键收拢成组。

    **入参是「一条 query 的命中列表」**，不是嵌套的组结构：`hybrid_search` 返回的
    是「每条 query 一个结果列表」，而 grouping 的分组体现在**同一个 paper_id 的
    命中相邻出现**上。调用方取过第一维再传进来。

    这里按首次出现顺序收拢而不是排序：Milvus 已经在服务端排好序了，
    **本层不重排**——重排一次就等于把服务端的融合结果换成自己的口径，
    而那个口径没有任何依据。
    """
    groups: list[list[RetrievedChunk]] = []
    index_of: dict[str, int] = {}
    for row in rows:
        chunk = _row_to_chunk(row, text_field)
        if chunk is None:
            continue
        entity = row.get("entity", row)
        key = str(entity.get(group_by_field) or chunk.paper_id)
        position = index_of.get(key)
        if position is None:
            index_of[key] = len(groups)
            groups.append([chunk])
        else:
            groups[position].append(chunk)
    _logger.info("grouping_parsed", groups=len(groups), rows=len(rows), requested=requested_groups)
    return ParsedGroups(groups=groups, requested_groups=requested_groups)


def to_paper_hits(
    parsed: ParsedGroups,
    *,
    titles: dict[str, str] | None = None,
    attributions: dict[str, Attribution] | None = None,
    cited_by_counts: dict[str, int] | None = None,
) -> list[PaperHit]:
    """组 → `PaperHit`。**不重排组内顺序**——Milvus grouping 保证组内有序。

    `titles` / `attributions` / `cited_by_counts` 由**论文级元数据侧**提供
    （SQLite/DuckDB 读 parquet，与全库精确分面同一个载体）：这些字段不在 chunk
    schema 里，因为它们是论文级属性，放进每个 chunk 会在 147 万行上重复存储，
    而标题改一次就要重写整篇论文的全部块。

    **接真实实现**：`titles` 等三个映射当前由调用方传入；正式实现里它们来自
    `PaperMetadataStore.fetch_many(paper_ids)`（DuckDB 一次 IN 查询）。
    传空时用 paper_id 占位——**宁可显示 ID 也不显示一个编出来的标题**。
    """
    titles = titles or {}
    attributions = attributions or {}
    cited_by_counts = cited_by_counts or {}

    hits: list[PaperHit] = []
    for group in parsed.groups:
        head = group[0]
        hits.append(
            PaperHit(
                paper_id=head.paper_id,
                title=titles.get(head.paper_id, head.paper_id),
                year=head.year,
                primary_category=head.primary_category,
                cited_by_count=cited_by_counts.get(head.paper_id),
                granularity=head.granularity,
                # has_fulltext 是冗余字段，等价于 granularity == FULLTEXT。
                # 冗余是刻意的：API 响应直接下发，前端不做推导（M6 §1.2）。
                has_fulltext=head.granularity is Granularity.FULLTEXT,
                chunks=group,
                license=head.license,
                source_dataset=head.source_dataset,
                attribution=attributions.get(head.paper_id, _UNKNOWN_ATTRIBUTION),
            )
        )
    return hits
