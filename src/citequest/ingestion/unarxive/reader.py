"""全文层读取与筛选（M1 §3.2 的 F1–F4）。

归档是 tar.xz，内部是按分片组织的 JSONL，一行一篇论文。**必须流式读**：4.84 GB
压缩、解压后更大，一次性载入内存是不可能的；`tarfile` 以流模式打开（`r|xz`）时
只能顺序遍历，这正好与「一行行读、一行行处理」的形态吻合。

四道 filter 的顺序是**按代价从小到大**排的，这不是风格问题：F1/F2 只看元数据，
F3 要看 `body_text` 是否非空，越早判掉越少做无用功。
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from citequest.config.corpus import CorpusSettings
from citequest.observability.logging import get_logger

__all__ = ["FilterStats", "UnarxiveRecord", "iter_records", "passes_filters"]

_logger = get_logger(__name__)


class UnarxiveRecord(BaseModel):
    """一条原始记录（**清洗前**的形态）。

    刻意不复用 `core.corpus.Paper`：那是数据层的**输出**契约，字段已经标准化过。
    在这里就用它，等于把「原始数据长什么样」和「我们要求它长什么样」混成一件事，
    而 F3（`body_text` 为空）这类质量问题恰恰只存在于原始形态里。
    """

    paper_id: str = Field(description="unarXive 的记录主键，通常是 arXiv ID")
    metadata: dict[str, Any] = Field(default_factory=dict)
    body_text: list[dict[str, Any]] = Field(default_factory=list, description="带 section 结构的段落序列")
    bib_entries: dict[str, Any] = Field(
        default_factory=dict, description="key → 参考文献；`ids.open_alex_id` 是引用边来源"
    )
    abstract: dict[str, Any] | None = None

    @property
    def primary_category(self) -> str | None:
        """`metadata.categories` 的 **primary**（第一个）类目。

        用 primary 而不是「任一类目含 cs」——后者会把大量以 CS 为交叉学科标签的
        物理/数学论文卷进来，稀释领域纯度（F1 的理由）。
        """
        categories = self.metadata.get("categories")
        if isinstance(categories, str):
            parts = categories.split()
            return parts[0] if parts else None
        if isinstance(categories, list) and categories:
            first = categories[0]
            return str(first) if first else None
        return None

    @property
    def submitted_year(self) -> int | None:
        """提交年。unarXive 各版本的时间字段名不一致，按优先级探测。

        探测而不是写死一个字段名：写死的失败形态是「新版本换了字段名 → 年份全为
        None → F2 把所有记录都过滤掉 → 全文层空了」，而这个过程不报错。
        """
        for key in ("update_date", "submitted", "created", "date"):
            raw = self.metadata.get(key)
            if isinstance(raw, str) and len(raw) >= 4 and raw[:4].isdigit():
                return int(raw[:4])
            if isinstance(raw, int):
                return raw
        return None

    @property
    def license_value(self) -> str | None:
        raw = self.metadata.get("license")
        return str(raw) if raw else None


class FilterStats(BaseModel):
    """逐道 filter 的淘汰计数（M1 §10 必跑统计脚本）。

    `rejected_empty_body` 是里面最重要的一个：它就是 **LaTeX 解析失败率**，而
    解析失败率直接决定「5 万篇够不够凑」。**这类记录必须计数而不是静默丢弃**。
    """

    seen: int = Field(default=0, ge=0)
    rejected_non_cs: int = Field(default=0, ge=0)
    rejected_year: int = Field(default=0, ge=0)
    rejected_empty_body: int = Field(default=0, ge=0)
    rejected_license: int = Field(default=0, ge=0)
    accepted: int = Field(default=0, ge=0)
    malformed_lines: int = Field(default=0, ge=0, description="JSON 解析失败的行。按行隔离，不阻断整条流水线")

    @property
    def empty_body_ratio(self) -> float:
        """`body_text` 为空占比 = LaTeX 解析失败率的直接观测（M1 §5.2）。"""
        return self.rejected_empty_body / self.seen if self.seen else 0.0


#: F4 的宽松许可白名单。unarXive open subset 本身即可再分发子集，这里再过一道是
#: 为了让「哪些记录可以进 samples/」有据可查（决策记录 D2 第 3 条）。
PERMISSIVE_LICENSE_MARKERS = ("cc-by", "cc0", "publicdomain", "cc-by-sa")


def passes_filters(record: UnarxiveRecord, settings: CorpusSettings, stats: FilterStats) -> bool:
    """F1–F4，按代价从小到大短路。

    F2 的年份下界是**可调旋钮**：解压后实测不足 5 万篇就把 `fulltext_year_from`
    放宽到 2017；超出则从最新年份往回截，保留时效性最好的那部分。所以这里读的是
    配置而不是常量——预案要能被一个环境变量执行，才叫预案。
    """
    stats.seen += 1

    category = record.primary_category
    if not category or not category.startswith(settings.fulltext_category_prefix):
        stats.rejected_non_cs += 1
        return False

    year = record.submitted_year
    if year is None or not (settings.fulltext_year_from <= year <= settings.fulltext_year_to):
        stats.rejected_year += 1
        return False

    if not record.body_text:
        stats.rejected_empty_body += 1
        return False

    license_value = (record.license_value or "").lower()
    if license_value and not any(marker in license_value for marker in PERMISSIVE_LICENSE_MARKERS):
        stats.rejected_license += 1
        return False

    stats.accepted += 1
    return True


def iter_records(
    archive_path: Path,
    settings: CorpusSettings | None = None,
    *,
    stats: FilterStats | None = None,
    limit: int | None = None,
    start_after_member: str | None = None,
) -> Iterator[UnarxiveRecord]:
    """流式遍历归档，按 F1–F4 过滤后逐条产出。

    :param start_after_member: 断点续跑用——跳过这个 tar 成员之前（含）的全部内容。
        粒度只到成员（分片文件）而不到行，是因为 `chunk_id` 的幂等构造让重跑同一
        分片是安全的（M1 §8.2），为行级续跑维护额外状态是白付成本。
    :param limit: 规模上界。对应「超出 5 万则从最新年份往回截」的另一半——
        调用方按年份倒序遍历分片并配合这个参数。
    """
    conf = settings or CorpusSettings()
    counters = stats if stats is not None else FilterStats()
    skipping = start_after_member is not None
    produced = 0

    # `r|*` 是流模式：不做随机寻址，内存占用与归档大小无关。
    # 代价是不能回头，而这正好与「只顺序扫一遍」的用法一致。
    with tarfile.open(archive_path, mode="r|*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith((".jsonl", ".json")):
                continue
            if skipping:
                if member.name == start_after_member:
                    skipping = False
                continue

            handle = archive.extractfile(member)
            if handle is None:  # pragma: no cover - 防御：流模式下极少发生
                _logger.warning("tar_member_unreadable", member=member.name)
                continue

            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    # 按行隔离：一行坏 JSON 不能让 4.84 GB 的遍历中止。
                    counters.malformed_lines += 1
                    continue
                record = _to_record(payload)
                if record is None:
                    counters.malformed_lines += 1
                    continue
                if not passes_filters(record, conf, counters):
                    continue
                yield record
                produced += 1
                if limit is not None and produced >= limit:
                    _logger.info("record_limit_reached", limit=limit, member=member.name)
                    return


def _to_record(payload: dict[str, Any]) -> UnarxiveRecord | None:
    """原始 dict → `UnarxiveRecord`，字段名按已知别名兼容。

    主键在不同版本里叫 `paper_id` 或 `id`。缺主键的记录直接丢——没有主键就无法
    幂等、无法回溯、无法撤稿失效，留着只会变成一条查不到出处的数据。
    """
    paper_id = payload.get("paper_id") or payload.get("id")
    if not paper_id:
        return None
    return UnarxiveRecord(
        paper_id=str(paper_id),
        metadata=payload.get("metadata") or {},
        body_text=payload.get("body_text") or [],
        bib_entries=payload.get("bib_entries") or {},
        abstract=payload.get("abstract"),
    )
