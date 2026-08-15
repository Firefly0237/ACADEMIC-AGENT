"""OpenAlex 摘要重建：倒排索引 → 连续文本（M1 §6.9 的实现细节）。

OpenAlex 出于版权合规，摘要以**倒排索引**格式存储：`{词: [位置1, 位置2, ...]}`。
用之前必须还原成连续文本，而还原有三类必须显式处理的异常——它们都**不报错**，
只会让某些论文的摘要变成一段读起来怪怪的文本：

| 异常 | 形态 | 处理 |
|---|---|---|
| 位置空洞 | 位置序列不连续（原文里有被去掉的词） | **跳过，不插占位符**。插 `[UNK]` 之类的标记会进 BM25 倒排，制造一个全库高频的垃圾 token |
| 位置冲突 | 两个词声明了同一个位置 | 保留先出现的，记 warning |
| 空索引 / 字段缺失 | `abstract_inverted_index` 为 None 或 {} | 标记该篇无摘要——**它不进摘要层**，只有 title 的 chunk 达不到 80 token 下限 |
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from citequest.observability.logging import get_logger

__all__ = ["AbstractRebuildResult", "rebuild_abstract"]

_logger = get_logger(__name__)

#: 位置上界。OpenAlex 的摘要典型 150–250 词，出现远超这个量级的位置索引意味着
#: 数据本身异常（曾见过把全文塞进摘要字段的记录），继续还原会产出一个几万词的
#: 「摘要」，破坏「摘要层一篇一 chunk」这条不变量。
_MAX_POSITION = 10_000


class AbstractRebuildResult(BaseModel):
    """还原结果 + 三类异常的计数。"""

    text: str = Field(default="", description="还原后的摘要。空串表示该篇无可用摘要")
    word_count: int = Field(default=0, ge=0)
    gap_count: int = Field(default=0, ge=0, description="位置空洞数")
    conflict_count: int = Field(default=0, ge=0, description="位置冲突数")

    @property
    def is_usable(self) -> bool:
        return bool(self.text.strip())


def rebuild_abstract(inverted_index: dict[str, Any] | None, *, paper_id: str | None = None) -> AbstractRebuildResult:
    """把 `abstract_inverted_index` 还原成连续文本。

    实现是「按位置填进稀疏表 → 按位置升序拼接」，而不是「按位置排序后 join」：
    前者能直接观测到空洞与冲突（两类必须被计数的异常），后者会把它们抹平。
    """
    if not inverted_index:
        return AbstractRebuildResult()

    slots: dict[int, str] = {}
    conflicts = 0
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for raw_pos in positions:
            try:
                pos = int(raw_pos)
            except (TypeError, ValueError):
                continue
            if pos < 0 or pos > _MAX_POSITION:
                continue
            if pos in slots:
                # 保留先出现的词。谁「先」由 dict 迭代顺序决定，而 Python 的 dict
                # 保序 + JSON 解析保序，所以这个行为是确定性的、可复现的。
                conflicts += 1
                continue
            slots[pos] = str(word)

    if not slots:
        return AbstractRebuildResult(conflict_count=conflicts)

    ordered = sorted(slots.items())
    # 空洞 = 声明过的位置区间里缺掉的槽位数。不插占位符，只计数。
    span = ordered[-1][0] - ordered[0][0] + 1
    gaps = span - len(ordered)
    if conflicts or gaps:
        _logger.debug("abstract_rebuild_anomaly", paper_id=paper_id, gaps=gaps, conflicts=conflicts)

    return AbstractRebuildResult(
        text=" ".join(word for _, word in ordered),
        word_count=len(ordered),
        gap_count=max(gaps, 0),
        conflict_count=conflicts,
    )
