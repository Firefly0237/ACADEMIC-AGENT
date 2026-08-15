"""搜索结果的片段与高亮（M6 §1.2 的展示层契约）。

**高亮只在应用侧做，不向 Milvus 要**：Milvus 的 BM25 通道给的是分数不是命中位置，
而 `TEXT_MATCH` 只判命中。要拿到高亮位置只能自己在返回的原文上再匹配一次——
这一步在候选集只有 60 条时是纯字符串操作，比引一个高亮引擎便宜得多。

**片段不重新分词**：高亮匹配用的是「归一化后的词面 + 大小写不敏感」，
**不复刻 english analyzer 的 stemmer**。理由是复刻必然对不齐——analyzer 在
Milvus 服务端（Tantivy），应用侧再实现一遍就是两套分词器人肉对齐，而这正是
D7 里否决 Elasticsearch 的同一条判据。代价写出来：`training` 命中的文档里
高亮不到 `train`，**少高亮几个词好过高亮错位置**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from citequest.core.search import RetrievedChunk

__all__ = ["Snippet", "build_snippet", "highlight_terms"]

#: 片段窗口的默认长度（字符）。取 320 是因为它比一条摘要句长、比一个 450-token
#: 块短——目标是让用户在结果列表里读到**一段完整的话**而不是半句。
#: 这个数只影响展示，不进任何检索路径，所以不进参数正本表。
_DEFAULT_WINDOW = 320

_WORD_RE = re.compile(r"[\w一-鿿]+")


@dataclass(slots=True, frozen=True)
class Snippet:
    """一条展示片段。

    `highlights` 是**字符区间**而不是插好标签的 HTML：由前端决定怎么渲染。
    在后端拼 `<em>` 会让这个字段既是数据又是标记，一旦前端换渲染方式就得改后端，
    而且拼 HTML 是一个默认不安全的操作（文本来自论文正文，不是我们生成的）。
    """

    text: str
    highlights: tuple[tuple[int, int], ...]
    truncated_head: bool
    truncated_tail: bool


def _query_terms(query: str) -> list[str]:
    """把 query 拆成用于高亮的词面。

    **短于 2 个字符的词丢掉**：单字母会在几乎每段英文里命中，高亮成一片就等于
    没有高亮。中文单字保留——中文单字是有意义的检索单位。
    """
    terms = [t.lower() for t in _WORD_RE.findall(query)]
    return [t for t in terms if len(t) >= 2 or not t.isascii()]


def highlight_terms(text: str, query: str) -> tuple[tuple[int, int], ...]:
    """返回 query 词面在 text 里的命中区间（大小写不敏感，按起点排序、不重叠）。

    重叠区间会合并：`"neural network"` 里 `neural` 与 `network` 各自命中时，
    前端拿到两个相邻区间会渲染出两个挨着的高亮块，中间那个空格反而突兀。
    """
    lowered = text.lower()
    spans: list[tuple[int, int]] = []
    for term in _query_terms(query):
        start = lowered.find(term)
        while start >= 0:
            spans.append((start, start + len(term)))
            start = lowered.find(term, start + len(term))
    if not spans:
        return ()

    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def build_snippet(chunk: RetrievedChunk, query: str, *, window: int = _DEFAULT_WINDOW) -> Snippet:
    """围绕第一个命中位置截一个窗口，并把高亮偏移换算到窗口内坐标。

    **窗口按句边界向外扩，不在词中间切**：切在词中间的片段读起来像乱码，
    而这一条与 M1 §6.8「永不在段落中间切」是同一个取向——展示层的残缺同样会
    让用户以为原文就是这样。

    没有任何命中时取开头（摘要块的开头本来就是最有信息量的部分），
    **不返回空片段**：空片段会让结果卡片塌成一行标题，用户无法判断这条结果的内容。
    """
    if window < 32:
        raise ValueError(f"snippet 窗口至少 32 字符，收到 {window}")

    text = chunk.text
    hits = highlight_terms(text, query)
    if len(text) <= window:
        return Snippet(text=text, highlights=hits, truncated_head=False, truncated_tail=False)

    anchor = hits[0][0] if hits else 0
    start = max(0, anchor - window // 3)
    end = min(len(text), start + window)
    # 向词边界外扩：只在两端各扫最多 20 个字符，扫不到就按原位切——
    # 无界回溯会在一段没有空格的长公式描述上退化成扫全文。
    while start > 0 and not text[start - 1].isspace() and anchor - start < window // 3 + 20:
        start -= 1
    while end < len(text) and not text[end].isspace() and end - start < window + 20:
        end += 1

    windowed = text[start:end].strip()
    offset = start + (len(text[start:end]) - len(text[start:end].lstrip()))
    shifted = tuple(
        (h_start - offset, h_end - offset)
        for h_start, h_end in hits
        if h_start >= offset and h_end - offset <= len(windowed)
    )
    return Snippet(
        text=windowed,
        highlights=shifted,
        truncated_head=start > 0,
        truncated_tail=end < len(text),
    )
