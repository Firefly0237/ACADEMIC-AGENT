"""阶段 A：归一化与信号提取（M3 §1 的 A 段、§1.2 信号表）。

**这一层是纯字符串操作**，预算 p50 1ms / p95 3ms（M3 §1，推算值未实测）。
它存在的唯一理由是让 60–70% 的 query 不必付 Level 2 小 LLM 的 120–300ms。

一条贯穿本模块的纪律（M3 §1.2 末段）：**同一段文本只解析一次**。
所有信号在这里一次抽完，装进 `QuerySignals` 向下传；Level 1 与 Level 2
都不再回头解析原始文本。这是本模块唯一的性能设计，其余部分不做优化——
它们本来就在毫秒以下。
"""

from __future__ import annotations

import re
import unicodedata

from citequest.core.query_plan import QuerySignals
from citequest.query.lexicons import (
    COMPARISON_WORDS,
    METHOD_WORDS,
    QUESTION_WORDS,
    RECENCY_WORDS,
    scan_lexicon,
)

__all__ = ["count_tokens", "extract_phrases", "extract_signals", "normalize_query", "strip_spans"]

#: CJK 统一表意文字。M3 §4.1：语言判别**只有两行**，没有分类模型、没有功能词表。
#: 判据是语料侧的事实——文档层是纯英文，语言判别只影响「回答用什么语言」
#: 与「要不要走 sparse 通道」，这两件事都不需要区分英语和德语。
_CJK_RE = re.compile(r"[一-鿿぀-ヿ]")

#: 零宽字符与 BOM。它们看不见但会破坏词表匹配与 BM25 分词，
#: 而且从网页复制的 query 里非常常见。
_ZERO_WIDTH_RE = re.compile(r"[​-‏  ﻿]")

_WHITESPACE_RE = re.compile(r"\s+")

#: 成对双引号包裹的片段（M3 §1.2 / §4.3）。中文全角引号一起收——
#: 用户从中文输入法打出的引号同样表达「确切短语」这个语义。
_PHRASE_RE = re.compile(r'"([^"]+)"|“([^”]+)”')

#: 粗切分：空白 + 标点。**不用分词器**——这一层不能付模型的钱（M3 §1.2）。
_TOKEN_RE = re.compile(r"[\w一-鿿]+")


def normalize_query(raw: str) -> str:
    """NFKC 归一化 + 零宽字符清理 + 空白合并。

    NFKC 而不是 NFC：全角字母数字（"ＢＥＲＴ"）和兼容字符会被折叠成 ASCII 形态，
    否则同一个词在 BM25 的 analyzer 里就是两个 token，而用户看不出差别。

    **归一化后为空**由调用方判为 `QUERY_EMPTY`，本函数不抛异常——
    它是一个纯转换，把「空 query 怎么处理」的策略留在路由层（M3 §8.2）。
    """
    text = unicodedata.normalize("NFKC", raw)
    text = _ZERO_WIDTH_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def extract_phrases(text: str) -> list[tuple[str, int, int]]:
    """抽出成对引号包裹的确切短语，返回 `(短语, 起, 止)` 三元组。

    偏移必须带上：短语要从检索文本里剥离（走 PHRASE_MATCH 而不是留在 query 里），
    没有偏移就只能做字符串 replace，而 replace 会误伤同名的其它片段。

    **单个未配对的引号被忽略**——把它当短语起点会把后半句全吞掉，
    而用户打一个引号的常见原因只是输入法误触。
    """
    spans: list[tuple[str, int, int]] = []
    for match in _PHRASE_RE.finditer(text):
        phrase = match.group(1) or match.group(2) or ""
        if phrase.strip():
            spans.append((phrase.strip(), match.start(), match.end()))
    return spans


def strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """把若干 `(起, 止)` 区间从文本里剥离并重新合并空白。

    用于 M3 §7.2 第 2 条：结构化条件从检索文本里剥离
    （`NLP papers after 2020 about attention` → filter + 检索文本 `attention`）。
    **剥离后为空由调用方回退用原文检索**，否则会出现一次空 query 检索——
    空串会让 BM25 Function 产出空 sparse 向量并静默返回随机结果。
    """
    if not spans:
        return text
    kept: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if start < cursor:  # 区间重叠时以先到的为准，避免负长度切片
            continue
        kept.append(text[cursor:start])
        cursor = max(cursor, end)
    kept.append(text[cursor:])
    return _WHITESPACE_RE.sub(" ", "".join(kept)).strip()


def count_tokens(text: str) -> int:
    """粗 token 计数（M3 §1.2）：空白 + 标点切分，**不用分词器**。

    它只被三条阈值使用（`≤3`、`>4`、`>6`），这个精度足够；
    为它引一个真分词器会把这一层从 1ms 拉到几十 ms，而阈值本身就是粗判据。
    """
    return len(_TOKEN_RE.findall(text))


def extract_signals(
    normalized: str, *, year_expr: tuple[int | None, int | None] | None = None, category_terms: list[str] | None = None
) -> QuerySignals:
    """一次扫完 M3 §1.2 全部信号。

    `year_expr` 与 `category_terms` 由 `query.extraction` 抽好后传进来，而不是在
    这里再解析一遍——它们的解析规则更复杂（区间、上下界、映射表），放在一起会让
    本函数同时承担「扫词表」和「解析结构化条件」两个变化原因。

    :param normalized: `normalize_query` 的输出。**本函数不再做归一化**，
        重复归一化会掩盖调用方漏调的 bug。
    """
    if normalized != normalized.strip():  # pragma: no cover - 防御：契约是已归一化
        raise ValueError("extract_signals 收到未归一化的文本，调用方必须先过 normalize_query")

    lowered = normalized.lower()
    phrases = [phrase for phrase, _, _ in extract_phrases(normalized)]

    return QuerySignals(
        has_cjk=bool(_CJK_RE.search(normalized)),
        phrase_spans=phrases,
        year_expr=year_expr,
        category_terms=list(category_terms or []),
        question_words=scan_lexicon(lowered, QUESTION_WORDS),
        comparison_words=scan_lexicon(lowered, COMPARISON_WORDS),
        method_words=scan_lexicon(lowered, METHOD_WORDS),
        recency_words=scan_lexicon(lowered, RECENCY_WORDS),
        token_count=count_tokens(normalized),
    )
