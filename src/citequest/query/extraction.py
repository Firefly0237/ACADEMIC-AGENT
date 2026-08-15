"""过滤条件抽取：从自然语言到 `QueryFilters`（M3 §7）。

**能下推的只有四个标量字段**：`granularity` / `year` / `primary_category` / `license`。
这条边界由 collection schema 决定，不由需求决定——「引用量大于 100」看起来最像过滤，
实际上它不在向量库里（走论文级元数据侧或排序模式，M3 §7.1 末行）。

三条实现纪律（M3 §7.2），每一条都在下面的代码里能指出对应位置：

1. **正则优先，模型补位，不为它单开一次调用**——本模块只做正则；复杂表述由
   Level 2 那一次 LLM 调用顺带输出（`intent.py`），不新增第二次调用。
2. **条件从检索文本里剥离，但留一条兜底**——剥离只对正则确定命中的片段做；
   剥离后为空则整体回退用原文检索，否则会出现一次空 query 检索。
3. **抽取失败就丢弃该条件并显式记录**——年份倒挂、类目映射不命中一律丢掉这一条、
   打 warning、用剩下的条件继续。**绝不静默改写用户的语义**（比如把解析失败的
   年份默认成「近 5 年」）。丢弃是可观测的，猜测不是。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from citequest.core.query_plan import PhraseMatchSpec, QueryFilters
from citequest.observability.logging import get_logger
from citequest.query.lexicons import match_category
from citequest.query.normalize import extract_phrases, strip_spans

__all__ = ["ExtractionResult", "extract_filters"]

_logger = get_logger(__name__)

#: 年份的合法范围。与 `QueryFilters` 的 `ge/le` 一致——两处不一致时，
#: 这里放过去的值会在构造 pydantic 模型时抛异常，把「丢弃该条件」变成一次 500。
_YEAR_MIN, _YEAR_MAX = 1900, 2100

#: `2021-2023` / `2021 到 2023` 区间。先于上下界匹配，否则 `2021` 会被单独吃掉。
_RANGE_RE = re.compile(r"\b(19\d{2}|20\d{2})\s*(?:-|–|—|~|to|至|到)\s*(19\d{2}|20\d{2})\b")

#: 下界：after / since / 自 / 以后 / 之后 / 以来。
_AFTER_RE = re.compile(r"(?:after|since|from|自|从)\s*(19\d{2}|20\d{2})|\b(19\d{2}|20\d{2})\s*年?(?:以后|之后|以来|后)")

#: 上界：before / prior to / 之前 / 以前。
_BEFORE_RE = re.compile(r"(?:before|prior\s+to|until|截至)\s*(19\d{2}|20\d{2})|\b(19\d{2}|20\d{2})\s*年?(?:以前|之前)")

#: 词面直出的 arXiv 类目（`cs.CL`）。这类写法不需要经映射表。
_CATEGORY_RE = re.compile(r"\b(cs\.[A-Z]{2})\b", re.IGNORECASE)


@dataclass(slots=True)
class ExtractionResult:
    """抽取产物。

    `search_query` 与 `filters` 分开返回而不是塞进一个大对象，是因为它们的
    消费者不同：前者进 `AnnSearchRequest.data`，后者进 Milvus 过滤表达式。
    """

    search_query: str
    filters: QueryFilters
    phrase_match: PhraseMatchSpec | None = None
    category_terms: list[str] = field(default_factory=list)
    year_expr: tuple[int | None, int | None] | None = None
    dropped: list[str] = field(default_factory=list)
    """被丢弃的条件描述。进日志与 span——**丢弃必须可观测**，否则用户会以为
    自己的条件生效了，而结果集只是「看起来少了几篇」。"""


def _first_year(match: re.Match[str] | None) -> int | None:
    """取匹配里第一个非空的年份分组。

    两个分组是因为英文（`after 2020`）与中文（`2020 年以后`）的数字位置相反，
    写成一条正则的两个分支比写两条正则再合并结果少一层状态。
    """
    if match is None:
        return None
    for group in match.groups():
        if group:
            return int(group)
    return None


def _extract_years(lowered: str, dropped: list[str]) -> tuple[int | None, int | None, list[tuple[int, int]]]:
    """抽年份下上界，同时返回要从检索文本里剥离的区间。

    区间表达式优先：`2021-2023` 里的两个年份如果先被上下界规则各吃一次，
    会得到 `year>=2021 and year>=2023` 这种**看起来合理但范围错了**的结果。
    """
    spans: list[tuple[int, int]] = []

    range_match = _RANGE_RE.search(lowered)
    if range_match is not None:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        spans.append(range_match.span())
        if lo > hi:
            # 上下界颠倒 → 丢掉整条。改成自动交换是「静默改写用户语义」，
            # 而用户写反的真实意图无法判断（可能只是打错了另一个数）。
            dropped.append(f"year_range_reversed({lo}-{hi})")
            return None, None, spans
        return lo, hi, spans

    after_match = _AFTER_RE.search(lowered)
    before_match = _BEFORE_RE.search(lowered)
    year_from = _first_year(after_match)
    year_to = _first_year(before_match)
    if after_match is not None:
        spans.append(after_match.span())
    if before_match is not None:
        spans.append(before_match.span())

    if year_from is not None and not _YEAR_MIN <= year_from <= _YEAR_MAX:
        dropped.append(f"year_from_out_of_range({year_from})")
        year_from = None
    if year_to is not None and not _YEAR_MIN <= year_to <= _YEAR_MAX:
        dropped.append(f"year_to_out_of_range({year_to})")
        year_to = None
    if year_from is not None and year_to is not None and year_from > year_to:
        dropped.append(f"year_bounds_reversed({year_from},{year_to})")
        return None, None, spans

    return year_from, year_to, spans


def _extract_category(normalized: str, lowered: str) -> tuple[str | None, list[str], list[tuple[int, int]]]:
    """抽 arXiv 类目：先认词面直出的 `cs.XX`，再走别名映射表。

    **映射不命中就不加过滤**（M3 §7.1）：把 "in NLP" 里的 NLP 交给检索本身处理，
    比猜一个类目安全——猜错的过滤会把正确结果整批滤掉，而且不报错。
    """
    spans: list[tuple[int, int]] = []
    direct = _CATEGORY_RE.search(normalized)
    if direct is not None:
        code = direct.group(1)
        # 大小写统一成官方写法（cs.CL 而不是 cs.cl），否则等值过滤永远不命中。
        prefix, _, suffix = code.partition(".")
        canonical = f"{prefix.lower()}.{suffix.upper()}"
        spans.append(direct.span())
        return canonical, [canonical], spans

    mapped = match_category(lowered)
    if mapped is None:
        return None, [], spans
    # 别名词面**不从检索文本里剥离**：`about NLP` 里的 NLP 同时是有效的检索词，
    # 剥掉它等于让「过滤 + 检索」重复表达同一个约束，反而缩小召回。
    return mapped, [mapped], spans


def extract_filters(normalized: str, *, default_slop: int = 0) -> ExtractionResult:
    """把归一化后的 query 拆成「检索文本 + 结构化条件 + 短语约束」。

    :param default_slop: PHRASE_MATCH 的 slop，取自 `RetrievalSettings.phrase_match_default_slop`
        （默认 0 = 严格相邻同序）。放宽档 1 / 2 由调用方按产品形态决定，
        **不在这里拍一个「智能」的自适应值**。
    """
    if not normalized:
        raise ValueError("extract_filters 收到空文本；空 query 应在路由入口判为 QUERY_EMPTY")

    dropped: list[str] = []
    lowered = normalized.lower()
    strip_ranges: list[tuple[int, int]] = []

    # ---- 引号短语（M3 §4.3）----
    # 短语必须从检索文本里剥离：它已经由 PHRASE_MATCH 表达成硬过滤，
    # 留在文本里会让 BM25 再按普通词项算一遍权重，等于同一个约束计两次。
    phrase_match: PhraseMatchSpec | None = None
    phrases = extract_phrases(normalized)
    if phrases:
        phrase_text, start, end = phrases[0]
        phrase_match = PhraseMatchSpec(phrase=phrase_text, slop=default_slop)
        strip_ranges.append((start, end))
        if len(phrases) > 1:
            # 多短语只取第一条：多条 PHRASE_MATCH 用 and 串起来会让选择率骤降，
            # 而过滤选择率一低 HNSW 就退化成暴力扫描（M3 §7.3）。
            dropped.append(f"extra_phrases({len(phrases) - 1})")

    year_from, year_to, year_spans = _extract_years(lowered, dropped)
    strip_ranges.extend(year_spans)

    category, category_terms, cat_spans = _extract_category(normalized, lowered)
    strip_ranges.extend(cat_spans)

    search_query = strip_spans(normalized, strip_ranges)
    if not search_query:
        # M3 §7.2 第 2 条的兜底：剥离后为空则回退，**绝不发一次空 query 检索**——
        # 空串会让 BM25 Function 产出空 sparse 向量并静默返回随机结果。
        # 纯引号 query（`"chain of thought prompting"`）回退到**短语文本本身**而不是
        # 带引号的原文：引号只是语法标记，把它留在检索文本里对 BM25 没有意义，
        # 而短语词项正是这条 sparse 单路要排序的东西。
        search_query = phrase_match.phrase if phrase_match is not None else normalized
        dropped.append("strip_left_empty_fallback")

    filters = QueryFilters(year_from=year_from, year_to=year_to, primary_category=category)

    if dropped:
        _logger.warning("query_filter_dropped", dropped=dropped, query_len=len(normalized))

    return ExtractionResult(
        search_query=search_query,
        filters=filters,
        phrase_match=phrase_match,
        category_terms=category_terms,
        year_expr=(year_from, year_to) if (year_from or year_to) else None,
        dropped=dropped,
    )
