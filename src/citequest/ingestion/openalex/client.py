"""摘要层获取：OpenAlex API 的 filter 串、游标分页、限速与断点续跑（M1 §3.3）。

filter 串是这一层唯一**必须一字不差可复现**的东西——它决定了 219,676 这个实测数
能不能被别人重跑出来。三个条件（M1 §3.3 的 G1/G2/G3）：

| # | 条件 | OpenAlex 表达 |
|---|---|---|
| G1 | 领域 = 计算机科学 | `primary_topic.field.id:fields/17` |
| G2 | primary source 为 arXiv | `primary_location.source.id:S4306400194` |
| G3 | 出版日期 2023-01-01 ~ 2025-12-31 | `from_publication_date` / `to_publication_date` |

G2 的作用不只是「同源」：它保证摘要层与全文层可以按 arXiv ID 对齐，且**两层在
时间上无缝不重叠**（全文层到 2022-12，摘要层从 2023-01 起），于是「一篇论文同时
出现在两层」这个去重难题在设计上就不存在。

**分页只用 cursor 不用 page**：OpenAlex 的 `page` 分页有 10,000 条的深度上限，
而这里要拉 22 万条。cursor 没有深度限制，而且它天然就是断点续跑的游标。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import httpx
from pydantic import BaseModel, Field

from citequest.config.corpus import CorpusSettings
from citequest.observability.logging import get_logger

__all__ = [
    "ARXIV_SOURCE_ID",
    "OPENALEX_WORKS_URL",
    "OpenAlexClient",
    "build_works_filter",
]

_logger = get_logger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
#: arXiv 在 OpenAlex 里的 Source ID。写成常量并注明出处，是因为它是 G2 的全部内容——
#: 埋在一个 f-string 里的话，别人复现时根本不知道该去哪儿核对这个值。
ARXIV_SOURCE_ID = "S4306400194"
#: OpenAlex 建议在 UA 或 mailto 参数里留邮箱换取 polite pool（更稳定的限速）。
_DEFAULT_USER_AGENT = "citequest/0.1 (https://github.com/citequest/citequest)"


def build_works_filter(settings: CorpusSettings, *, year: int | None = None) -> str:
    """拼 G1+G2+G3 的 filter 串。

    :param year: 给定则把日期窗口收到这一年——**按年分片拉取**是有意的：
        22 万条如果一个游标拉到底，中途失败就要从头再来；分年之后失败只影响一年，
        而且每年的实测篇数（67,007 / 75,538 / 77,131）可以逐年核对，
        对不上立刻知道是哪一年的 filter 出了问题。
    """
    date_from = f"{year}-01-01" if year else f"{settings.abstract_year_from}-01-01"
    date_to = f"{year}-12-31" if year else f"{settings.abstract_year_to}-12-31"
    clauses = [
        f"primary_topic.field.id:{settings.openalex_field_id}",
        f"from_publication_date:{date_from}",
        f"to_publication_date:{date_to}",
    ]
    if settings.openalex_require_arxiv_primary_source:
        clauses.append(f"primary_location.source.id:{ARXIV_SOURCE_ID}")
    # OpenAlex 的多条件是逗号连接（AND 语义），顺序不影响结果但影响可读性，
    # 所以固定成「领域 → 时间 → 来源」这个与文档表格一致的顺序。
    return ",".join(clauses)


class OpenAlexPage(BaseModel):
    """一页响应。`next_cursor` 为 None 即遍历结束。"""

    results: list[dict[str, Any]] = Field(default_factory=list)
    next_cursor: str | None = None
    total_count: int | None = None


class OpenAlexClient:
    """带限速与重试的游标分页客户端。

    三条纪律：

    1. **限速是主动的，不是被动等 429**。OpenAlex 是免费公共服务，靠撞限流来找
       速率上限既不礼貌也不稳定——被限流之后的退避会让整体吞吐比主动限速更低。
    2. **重试只对幂等的 GET 做，且只对可重试状态码做**。4xx（filter 写错）重试
       一万次也不会变对，只会把一个立刻可见的错误拖成一个缓慢的错误。
    3. **游标必须能被外部拿走**（`iter_works` 每页都回吐 cursor），否则断点续跑
       就只能靠「已处理条数」，而条数在结果集变化时会指向完全不同的位置。
    """

    def __init__(
        self,
        *,
        mailto: str | None = None,
        per_page: int = 200,
        min_interval_s: float = 0.12,
        max_retries: int = 3,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not 1 <= per_page <= 200:
            raise ValueError(f"per_page 必须在 [1,200]（OpenAlex 官方上限 200），收到 {per_page}")
        self._mailto = mailto
        self._per_page = per_page
        self._min_interval_s = min_interval_s
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._client = client or httpx.Client(headers={"User-Agent": _DEFAULT_USER_AGENT}, timeout=timeout_s)
        self._last_call_at = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAlexClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _throttle(self) -> None:
        """按最小调用间隔主动限速（上面第 1 条纪律）。"""
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)
        self._last_call_at = time.monotonic()

    def fetch_page(self, filter_expr: str, cursor: str = "*") -> OpenAlexPage:
        """拉一页。指数退避重试，只对 429/5xx 与网络错误重试。"""
        params: dict[str, str | int] = {
            "filter": filter_expr,
            "per-page": self._per_page,
            "cursor": cursor,
        }
        if self._mailto:
            # polite pool：官方明确说带 mailto 的请求走更宽松、更稳定的池子。
            params["mailto"] = self._mailto

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                response = self._client.get(OPENALEX_WORKS_URL, params=params, timeout=self._timeout_s)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"OpenAlex 返回 {response.status_code}", request=response.request, response=response
                    )
                response.raise_for_status()
                payload = response.json()
                meta = payload.get("meta") or {}
                return OpenAlexPage(
                    results=payload.get("results") or [],
                    next_cursor=meta.get("next_cursor"),
                    total_count=meta.get("count"),
                )
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status < 500 and status != 429:
                    # 4xx 不重试：filter 写错重试一万次也不会变对。
                    raise
                last_error = exc
                backoff = 2**attempt
                _logger.warning("openalex_retry", attempt=attempt, backoff_s=backoff, error=str(exc))
                time.sleep(backoff)

        raise RuntimeError(f"OpenAlex 请求重试 {self._max_retries} 次仍失败：{last_error}")

    def iter_works(self, filter_expr: str, *, start_cursor: str = "*") -> Iterator[tuple[dict[str, Any], str | None]]:
        """游标遍历，逐条产出 `(work, 本页之后的 next_cursor)`。

        把 cursor 与记录一起吐出来，调用方就能在**任意一条**之后写 checkpoint，
        而不是只能在整页处理完之后写。这一点在单页 200 条、单条处理较慢时很值钱。

        :param start_cursor: 续跑时传上次 checkpoint 里的 `last_cursor`。
        """
        cursor: str | None = start_cursor
        while cursor:
            page = self.fetch_page(filter_expr, cursor)
            if not page.results:
                return
            for work in page.results:
                yield work, page.next_cursor
            if page.next_cursor == cursor:
                # 防御一种真实发生过的形态：服务端返回的 next_cursor 与当前相同，
                # 于是遍历永远不结束。宁可少拉也不能空转。
                _logger.error("openalex_cursor_stuck", cursor=cursor)
                return
            cursor = page.next_cursor
