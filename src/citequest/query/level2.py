"""Level 2 小 LLM 兜底分类器（M3 §2.3）。

**一次调用、一个 schema、一次校验，输出不合 schema 就整体走默认值，不做部分采纳。**
「部分采纳」听起来更宽容，实际后果是拿到一个内部矛盾的 plan（判成 SEARCH_ONLY
却又标 needs_fulltext=true），而这种矛盾在下游表现为「过滤生效了但没人生成回答」。

模型规模 1.5B–3B（`GenerationSettings.router_model_id`），温度 0.0
（`router_temperature`）——路由必须可复现，同一条 query 两次给出不同 route 会让
缓存、监控和回归测试同时失效。

**这一层不设自己的超时常数**：deadline 由调用方按 `TimeoutSettings.router_ms`（800ms）
算好传进来。两处各拍一个超时值时，靠后的那个永远是死代码。
"""

from __future__ import annotations

import json

from citequest.config.rag import GenerationSettings
from citequest.core.enums import RewriteStrategy, RouteType
from citequest.core.query_plan import QuerySignals
from citequest.llm.chat import ChatClient, Deadline
from citequest.observability.logging import get_logger
from citequest.query.intent import Level2Output
from citequest.rag.prompts.router import build_router_messages

__all__ = ["LlmLevel2Classifier"]

_logger = get_logger(__name__)

#: 类目字段的形态校验。模型很容易输出 "NLP" 或 "cs.cl"——前者不是类目、
#: 后者大小写不对，两者都会让等值过滤永远不命中，且**不报错**。
_CATEGORY_PREFIX = "cs."
_YEAR_MIN, _YEAR_MAX = 1900, 2100


class LlmLevel2Classifier:
    """把一次 chat 调用的 JSON 输出解析成 `Level2Output`。

    解析失败一律**抛异常**而不是返回默认值：默认值的构造在
    `QueryRouter._run_level2` 里，那里同时负责打降级点与关掉缓存。
    在两个地方各构造一次默认值，迟早会出现「降级了但没打点」。
    """

    def __init__(self, chat: ChatClient, settings: GenerationSettings) -> None:
        self._chat = chat
        self._settings = settings

    async def classify(
        self,
        normalized: str,
        signals: QuerySignals,
        *,
        deadline: Deadline | None = None,
    ) -> Level2Output:
        """一次调用产出 route / needs_fulltext / filters / rewrite_strategy 四个字段。

        :param signals: 规则层已抽好的信号。**这里不再解析原文**（M3 §1.2：
            同一段文本只解析一次），它只用于日志归因——「Level 1 明明有疑问词
            却没命中」这类分布异常要能在日志里被看见。
        """
        raw = await self._chat.complete(
            build_router_messages(normalized),
            model=self._settings.router_model_id,
            temperature=self._settings.router_temperature,
            max_tokens=128,
            deadline=deadline,
        )
        output = _parse(raw)
        _logger.info(
            "level2_classified",
            route=output.route.value,
            needs_fulltext=output.needs_fulltext,
            token_count=signals.token_count,
        )
        return output


def _parse(raw: str) -> Level2Output:
    """严格解析。任何一处不合法都整体拒绝。

    只截取第一个 `{...}` 区间：小模型经常在 JSON 前后附一句解释，
    而「模型不按纯 JSON 输出」是一个必然出现、且不值得用重试去对抗的形态。
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Level 2 输出里没有 JSON 对象")
    payload = json.loads(raw[start : end + 1])

    route = RouteType(str(payload["route"]).strip().upper())
    needs_fulltext = bool(payload.get("needs_fulltext", False))

    strategy_raw = str(payload.get("rewrite_strategy", "direct")).strip().lower()
    strategy = RewriteStrategy(strategy_raw)

    year_from = _coerce_year(payload.get("year_from"))
    year_to = _coerce_year(payload.get("year_to"))
    if year_from is not None and year_to is not None and year_from > year_to:
        # 与正则抽取同一条纪律（M3 §7.2 第 3 条）：丢弃该条件，不静默交换。
        _logger.warning("level2_year_bounds_reversed", year_from=year_from, year_to=year_to)
        year_from = year_to = None

    category = payload.get("primary_category")
    if category is not None:
        category = str(category).strip()
        if not category.startswith(_CATEGORY_PREFIX) or len(category) != 5:
            _logger.warning("level2_category_dropped", value=category)
            category = None
        else:
            category = f"{_CATEGORY_PREFIX}{category[3:].upper()}"

    return Level2Output(
        route=route,
        needs_fulltext=needs_fulltext,
        year_from=year_from,
        year_to=year_to,
        primary_category=category,
        rewrite_strategy=strategy,
    )


def _coerce_year(value: object) -> int | None:
    """把模型给的年份转成 int，越界或非数字一律丢弃。

    不抛异常是刻意的：年份是**可选**条件，丢掉它只是少一个过滤；
    而 route 解析失败会让整条链路没有落点，那才该整体拒绝。
    """
    if value is None:
        return None
    try:
        year = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    return year if _YEAR_MIN <= year <= _YEAR_MAX else None
