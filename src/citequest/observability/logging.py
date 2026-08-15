"""结构化日志初始化（M6 §6.5）。

三条决定写在这里而不是散在调用处：

1. **日志一律结构化**（生产 JSON / 开发彩色），不拼字符串。理由很实际：
   本项目最重要的几个监控项——降级占比、group_count 与请求组数之比、缓存命中率
   ——全部是「按字段聚合」而不是「按关键词 grep」，字符串日志聚合不出来。
2. **`request_id` 由 processor 自动注入**，不靠每个调用点手写。手写必然漏，
   而漏掉的那条恰好是出事时最想看的那条。
3. **不记 query 原文之外的用户输入**，且 `request_id` 进日志前已在
   `observability.context` 里做过字符集校验——这个字段会原样落盘，不校验就是
   一个日志注入面（M6 §1.3）。
"""

from __future__ import annotations

import logging
import sys
from typing import Any, TextIO

import structlog

from citequest.observability.context import current_request_id

__all__ = ["configure_logging", "get_logger"]


def _inject_request_id(
    _logger: Any,
    _method_name: str,
    event_dict: structlog.typing.EventDict,
) -> structlog.typing.EventDict:
    """把当前上下文的 request_id 注入每条日志。

    未绑定时**不写这个键**而不是写 None：离线建库、评测脚本本来就没有请求上下文，
    写一堆 `request_id: null` 只会污染索引。
    """
    request_id = current_request_id()
    if request_id is not None:
        event_dict["request_id"] = request_id
    return event_dict


def configure_logging(
    *,
    level: str = "INFO",
    fmt: str = "json",
    stream: TextIO | None = None,
) -> None:
    """初始化 structlog 与标准库 logging。

    进程内只应调用一次（API 的 lifespan、CLI 的入口、评测 runner 各一次）。
    重复调用是幂等的——structlog.configure 是整体替换而非追加。

    :param level: 日志级别名，取自 `Settings.log_level`
    :param fmt: `json`（容器）或 `console`（本地开发），取自 `Settings.log_format`
    :param stream: 日志输出流，默认 stdout。**MCP server 必须显式传 stderr**——
        stdio 传输下 stdout 是 JSON-RPC 的协议通道，往里写一行日志就会破坏帧结构，
        表现是宿主端报一个与日志内容毫无关系的解析错误。这个参数存在的唯一理由
        就是它，所以不要为了「统一」把它去掉。
    """
    if fmt not in ("json", "console"):
        raise ValueError(f"不支持的日志格式：{fmt}（只有 json / console）")

    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if numeric_level is None:
        raise ValueError(f"未知日志级别：{level}")

    target = stream if stream is not None else sys.stdout

    # 标准库 logging 是底座：uvicorn、pymilvus、httpx 都往它写，
    # 不接管它就会出现「一半日志是 JSON 一半是纯文本」的局面。
    # force=True：basicConfig 默认在已有 handler 时直接静默返回，
    # 于是「第二次调用换了 stream」会毫无提示地不生效——正是 MCP 那条路会踩的坑。
    logging.basicConfig(format="%(message)s", stream=target, level=numeric_level, force=True)

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        _inject_request_id,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if fmt == "json" else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        # 必须把 target 传进去：PrintLoggerFactory 默认写 stdout，
        # 只改上面 basicConfig 的 stream 是不够的——structlog 自己的 logger
        # 根本不走标准库那条路，MCP 下依然会污染协议通道。
        logger_factory=structlog.PrintLoggerFactory(file=target),
        # True 会让第一次 configure 之后的再次配置失效，
        # 而测试里需要按用例切换格式，所以不缓存。
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """取一个绑定了模块名的 logger。

    统一走这里而不是各处 `structlog.get_logger()`，是为了保证 `logger` 字段
    永远是模块路径——按模块过滤日志是排查时的第一个动作。
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
