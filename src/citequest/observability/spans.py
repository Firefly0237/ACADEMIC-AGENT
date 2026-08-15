"""span 记录：一次请求内各阶段的耗时、降级与关键属性（M6 §6.5）。

span 树固定为 `route → search → rerank → hyde → generate → ground → postprocess`，
节点名取自 `core.enums.PipelineStage`——**三处共用同一套字面值**（span 名、
Prometheus 直方图标签、结构化日志的 stage 字段）。三处各写各的字符串，聚合时就对不上。

本模块刻意做得很薄：它只负责「量出来 + 记下来」。真正的 trace 后端（Langfuse）
通过 `set_trace_exporter` 注入，缺省时 span 只落结构化日志——**没有后端也必须能跑**，
因为评测脚本与离线建库不该被一个可观测组件绑架。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from citequest.core.enums import DegradeReason, PipelineStage
from citequest.observability.context import current_request_id
from citequest.observability.logging import get_logger

__all__ = ["SpanRecord", "record_degrade", "set_trace_exporter", "span"]

_log = get_logger(__name__)

#: trace 后端导出函数。None 表示只落日志。
_exporter: Callable[[SpanRecord], None] | None = None


@dataclass(slots=True)
class SpanRecord:
    """一个 span 的完整记录。

    `degraded` 放在 span 上而不是只放在响应里，是因为**降级必须能按阶段归因**：
    「本次请求降级了」没有排查价值，「rerank 阶段因排队超时降级」才有。
    """

    stage: PipelineStage
    request_id: str | None
    started_at: float
    duration_ms: float = 0.0
    ok: bool = True
    degraded: DegradeReason | None = None
    error_type: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


def set_trace_exporter(exporter: Callable[[SpanRecord], None] | None) -> None:
    """注入 trace 后端导出函数（Langfuse 等）。

    传 None 关闭导出。用注入而不是在本模块直接 import Langfuse SDK，
    是为了让 core 到 observability 这条链路不产生对外部 SaaS 的编译期依赖——
    单元测试与离线脚本不该因为少装一个可选依赖就 import 失败。
    """
    # 进程级单例，用模块变量是最简形态——为它包一个类只会多一层没有状态的间接。
    global _exporter
    _exporter = exporter


@contextmanager
def span(stage: PipelineStage, **attrs: Any) -> Iterator[SpanRecord]:
    """记录一个阶段的 span。

    用法：

        with span(PipelineStage.SEARCH, group_limit=16, ef=192) as sp:
            result = do_search()
            sp.attrs["group_count"] = result.group_count

    时间基准用 `perf_counter` 而不是 `time.time()`：后者会被 NTP 校时影响，
    在毫秒量级的阶段上足以量出负数耗时。

    异常**只记录不吞掉**——吞异常等于把故障变成静默的错误结果，
    而本项目的降级语义要求降级是显式的（由调用方捕获后填 `sp.degraded`）。
    """
    record = SpanRecord(
        stage=stage,
        request_id=current_request_id(),
        started_at=time.time(),
        attrs=dict(attrs),
    )
    started = time.perf_counter()
    try:
        yield record
    except Exception as exc:
        record.ok = False
        record.error_type = type(exc).__name__
        raise
    finally:
        record.duration_ms = (time.perf_counter() - started) * 1000.0
        _emit(record)


def _emit(record: SpanRecord) -> None:
    """落日志 + 导出到 trace 后端。

    导出失败**绝不能影响业务**：可观测组件挂掉时正确的行为是丢掉这条 trace，
    而不是把一个成功的请求变成 500。
    """
    _log.info(
        "span",
        stage=record.stage.value,
        duration_ms=round(record.duration_ms, 2),
        ok=record.ok,
        degraded=record.degraded.value if record.degraded else None,
        error_type=record.error_type,
        **record.attrs,
    )
    if _exporter is None:
        return
    try:
        _exporter(record)
    except Exception as exc:
        # 这里**故意捕获 Exception 基类**：可观测层的异常必须就地终结。
        # 放它往上冒，等于让一个 trace 后端的抖动去决定业务请求的成败。
        _log.warning("trace_export_failed", error_type=type(exc).__name__)


def record_degrade(record: SpanRecord, reason: DegradeReason) -> None:
    """在 span 上标记一次降级。

    单列一个函数而不是让调用方直接赋值，是因为降级点分散在十几处，
    集中一个入口后才有地方挂 Prometheus 计数器——
    `degraded` 非空的请求占比是三个「静默失败」监控项之一（M6 §6.5）：
    系统长期跑在降级状态时，所有健康检查都是绿的。
    """
    record.degraded = reason
    # TODO(接真实实现): 这里累加 Prometheus Counter
    #   citequest_degrade_total{stage=..., reason=...}
    #   现在只落日志，指标端点由 api 层接上 prometheus_client 后补。
