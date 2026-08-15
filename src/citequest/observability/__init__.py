"""可观测基座：结构化日志、request_id 上下文、span 树、召回明细、badcase 归因。

**一套 ID 贯穿全链路**（M6 §6.5）：`request_id` 同时是 HTTP 响应头、结构化日志
字段、trace id 和用户反馈的关联键。不引第二套 ID——两套 ID 意味着每次排查都要
先做一次关联，而关联表本身会成为新的故障点。

三层分工不重叠（M7 §11.1）：

| 层 | 载体 | 回答的问题 |
|---|---|---|
| **Trace** | `trace.py` span 树（导出到 Langfuse） | 「**这一次请求**发生了什么」 |
| **Metrics** | 结构化日志 → Prometheus / Grafana | 「**整体趋势**怎么样」 |
| **明细落盘** | `retrieval_log.py` 的 JSONL | 「模型到底召回了哪些块」 |

聚合统计发现异常，trace 定位到具体请求，明细落盘还原那一次检索的完整决策过程。

模块划分：

- `context.py`       —— request_id 的生成、注入与读取（ULID，字典序即时间序）
- `logging.py`       —— 结构化日志的统一初始化
- `spans.py`         —— 阶段计时与降级打点（扁平的 `PipelineStage` 上下文管理器）
- `trace.py`         —— span 树 `request → turn → tool_call → llm_call` 与必记字段
- `cost.py`          —— 单价表与三维成本归因（step / tool / model）
- `retrieval_log.py` —— **召回明细落盘**：五类 badcase 的判定信号与 citation 验证依据
- `badcase.py`       —— 五类归因矩阵，判定顺序自上而下写死

本包**不依赖**任何外部可观测 SaaS：trace 后端通过 `set_trace_exporter` 注入，
不注入时 span 只落日志。评测脚本与离线建库不该被一个可选组件绑架。
"""

from citequest.observability.badcase import (
    BadcaseClass,
    BadcaseVerdict,
    classify_badcase,
    conflicting_evidence_present,
)
from citequest.observability.context import (
    REQUEST_ID_PATTERN,
    bind_request_id,
    current_request_id,
    new_request_id,
    reset_request_id,
    sanitize_client_request_id,
)
from citequest.observability.cost import CostAttribution, ModelPrice, PriceTable, attribute_cost
from citequest.observability.logging import configure_logging, get_logger
from citequest.observability.retrieval_log import RetrievalDetail, RetrievalDetailWriter, build_details
from citequest.observability.spans import SpanRecord, record_degrade, set_trace_exporter, span
from citequest.observability.trace import SamplingPolicy, SpanKind, SpanNode, TraceTree

__all__ = [
    "REQUEST_ID_PATTERN",
    "BadcaseClass",
    "BadcaseVerdict",
    "CostAttribution",
    "ModelPrice",
    "PriceTable",
    "RetrievalDetail",
    "RetrievalDetailWriter",
    "SamplingPolicy",
    "SpanKind",
    "SpanNode",
    "SpanRecord",
    "TraceTree",
    "attribute_cost",
    "bind_request_id",
    "build_details",
    "classify_badcase",
    "configure_logging",
    "conflicting_evidence_present",
    "current_request_id",
    "get_logger",
    "new_request_id",
    "record_degrade",
    "reset_request_id",
    "sanitize_client_request_id",
    "set_trace_exporter",
    "span",
]
