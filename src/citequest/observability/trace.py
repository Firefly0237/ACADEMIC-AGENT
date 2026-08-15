"""span 树：`request → turn → tool_call → llm_call`（M7 §11.2、§11.3）。

```
request                                    # 一次 HTTP / MCP 调用
└── turn                                   # 一轮对话（RAG 路一轮 = 一次问答）
    ├── route                              # 查询理解与路由
    ├── tool_call                          # 一次工具调用（检索类工具）
    │   ├── search        (hybrid / dense_only / sparse_only)
    │   ├── rerank
    │   └── aggregate     (chunk → paper)
    ├── llm_call                           # 一次模型调用
    │   ├── hyde / generate / ground
    └── postprocess                        # 引用校验、降级标记、SSE 封装
```

**四层的分层依据是「谁能失败、谁能被重试、谁单独计价」**——每一层都是一个独立的
失败域与成本归因单元。不是按代码结构分的，所以不能改成「按模块一层」。

**与编排层共用同一套 `request_id`，不引第二套 ID**（`observability.context`）。
这不只是省事：编排图的节点与 span 树的 span 是**同一个模型**，意味着「执行到哪
一步」和「观测到哪一步」永远对得上，不需要在两套 ID 之间做映射。

与 `spans.py` 的分工：那边是**阶段计时与降级打点**（一个扁平的 `PipelineStage`
上下文管理器，业务代码里到处在用）；这里是**树结构与必记字段**（父子关系、
token 三分、成本、版本号、采样）。两者共用 `PipelineStage` 字面值，所以一条
`SpanNode` 既能挂在树上，也能对上 Prometheus 的直方图标签。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from citequest.core.enums import DegradeReason, PipelineStage
from citequest.observability.context import current_request_id
from citequest.observability.logging import get_logger

__all__ = ["SamplingPolicy", "SpanKind", "SpanNode", "TraceTree"]

_logger = get_logger(__name__)


class SpanKind(StrEnum):
    """span 树的四层。**层级是固定的，不允许运行时新增一层。**

    新增一层意味着成本归因的维度变了，而成本归因表（§11.4）是按这四层写死的
    ——多出来的那层会成为一个没人统计的黑洞。
    """

    REQUEST = "request"
    TURN = "turn"
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"


@dataclass(slots=True)
class SpanNode:
    """一个 span 的完整记录。**九个必记字段（§11.3）一个都不能省。**

    每个字段都对应一个具体的排查场景，不是「能记的都记」：

    - `model_id` 精确到版本（含 TEI 镜像 tag、reranker 版本）→ serving 维归因；
    - `prompt/completion/cached_tokens` **三者分开** → 缓存命中率与成本账；
    - `prompt_version` → prompt 改了但指标没记版本，两次评测的差异就无法归因；
    - `index_version` → 与缓存 key 共用同一个值，换版本即一键全量失效；
    - `error_class` 区分**参数/业务错误**与**基础设施错误** → 两类失败处置不同。
    """

    kind: SpanKind
    name: str
    span_id: str
    parent_span_id: str | None
    request_id: str | None
    started_at: float
    stage: PipelineStage | None = None
    duration_ms: float = 0.0
    model_id: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_estimate: float = 0.0
    prompt_version: str | None = None
    index_version: str | None = None
    status: str = "ok"
    error_class: str | None = None
    degraded: DegradeReason | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """计价口径的总量。**cached_tokens 不计入**——它单独计价（通常更便宜），
        混进来会让成本估算系统性偏高，而偏高的成本账没人会去查。"""
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class SamplingPolicy:
    """采样策略（§11.5）。

    **错误请求与降级请求一律 100% 采样**——它们才是要查的那部分。按固定比例采样
    所有请求的后果是：出问题的那 0.5% 恰好没被采到，而这正是最需要 trace 的时候。
    """

    base_rate: float = 1.0
    always_sample_errors: bool = True
    always_sample_degraded: bool = True

    def should_export(self, root: SpanNode, *, has_degrade: bool) -> bool:
        if self.always_sample_errors and root.status != "ok":
            return True
        if self.always_sample_degraded and has_degrade:
            return True
        if self.base_rate >= 1.0:
            return True
        # 用 request_id 的哈希而不是随机数：同一个请求在多个进程里被判定的结果
        # 必须一致，否则一条 trace 会被采到一半。
        digest = hash(root.request_id or root.span_id) & 0xFFFF
        return digest / 0xFFFF < self.base_rate


class TraceTree:
    """一次请求内的 span 树。**每请求一棵，不是进程级单例。**

    它刻意不做导出：导出后端（Langfuse）由 `spans.set_trace_exporter` 注入，
    这里只负责结构与统计。分开的理由与 `spans.py` 相同——**没有后端也必须能跑**，
    评测脚本与离线建库不该被一个可观测组件绑架。
    """

    def __init__(self, *, index_version: str | None = None, policy: SamplingPolicy | None = None) -> None:
        self._nodes: list[SpanNode] = []
        self._stack: list[SpanNode] = []
        self._index_version = index_version
        self._policy = policy or SamplingPolicy()
        self._counter = 0

    @property
    def nodes(self) -> list[SpanNode]:
        """按结束顺序排列的全部 span。父节点排在子节点之后（它最后才结束）。"""
        return self._nodes

    @property
    def root(self) -> SpanNode | None:
        return next((n for n in self._nodes if n.parent_span_id is None), None)

    @contextmanager
    def span(
        self,
        kind: SpanKind,
        name: str,
        *,
        stage: PipelineStage | None = None,
        model_id: str | None = None,
        prompt_version: str | None = None,
        **attrs: Any,
    ) -> Iterator[SpanNode]:
        """开一个 span，自动挂到当前栈顶作为父节点。

        用法：

            with tree.span(SpanKind.REQUEST, "POST /v1/answer") as req:
                with tree.span(SpanKind.LLM_CALL, "generate", model_id="qwen2.5-7b") as call:
                    call.completion_tokens = 512

        计时用 `perf_counter`（与 `spans.py` 一致）：`time.time()` 会被 NTP 校时
        影响，在毫秒量级的 span 上足以量出负数耗时。

        **异常只记录不吞掉**：吞异常等于把故障变成静默的错误结果，而本项目的降级
        语义要求降级是显式的（由调用方捕获后填 `degraded`）。
        """
        self._counter += 1
        node = SpanNode(
            kind=kind,
            name=name,
            span_id=f"{kind.value}-{self._counter}",
            parent_span_id=self._stack[-1].span_id if self._stack else None,
            request_id=current_request_id(),
            started_at=time.time(),
            stage=stage,
            model_id=model_id,
            prompt_version=prompt_version,
            index_version=self._index_version,
            attrs=dict(attrs),
        )
        self._stack.append(node)
        started = time.perf_counter()
        try:
            yield node
        except Exception as exc:
            node.status = "error"
            node.error_class = _classify_error(exc)
            raise
        finally:
            node.duration_ms = (time.perf_counter() - started) * 1000.0
            self._stack.pop()
            self._nodes.append(node)
            _logger.debug(
                "span_closed",
                kind=node.kind.value,
                name=node.name,
                duration_ms=round(node.duration_ms, 2),
                status=node.status,
            )

    def by_kind(self, kind: SpanKind) -> list[SpanNode]:
        return [n for n in self._nodes if n.kind is kind]

    def should_export(self) -> bool:
        """这棵树要不要送到 trace 后端。错误与降级恒为 True。"""
        root = self.root
        if root is None:
            return False
        return self._policy.should_export(root, has_degrade=any(n.degraded is not None for n in self._nodes))


def _classify_error(exc: Exception) -> str:
    """把异常分成**参数/业务错误**与**基础设施错误**两类（§11.3 `error_class`）。

    两类的处置完全不同：参数错误要改调用方，基础设施错误要看依赖服务——把它们
    混在一个 `error_rate` 里，告警一响就得先人工分拣一遍。

    判据用 `CiteQuestError` 这条继承边而不是逐个异常类型枚举：本项目自定义的
    异常都是业务语义的（`core.errors`），其余（连接、超时、序列化）一律算基础设施。
    """
    from citequest.core.errors import CiteQuestError

    return "business" if isinstance(exc, CiteQuestError) else "infrastructure"
