"""工具注册表：**数据结构而非 if-else 链**（M5 §12.2、决策记录 K10）。

```
name → {schema, handler, token_budget, timeout, retry, quota}
```

**加一个工具是加一条数据，不是加一段控制流。** 这是「平台化」在这个规模下唯一
诚实的体现——不会说做了一个平台，但注册表是数据结构这件事是真的，而且可验证：
`ToolRegistry` 上没有任何一处按工具名分支的代码。

两类失败语义（M5 §12.2，与 MCP 规范的两类错误划分同构）：

| 类别 | 情形 | 处理 | 为什么 |
|---|---|---|---|
| 参数 / 业务错误 | schema 不合法、paper_id 不存在、年份区间反了 | 返回结构化错误对象**喂回模型**自我修正；同工具同 error_code 连续 2 次 → 本轮拉黑 | 这类错误模型改得了 |
| 基础设施错误 | 检索超时、重排不可用、元数据库不可达 | **不喂回模型**，向上抛给图层：重试 1 次后熔断 | 这类错误模型改不了，喂回去只会让它换个说法再试一次 |
| 空结果 | 命中 0 条 | **不是错误**，返回 `{ok: true, results: [], note: "no match"}` | 把空结果当错误，模型会去「修复」一个没坏的东西 |

**返回裁剪**是这一层第二重要的东西（M5 §11.2）：工具原始返回 ≤48 chunk ≈ 21,600
token，进生成上下文 8 条 ≈ 3,600 token，而进 planner 的观察摘要上限 **800 token/轮**。
两次收敛（6× 与 4.5×）才是 token 预算能收敛的根本原因——工具描述省的是常驻，
观察层裁剪省的是**随轮数线性增长**的那部分，后者更要命，因为它会累积。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from citequest.core.errors import UpstreamTimeoutError
from citequest.observability.logging import get_logger

__all__ = [
    "ToolCallLog",
    "ToolErrorClass",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "clip_observation",
]

_logger = get_logger(__name__)

#: 观察层裁剪的粗估系数，与 `rag.context.ContextBudget.chars_per_token` 同口径。
#: 这里刻意不引真分词器：裁剪是一道闸不是一次计费，差 5% 不改变任何决策。
_CHARS_PER_TOKEN = 4.0


class ToolErrorClass(StrEnum):
    """失败的两类 + 一个「不是失败」。分类决定的是**错误往哪走**，不是措辞。"""

    #: 参数 / 业务错误 → 结构化错误对象喂回模型
    PARAM = "param"
    #: 基础设施错误 → 不喂回模型，抛给图层重试后熔断
    INFRA = "infra"


@dataclass(slots=True)
class ToolResult:
    """一次工具调用的产物。

    `note` 存在的唯一理由是把「空结果」和「出错」在结构上分开：空结果的
    `ok` 仍然是 True，只是带一句 `no match`。这两件事给模型的下一步动作完全不同。
    """

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_class: ToolErrorClass | None = None
    #: 给模型看的修正提示。只在 PARAM 类错误上有意义——INFRA 类不喂回模型，
    #: 写一句提示给它也没有用武之地。
    hint: str | None = None
    note: str | None = None
    truncated: bool = False
    observation_tokens: int = 0

    @classmethod
    def empty(cls, note: str = "no match") -> ToolResult:
        """空结果的标准形态。**它是成功不是失败。**"""
        return cls(ok=True, data={"results": []}, note=note)

    @classmethod
    def param_error(cls, code: str, hint: str) -> ToolResult:
        """参数 / 业务错误：结构化对象，喂回模型让它自己改。"""
        return cls(ok=False, error_code=code, error_class=ToolErrorClass.PARAM, hint=hint)

    def feedable_to_model(self) -> bool:
        """这条结果能不能进模型上下文。

        INFRA 类**一律不能**：模型对基础设施故障无能为力，喂回去只会浪费一轮
        预算，还制造「模型很努力」的假象。
        """
        return self.error_class is not ToolErrorClass.INFRA


#: 工具处理函数签名。入参是已按 schema 校验过的 dict，出参是 `ToolResult`。
#: 用统一签名而不是各自定制，是注册表能成为数据结构的前提。
ToolHandler = Callable[[dict[str, Any]], Awaitable[ToolResult]]


@dataclass(slots=True, frozen=True)
class ToolSpec:
    """一个工具的完整声明。字段与 K10 那条 `name → {...}` 逐项对应。"""

    name: str
    #: 一句话触发描述。渐进式披露下它是**索引行**（≈25 token），也是单点故障：
    #: 描述写不准 → 该命中的不命中，模型不知道自己有这个能力（M5 §11.1）。
    description: str
    #: JSON Schema。MCP 的 `inputSchema` 直接用它，不再写第二份。
    input_schema: dict[str, Any]
    handler: ToolHandler
    #: 返回裁剪上限。`search_chunks` 回 chunk 原文，是最贵的工具，所以它的预算最紧。
    token_budget: int
    timeout_ms: int
    #: 只读幂等是这一层的前提：同一个调用做十次结果一样，所以重试是安全的。
    retry: int = 0
    #: 单会话调用次数上限。MCP 走 stdio、没有 IP，令牌桶的 key 维度不成立，
    #: 配额只能换成「单会话次数 + 按 token 计的预算」（M6 §4.5）。
    quota_per_session: int = 30
    readonly: bool = True
    idempotent: bool = True


@dataclass(slots=True)
class ToolCallLog:
    """一次调用的轨迹记录（K10：span 埋点与轨迹 JSONL 落盘）。

    `error_code` 单独留字段是为了支撑「同工具同 error_code 连续 2 次 → 本轮拉黑」
    这条规则：1 次太急（模型常常第二次就改对了），3 次浪费 2 轮预算。
    """

    tool: str
    ok: bool
    took_ms: int
    error_code: str | None = None
    observation_tokens: int = 0
    truncated: bool = False


def clip_observation(text: str, token_budget: int) -> tuple[str, bool, int]:
    """把工具返回裁剪到观察层预算内（M5 §11.2）。

    返回 `(裁剪后文本, 是否裁过, 估算 token)`。**必须把「裁过」这件事带出来**——
    裁剪直接影响模型看到的信息量，而它在结果里的表现只是「内容少了点」，
    不打点就没人知道发生过。

    截断策略是**尾部截断 + 显式标注**，不是中间省略：模型对「后面还有」这件事
    需要一个可见的信号，否则它会把不完整的列表当成完整的。
    """
    budget_chars = int(token_budget * _CHARS_PER_TOKEN)
    estimated = int(len(text) / _CHARS_PER_TOKEN) + 1
    if len(text) <= budget_chars:
        return text, False, estimated
    marker = "\n…（已按观察层预算截断，如需更多请收窄条件后重查）"
    clipped = text[: max(budget_chars - len(marker), 0)] + marker
    return clipped, True, token_budget


class ToolRegistry:
    """`name → ToolSpec` 的字典，外加一次调用的统一执行路径。

    **执行路径里没有任何一处按工具名分支**——这正是「注册表是数据结构」这句话
    可以被验证的地方。超时、重试、裁剪、配额四件事对所有工具是同一段代码，
    差异全部落在 `ToolSpec` 的字段值上。
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._calls: dict[str, int] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        """注册一个工具。重名直接报错——静默覆盖会让两份实现只有一份生效。"""
        if spec.name in self._specs:
            raise ValueError(f"工具重名：{spec.name}")
        if not spec.readonly or not spec.idempotent:
            # 只读幂等是这一层的设计前提：正因为都是只读的，MCP 那条路才不需要
            # 审批断点，重试才是安全的。放进来一个有副作用的工具会同时推翻这两条。
            raise ValueError(f"工具 {spec.name} 必须是只读幂等的（决策记录 D18）")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self) -> list[ToolSpec]:
        """按名字排序返回，保证 MCP `list_tools` 的顺序稳定。

        顺序稳定不是洁癖：工具列表每次都换顺序会让宿主端的缓存与 diff 全部失效。
        """
        return [self._specs[name] for name in sorted(self._specs)]

    def index_lines(self) -> list[str]:
        """渐进式披露的索引行：名字 + 一句话描述（M5 §11.1）。

        本系统只有 5 个原子工具，**四条收敛路径一条都不需要**——工具集小到
        不构成那个问题。这个方法留在这里是因为它同时是 MCP 那侧的描述来源，
        不是因为我们在做渐进式披露。
        """
        return [f"{spec.name}: {spec.description}" for spec in self.specs()]

    def quota_left(self, name: str) -> int:
        spec = self._specs[name]
        return max(spec.quota_per_session - self._calls.get(name, 0), 0)

    async def call(self, name: str, arguments: dict[str, Any]) -> tuple[ToolResult, ToolCallLog]:
        """统一执行：配额 → 超时 → handler → 裁剪 → 记账。

        **异常在这里被分成两类**（K10）：`ValueError` 一律当参数/业务错误喂回模型；
        其余异常当基础设施错误**向上抛**——由图层决定重试还是熔断。
        把它们混成一个 `except Exception` 是最常见的错误，代价是既重试了不该重试的，
        又没重试该重试的。
        """
        spec = self._specs.get(name)
        if spec is None:
            # 未知工具是参数错误：模型改得了（换一个存在的工具名）。
            result = ToolResult.param_error("UNKNOWN_TOOL", f"可用工具：{', '.join(self.names())}")
            return result, ToolCallLog(tool=name, ok=False, took_ms=0, error_code="UNKNOWN_TOOL")

        if self.quota_left(name) <= 0:
            result = ToolResult.param_error("QUOTA_EXCEEDED", f"{name} 本会话调用次数已达上限 {spec.quota_per_session}")
            return result, ToolCallLog(tool=name, ok=False, took_ms=0, error_code="QUOTA_EXCEEDED")

        self._calls[name] = self._calls.get(name, 0) + 1
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(spec.handler(arguments), timeout=spec.timeout_ms / 1000)
        except ValueError as exc:
            # 参数 / 业务错误：结构化对象喂回模型，让它自己改。
            result = ToolResult.param_error("INVALID_ARGUMENT", str(exc))
        except TimeoutError as exc:
            # 超时是基础设施错误：**不喂回模型**，翻译成领域异常抛给图层，
            # 由图层决定重试 1 次还是熔断。这里不吞掉它，也不返回一个假的空结果——
            # 假空结果会让调用方以为「查过了、没有」，而事实是「没查成」。
            raise UpstreamTimeoutError(f"工具 {name} 超时（预算 {spec.timeout_ms}ms）") from exc
        took_ms = int((time.perf_counter() - started) * 1000)

        if result.ok and result.data:
            payload = json.dumps(result.data, ensure_ascii=False, separators=(",", ":"))
            _, truncated, tokens = clip_observation(payload, spec.token_budget)
            result.truncated = truncated
            result.observation_tokens = tokens

        log = ToolCallLog(
            tool=name,
            ok=result.ok,
            took_ms=took_ms,
            error_code=result.error_code,
            observation_tokens=result.observation_tokens,
            truncated=result.truncated,
        )
        _logger.info(
            "tool_called",
            tool=name,
            ok=result.ok,
            took_ms=took_ms,
            error_code=result.error_code,
            observation_tokens=result.observation_tokens,
            truncated=result.truncated,
        )
        return result, log
