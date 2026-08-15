"""MCP 工具的 adapter：**复用 `orchestration.tool_specs` 的定义，不重写一份**。

这一层做的事只有两件：把 `ToolSpec` 翻译成 MCP 的工具描述结构，把 `ToolResult`
翻译成 MCP 的返回结构。**没有一行检索逻辑。**

叙事必须是：「我把检索能力做成了**协议无关的 service 层**，MCP 是其中一个 adapter，
HTTP API 是另一个」，而不是「我做了一个 MCP server」。区别不是措辞，是可验证的
结构差异——两个 adapter 共用同一套 service 层，不是两套业务逻辑。如果是两套，
第二套一定会在某次改动后与第一套不一致，而这个不一致只有在用户报障时才会被发现。

**配额换了维度**（M6 §4.5）：MCP 走 stdio、是本地进程、没有 IP 概念，令牌桶的
key 维度不成立。所以配额换成两条——单会话的工具调用次数上限（`ToolSpec.quota_per_session`），
以及**按 token 而不是按次数计的预算**（`ToolSpec.token_budget`：一个 `search_chunks`
返回 48 个 chunk 的文本，token 成本远高于一次 `get_paper_detail`）。

5 个工具全部只读、幂等、无副作用，所以**不需要审批断点**——这正是把「只读」写进
工具设计约束的收益。
"""

from __future__ import annotations

import json
from typing import Any

from citequest.observability.logging import get_logger
from citequest.orchestration.tools import ToolRegistry, ToolResult, ToolSpec

__all__ = ["describe_tools", "index_lines", "quota_snapshot", "render_result", "run_tool", "spec_of"]

_logger = get_logger(__name__)


def describe_tools(registry: ToolRegistry) -> list[dict[str, Any]]:
    """`list_tools` 的载荷：名字 + 一句话描述 + JSON Schema。

    `inputSchema` 直接用 `ToolSpec.input_schema`，**不再写第二份**：schema 写两份
    必然漂移，而漂移的表现是宿主端按一份 schema 构造参数、服务端按另一份校验，
    然后模型收到一个它无法理解的参数错误。

    描述是**单点故障**：写不准 → 该命中的不命中，模型不知道自己有这个能力。
    渐进式披露把「能力好不好」的问题转移成了「描述写得准不准」，
    这是真实的成本转移不是消除。
    """
    return [
        {"name": spec.name, "description": spec.description, "inputSchema": spec.input_schema}
        for spec in registry.specs()
    ]


def render_result(result: ToolResult) -> dict[str, Any]:
    """`ToolResult` → MCP 返回结构。

    **两类失败在这里分道扬镳**（与 MCP 规范的两类错误划分同构）：

    - 参数 / 业务错误 → `isError: true` 的**工具结果**，内容是结构化错误对象，
      模型看得见、改得了；
    - 基础设施错误 → 根本不会走到这里：它在 `ToolRegistry.call` 里就被抛成
      `UpstreamTimeoutError` 向上传播了，**不喂回模型**。喂回去只会让它换个说法
      再试一次，浪费一轮预算还制造「模型很努力」的假象。

    空结果走的是**成功**分支（`ok=True` + `note`）：把空结果当错误，
    模型会去「修复」一个没坏的东西。
    """
    if not result.ok:
        payload = {"error_code": result.error_code, "hint": result.hint}
        return {"isError": True, "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}

    body: dict[str, Any] = dict(result.data)
    if result.note:
        body["note"] = result.note
    if result.truncated:
        # 裁剪必须对模型可见：不告诉它「后面还有」，它会把不完整的列表当成完整的，
        # 然后基于一个被截断的证据集下结论。
        body["truncated"] = True
    return {"isError": False, "content": [{"type": "text", "text": json.dumps(body, ensure_ascii=False)}]}


async def run_tool(registry: ToolRegistry, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """执行一次工具调用。超时、重试、裁剪、配额四件事全在注册表里，这里不重复。"""
    result, log = await registry.call(name, arguments or {})
    _logger.info(
        "mcp_tool_call",
        tool=log.tool,
        ok=log.ok,
        took_ms=log.took_ms,
        error_code=log.error_code,
        observation_tokens=log.observation_tokens,
    )
    return render_result(result)


def quota_snapshot(registry: ToolRegistry) -> dict[str, int]:
    """各工具的剩余配额，供 server 侧诊断。

    单列出来是因为 stdio 下**没有 HTTP 响应头可以挂配额信息**：
    调用方看不见 `X-RateLimit-Remaining`，配额用完时只会收到一个突然的业务错误。
    有这个快照，至少排查时能一眼看出是配额而不是逻辑问题。
    """
    return {spec.name: registry.quota_left(spec.name) for spec in registry.specs()}


def index_lines(registry: ToolRegistry) -> list[str]:
    """工具索引行（名字 + 一句话，≈25 token/条）。

    本系统只有 5 个原子工具，**渐进式披露的四条收敛路径一条都不需要**——
    工具集小到不构成那个问题。「我的工具集小到不构成这个问题」比假装用了一堆
    收敛技巧更可信。这个函数留着，是因为它同时是 `describe_tools` 的描述来源。
    """
    return registry.index_lines()


def spec_of(registry: ToolRegistry, name: str) -> ToolSpec | None:
    """按名取 spec，供 server 侧在调用前展示工具输入（MCP 要求 human in the loop）。"""
    return registry.get(name)
