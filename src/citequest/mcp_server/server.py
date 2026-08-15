"""MCP server 装配与 stdio 启动（决策记录 D18、M5 §13）。

> **这一节的定位必须先说清楚：它不是本项目的亮点。** 它证明的是「了解协议的能力
> 模型和 adapter 分层」，不是架构能力。

严格限定的范围：

| 项 | 取值 | 理由 |
|---|---|---|
| SDK | **官方 Python SDK** | 不写协议实现 |
| 传输 | **stdio only** | 一句自足的工程理由：**不引入远程鉴权面**。不引具体规范条款——拿过期的规范条款去辩护当前选择，是自己给自己挖坑 |
| 原语 | **仅 tools**，不做 resources / prompts | 本场景没有真实消费方。它们不是不好，是**这里没有东西要放进去** |
| 鉴权 | 不做 OAuth Resource Server | 那套是为多租户准备的，单机接入场景下是纯负担 |
| 工具数 | **5 个只读幂等工具** | 与 service 层已有能力一一对应，边际成本≈0 |

**差异化在哪**：市面上已有大量 OpenAlex / arXiv 的 MCP server，**再做一个 API
转发是零信号**。这里暴露的是自己的混合检索 + 重排 + 带引用校验的 RAG——
这是那些转发型 server 都没有的东西。

**明确不做的**：不把整个 Agent 循环包成一个 MCP tool。两条硬理由——① 一个内部还
调 LLM 的工具，它的 token 消耗不在调用方的预算账本里，**调用方的全部预算护栏对
它失效**；② 它对外是个跑几十秒、中途不可观察也不可中断的黑盒，而 MCP 的设计意图
是 model-controlled 且对 client 可见（规范要求 client 在调用前把工具输入展示给
用户、保持 human in the loop）。**违背的是协议意图而不只是风格偏好。**

---

## SDK 版本口径（必须写清楚，因为这里与决策记录不一致）

决策记录 V9 记的是：规范锚点 **2026-07-28**，当时官方 README 的写法是
`from mcp.server import MCPServer`（SDK v2 rework），并要求「类名与 `run()` 传输
参数字面值以官方 README 当前版本为准，禁止凭记忆写」。

**本文件按实际锁定的依赖写，而不是按那条记录写**，理由是可核对的：

- `pyproject.toml` 里锁的是 `mcp>=1.9,<2.0`，v2 的 `MCPServer` 不在这个区间内；
- 本机实际安装的是 **mcp 1.26.0**，`mcp.server` 导出的是 `Server` / `FastMCP`，
  **没有 `MCPServer`**（已用 `dir(mcp.server)` 核实）。

所以这里用 1.x 的 lowlevel 形态：`Server` + `stdio_server()` + `InitializationOptions`。
**要升到 v2 时，需要同时改两处**：`pyproject.toml` 的版本区间，和本文件的导入与
`run()` 调用——除此之外的全部代码与 SDK 版本无关，这正是 adapter 分层的收益。
"""

from __future__ import annotations

import sys

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from citequest.config.settings import Settings, get_settings
from citequest.mcp_server.tools import describe_tools, quota_snapshot, run_tool
from citequest.observability.logging import configure_logging, get_logger
from citequest.pipelines.factory import Services, build_services

__all__ = ["SERVER_NAME", "SERVER_VERSION", "build_server", "run_stdio"]

_logger = get_logger(__name__)

SERVER_NAME = "citequest"
SERVER_VERSION = "0.1.0"


def build_server(services: Services) -> Server:
    """装配 server：注册 `list_tools` 与 `call_tool` 两个 handler，**仅 tools 原语**。

    两个 handler 都是薄壳：工具定义来自 `orchestration.tool_specs`，执行来自
    `orchestration.tools.ToolRegistry`。**这一层没有任何检索逻辑**——
    这正是「service 层 + 多个 adapter」这个分层可验证的地方。
    """
    server: Server = Server(SERVER_NAME)
    registry = services.tools

    # SDK 的两个 handler 装饰器在 1.x 里没有类型标注，strict 模式下会把被装饰的
    # 函数整体判成 untyped。**只在这两行豁免**，不放宽全局 strict——
    # 放宽全局等于用一个第三方库的标注缺口换掉整个包的类型检查。
    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[types.Tool]:
        """5 个只读幂等工具的描述。顺序稳定（按名排序）。

        顺序稳定不是洁癖：工具列表每次都换顺序会让宿主端的缓存与 diff 全部失效。

        `inputSchema` 直接来自 `ToolSpec.input_schema`，**不在这里另写一份**：
        schema 写两份必然漂移，而漂移的表现是宿主端按一份构造参数、服务端按另一份
        校验，然后模型收到一个它无法理解的参数错误。
        """
        described = describe_tools(registry)
        _logger.info("mcp_list_tools", count=len(described), quota=quota_snapshot(registry))
        return [
            types.Tool(name=item["name"], description=item["description"], inputSchema=item["inputSchema"])
            for item in described
        ]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, object] | None = None) -> list[types.TextContent]:
        """执行一次工具调用。

        **不在这里 catch 基础设施错误**：它们由注册表抛成 `UpstreamTimeoutError`
        向上传播，交给 SDK 翻译成协议级错误。参数/业务错误则已经被注册表变成了
        一段结构化错误文本喂回模型——**两类错误走两条路**，把它们混成一个
        `except Exception` 会既重试了不该重试的，又没重试该重试的。
        """
        rendered = await run_tool(registry, name, dict(arguments or {}))
        return [types.TextContent(type="text", text=part["text"]) for part in rendered["content"]]

    _logger.info("mcp_server_built", tools=registry.names(), sdk="mcp 1.x lowlevel Server")
    return server


async def run_stdio(settings: Settings | None = None) -> None:
    """stdio 传输启动。

    **stdout 是协议通道**：任何写到 stdout 的内容都会破坏 JSON-RPC 的帧结构，
    而宿主端报出来的会是一个与日志内容毫无关系的解析错误。所以日志必须走 stderr
    ——这是本入口唯一一处与其它入口不同的地方，也是最容易踩的一处。
    """
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format, stream=sys.stderr)

    services = build_services(settings)
    server = build_server(services)
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=SERVER_NAME,
                    server_version=SERVER_VERSION,
                    # 只声明 tools 能力：不做 resources / prompts，
                    # 声明了却没有内容会让宿主端白跑一次列举。
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        # stdio 下宿主退出即会话结束，这里必须释放 Milvus 连接与 HTTP 客户端——
        # 不释放的表现是宿主重启几次之后 Milvus 侧连接数打满。
        await services.aclose()
        _logger.info("mcp_server_stopped")
