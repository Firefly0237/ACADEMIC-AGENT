"""MCP 服务端：官方 Python SDK，**stdio only**，5 个只读幂等工具。

- `server.py` —— server 装配与 stdio 启动
- `tools.py`  —— 5 个工具定义，内部直接复用 `pipelines/` 与 `retrieval/`

工具清单（全部只读、全部幂等）：
`search_papers` / `search_chunks` / `get_paper_detail` / `structured_filter` /
`expand_by_citation`。只用 tools 原语，不用 resources 与 prompts。

命名为 `mcp_server` 而不是 `mcp`，纯粹是可读性考虑：绝对导入本身不会与 pip 包
`mcp` 冲突，但读代码的人需要一眼分清「我们的 server」和「官方 SDK」。

两条边界：

1. **只读**——没有任何工具会写库。这让「MCP 暴露给外部宿主」这件事的风险面
   收敛到只剩信息读取，不需要再设计一套授权模型。
2. **不重写检索逻辑**——工具函数是薄壳，调的是 `pipelines` 里那份同样被 HTTP
   用着的 service。两条入口各写一份编排，行为迟早静默不一致。

SDK 的类名与 `run()` 的传输参数字面值**以当前 README 为准，禁止凭记忆写**
——这个 SDK 的 API 在 1.x 期间改过不止一次。**唯一需要核对的那一行在
`server.py` 的导入处**，其余代码与 SDK 版本无关。

本包**刻意不在 `__init__` 里 import `server`**：`server.py` 顶层要 import 官方
`mcp` 包，而 `tools.py`（纯 adapter 逻辑）不需要。分开之后，没装 mcp 依赖的环境
仍然能 import 并单测 adapter 那一半——那一半才是这里真正有逻辑的部分。
"""

from citequest.mcp_server.tools import describe_tools, render_result, run_tool

__all__ = ["describe_tools", "render_result", "run_tool"]
