"""CiteQuest：面向计算机论文的学术检索与 RAG 系统。

包结构按**能力域**分一级目录，不按 MVC 分层：

- `config`       —— 参数正本。决策记录 §4 的每个数值在这里有且只有一份默认值
- `core`         —— 领域层：枚举、数据契约、异常。不依赖任何外部服务
- `observability`—— 结构化日志、request_id 注入、span 记录
- `ingestion`    —— 语料获取与切块（M1）
- `indexing`     —— Milvus schema、Function 声明与灌库（M2）
- `retrieval`    —— 混合检索、RRF 融合、grouping、rerank（M2/M4）
- `query`        —— 查询理解、路由、HyDE、decompose（M3）
- `rag`          —— 生成、引用、grounding、流式（M4/M5）
- `graph`        —— 唯一的 LangGraph 部分：一条窄的检索纠正循环
- `orchestration`—— 工具定义与注册表、研究会话记忆、冲突检测（M8）
- `pipelines`    —— workflow 编排，把上面各层串成 search / answer 两条链路
- `llm`          —— 外部模型服务的客户端封装（只做协议适配）
- `api`          —— FastAPI 薄层：协议转换 + 依赖注入，无业务逻辑
- `mcp_server`   —— MCP server，stdio only，5 个只读幂等工具
- `evaluation`   —— 评测作为一等模块：数据集适配、指标、跑分
- `cli`          —— `citequest` 命令行入口，与 `api` / `mcp_server` 同为协议适配层

**依赖方向是单向的**：core / config 不 import 任何上层；api / mcp_server 只 import
pipelines，不直接碰 retrieval 与 rag。这条约束让 MCP 那条路与 HTTP 那条路
共用同一个 service 层，而不是各自复制一份（K11）。
"""

from citequest.config.settings import Settings, get_settings

__all__ = ["Settings", "__version__", "get_settings"]

__version__ = "0.1.0"
