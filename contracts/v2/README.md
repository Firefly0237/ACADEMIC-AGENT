# 领域与工具协议

本目录保存CiteQuest的JSON Schema、接口定义和结构化工作流协议。

| 文件 | 职责 |
| --- | --- |
| `domain.schema.json` | 文献、来源、文件、任务、记忆和索引的数据类型 |
| `openapi.json` | HTTP接口与请求/响应类型 |
| `policy.json` | 业务状态和运行策略 |
| `tools.json` | 工具输入与输出 |
| `roles.json` | 研究角色的结构化交接 |
| `evaluation.json` | 评估任务与指标定义 |
| `fixtures.json` | 协议示例和测试数据 |
| `traceability.json` | 功能与协议对象的映射 |

各组件通过JSON Schema引用共享类型。系统关系见[架构](../../docs/architecture.md)。
