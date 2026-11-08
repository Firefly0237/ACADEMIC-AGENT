# 开发指南

## 目录

- `src/citequest/`：后端业务、检索与 Agent 工作流。
- `web/src/`：React 界面。
- `deploy/core/`：正式容器配方和配置样例。
- `runtime/`：各运行角色的 manifest 与依赖锁。
- `contracts/`、`migrations/`、`tests/`：协议、迁移和测试。

## 工作方式

- 先阅读相关实现，保持修改集中并保留其他协作者的改动。
- 复用现有 LangGraph、MCP、Provider 和领域服务。
- 使用对应角色的锁定依赖，按改动选择必要检查。
- 数据库变更通过显式迁移；持久数据和容器资源按所属服务管理。
- 密钥使用秘密文件引用，日志保持脱敏。
- 外部模型、云服务和资源操作遵循当前任务授权。
- 交付说明写清实际变更、执行的检查与结果。

安装与运行见 [快速开始](docs/quickstart.md)，提交约定见 [贡献指南](CONTRIBUTING.md)。
