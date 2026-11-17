# 贡献指南

欢迎为 CiteQuest 贡献代码、文档和问题反馈。

## 开发环境

- Python 3.12 与 uv
- Node.js 24、npm 11
- Docker Compose v2

从仓库根目录安装相应运行角色的依赖：

```bash
uv sync --locked --project runtime/s07
uv sync --locked --project runtime/s08 --extra search
cd web
npm ci
```

服务配置与启动方式见[快速开始](docs/quickstart.md)，模块关系见[架构](docs/architecture.md)。

## 提交修改

1. 为功能或修复建立独立分支。
2. 将业务实现放在 `src/citequest/`，界面放在 `web/src/`。
3. 为修改补充合适的单元或契约测试，更新相关文档。
4. 在对应锁定环境中检查受影响模块。
5. 提交 Pull Request，说明解决的问题、主要变化及验证结果。

前端常用命令：

```bash
cd web
npm run dev
npx tsc -b
npm run build
```

Python 测试位于 `tests/`，使用对应角色环境中的 pytest 运行。依赖更新应同时提交 manifest 与锁文件。

## 代码约定

- 复用现有 LangGraph、MCP、Provider 和领域服务。
- 使用明确的类型、结构化协议和稳定的任务身份。
- 数据库变化通过迁移管理，部署参数放入配置样例。
- 凭据采用秘密文件引用；用户材料和模型资产通过持久存储管理。
- 提交保持主题集中，便于阅读和回滚。

## 问题反馈

请提供复现步骤、期望行为、实际行为与相关组件版本。日志和截图应先移除凭据及个人数据。

提交贡献即表示同意按项目的 [Apache License 2.0](LICENSE) 发布贡献内容。
