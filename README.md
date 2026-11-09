# CiteQuest

面向学术文献的智能搜索引擎与 AI 科研工作站。

[快速开始](docs/quickstart.md) · [架构](docs/architecture.md) · [轨迹评估](docs/evaluation.md) · [Google Cloud](deploy/gcp/workloads/README.md) · [贡献指南](CONTRIBUTING.md)

CiteQuest 将文献检索、RAG 问答和 Agent 工作流放在同一研究空间中：从搜索论文、追溯引用，到整理文献、分析观点和组织研究计划。

## 功能

- **学术搜索**：BM25 与向量混合检索、可选 cross-encoder 精排、稳定分页、文献详情、原文与 Harvard 引用。
- **文献问答**：围绕论文或上传材料进行多轮对话，回答关联可定位的原文证据。
- **摘要与比较**：逐篇阅读与归纳，再按研究问题组织多文献比较。
- **观点考证**：选择支持或反驳方向，收集证据并呈现论断、引用和证据缺口。
- **研究计划**：多轮信息补全、上海交大模板配置、提纲/文献/方法/写作角色协作、章节与全文审查。
- **稿后协作**：可选三 Agent 评审、发现项选择、定向修订、稿件版本与版本绑定导出。
- **实验设计**：根据研究问题、数据与资源组织实验方案，保留用户确认事实与待补充信息。
- **个人工作站**：文献收藏、浏览/搜索历史、上传材料相似搜索、项目记忆、版本化 Skill 和主题切换。
- **可观察执行**：真实工具与模型回执、多轮历史选择、用量账本及 DecisionRAG 文献→片段概率导航。

## 架构

React 提供搜索与研究界面，FastAPI 承载 HTTP 服务，LangGraph 编排持久化 Agent 工作流，MCP 连接检索和证据工具。PostgreSQL 保存业务数据、任务和记忆，Milvus 与 TEI 提供检索，CAS 保存来源原文。Agentic RAG 将需求、证据、结构化论断与独立来源审查连接；源更新与索引发布使用独立作业和不可变 generation。

更多流程说明见[架构文档](docs/architecture.md)。

## 快速开始

使用 Docker Compose 部署。先按[快速开始](docs/quickstart.md)准备数据库、检索服务、身份认证和运行配置，然后在仓库根目录执行：

```bash
cp .env.example .env
# 编辑 .env，配置服务网络、配置目录、密钥目录和文献存储。
docker compose --env-file .env -f deploy/core/compose.yaml --profile agents build
docker compose --env-file .env -f deploy/core/compose.yaml --profile agents up -d --no-build
```

Web 默认监听 `http://127.0.0.1:18000`，公开访问地址由 HTTPS 入口与 OIDC 配置提供。

## 项目结构

```text
src/citequest/   后端、检索、RAG 与 Agent
web/            React 界面
deploy/         容器、Compose 与 Google Cloud 模块
runtime/        各运行角色的依赖锁与配置
assets/         文档模板、引用样式与模型元数据
contracts/      结构化协议
migrations/     数据库迁移
scripts/        数据、索引与运行组件入口
tests/          自动化测试
tools/          工程管理命令
docs/           使用与架构文档
```

## 贡献

欢迎提交问题、改进建议和 Pull Request。开发环境与提交方式见[贡献指南](CONTRIBUTING.md)。

## 许可证

代码采用 [Apache License 2.0](LICENSE)。模型与数据遵循各自许可证。
