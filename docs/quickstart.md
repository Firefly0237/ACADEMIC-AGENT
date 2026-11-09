# 快速开始

通过 [Docker Compose](../deploy/core/compose.yaml)运行 CiteQuest。

## 环境

- Docker 与 Docker Compose v2
- PostgreSQL 17，包含应用迁移及 LangGraph Saver/Store 表
- Milvus、TEI 与对应文献索引
- OIDC 身份服务与 HTTPS 入口
- 文献 CAS、tokenizer 和模型API凭据

## 1. 配置部署路径

在仓库根目录执行：

```bash
cp .env.example .env
```

编辑 `.env`，设置运行目录和服务网络：

| 变量 | 用途 |
| --- | --- |
| `CITEQUEST_CORE_CONFIG_DIR` | JSON配置目录 |
| `CITEQUEST_CORE_WEB_SECRETS_DIR` | Web秘密目录 |
| `CITEQUEST_CORE_AGENT_SECRETS_DIR` | Agent秘密目录 |
| `CITEQUEST_CORE_CAS_DIR` | 应用CAS目录 |
| `CITEQUEST_CORE_TOKENIZER_FILE` | tokenizer文件 |
| `CITEQUEST_CORE_BUSINESS_NETWORK` | 数据库网络 |
| `CITEQUEST_CORE_INDEX_NETWORK` | Milvus/TEI网络 |
| `CITEQUEST_CORE_IDENTITY_NETWORK` | 身份服务网络 |
| `CITEQUEST_CORE_HTTP_PORT` | Web宿主端口，默认18000 |

目录与资产供容器UID 10001读取，CAS供Web写入。

## 2. 配置服务

将以下样例复制到配置目录，并使用表中部署文件名：

| 样例 | 部署文件 |
| --- | --- |
| [Web](../deploy/core/config/web.example.json) | `web.json` |
| [搜索](../deploy/core/config/search.example.json) | `search.json` |
| [Agent API](../runtime/s08/api.example.json) | `api.json` |
| [Agent worker](../runtime/s08/worker.example.json) | `worker.json` |
| [材料证据](../deploy/core/config/material.example.json) | `material.json` |
| [材料索引](../deploy/core/config/material-native.example.json) | `material-native.json` |

在 `web.json` 填写外部HTTPS地址、OIDC issuer和client ID。将 `${origin}/auth/callback` 注册为OIDC回调地址。

在搜索与材料配置中填写Milvus/TEI端点、集合和索引发行包对应的profile/policy/evaluation哈希；在worker配置中填写模型元数据回执文件。tokenizer使用[runtime/s04/tokenizer-profile.json](../runtime/s04/tokenizer-profile.json)指定的版本。

## 3. 配置秘密

按各服务配置中的文件引用，准备下列秘密：

| 文件 | 内容 |
| --- | --- |
| `business_sqlalchemy_dsn` | Web、检索与材料服务的数据库连接 |
| `agent_business_dsn` | Agent业务数据库连接 |
| `checkpoint_dsn`、`memory_dsn` | Saver与Store连接 |
| `cursor_key` | 分页签名密钥 |
| `oidc.json` | Web的`client_secret`与`session_secret` |
| `model-key` | Agent模型API密钥 |

`oidc.json`中两个secret均使用至少32字符的随机值。Web和Agent使用各自的秘密目录，通过 `/run/core-secrets` 读取；JSON配置挂载到 `/run/core-config`。

## 4. 构建与运行

```bash
docker compose --env-file .env -f deploy/core/compose.yaml --profile agents build
docker compose --env-file .env -f deploy/core/compose.yaml --profile agents up -d --no-build
docker compose --env-file .env -f deploy/core/compose.yaml --profile agents ps
```

Web监听 `http://127.0.0.1:18000`。通过配置的HTTPS地址访问并完成OIDC登录，即可进入搜索与研究工作站。

查看日志和停止服务：

```bash
docker compose --env-file .env -f deploy/core/compose.yaml --profile agents logs --tail=100
docker compose --env-file .env -f deploy/core/compose.yaml --profile agents stop
```

## 可选组件

- **DecisionRAG导航**：使用[导航API配置](../runtime/s08/api-decisionrag.example.json)、[导航worker配置](../runtime/s08/worker-decisionrag.example.json)及[概率服务](../runtime/decisionrag/README.md)。
- **文档导出**：配置独立[导出工具链](../runtime/s13/README.md)，运行 `python -m citequest.exports_v3.cli --config <配置文件> --serve`。
- **合格原生语料**：设置 `CITEQUEST_CORE_NATIVE_CHAIN_FILE`，在Compose命令中增加 `-f deploy/core/compose.native-qualified.yaml`。

## 本地开发

后端环境使用 `runtime/s07` 与 `runtime/s08` 的uv锁，前端使用 `web/package-lock.json`。

```bash
uv sync --locked --project runtime/s07
uv sync --locked --project runtime/s08 --extra search
cd web
npm ci
npm run dev
```

前端通过Vite开发服务器运行。更多开发约定见[贡献指南](../CONTRIBUTING.md)。

## Google Cloud

[Foundation](../deploy/gcp/foundation/README.md) 配置网络、Cloud SQL、存储与服务身份；[Workloads](../deploy/gcp/workloads/README.md) 配置 HTTPS 入口、多可用区 Web 实例和独立工作角色。

服务器镜像使用 `deploy/core/Dockerfile` 的 `web`、`agent`、`updater`、`indexer`、`export_worker` 构建目标。Web、Agent 与索引角色分别锁定在 `runtime/server-web`、`runtime/server-agent`、`runtime/server-indexer`；源更新使用 `runtime/s05-updates-linux`。

Web 在负载均衡器后使用 [GCP 配置](../deploy/core/config/web.gcp.example.json)，按部署网络设置 `trusted_proxy_networks`。配置发布按对象 generation 与 SHA-256 固定版本，镜像使用 registry digest，秘密通过 Secret Manager 的固定版本挂载。应用迁移、Saver/Store 初始化与角色授权在工作进程启动前完成。

文档渲染镜像与发布配置通过 [Export release](../runtime/s13/README.md) 工作流生成，Web 与导出消费者使用同一发行身份。
