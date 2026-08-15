# 开发指南

本文覆盖本地环境、测试分层、代码规范，以及三类最常见的扩展动作：
新增一个检索策略、新增一个 MCP 工具、新增一个配置参数。

## 环境

依赖用 [uv](https://docs.astral.sh/uv/) 管理，锁文件是 `uv.lock`。
**所有命令都走 `uv run`**，保证本地、pre-commit 钩子与 CI 用的是同一套版本。

```bash
make install          # uv sync + pre-commit install
uv run citequest --help
```

`make install` 里刻意**不加** `--frozen`：新克隆的仓库还没有 `uv.lock` 时 `--frozen` 会直接报错，
而这是 quickstart 的第一步。首次 `make install`（或 `make lock`）解析出的 `uv.lock` **必须提交**——
CI（`uv sync --frozen`）与镜像构建（`COPY pyproject.toml uv.lock`）都只认仓库里的那份锁文件。

Python 版本由 `.python-version` 固定为 3.12（`requires-python = ">=3.12,<3.14"`）。

依赖分档的原则是**在线路径必需的才进主依赖**：

| 组 | 装什么 | 为什么单列 |
|---|---|---|
| 主依赖 | pymilvus / fastapi / pydantic-settings / langgraph / mcp / duckdb | 在线路径必需 |
| `models`（extra） | torch / transformers / sentence-transformers | API 容器与离线建库容器需要的模型栈不同，混在一起会让 API 镜像凭空多几个 GB |
| `ingest`（extra） | datasets / tqdm / orjson | 只有建库要用 |
| `dev`（PEP 735 组，默认装） | pytest / ruff / mypy / pre-commit | — |
| `eval`（PEP 735 组） | ranx / pandas / scipy / matplotlib | 评测专用，不污染主依赖 |

```bash
uv sync --frozen --extra models        # 需要跑 rerank / HHEM 时
uv run --group eval citequest evaluate # 评测
```

改了 `pyproject.toml` 一定要重新 `uv lock`（`make lock`）——忘了 lock 是 CI 上最常见的一类假失败，
pre-commit 里的 `uv lock --check` 钩子就是拦这个的。

## 本地依赖服务

```bash
cp docker/.env.example docker/.env
make up      # Milvus standalone（含 etcd + minio）+ TEI
make logs    # 跟随日志
make down    # 停掉并保留数据卷
```

两个容易踩的点：

1. **TEI 镜像 tag 必须匹配本机 GPU 架构**（默认 `89-1.9` = Ada Lovelace / RTX 4000 系）。
   compute capability < 7.5 的卡（V100、GTX 10 系）TEI 不支持，只能退 CPU 镜像。见 `docker/tei/README.md`。
2. **`CITEQUEST_EMBEDDING__ENDPOINT` 填的是 Milvus 容器视角的地址**（`http://tei:80`）。
   去调 TEI 的是 Milvus 服务端而不是应用进程，填 `localhost` 会在 Milvus 容器内解析到它自己。
3. compose 里**故意没有挂 `milvus.yaml`**：官方配置无需改动即可用 TEI provider，
   而挂载会整份替换镜像内的默认配置——挂一份只写了几行的 yaml 等于把其余几百项默认值清空，Milvus 起不来。

## 测试分层

```bash
make test-unit          # 不连任何外部服务，CI 跑的就是这一档
make test-integration   # 需要本机已 make up（Milvus + TEI）
make test-e2e           # FastAPI TestClient 打通 search / answer（含 SSE）
make test               # 全部
```

marker 定义在 `pyproject.toml`：`integration`（需要本地 Milvus / TEI，CI 上默认跳过）、
`e2e`（TestClient 全链路）、`slow`（单条耗时超过数秒）。
`asyncio_mode = "auto"`，异步测试不需要写 `@pytest.mark.asyncio`。

pytest 用 `--import-mode=importlib`：src layout 下测试跑的是**安装后的包**，
而不是仓库里那份源码副本——打包错误会在测试阶段就暴露，而不是等到别人 pip install 才发现。

三条测试纪律：

1. **测试不得复制或重写业务逻辑。** 测试只组装输入、断言输出，业务路径必须完全复用生产代码。
   这一条在 `evaluation/` 上同样成立：评测复用生产管线，评测代码里另写一份检索，测的就不是线上跑的那套。
2. **测试输入直接由变量给定**，不用命令行参数解析来构造。
3. **临时文件写到 pytest 的 `tmp_path`**，不污染仓库工作区。

`tests/unit/test_settings_defaults.py` 是一个特殊的测试：它拦的是「设计参数被悄悄改掉」。
改了 `config/` 里的默认值而没有同步设计文档，这条会红。

## 代码规范

质量门是 `make check`（`lint` + `type` + `test-unit`，其中 `lint` 含 `ruff format --check`），
与 CI 的四步逐字一致。提交前跑一次。

**ruff**（`line-length = 120`，规则集 `E,F,I,UP,B,SIM,C4,RUF,ANN,T20`）：

- `ANN`：类型注解必须齐全（允许 `**kwargs: Any`）。
- `T20`：**禁止 `print`**，输出一律走 `observability.logging` 的结构化日志。
  这条在 MCP 那条路上是硬性的——stdio 传输下 stdout 是协议通道，任何写到 stdout 的内容都会破坏 JSON-RPC 帧结构，
  而宿主端报出来的会是一个与日志内容毫无关系的解析错误。
- `RUF001/002/003` 已关闭：中文注释里必然出现全角标点，否则会被全部报成 ambiguous unicode。
- `E501` 由 formatter 管：中文注释按视觉宽度算会被误判。

**mypy** 跑 `strict` + `warn_unreachable`，带 pydantic 插件。第三方库存根参差不齐，
所以对 `pymilvus` / `langgraph` / `mcp` / `ranx` 等显式放开 `ignore_missing_imports`。

**语言分工**（全仓库一致）：注释与文档用中文，标识符、模块名、配置键、日志 event 名、
MCP 工具名与描述用英文。MCP 工具描述面向模型，必须是英文且要写准——
描述是渐进式披露下的单点故障，写不准就等于该命中的不命中。

**注释写什么**：写「为什么是这个取值」和「改了会怎样」，不写代码已经说清的事。
一个可以照抄的形态是：结论 + 依据（官方文档 / 论文 / 自选理由）+ 代价。
凡是数值，必须能在 [configuration.md](configuration.md) 里查到；查不到的数值不许写进代码。

**pre-commit** 的 hook 一律是 `local` + `uv run --frozen`，不用各工具的官方 mirror repo——
mirror 的 `rev` 与 `uv.lock` 是两套独立的真相，迟早出现「本地过了 CI 没过」。

## 新增一个检索策略

「策略」在本项目里指的是**一条 query 走哪几路、用什么过滤、怎么排序**，
而不是一个新的融合器。加策略只碰三个位置：

1. **路由输出**（`query/intent.py`、`core/query_plan.py`）——让 `QueryPlan` 能表达这个新分支。
   如果新分支需要新的枚举取值，加在 `core/enums.py`，并在 docstring 里写清它「不做会怎样」。
2. **请求装配**（`indexing/ann_requests.py`）——决定组几路 `AnnSearchRequest`、每路的 `search_params`。
   注意 `AnnSearchRequest` **不支持** grouping 的四个参数（`group_by_field` / `group_size` /
   `strict_group_size` / `rank_group_scorer`），它们只能在 `hybrid_search` 顶层传。
3. **执行与打点**（`retrieval/hybrid.py`）——设置 `RetrievalMode`，让降级可度量。

三条不可动的约束，新策略必须遵守：

- **融合器只有 `RRFRanker(k=60)`。** 需要通道偏置就做成**单路不融合**（单路下量纲问题不存在），
  不要引入加权器。想加 WeightedRanker 之前先想清楚：两路分数不在同一个测度空间，
  归一化只压值域不对齐分布。
- **不许拿 RRF 分数做阈值。** 全系统唯一能做阈值的是 rerank 的归一化分。
- **新策略必须能被评测证伪。** 加策略的同时在 `evaluation/ablation.py` 里加一档，
  跑 `citequest evaluate --config <新档>` 与现有档位对照；**止损判据事先写死**
  （HyDE 与纠正循环都是这么定的），跑不赢就下线，而不是留着「以后再调」。

## 新增一个 MCP 工具

**加一个工具只改一个文件**：`orchestration/tool_specs.py`。
注册表（`orchestration/tools.py`）里没有任何按工具名分支的代码——超时、重试、裁剪、配额
对所有工具是同一段逻辑，差异全部落在 `ToolSpec` 的字段值上。

```python
registry.register(
    ToolSpec(
        name="...",              # 英文，snake_case
        description="...",       # 英文一句话。它是索引行，也是单点故障
        input_schema=...,        # JSON Schema。MCP 的 inputSchema 直接用它，不写第二份
        handler=...,             # 薄壳：调 pipelines 里的 service，不写检索逻辑
        token_budget=...,        # 返回裁剪上限，按返回粒度定
        timeout_ms=...,
        quota_per_session=...,   # 最贵的工具配额最紧
    )
)
```

四条硬约束：

1. **必须只读幂等**。注册表会直接拒绝 `readonly=False` 或 `idempotent=False` 的 spec——
   正因为都是只读的，MCP 那条路才不需要审批断点，重试才是安全的。
2. **handler 是薄壳**，调的是 `pipelines/` 里那份同样被 HTTP 用着的 service。
   两条入口各写一份编排，行为迟早静默不一致，而不一致的表现是「同一个问题在 MCP 宿主里和在网页里答案不同」。
3. **工具边界按返回粒度划**。`search_papers` 与 `search_chunks` 拆成两个，就是因为
   合成一个会让调用方混淆「我是在找论文还是在找证据」，而这类混淆是工具选择错误的主要来源。
4. **不要把多步流程做成一个工具**。带内部 LLM 调用的长流程会让调用方的预算护栏全部失效，
   而且对外是个跑几十秒、中途不可观察也不可中断的黑盒，违背 MCP 的协议意图。这类流程应该做成 Skill。

错误处理分两类，不要混：参数 / 业务错误返回 `isError: true` 的结构化工具结果（模型看得见、改得了）；
基础设施错误直接向上抛（不喂回模型——喂回去只会让它换个说法再试一次）。
空结果走**成功**分支（`ok=True` + `note`）：把空结果当错误，模型会去「修复」一个没坏的东西。

## 新增一个配置参数

1. 加到 `src/citequest/config/` 里对应的分组模型上，写 `default` 与 `description`。
   **description 里要有依据**：官方文档 / 论文 / 自选理由，三选一。
2. 在 [configuration.md](configuration.md) 的对应表里补一行。
3. 如果它与别的参数有跨组约束（改一个不改另一个会静默出错），加进 `Settings` 的 model validator——
   现有的五条跨组校验都是这么来的，它们的共同特征是「单看每组都合法，放在一起才暴露矛盾」。
4. 如果它是**部署差异**（地址、密钥、开关），在 `.env.example` 里加一行示例；
   如果它是**设计参数**，不要加进 `.env.example`——环境变量不该用来改设计参数。
5. 建库前不可逆的参数（analyzer、BM25 参数、`text_exact`、占位符策略）额外标 ⚠ 并写清返工代价。

改完跑 `uv run citequest show-config` 确认它出现在输出里。

## 建库与评测

```bash
uv run citequest build-index --dry-run              # 只做 schema 校验与规模估算
uv run citequest build-index --granularity fulltext # 只灌某一层
uv run --group eval citequest evaluate --benchmark litsearch --config hybrid_rerank
```

`build-index` 是一条**昂贵且不可逆**的命令：全量重灌是 6–10 小时量级，
而且四项建库前参数一旦写进 collection 就改不动。`--dry-run` 不是可选的礼貌，是默认应该先跑一遍的步骤。
建库参数会被完整打进日志——出问题时第一件要查的事就是「当初这一库到底是用什么参数建的」。

评测报告落到 `reports/`。**产不出真实 run 就不该落一份报告**：报告一旦落盘就会被引用。

## 服务与调试

```bash
make serve            # uvicorn，热重载
make mcp              # MCP server，stdio
```

`serve` 默认单 worker，多数情况下不该改：并发闸门与 GPU 信号量是进程内状态，
多 worker 会让「rerank 并发 4」实际变成 4 × worker 数，把显存账直接打穿。
要扩容就起多个容器，让闸门额度随实例数拆分。

调试时优先看结构化日志里的 `request_id`——它贯穿 HTTP、pipeline、Milvus 调用与 SSE 事件。
三个值得先看的字段：`retrieval_mode`（是不是降级成单路了）、`degraded`（哪一环降级了）、
`group_count`（是不是没凑满组）。这三项的共同特征是不报错、只让指标悄悄下滑。

日志输出建议落文件而不是只看终端回显：

```bash
CITEQUEST_LOG_FORMAT=json make serve > /tmp/citequest.log 2>&1
```

## 提交

分支从 `main` 切出，提交前 `make check` 必须绿。CI（`.github/workflows/ci.yml`）跑的是
`ruff format --check` → `ruff check` → `mypy` → `pytest tests/unit`，用的是 `uv sync --frozen`——
CI 绝不隐式更新锁文件，否则会出现「同一个 commit 昨天绿今天红」。

改动涉及设计参数或架构决策时，同步更新 [configuration.md](configuration.md) 或
[architecture.md](architecture.md)，并在 [CHANGELOG.md](../CHANGELOG.md) 里记一条。
