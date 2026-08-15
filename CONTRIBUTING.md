# 参与贡献

感谢关注 CiteQuest。本文说明在哪里提问、接受什么样的改动，以及提交前需要过哪几关。

## 去哪里提问

- **用法问题、复现不出来、命令报错** → 开一个 GitHub Issue，附上 `uv run citequest show-config` 的输出、
  出错命令的完整日志（`CITEQUEST_LOG_FORMAT=json`）与其中的 `request_id`。
- **对某个设计决策的疑问**（为什么是 RRF 而不是加权融合、为什么 rerank 放应用侧……）→
  先看 [docs/architecture.md](docs/architecture.md) 与 [docs/configuration.md](docs/configuration.md)，
  那里每条决策都写了依据与代价。仍有疑问就开 Issue，讨论会被回填进文档。
- **本地开发环境问题** → 见 [docs/development.md](docs/development.md)。

## 接受什么样的改动

本项目接受 Pull Request。适合直接提 PR 的：文档订正、死链修复、缺陷修复、测试补充、
以及在现有架构内的小范围增强（新增一个检索策略档位、新增一个只读 MCP 工具）。

**改动前请先开 Issue 讨论**的情况：

- 改动任何**建库前不可逆**的参数（analyzer 类型、`bm25_k1` / `bm25_b`、`text_exact` 是否存在、占位符策略）。
  这类改动的代价是一次全量重灌，不是改一行代码。
- 引入新的常驻服务（Elasticsearch、Postgres、独立缓存服务等）。现有架构刻意不新增有状态服务。
- 触碰几条写死的约束：融合器唯一 `RRFRanker(k=60)`；RRF 分数不得用作阈值；
  dense 编码不得回到客户端做；MCP 工具必须只读幂等。
  这些不是风格偏好，每一条背后都有具体的失败模式，见 architecture.md。

**不会被接受的**：加入未实测的性能数字或相对提升百分比；把设计文档整段搬进 README；
新增只有 `pass` 的 `try/except`；在测试里复制一份业务逻辑。

## 提交前

```bash
make check        # ruff format --check + ruff check + mypy + 单元测试，与 CI 完全一致
```

另外确认：

1. 新增或改动的数值参数，在 [docs/configuration.md](docs/configuration.md) 里有对应行，且写了依据；
2. 改了 `pyproject.toml` 就重新 `uv lock`（`make lock`），并把 `uv.lock` 一起提交；
3. 注释与文档用中文，标识符、模块名、配置键、MCP 工具名与描述用英文；
4. 面向用户可见的改动在 [CHANGELOG.md](CHANGELOG.md) 的 `Unreleased` 段落记一条。

pre-commit 钩子会在 `make install` 时装好，本地提交时自动跑格式化、lint、类型检查与 `uv lock --check`。

## 分支与提交信息

从 `main` 切分支，一个 PR 只做一件事。提交信息用祈使句写清「改了什么、为什么」，
涉及设计决策时在正文里说明取舍与代价——这与代码注释的要求是同一条：写「为什么是这个取值」和「改了会怎样」。

## 行为准则

请保持技术讨论就事论事。对设计决策的质疑是受欢迎的，但请连同你的依据一起给出。
