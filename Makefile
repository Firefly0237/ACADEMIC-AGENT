# CiteQuest 统一任务入口。所有命令内部一律走 `uv run`，
# 保证本地、pre-commit hook 与 CI 用的是 uv.lock 锁定的同一套版本。
.DEFAULT_GOAL := help
.PHONY: help install lock lint format type test test-unit test-integration test-e2e check \
        up down logs index eval serve mcp clean

help:  ## 显示本文件里所有带说明的目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---- 环境 ----
install:  ## 装齐运行依赖 + dev 组。uv.lock 不存在时先解析生成，生成后必须提交
	# 这里**不用** --frozen：新克隆的仓库如果还没有 uv.lock，--frozen 会直接报错退出，
	# 而这条命令正是 quickstart 的第一步。CI 与镜像构建那两处才用 --frozen——
	# 它们面对的是「锁文件已经在仓库里」的既定事实，绝不该隐式更新它。
	uv sync
	uv run pre-commit install

lock:  ## 重新解析依赖并更新 uv.lock（改完 pyproject.toml 必跑，pre-commit 会校验）
	uv lock

# ---- 质量门（CI 跑的就是这四个）----
lint:  ## ruff 格式检查 + lint（两条与 CI 的前两步逐字一致，否则 make check 绿而 CI 红）
	uv run ruff format --check src tests scripts
	uv run ruff check src tests scripts

format:  ## ruff 格式化
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

type:  ## mypy 严格模式类型检查
	uv run mypy

test-unit:  ## 单元测试：不连任何外部服务
	uv run pytest tests/unit -m "not integration and not e2e"

test-integration:  ## 集成测试：需要本机已起 Milvus + TEI（make up）
	uv run pytest tests/integration -m integration

test-e2e:  ## 端到端：FastAPI TestClient 打通 search / answer
	uv run pytest tests/e2e -m e2e

test:  ## 全部测试
	uv run pytest

check: lint type test-unit  ## CI 门禁的本地等价物

# ---- 本地依赖服务 ----
up:  ## 拉起 Milvus standalone + TEI（+ API）
	docker compose -f docker/docker-compose.yml up -d

down:  ## 停掉并保留数据卷
	docker compose -f docker/docker-compose.yml down

logs:  ## 跟随查看容器日志
	docker compose -f docker/docker-compose.yml logs -f

# ---- 业务流程 ----
index:  ## 跑 ingestion → indexing 全流程灌库（6-10 小时量级，别在半夜以外的时间随手跑）
	uv run citequest build-index

eval:  ## 跑评测并把报告写到 reports/
	uv run --group eval citequest evaluate

serve:  ## 起 API（开发模式，带热重载）
	uv run citequest serve --reload

mcp:  ## 起 MCP server（stdio，供 Claude Desktop 之类的宿主接入）
	uv run citequest mcp

# ---- 清理 ----
clean:  ## 清掉工具缓存与构建产物（不碰 data/）
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
