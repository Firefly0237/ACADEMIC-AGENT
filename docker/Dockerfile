# CiteQuest API 镜像。
#
# 两段式构建：builder 装依赖，runtime 只拿 .venv 与源码。
# 这里**没有 torch / transformers**——在线 dense 编码走 TEI，API 进程不做任何推理。
# reranker 与 HHEM 需要模型栈时，用 `--build-arg EXTRAS=models` 单独构建一个镜像，
# 而不是把几个 GB 的 CUDA 依赖塞进所有部署单元。

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 先只拷依赖清单再 sync：源码改动不会让依赖层缓存失效，
# 而依赖层是这个镜像里唯一耗时的一层。
COPY pyproject.toml uv.lock ./
# --no-install-project：此时源码还没拷进来，装项目本身会失败。
# --no-dev：dev 组（pytest/ruff/mypy）不进运行镜像。
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# curl 是给 healthcheck 用的，别删。
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 不用 root 跑服务。
RUN useradd --create-home --uid 10001 citequest
WORKDIR /app

COPY --from=builder --chown=citequest:citequest /app/.venv /app/.venv
COPY --from=builder --chown=citequest:citequest /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CITEQUEST_LOG_FORMAT=json

USER citequest
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# 容器里不开 --reload，也不开多 worker：并发闸门与 GPU 信号量是**进程内**状态
# （决策记录 §4.9），多 worker 会让「rerank 并发 4」实际变成 4×worker 数。
# 要横向扩容就起多个容器，让闸门额度随之按实例数拆分。
CMD ["uvicorn", "citequest.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
