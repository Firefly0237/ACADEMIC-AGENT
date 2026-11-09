# Build targets share source and use independently locked Linux server roles.
ARG PYTHON_IMAGE=python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
ARG NODE_IMAGE=node:24.13.0-bookworm-slim

FROM ${NODE_IMAGE} AS frontend
ENV NPM_CONFIG_ENGINE_STRICT=true
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM ${PYTHON_IMAGE} AS source
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/citequest/src LANGGRAPH_STRICT_MSGPACK=true LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false
WORKDIR /opt/citequest
COPY src/citequest src/citequest
COPY contracts contracts
COPY assets/export assets/export
COPY runtime/s13/release-profile.json runtime/s13/release-profile.json
COPY runtime/s04/tokenizer-profile.json runtime/s04/tokenizer-profile.json
COPY runtime/s08/skills runtime/s08/skills
RUN mkdir -p data/runtime-s04/tokenizer data/runtime-v2-app/blobs /run/core-config /run/core-secrets \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin citequest

FROM source AS web
COPY runtime/server-web/requirements.txt /tmp/web-requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/web-requirements.txt
COPY --from=frontend /build/web/dist web/dist
USER 10001:10001
EXPOSE 8080
CMD ["python", "-m", "citequest.web_v3.main", "--config", "/run/core-config/web.json"]

FROM source AS agent
# Exported with --extra search: the MCP reader needs the locked splitter/tokenizer.
COPY runtime/server-agent/requirements.txt /tmp/agent-requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/agent-requirements.txt
USER 10001:10001
CMD ["python", "-m", "citequest.agents_v3.worker", "--config", "/run/core-config/worker.json", "--drain-on-signal"]

FROM agent AS export_worker
CMD ["python", "-m", "citequest.exports_v3.cli", "--config", "/run/core-config/export-worker.json", "--serve"]

FROM source AS updater
COPY runtime/s05-updates-linux/requirements.txt /tmp/updater-requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/updater-requirements.txt
USER 10001:10001
CMD ["python", "-m", "citequest.updates_v2.broker", "--config", "/run/core-config/source-broker.json"]

FROM source AS indexer
COPY runtime/server-indexer/requirements.txt /tmp/indexer-requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/indexer-requirements.txt
USER 10001:10001
CMD ["python", "-m", "citequest.updates_v2.publication_cli", "--config", "/run/core-config/publication.json"]
