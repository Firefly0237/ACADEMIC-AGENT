ARG ASGI_IMAGE=citequest-s07-web:0.9.1-core
ARG CPU_IMAGE=citequest-s04-linux:0.3.0-cpu-xyz
FROM ${ASGI_IMAGE} AS asgi
USER 0
RUN python - <<'PY'
import importlib.metadata as metadata
import pathlib
import shutil
base = pathlib.Path('/usr/local/lib/python3.12/site-packages')
target = pathlib.Path('/asgi-only')
for name in ('fastapi', 'starlette', 'uvicorn', 'opentelemetry-api'):
    dist = metadata.distribution(name)
    for entry in dist.files or []:
        source = pathlib.Path(dist.locate_file(entry)).resolve()
        if source.is_relative_to(base) and source.is_file() and source.suffix != '.pyc':
            destination = target / source.relative_to(base)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
PY

FROM ${CPU_IMAGE} AS service
USER 0
COPY --from=asgi /asgi-only/ /usr/local/lib/python3.12/site-packages/
WORKDIR /opt/citequest
ENV PYTHONPATH=/opt/citequest/src:/opt/citequest TRANSFORMERS_OFFLINE=1
COPY src/citequest src/citequest
COPY contracts contracts
COPY scripts/s14 scripts/s14
COPY runtime/decisionrag/service-runtime.lock.json runtime/decisionrag/service-runtime.lock.json
COPY runtime/decisionrag/verify_service_runtime.py runtime/decisionrag/verify_service_runtime.py
RUN python runtime/decisionrag/verify_service_runtime.py && python -m pip check
LABEL org.citequest.role="decisionrag-private-CPU-service" org.citequest.runtime="citequest-decisionrag-service-runtime/1"
USER 10001:10001
CMD ["python", "-m", "scripts.s14.serve_decisionrag", "--config", "/run/core-config/decisionrag.json", "--execute-local-cpu"]
