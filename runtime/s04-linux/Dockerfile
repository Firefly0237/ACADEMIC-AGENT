FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libseccomp2 && rm -rf /var/lib/apt/lists/*
COPY runtime/s04-linux/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt
COPY src/citequest src/citequest
COPY contracts contracts
COPY runtime/s04 runtime/s04
COPY runtime/s04-linux runtime/s04-linux
COPY runtime/v2/uv.lock runtime/v2/uv.lock
COPY runtime/s04-app/uv.lock runtime/s04-app/uv.lock
COPY scripts/s04_linux scripts/s04_linux
COPY reports/stages/v2/S02/parser/models-initial/model-download.json reports/stages/v2/S02/parser/models-initial/model-download.json
COPY data/runtime-s04/tokenizer/tokenizer.json data/runtime-s04/tokenizer/tokenizer.json
COPY source-manifest.json source-manifest.json
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin cqparser && python -c "import ctypes.util; assert ctypes.util.find_library('seccomp'), 'libseccomp required'"
USER 10001:10001
CMD ["python", "-S", "scripts/s04_linux/worker_entry.py"]
