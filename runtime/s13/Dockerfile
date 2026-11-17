# Candidate Linux exporter, separate from the online API and paid agent worker.
FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
ARG DEBIAN_SNAPSHOT=20261105T000000Z
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src HOME=/tmp/cqexport XDG_CACHE_HOME=/tmp/cqexport/cache
WORKDIR /app
# The pinned base determines Debian's codename. The snapshot fixes its package
# universe; the build report must additionally capture installed dpkg versions.
RUN set -eu; . /etc/os-release; test "$ID" = debian; \
    rm -f /etc/apt/sources.list.d/debian.sources; \
    printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s/ %s main\n' "$DEBIAN_SNAPSHOT" "$VERSION_CODENAME" > /etc/apt/sources.list; \
    apt-get -o Acquire::Check-Valid-Until=false update; \
    apt-get install -y --no-install-recommends libreoffice-writer libreoffice-math fonts-noto-cjk fonts-noto-core fonts-dejavu-core fonts-liberation fonts-crosextra-carlito fontconfig poppler-utils; \
    rm -rf /var/lib/apt/lists/*
COPY runtime/s13/font-packages.lock.json /tmp/font-packages.lock.json
COPY scripts/s13/install_fonts.py /tmp/install_fonts.py
RUN apt-get -o Acquire::Check-Valid-Until=false update && python /tmp/install_fonts.py && rm -rf /var/lib/apt/lists/*
COPY runtime/s13/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt
COPY runtime/s13/fontconfig.xml /etc/fonts/conf.d/99-citequest-export.conf
RUN fc-cache -f
COPY src/citequest src/citequest
COPY assets/export assets/export
COPY scripts/s13 scripts/s13
COPY runtime/s13 runtime/s13
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin cqexport
USER 10001:10001
ENTRYPOINT ["python", "scripts/s13/export_entry.py"]
