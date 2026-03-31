FROM python:3.10-slim AS builder

ARG INCLUDE_ML=false

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    gfortran \
    libffi-dev \
    libpq-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /tmp/build

COPY requirements.txt requirements-ml.txt ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir Babel==2.13.1 && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ "$INCLUDE_ML" = "true" ]; then \
        pip install --no-cache-dir -r requirements-ml.txt && \
        pip install --no-cache-dir Cython && \
        pip install --no-cache-dir --no-build-isolation pyTMHMM==1.3.6; \
    fi


FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    FLASK_DEBUG=False \
    MALLOC_ARENA_MAX=2 \
    MKL_NUM_THREADS=1 \
    DEBUG=False \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMEXPR_NUM_THREADS=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_DISABLE_CACHING=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONWARNINGS="ignore:.*torch_dtype.*:FutureWarning" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TMBED_MODEL_DIR=/var/app/data/tmbed-models \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    VIRTUAL_ENV=/opt/venv \
    LD_LIBRARY_PATH=/usr/local/lib \
    OPTIONAL_TM_TOOL_HOME=/opt/metamp-optional-tools \
    PATH="/opt/venv/bin:/opt/metamp-optional-tools/bin:/opt/metamp-optional-tools/wrappers:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    curl \
    git \
    libdrm2 \
    libeigen3-dev \
    libexpat1 \
    libcurl4-openssl-dev \
    libfontconfig1 \
    libgbm1 \
    libglib2.0-0 \
    libgomp1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpq5 \
    libstdc++6 \
    libzip-dev \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    nlohmann-json3-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /var/app /var/app/data /var/app/data/models /var/app/data/tmbed-models /var/app/logs /opt/metamp-optional-tools/bin /opt/metamp-optional-tools/wrappers /opt/metamp-optional-tools/packages /tmp/matplotlib /tmp/numba_cache && \
    ln -sfn /var/app/data/models /var/app/models && \
    chown -R appuser:appuser /var/app /opt/metamp-optional-tools /tmp/matplotlib /tmp/numba_cache

WORKDIR /var/app

COPY --from=builder /opt/venv /opt/venv

COPY app.py celery_app.py manage.py server.py requirements.txt ./
COPY config ./config
COPY database ./database
COPY migrations ./migrations
COPY public ./public
COPY src ./src
COPY utils ./utils
COPY vendor ./vendor
COPY serverConfig/fix_vegafusion_issues.py ./serverConfig/fix_vegafusion_issues.py
COPY serverConfig/fix_tmbed_issues.py ./serverConfig/fix_tmbed_issues.py
COPY .env.production ./.env.production
COPY .env.docker.deployment ./.env.docker.deployment

RUN cp -R /var/app/vendor/optional_tm_tools/. /opt/metamp-optional-tools/

RUN GEMMI_VERSION=0.7.0 && \
    cd /tmp && \
    curl -L -O https://github.com/project-gemmi/gemmi/archive/refs/tags/v${GEMMI_VERSION}.tar.gz && \
    tar -xzf v${GEMMI_VERSION}.tar.gz && \
    rm v${GEMMI_VERSION}.tar.gz && \
    cd gemmi-${GEMMI_VERSION} && \
    cmake -B build && \
    make -j"$(nproc)" -C build && \
    make -C build install && \
    rm -rf /tmp/gemmi-${GEMMI_VERSION}

RUN PUGIXML_VERSION=1.14 && \
    cd /tmp && \
    mkdir -p contrib && \
    cd contrib && \
    curl -L -O https://github.com/zeux/pugixml/archive/refs/tags/v${PUGIXML_VERSION}.tar.gz && \
    tar -xzf v${PUGIXML_VERSION}.tar.gz && \
    rm v${PUGIXML_VERSION}.tar.gz && \
    ln -sfn pugixml-${PUGIXML_VERSION} pugixml

RUN TMDET_VERSION=4.1.2 && \
    cd /tmp && \
    git clone --branch ${TMDET_VERSION} --depth 1 https://github.com/brgenzim/TmDet.git && \
    cd TmDet && \
    cmake -B build && \
    make -j"$(nproc)" -C build && \
    make -C build install && \
    ldconfig && \
    rm -rf /tmp/TmDet /tmp/contrib

RUN set -eu; \
    SIGNALP_PACKAGE_ROOT=/opt/metamp-optional-tools/packages; \
    SIGNALP_SRC=""; \
    SIGNALP_TMP=""; \
    if [ -d "${SIGNALP_PACKAGE_ROOT}/signalp-6-package" ]; then \
        SIGNALP_SRC="${SIGNALP_PACKAGE_ROOT}/signalp-6-package"; \
    else \
        SIGNALP_ARCHIVE="$(find "${SIGNALP_PACKAGE_ROOT}" -maxdepth 1 -type f \( -name 'signalp-6-package*.tar.gz' -o -name 'signalp6*.tar.gz' -o -name 'signalp*.tar.gz' \) | head -n 1 || true)"; \
        if [ -n "${SIGNALP_ARCHIVE}" ]; then \
            SIGNALP_TMP=/tmp/metamp-signalp-package; \
            rm -rf "${SIGNALP_TMP}"; \
            mkdir -p "${SIGNALP_TMP}"; \
            tar -xzf "${SIGNALP_ARCHIVE}" -C "${SIGNALP_TMP}"; \
            SIGNALP_SRC="$(find "${SIGNALP_TMP}" -maxdepth 2 -type d -name 'signalp-6-package' | head -n 1 || true)"; \
            if [ -z "${SIGNALP_SRC}" ] && [ -f "${SIGNALP_TMP}/setup.py" -o -f "${SIGNALP_TMP}/pyproject.toml" ]; then \
                SIGNALP_SRC="${SIGNALP_TMP}"; \
            fi; \
        fi; \
    fi; \
    if [ -n "${SIGNALP_SRC}" ]; then \
        python -m venv /opt/metamp-optional-tools/signalp_venv; \
        /opt/metamp-optional-tools/signalp_venv/bin/pip install --upgrade pip setuptools wheel; \
        /opt/metamp-optional-tools/signalp_venv/bin/pip install --no-cache-dir 'numpy<2' 'torch<2'; \
        /opt/metamp-optional-tools/signalp_venv/bin/pip install --no-cache-dir "${SIGNALP_SRC}"; \
        SIGNALP_DIR="$(/opt/metamp-optional-tools/signalp_venv/bin/python -c 'import os, signalp; print(os.path.dirname(signalp.__file__))')"; \
        mkdir -p "${SIGNALP_DIR}/model_weights"; \
        if [ -d "${SIGNALP_SRC}/models" ]; then \
            cp -r "${SIGNALP_SRC}/models/." "${SIGNALP_DIR}/model_weights/"; \
        fi; \
    fi; \
    rm -rf /tmp/metamp-signalp-package

RUN set -eu; \
    CCTOP_PACKAGE_ROOT=/opt/metamp-optional-tools/packages; \
    CCTOP_TARGET=/opt/metamp-optional-tools/cctop; \
    rm -rf "${CCTOP_TARGET}"; \
    if [ -d "${CCTOP_PACKAGE_ROOT}/cctop" ]; then \
        cp -R "${CCTOP_PACKAGE_ROOT}/cctop" "${CCTOP_TARGET}"; \
    else \
        CCTOP_ARCHIVE="$(find "${CCTOP_PACKAGE_ROOT}" -maxdepth 1 -type f \( -name 'cctop*.tgz' -o -name 'cctop*.tar.gz' \) | head -n 1 || true)"; \
        if [ -n "${CCTOP_ARCHIVE}" ]; then \
            tar -xzf "${CCTOP_ARCHIVE}" -C /opt/metamp-optional-tools; \
        fi; \
    fi; \
    if [ -d "${CCTOP_TARGET}/Lib" ] && [ -d "${CCTOP_TARGET}/Standalone" ] && [ -f /usr/local/include/hmmtop/hmmtop.h ] && [ -e /usr/local/lib/libhmmtop.so -o -e /usr/local/lib/libHMMTOP.so -o -e /usr/local/lib/libhmmtop_lib.so -o -e /usr/local/lib/libhmmtop_lib.a -o -e /usr/local/lib/hmmtop_lib ]; then \
        cd "${CCTOP_TARGET}/Lib" && \
        mkdir -p build && \
        cd build && \
        cmake .. && \
        make -j"$(nproc)" && \
        cd "${CCTOP_TARGET}/Standalone" && \
        mkdir -p build && \
        cd build && \
        cmake .. && \
        make -j"$(nproc)"; \
    fi; \
    if [ -d "${CCTOP_TARGET}" ]; then \
        find "${CCTOP_TARGET}" -type f \( -name "cctop" -o -name "*.sh" \) -exec chmod +x {} + || true; \
    fi

RUN python /var/app/serverConfig/fix_vegafusion_issues.py || true
RUN python /var/app/serverConfig/fix_tmbed_issues.py || true

RUN if [ -d /opt/venv/lib/python3.10/site-packages/tmbed ]; then \
        rm -rf /opt/venv/lib/python3.10/site-packages/tmbed/models && \
        ln -sfn /var/app/data/tmbed-models /opt/venv/lib/python3.10/site-packages/tmbed/models; \
    fi && \
    find /var/app/vendor /opt/metamp-optional-tools -type f \( -name "*.sh" -o -name "metamp-*" -o -name "metamp-run-*" \) -exec chmod +x {} + || true && \
    chown -R appuser:appuser /var/app/data/tmbed-models /var/app/vendor /opt/metamp-optional-tools

USER appuser

EXPOSE 8081

CMD ["sh", "-c", "mkdir -p /var/app/data/models/semi-supervised /var/app/data/tmbed-models /var/app/logs /tmp/matplotlib /tmp/numba_cache && exec gunicorn --workers ${GUNICORN_WORKERS:-1} --timeout ${GUNICORN_TIMEOUT:-180} --graceful-timeout ${GUNICORN_GRACEFUL_TIMEOUT:-60} --max-requests ${GUNICORN_MAX_REQUESTS:-200} --max-requests-jitter ${GUNICORN_MAX_REQUESTS_JITTER:-25} --worker-tmp-dir /dev/shm -k gevent --bind 0.0.0.0:8081 --access-logfile - --error-logfile - server:app"]
