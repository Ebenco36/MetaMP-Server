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
    if [ "$INCLUDE_ML" = "true" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi


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
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    curl \
    libdrm2 \
    libexpat1 \
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
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /var/app /var/app/data /var/app/data/models /var/app/data/tmbed-models /var/app/logs /tmp/matplotlib /tmp/numba_cache && \
    ln -sfn /var/app/data/models /var/app/models && \
    chown -R appuser:appuser /var/app /tmp/matplotlib /tmp/numba_cache

WORKDIR /var/app

COPY --from=builder /opt/venv /opt/venv

COPY app.py celery_app.py manage.py server.py requirements.txt ./
COPY config ./config
COPY database ./database
COPY migrations ./migrations
COPY public ./public
COPY src ./src
COPY utils ./utils
COPY serverConfig/fix_vegafusion_issues.py ./serverConfig/fix_vegafusion_issues.py
COPY serverConfig/fix_tmbed_issues.py ./serverConfig/fix_tmbed_issues.py
COPY .env.production ./.env.production
COPY .env.docker.deployment ./.env.docker.deployment

RUN python /var/app/serverConfig/fix_vegafusion_issues.py || true
RUN python /var/app/serverConfig/fix_tmbed_issues.py || true

RUN if [ -d /opt/venv/lib/python3.10/site-packages/tmbed ]; then \
        rm -rf /opt/venv/lib/python3.10/site-packages/tmbed/models && \
        ln -sfn /var/app/data/tmbed-models /opt/venv/lib/python3.10/site-packages/tmbed/models; \
    fi && \
    chown -R appuser:appuser /var/app/data/tmbed-models

USER appuser

EXPOSE 8081

CMD ["sh", "-c", "mkdir -p /var/app/data/models/semi-supervised /var/app/data/tmbed-models /var/app/logs /tmp/matplotlib /tmp/numba_cache && exec gunicorn --workers ${GUNICORN_WORKERS:-1} --timeout ${GUNICORN_TIMEOUT:-180} --graceful-timeout ${GUNICORN_GRACEFUL_TIMEOUT:-60} --max-requests ${GUNICORN_MAX_REQUESTS:-200} --max-requests-jitter ${GUNICORN_MAX_REQUESTS_JITTER:-25} --worker-tmp-dir /dev/shm -k gevent --bind 0.0.0.0:8081 --access-logfile - --error-logfile - server:app"]
