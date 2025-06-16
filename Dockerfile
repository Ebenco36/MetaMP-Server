# Stage 1: Builder
FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Add PostgreSQL 16 repo
RUN apt-get update && apt-get install -y wget gnupg lsb-release && \
    echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gcc \
    gfortran \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    redis-server \
    postgresql-16 \
    postgresql-client-16 \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /var/app

COPY requirements.txt .

RUN pip install --upgrade pip==25.1.1 setuptools wheel --verbose && \
    pip install Babel==2.13.1 --no-deps && \
    pip install -r requirements.txt supervisor --no-deps --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Install TMbed
RUN git clone --depth=1 https://github.com/BernhoferM/TMbed.git tmbed && \
    cd tmbed && \
    pip install . --no-cache-dir --timeout=60 --retries=10

# Stage 2: Runtime
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    FLASK_DEBUG=False \
    DEBUG=False \
    NUMBA_DISABLE_CACHING=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/MPLCONFIGDIR/ \
    NUMBA_DEBUG=1

# Add PostgreSQL 16 repo
RUN apt-get update && apt-get install -y wget gnupg lsb-release && \
    echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libev-dev \
    libevent-dev \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    redis-server \
    postgresql-17 \
    postgresql-client-17 \
    postgresql-contrib-17 \
    nginx \
    supervisor \
    gfortran \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


WORKDIR /var/app

# Copy from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /var/app /var/app

# Copy project files
COPY . .

# Patch UMAP for Numba compatibility
RUN python -c "import umap, os; path=os.path.join(os.path.dirname(umap.__file__), 'layouts.py'); \
    data=open(path).read().replace('@numba.njit', '@numba.njit(cache=False)'); open(path, 'w').write(data)"

# Copy dump file for restoration in entrypoint
COPY all_tables.dump /var/app/initdb/all_tables.dump

# Configure supervisor
RUN mv serverConfig/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Configure nginx
RUN mv nginx/nginx.conf /etc/nginx/sites-available/mpvis.com && \
    ln -s /etc/nginx/sites-available/mpvis.com /etc/nginx/sites-enabled/ && \
    rm /etc/nginx/sites-enabled/default

# Entrypoint
COPY serverConfig/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY .env.production .env

ENTRYPOINT ["/entrypoint.sh"]
EXPOSE 8081 8090
