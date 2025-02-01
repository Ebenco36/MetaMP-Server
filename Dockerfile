# Stage 1: Builder
FROM python:3.10-slim AS builder

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    NUMBA_DISABLE_CACHING=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/MPLCONFIGDIR/ \
    NUMBA_DEBUG=1

# Install only build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /var/app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --verbose\
    && pip install --no-cache-dir supervisor


# Ensure /tmp is writable
RUN mkdir -p /tmp/numba_cache && chmod -R 777 /tmp/numba_cache

RUN sed -i 's/@numba.njit/@numba.njit(cache=False)/g' /usr/local/lib/python3.10/site-packages/umap/layouts.py

# Stage 2: Final Image
FROM python:3.10-slim

# Environment variables
ENV FLASK_DEBUG=False \
    DEBUG=False \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive\
    NUMBA_DISABLE_CACHING=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/MPLCONFIGDIR/ \
    NUMBA_DEBUG=1

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libev-dev \
    libevent-dev \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    nginx \
    supervisor \
    gfortran \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /var/app

# Copy application code and dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Configure directories and permissions
RUN mv serverConfig/supervisord.conf /etc/supervisor/conf.d/supervisord.conf && \
    mv nginx/nginx.conf /etc/nginx/sites-available/mpvis.com && \
    ln -s /etc/nginx/sites-available/mpvis.com /etc/nginx/sites-enabled/ && \
    rm /etc/nginx/sites-enabled/default

# Ensure /tmp is writable
RUN mkdir -p /tmp/numba_cache && chmod -R 777 /tmp/numba_cache

RUN sed -i 's/@numba.njit/@numba.njit(cache=False)/g' /usr/local/lib/python3.10/site-packages/umap/layouts.py

# Make the entrypoint script executable
COPY serverConfig/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY .env.production .env

# Set the script to be executed when the container starts
ENTRYPOINT ["/entrypoint.sh"]

# Expose required ports
EXPOSE 8081 8090
