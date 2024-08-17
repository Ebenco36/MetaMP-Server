# Use an Ubuntu base image with the latest Python 3.11
FROM ubuntu:20.04

# setting environment variables
ENV FLASK_ENV=production \
    DEBUG=False \
    NUMBA_DISABLE_CACHING=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    libev-dev \
    libevent-dev \
    python3.11 \
    python3.11-venv \
    python3-pip \
    gcc \
    g++ \
    python3.11-dev \
    libc-dev \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    nginx \
    supervisor \
    build-essential \
    libpcre3-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Set the working directory in the container
WORKDIR /var/app

# Copy the rest of the application code into the container
COPY . /var/app

RUN pip install -r requirements.txt \
    && pip install apache-airflow[kubernetes] \
    && pip install supervisor

# Create Airflow directories and ensure the right permissions
RUN mkdir -p /var/app/airflow_home/logs/scheduler && \
    chown -R www-data:www-data /var/app /var/app/airflow_home && \
    chmod -R 775 /var/app/airflow_home && \
    mv serverConfig/supervisord.conf /etc/supervisor/conf.d/supervisord.conf && \
    mv nginx/nginx.conf /etc/nginx/sites-available/mpvis.com && \
    ln -s /etc/nginx/sites-available/mpvis.com /etc/nginx/sites-enabled/ && \
    rm /etc/nginx/sites-enabled/default


# Make the entrypoint script executable
COPY serverConfig/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the script to be executed when the container starts
ENTRYPOINT ["/entrypoint.sh"]

# Expose the port the app runs on
EXPOSE 8081 8090