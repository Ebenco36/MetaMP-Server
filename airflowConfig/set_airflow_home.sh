#!/bin/bash
# This script sets the AIRFLOW_HOME environment variable and ensures no duplicates
# File path
BASHRC="/root/.bashrc"
ENVIRONMENT="/etc/environment"

# Set the root directory
ROOT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"

# Load the appropriate .env file based on FLASK_ENV
if [ "$FLASK_ENV" = "production" ]; then
    if [ -f "$ROOT_DIR/.env.production" ]; then
        export $(grep -v '^#' "$ROOT_DIR/.env.production" | xargs)
    else
        echo "Error: .env.production file not found."
        exit 1
    fi
else
    if [ -f "$ROOT_DIR/.env.development" ]; then
        export $(grep -v '^#' "$ROOT_DIR/.env.development" | xargs)
    else
        echo "Error: .env.development file not found."
        exit 1
    fi
fi

# Function to add a line to a file if the line doesn't already exist
add_line_if_not_exists() {
    local file=$1
    local line=$2
    grep -qxF "$line" "$file" || echo "$line" >> "$file"
}

# For a specific user's .bashrc
add_line_if_not_exists $BASHRC "export AIRFLOW_HOME=/var/app/airflow_home"
add_line_if_not_exists $BASHRC "export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=$DATABASE_URL_AIRFLOW"
add_line_if_not_exists $BASHRC "export AIRFLOW__CORE__LOAD_EXAMPLES=False"
add_line_if_not_exists $BASHRC "export AIRFLOW__WEBSERVER__WARN_DEPLOYMENT_EXPOSURE=False"

# For global environment settings
add_line_if_not_exists $ENVIRONMENT "AIRFLOW_HOME=\"/var/app/airflow_home\""
