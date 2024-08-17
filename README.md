# MetaMP README

## Overview

MetaMP is a platform designed for managing and visualizing membrane protein data. This documentation provides all the necessary commands and guidelines for setting up the development environment, running the application, and managing the associated database migrations.

## Table of Contents

1. [Features](#features)
2. [Built With](#built-with)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Running the App](#running-the-app)
6. [Usage](#usage)
7. [Environment Setup](#environment-setup)
8. [Installing Dependencies](#installing-dependencies)
9. [Database Migrations](#database-migrations)
10. [Running the Application](#running-the-application)
11. [Working with Redis](#working-with-redis)
12. [Working with RabbitMQ](#working-with-rabbitmq)
13. [Docker Commands](#docker-commands)
15. [Common Issues & Fixes](#common-issues--fixes)
16. [Project Structure](#project-structure)
17. [Development Stages Overview](#development-stages-overview)
18. [Recent Updates](#recent-updates)
19. [Hosting the Platform](#hosting-the-platform)

## Features
- Gunicorn server setup
- Database migration setup for SQLAlchemy Models
- JWT authentication at `<your-host>/api/v1/login`
- URL to register your admin and users at `<your-host>/api/v1/signup`
- Provisioned route file for all your routes with admin blueprints
- Test environment setup

## Built With

This App was developed with the following stack:

- Python
- Flask
- Flask-RESTful
- Postgres DB
- Gunicorn Web Server

## Requirements
- Python 3.6+
- Python pip
- Postgres SQL

## Installation
- Fork this repository.
- Create a `.env` file as shown in the `env_example` file.
- Set up your database.
- On the terminal, `cd` into the app folder.
- Run `pip install -r requirements.txt` to install required modules.
- Run `python manage.py db init` to set up Alembic migrations.
- Run `python manage.py db migrate -m='<your migration message>'` to create migration files.
- Then run `python manage.py db upgrade` to create tables.

## Running the App
- On the terminal, run `gunicorn main:app`.
- To run the app on a specific port, use `gunicorn -127.0.0.1:<port> main:app`.

## Usage
- `src/api/resources` — Flask-RESTful resources for your project.
- `src/models` — SQLAlchemy models and schema.
- `src/routes/api` — Contains all your route definitions.
- `src/utils` — Contains validations, security, and helper files.
- `src/middlewares` — Define your middleware files here.
- You can modify the app to suit your needs.
- Happy usage!

## Environment Setup

### Conda Environment (Linux/MacOS)
1. Create a virtual environment:
    ```bash
    conda create -n venv_mpvis_flask python=3.9
    ```
2. Activate the environment:
    ```bash
    conda activate venv_mpvis_flask
    ```

### Conda Environment (Windows)
1. Create a virtual environment:
    ```bash
    conda create -n _wvenv_mpvis python=3.9
    ```
2. Activate the environment:
    ```bash
    conda activate _wvenv_mpvis
    ```

### Python Virtual Environment (Linux/MacOS)
1. Create a virtual environment:
    ```bash
    python3.9 -m venv .venv_mpvis
    ```
2. Activate the environment:
    ```bash
    source .venv_mpvis/bin/activate
    ```

### Python Virtual Environment (Windows)
1. Activate the environment:
    ```bash
    .venv_mpvis_\Scripts\activate
    ```

## Installing Dependencies

1. Install necessary packages:
    ```bash
    pip install -r requirements.txt
    ```
2. Install additional required modules:
    ```bash
    pip install python-dotenv pyjwt pillow hupper hdbscan missingno bioseq biopython
    ```

## Database Migrations

1. Initialize Alembic migrations:
    ```bash
    python manage.py db init
    ```
2. Create migration files:
    ```bash
    python manage.py db migrate -m "<your migration message>"
    ```
3. Apply migrations to the database:
    ```bash
    python manage.py db upgrade
    ```
4. Running database migration and seeding all at once
    ```
    flask sync-protein-database
    flask sync-question-with-database
    flask sync-system_admin-with-database
    flask sync-feedback-questions-with-database
    ```

### Flask Database Commands
- Stamp the current revision to the latest:
    ```bash
    flask db stamp head
    ```
- Create a new migration:
    ```bash
    flask db migrate -m "Reset database"
    ```
- Apply the migration:
    ```bash
    flask db upgrade
    ```

## Running the Application

### Using Gunicorn
1. Start the application (Linux/MacOS):
    ```bash
    gunicorn main:app --reload
    ```
2. Start the application on a specific port:
    ```bash
    gunicorn -127.0.0.1:<port> main:app --reload
    ```

### Using Waitress (Windows)
1. Start the application:
    ```bash
    waitress-serve --listen=127.0.0.1:8000 main:app
    ```

## Working with Redis

1. Start Redis:
    ```bash
    brew services start redis
    ```
2. Check Redis service status:
    ```bash
    brew services info redis
    ```
3. Stop Redis:
    ```bash
    brew services stop redis
    ```
4. Connect to Redis CLI:
    ```bash
    redis-cli
    ```

## Working with RabbitMQ

1. Start RabbitMQ service:
    ```bash
    brew services start rabbitmq
    ```
2. If you do not need a background service, run:
    ```bash
    CONF_ENV_FILE="/opt/homebrew/etc/rabbitmq/rabbitmq-env.conf" /opt/homebrew/opt/rabbitmq/sbin/rabbitmq-server
    ```

### RabbitMQ Management Plugin
- Management Plugin is enabled by default at: `http://localhost:15672`
- Default credentials:
    - Username: username
    - Password: password

## Docker Commands

1. Build the Docker image:
    ```bash
    docker build -t mpvis .
    ```
2. Tag and push the Docker image:
    ```bash
    docker tag mpvis-flask-app ebenco36/mpvis:latest
    docker push ebenco36/mpvis:latest
    ```
3. Pull the Docker image:
    ```bash
    docker pull ebenco36/mtest_docker:latest
    ```
4. Start the application using Docker Compose:
    ```bash
    docker-compose -f docker2-compose.yml up -d
    ```
5. Free up space by pruning Docker system:
    ```bash
    docker system prune -a
    ```

## Common Issues & Fixes

1. Fix most issues by reinstalling dependencies:
    ```bash
    pip install --upgrade --force-reinstall -r requirements.txt
    ```
2. For attribute errors, upgrade `attrs`:
    ```bash
    pip install --upgrade attrs
    ```
3. To handle MySQL issues:
    ```bash
    export LDFLAGS="-L/opt/homebrew/opt/mysql-client/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/mysql-client/include"
    export MYSQLCLIENT_CFLAGS=`mysql_config --cflags`
    export MYSQLCLIENT_LDFLAGS=`mysql_config --libs`
    ```

## Project Structure

- `src/api/resources` - Contains Flask-RESTful resources.
- `src/models` - SQLAlchemy models and schema.
- `src/routes/api` - Contains all route definitions.
- `src/utils` - Contains validations, security, and helper files.
- `src/middlewares` - Define middleware files here.

You can modify the app to suit your needs.

## Development Stages Overview

**Stage 1:** Focus on MPstruct analysis and visualization (e.g., tree diagrams, visuals).

**Stage 2:** Focus on enriched MPstruct data, excluding MPstruct itself (e.g., summary statistics, relevant visualizations).

**Stage 3:** Integration of both curated MPstruct and enriched data:
  - Analyze and visualize data with respect to resolution methods.
  - Validate new/unclassified MP structures through clustering.
  - Outlier detection.

## Recent Updates

### Issue Discovery
Some membrane proteins listed on MPstruct cannot be found anywhere, including PDB (e.g., 7ROW, 7UUV).

### Problematic Entries
Certain codes in MPstruct have been updated on PDB, causing discrepancies (e.g., 5W7L replaced by 8G1N).

## Hosting the Platform

To host the platform for testing or production, use the following commands:
- Convert images to PDF for reports:
    ```bash
    convert ourLogo.png dashboard.png summaryStats.png details.png trainingStart.png Answer1.png TrainingQuestion.png testSummary.png Dimensionality.png trainingEnd.png table.png exploration1.png exploration2.png ML.png ML2.png about.png output.pdf
    ```
