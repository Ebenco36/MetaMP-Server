# MetaMP Application

MetaMP is a web application designed to dynamically curate structure determination metadata for resolved Membrane Proteins (MPs). It provides several interactive views to explore enriched MP structure data and associated metadata, supporting advanced analysis and data-driven decision-making.

## Table of Contents

1. [About the Project](#about-the-project)
2. [Abstract](#abstract)
3. [Key Features](#key-features)
4. [Built On](#built-on)
5. [Requirements](#requirements)
6. [Jupyter Notebook Test](#jupyter-notebook-test)
7. [Getting started](#getting-started)
8. [Installation and Running](#installation-and-running)
9. [Project Folder Structure](#project-folder-structure)
10. [Application Setup and Configuration](#application-setup-and-configuration)
11. [Seeding Process](#seeding-process)
12. [Running the Application](#running-the-application)
    - [Running with Flask Development Server](#running-with-flask-development-server)
    - [Running with Gunicorn (macOS/Linux)](#running-with-gunicorn-macoslinux)
    - [Running with Waitress (Windows)](#running-with-waitress-windows)
13. [Using Docker](#using-docker)
14. [Flask Database Commands](#flask-database-commands)
15. [Working with Redis](#working-with-redis)
16. [Working with RabbitMQ](#working-with-rabbitmq)
17. [Performance Considerations](#performance-considerations)
18. [Data Availability Statement](#data-availability-statement)
19. [License](#license)
20. [Contact](#contact)
21. [Acknowledgments](#acknowledgments)

## About the Project

**MetaMP** allows users to visualize, analyze, and curate MP structure metadata through various interactive views. These views provide insights into MP structures resolved by different experimental methods, discrepancies in data across databases, outlier detection, and more. The application is modular, with each component having its own route, view, service, and model files.

## Abstract

Structural biology has made significant progress in the determination of membrane proteins, leading to a remarkable increase in the number of structures accessible from several dedicated databases.
Despite these advances, the lack of integrated databases results in inaccuracies, inconsistencies, and duplication of effort, hindering a coherent understanding of these structures. 
The inherent complexity of membrane protein structures, coupled with challenges such as missing data and computational barriers from disparate sources, underscores the need for improved database integration.
To address this gap in membrane protein structure analysis, we present MetaMP, the first computational application to use machine learning to categorize membrane protein structures. 
In addition to curating structural determination methods, MetaMP enriches membrane protein data with comprehensive and accurate metadata, all enhanced by a user-friendly landing page and seven interactive views.
MetaMP provides an unprecedented overview of the discrepancies between membrane protein databases. 
A first user evaluation confirmed the effectiveness of training on MetaMP, highlighting its advantages across tasks of varying difficulty, with no significant effect of completion speed on accuracy. 
This underscores the need to assist experts in critical analytical tasks, such as classifying newly published structures and detecting outliers. 
MetaMP was validated by both professionals and academics, focusing on the calculation of summary statistics and the identification of outliers. 
Participants completed three tasks in less than 10 minutes and showed significant improvements from training to testing, particularly in the early summary statistics and the identification of outliers. 
MetaMP not only finds and resolves 77% of the data discrepancies compared to expert validation, it also predicts the correct class of newly resolved membrane proteins 98% of the time.
In summary, MetaMP is an awaited resource that synthesizes current knowledge and improves the understanding of membrane protein structures, enabling more informed and rigorous scientific investigations.


![MetaMP Architecture](public/MetaMPArchitecture.png)

## Key Features

### 1. Overview View
Provides high-level visualizations of MP structures, categorized by experimental methods, taxonomic domains, and groups. It includes interactive charts for exploring trends such as the cumulative sum of resolved MP structures over time.

### 2. Summary Statistics View
Offers an on-demand analysis of MP structure metadata. A bar chart displays the cumulative sum of resolved structures by experimental method, with a table showing detailed data points. Users can dynamically filter data by various attributes, such as molecular type, resolution, or growth method, to update the visualization and table.

### 3. Data Discrepancy View
Identifies and displays discrepancies between database entries for MP structures. It provides a line chart to visualize discrepancies over time and a detailed table for inspecting metadata differences. A form is available for users to provide feedback to resolve discrepancies.

### 4. Outlier Detection View
Focuses on detecting outliers in MP structure data using Principal Component Analysis (PCA) and DBSCAN clustering. It includes a whisker plot and Scatter Plot Matrix (SPLOM) for analyzing outliers and understanding data variability. This view enables users to interactively examine anomalies for further investigation or correction.

### 5. Database View
Features a customizable tabular interface for exploring the enriched MetaMP database. Users can filter, sort, and export data based on criteria such as taxonomic domain, experimental method, and resolution. This view facilitates detailed analysis, comparison, and reporting.

### 6. Exploration View
Supports interactive exploration of MP structure data through a dynamic dashboard with customizable filters and visualization options. Users can analyze relationships between attributes like molecular type and experimental method to identify patterns and generate insights.

### 7. Grouping View
Utilizes AI to suggest categorizations of MP structures into predefined groups based on attributes. Experts review and refine these AI-generated groupings to ensure accuracy, enabling efficient and nuanced data curation.

## Built On

<!-- - **Python**: Programming language used for backend development. -->
- **Flask**: Web framework used for creating the server-side application.
- **Docker**: Containerization platform for easy deployment.
- **Redis**: In-memory data structure store for caching.
- **RabbitMQ**: Message-broker software for handling asynchronous tasks.

## Requirements

- Python 3.9+
- Flask 2.0+
- Docker (Docker version 27.4.0, build bde2b89)
- Redis
- RabbitMQ
- PostgreSQL


## Jupyter Notebook Test

### Overview
We have developed a Jupyter Notebook, **JupyterNotebookTest.ipynb**, to test and visualize various implementations, including machine learning techniques such as Dimensionality Reduction and Semi-Supervised Learning. This notebook serves as an interactive tool to explore our methodologies and outcomes in detail.

### Key Features
1. **Visualization of Results**: The notebook includes a variety of interactive charts and visualizations designed to effectively communicate insights derived from our analyses.
2. **Implementation Testing**: Evaluate the performance of Dimensionality Reduction and Semi-Supervised Learning models with provided sample datasets.

### Accessing the Required Data
To utilize the notebook, you will need to download the required dataset.

[Click here to download the dataset](https://drive.google.com/file/d/1L7mArSRHRbpp6hq0z74XpyHYvJBM8Kjt/view?usp=drive_link)

After downloading the dataset, ensure that the file is saved to the specified filepath mentioned in the notebook to guarantee seamless operation.

### Instructions
1. Download the dataset using the link above.
2. Open the Jupyter Notebook **JupyterNotebookTest.ipynb** and verify that the filepath to the dataset matches your local setup.
3. Follow the instructions in the notebook to execute the code cells and interact with the visualizations.

We aim to provide a streamlined and intuitive experience for exploring our machine learning implementations and results.



## Getting started
To minimize installation issues and version conflicts, we have deployed a Docker image for both the frontend and backend, ensuring that everyone can easily test the complete application.


### Environment Setup

To set up the development environment, follow these steps:

1. Install Python (3.8 or higher) from the [official Python website](https://www.python.org/) - (Unstable Installation).
2. Install Docker and Docker Compose from the [official Docker website](https://www.docker.com/).





#### Supported Platforms

Our Docker images are built to support the following platforms:

##### Supported Architectures

1. **Linux/amd64**  
   - 64-bit architecture for most modern Linux desktops and servers.
   - Commonly used on Intel and AMD processors.

2. **Linux/arm64**  
   - 64-bit ARM architecture.
   - Supported on devices such as Raspberry Pi (64-bit models), ARM-based servers, and Apple Silicon (M1/M2).

##### Compatibility

- **Linux/amd64**: Compatible with distributions like Ubuntu, Debian, CentOS, Fedora, and others running on 64-bit Intel or AMD processors.

- **Linux/arm64**: Compatible with ARM-based Linux distributions, including Ubuntu ARM, Debian ARM, and Raspberry Pi OS (64-bit).

## Installation and Running
### Stable
The easiest way to install the tool is by using our latest Docker image:

1. Install Docker and Docker Compose from the [official Docker website](https://www.docker.com/).
2. Run the command below from your Desktop or any location you prefer.

#### Docker Deployment for Backend and Frontend
Run the following command from your terminal:
```bash
$ ( [ -d "metamp_app" ] && rm -rf metamp_app ); mkdir metamp_app && cd metamp_app && \
  curl -o .env.docker.deployment https://raw.githubusercontent.com/Ebenco36/MetaMP-Server/refs/heads/main/.env.docker.deployment && \
  curl -o docker-compose.yml https://raw.githubusercontent.com/Ebenco36/MetaMP-Server/refs/heads/main/docker-compose.yml && \
  docker compose --env-file .env.docker.deployment -f docker-compose.yml up -d
```
![Docker-Deployment-for-Backend](public/backend-completion-status.png)

Once the command above has been successfully executed (as illustrated in the image above):

- The backend will run at: [http://localhost:5400/api/v1/dashboard](http://localhost:5400/api/v1/dashboard)
- The frontend will be accessible at: [http://localhost/#/dashboard](http://localhost/#/dashboard)

### Note:
You can use the localhost IP or your own custom hostname to access the app.

### Unstable
*Caution! Use at your own

Step 1: Clone the repository
    ```bash
        git clone https://github.com/Ebenco36/MPVIS-V2.git
        cd MetaMP
    ```

Step 2: Set up a virtual environment:
   ```bash
        python -m venv venv_metamp
        source venv_metamp/bin/activate  # On Windows use `venv_metamp\Scripts\activate`
   ```

Step 3: Install Dependencies:
    ```bash
        pip install -r requirements.txt
    ```

Step 4: Configure Environment Variables:
    Copy .env content from our example env.
    ```bash
        cp env_example .env
    ```

## Project Folder Structure
```bash
/MetaMP
|-- /config         # Here is where we have different environment configuration
|-- /database       # Database connection class is implemented here
|-- /datasets       # A directory for processed datasets
|-- /logs           # Save application logs
|-- /migrations     # MetaMP migration folder
|-- /models         # machine learning models are saved here
|-- /nginx          # Nginx configuration for docker container is here
|-- /public         # Public saves image and other static content
|-- /serveConfig    # This folder contains files required for docker container to start properly
|-- /src            # MetaMP implementations are here
    |-- /app (Module)
    |   |-- /routes       # Flask route files for each view
    |   |-- /views        # HTML and Jinja templates for views
    |   |-- /services     # Business logic and service files
    |   |-- /models       # Database models
|-- /tests          # test are written here
|-- /utils          # MetaMP utils
|-- Dockerfile        # Docker configuration
|-- docker-compose-dev.yml        # Dev Containers
|-- docker-compose.yml # Deployed Containers
|-- README.md         # Documentation file
|-- requirements.txt  # Python dependencies
|-- env_example      # Example environment configuration
|-- /manage.py, app.py and server.py # Server implementation
```

## Application Setup and Configuration

1. Configure the .env file with your database and other settings.
2. Run migrations and seed to set up the database schema:
    ```bash
    flask sync-protein-database
    ```
## Running the Application

### Running with Flask Development Server
To run the application using Flask's development server:
    ```bash
    python manage.py runserver
    ```
### Running with Gunicorn (macOS/Linux)
To run the application using Gunicorn:
    ```bash
    gunicorn -w 4 --graceful-timeout 30 -k gevent -b 0.0.0.0:5400 --reload server:app
    ```
### Running with Waitress (Windows)
To run the application using Waitress:
    ```bash
    waitress-serve --port=5000 app:app
    ```
## Using Docker
1. Build the Docker image:
    ```bash
    docker build -t mpvis .
    ```
2. Start the application using Docker Compose:
    ```bash
    docker-compose -f docker-compose.yml up -d
    ```
3. Free up space by pruning Docker system:
    ```bash
    docker system prune -a
    ```
## Flask Database Commands
1. flask db init: Initialize the database.
2. flask db migrate: Create a new migration.
3. flask db upgrade: Apply migrations.
4. flask db downgrade: Revert migrations.


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
## Performance Considerations
To optimize the performance of MetaMP:

1. Use Redis for caching frequently accessed data.
2. Utilize RabbitMQ for handling asynchronous tasks.
3. Optimize database queries and use indexes where necessary.
4. Use connection pooling and load balancing for handling high traffic.


## Data Availability Statement

The data supporting the findings of this manuscript are derived from publicly available databases widely used in membrane protein research. These sources include:

- **MPstruc**: http://blanco.biomol.uci.edu/mpstruc/
- **OPM (Orientations of Proteins in Membranes)**: https://opm.phar.umich.edu/
- **PDB (Protein Data Bank)**: https://www.rcsb.org/
- **UniProt**: https://www.uniprot.org/uniprotkb

These datasets provide detailed information on membrane protein structures and related characteristics. All data are publicly accessible and do not contain sensitive or personally identifiable information.

The aggregated and processed data used in this study will be made available upon reasonable request to the corresponding author. Requests must include details about the intended use of the data. For further inquiries or access requests, please contact the corresponding author.

## License

Licensed under the GNU General Public License, Version 3.0 ([LICENSE](./LICENSE) or https://www.gnu.org/licenses/gpl-3.0.en.html)

### Contribution

Any contribution intentionally submitted for inclusion in the work by you, shall be licensed under the GNU GPLv3.


## Contact
ebenco94@gmail.com, georgeshattab@gmail.com


## Acknowledgments
1. Flask for the web framework.
2. Docker for containerization.
3. Redis for caching.
4. RabbitMQ for message brokering.