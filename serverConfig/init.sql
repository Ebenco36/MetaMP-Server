-- Create the first database
CREATE DATABASE mpvis_airflow_db;
GRANT ALL PRIVILEGES ON DATABASE mpvis_airflow_db TO mpvis_user;
-- Additional setup can include creating roles, schemas, or preloading data
-- For example, setting up a specific schema in mydatabase1
\c mpvis_airflow_db