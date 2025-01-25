#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e


airflow_logs_dir="/var/app/airflow_home/logs"

# Check if the logs directory exists, create it if not
if [ ! -d "$airflow_logs_dir" ]; then
    mkdir -p "$airflow_logs_dir"
    echo "Airflow logs directory created."
fi

# Set ownership and permissions for the logs directory
chown -R www-data:www-data "$airflow_logs_dir"
chmod -R 775 "$airflow_logs_dir"
echo "Permissions and ownership set for Airflow logs directory."

# You can also specifically handle the subdirectories if needed
chown -R www-data:www-data "$airflow_logs_dir/scheduler"
chmod -R 775 "$airflow_logs_dir/scheduler"

export AIRFLOW_HOME=/var/app/airflow_home
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="$DATABASE_URL_AIRFLOW"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__WEBSERVER__WARN_DEPLOYMENT_EXPOSURE=False
export AIRFLOW__CORE__DAGS_FOLDER=/var/app/airflow_home/dags
export AIRFLOW__LOGGING__BASE_LOG_FOLDER=/var/app/airflow_home/logs
export AIRFLOW__LOGGING__DAG_PROCESSOR_MANAGER_LOG_LOCATION=/var/app/airflow_home/logs/dag_processor_manager/dag_processor_manager.log
export AIRFLOW__WEBSERVER__CONFIG_FILE=/var/app/airflow_home/webserver_config.py
export AIRFLOW__SCHEDULER__CHILD_PROCESS_LOG_DIRECTORY=/var/app/airflow_home/logs/scheduler
export AIRFLOW__SCHEDULER__CHILD_PROCESS_LOG_DIRECTORY=/var/app/airflow_home/logs/scheduler
export AIRFLOW__CORE__PLUGINS_FOLDER=/var/app/airflow_home/plugins
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


 export MPLCONFIGDIR=/tmp/matplotlib
 mkdir -p $MPLCONFIGDIR
 chown www-data:www-data /tmp/matplotlib

# Set environment variable for Python path
# Set the PYTHONPATH
export PYTHONPATH="/var/app:$PYTHONPATH"

# Ensure the logs directory is writable
chmod -R 744 /var/app/logs

# touch /var/app/__init__.py
echo "Grant CHMOD"

log_dir="/var/app/logs"
log_file="${log_dir}/error.log"
access_log_file="${log_dir}/access.log"

# Check if the log directory exists, create it if it does not
if [ ! -d "$log_dir" ]; then
    touch -p "$log_dir"
    echo "Log directory created."
fi

# Set the correct permissions for the log directory
chmod 755 "$log_dir"
echo "Permissions set for log directory."

# Check if the log file exists, create it if it does not
if [ ! -f "$log_file" ]; then
    touch "$log_file"
    touch "$access_log_file"
    echo "Log file created."
fi
chmod 766 /var/app/logs/error.log
chmod 766 /var/app/logs/access.log


# Proceed with your setup or server start commands
echo "Starting services..."

# Ensure DBFile.sh is executable
chmod +x /var/app/serverConfig/DBFile.sh
echo "Copied DBFile.sh"

#chmod -R 777 /var/app/airflow_home

# Debugging: Final setup check
# Function to check if Nginx is running
check_nginx() {
    echo "Checking if Nginx is running..."
    if pgrep nginx > /dev/null 2>&1; then
        sudo service nginx restart
        echo "Nginx is running."
    else
        echo "Nginx is not running. Starting Nginx..."
        sudo service nginx start
        if [ $? -eq 0 ]; then
            echo "Nginx started successfully."
        else
            echo "Failed to start Nginx."
            exit 1
        fi
    fi
}

# Call the function to check Nginx
check_nginx

echo "Setup completed. Checking Nginx status:"
sudo service nginx status
sudo service nginx reload

nginx -t

# Configure and start Airflow services
airflow db migrate
airflow users create --role Admin --username admin --email admin@yourdomain.com --firstname Admin --lastname User --password securepassword
airflow variables set script_file_path "/var/app/serverConfig/DBFile.sh"

sudo chmod -R 777 /var/app/airflow_home/logs


# Start Supervisor
supervisord -c /etc/supervisor/supervisord.conf

# Re-read and update Supervisor configurations to apply changes
echo "Updating Supervisor configuration..."
supervisorctl reread
supervisorctl update
#sudo supervisorctl restart airflow_scheduler
#sudo supervisorctl restart airflow_webserver
supervisorctl restart all

# Output the status of services to confirm they're all running as expected
supervisorctl status
# Check the status of all programs under Supervisor's management
echo "Checking status of all managed services:"
supervisorctl status

# Start or restart services via Supervisor
# supervisorctl restart all

flask sync-protein-database
