#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export NUMBA_DISABLE_CACHING=1
export MPLCONFIGDIR=/tmp/matplotlib
export FLASK_ENV=production

mkdir -p $MPLCONFIGDIR
chown www-data:www-data /tmp/matplotlib
chown -R www-data:www-data /var/app
chmod -R 750 /var/app

# Set environment variable for Python path
export PYTHONPATH="/var/app:$PYTHONPATH"

# Ensure the logs directory is writable
chmod -R 744 /var/app/logs

log_dir="/var/app/logs"
log_file="${log_dir}/error.log"
access_log_file="${log_dir}/access.log"

# Check if the log directory exists, create it if it does not
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
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
chmod 766 "$log_file"
chmod 766 "$access_log_file"


# Create and set permissions for the safe directory for Celery beat schedule
safe_dir="/var/app/celery-beat"
mkdir -p $safe_dir
chown www-data:www-data $safe_dir
chmod 766 $safe_dir
echo "Safe directory for Celery beat schedule prepared."

# If the schedule file exists, ensure it also has appropriate permissions
if [ -f ./celery-beat/celerybeat-schedule ]; then
    chown www-data:www-data ./celery-beat/celerybeat-schedule
    chmod 766 ./celery-beat/celerybeat-schedule
fi

echo "Updating VegaFusion runtime.py file..."
python /var/app/serverConfig/fix_vegafusion_issues.py

# Proceed with setup or server start commands
echo "Starting services..."

# Ensure DBFile.sh is executable
chmod +x /var/app/serverConfig/DBFile.sh
echo "Copied DBFile.sh"

# Function to check if Nginx is running
check_nginx() {
    echo "Checking if Nginx is running..."
    if pgrep nginx > /dev/null 2>&1; then
        service nginx restart
        echo "Nginx is running."
    else
        echo "Nginx is not running. Starting Nginx..."
        service nginx start
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
service nginx status
service nginx reload

nginx -t

# Execute the .sh script
echo "Executing script..."
/var/app/serverConfig/DBFile.sh


# Start Supervisor
supervisord -c /etc/supervisor/supervisord.conf

# Re-read and update Supervisor configurations to apply changes
echo "Updating Supervisor configuration..."
supervisorctl reread
supervisorctl update
supervisorctl restart all

# Output the status of services to confirm they're all running as expected
echo "Checking status of all managed services:"
supervisorctl status
