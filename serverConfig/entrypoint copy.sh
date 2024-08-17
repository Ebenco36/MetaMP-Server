#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e


# Set environment variable for Python path
export PYTHONPATH=/var/app

# Ensure the logs directory is writable
chmod -R 744 /var/app/logs

chmod +x /var/app/serverConfig/DBFile.sh

# Copy and configure system service files and nginx configuration if they exist
if [ -f /var/app/serverConfig/services/mpvis-app.service ]; then
  rm -f /etc/systemd/system/mpvis-app.service
  cp -f -p /var/app/serverConfig/services/mpvis-app.service /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable mpvis-app.service
  systemctl start mpvis-app.service
  systemctl status mpvis-app.service
else
  echo "Error: service config not found."
  exit 1
fi

if [ -f /var/app/nginx/nginx.conf ]; then
  rm -f /etc/nginx/sites-available/mpvis.com
  rm -f /etc/nginx/sites-enabled/mpvis.com
  cp -f -p /var/app/nginx/nginx.conf /etc/nginx/sites-available/mpvis.com
  ln -s /etc/nginx/sites-available/mpvis.com /etc/nginx/sites-enabled/
  systemctl restart nginx
  nginx -t && systemctl reload nginx
  ufw allow 'Nginx Full'
  ufw reload
else
  echo "Error: nginx.conf not found."
  exit 1
fi

# Debugging: Final setup check
echo "Setup completed. Checking Services and Nginx status:"
systemctl status nginx

airflow config list --defaults > "${AIRFLOW_HOME}/airflow.cfg"
chmod +x /var/app/airflowConfig/set_airflow_home.sh
/var/app/airflowConfig/set_airflow_home.sh
rm -f /etc/systemd/system/airflow-scheduler.service
rm -f /etc/systemd/system/airflow-webserver.service
cp -f -p /var/app/airflowConfig/services/airflow-scheduler.service /etc/systemd/system/
cp -f -p /var/app/airflowConfig/services/airflow-webserver.service /etc/systemd/system/
airflow db init
airflow users create --role Admin --username admin --email admin@yourdomain.com --firstname Admin --lastname User --password securepassword
if [[ $(systemctl is-active airflow-webserver.service) == "active" ]]; then
  echo "Airflow webserver is running, restarting service..."
  systemctl restart airflow-webserver.service
else
  echo "Airflow webserver is not active, starting service..."
  systemctl enable airflow-webserver.service
  systemctl start airflow-webserver.service
fi

if [[ $(systemctl is-active airflow-scheduler.service) == "active" ]]; then
  echo "Airflow scheduler is running, restarting service..."
  systemctl restart airflow-scheduler.service
else
  echo "Airflow scheduler is not active, starting service..."
  systemctl enable airflow-scheduler.service
  systemctl start airflow-scheduler.service
fi
systemctl restart mpvis-app.service

# supervisord -c /etc/supervisor/conf.d/supervisord.conf
#cd /var/app

rm __init__.py

flask sync-protein-database
