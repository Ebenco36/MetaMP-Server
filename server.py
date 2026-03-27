import os
from app import app
import logging
from logging.handlers import RotatingFileHandler

port = os.getenv("FLASK_RUN_PORT")
host = os.getenv("FLASK_RUN_HOST")
app.config.from_object(os.getenv('APP_SETTINGS'))

# Configure logging
if not app.debug:
    # Set the log level
    log_level = logging.INFO

    # Create a file handler that logs messages to a file
    log_file = './error.log'
    file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 10, backupCount=5)
    file_handler.setLevel(log_level)

    # Create a log formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add the file handler to the app's logger
    app.logger.addHandler(file_handler)
    
if __name__ == '__main__':
    app.run(host=host, port=port)
