import os
def str_to_bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")
    
class Config(object):
    DEBUG = False
    TESTING = False
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_SILENCE_UBER_WARNING=1
    SQLALCHEMY_COMMIT_ON_TEARDOWN=False
    
    SECRET_KEY=os.getenv('SECRET_KEY')
    JWT_SECRET_KEY=os.getenv('JWT_SECRET_KEY')
    SECURITY_PASSWORD_SALT=os.getenv('SECURITY_PASSWORD_SALT')
    FLASK_APP=os.getenv('FLASK_APP')
    FLASK_RUN_HOST=os.getenv('FLASK_RUN_HOST')
    FLASK_RUN_PORT=os.getenv('FLASK_RUN_PORT')
    #APP CONFIG
    DB_USER=os.getenv('DB_USER')
    DB_PASSWORD=os.getenv('DB_PASSWORD')
    DB_HOST=os.getenv('DB_HOST')
    DB_PORT=os.getenv('DB_PORT')
    DB_NAME=os.getenv('DB_NAME')

    ADMIN_NAME=os.getenv('ADMIN_NAME')
    ADMIN_EMAIL=os.getenv('ADMIN_EMAIL')
    ADMIN_USERNAME=os.getenv('ADMIN_USERNAME')
    ADMIN_PASSWORD=os.getenv('ADMIN_PASSWORD')
    ADMIN_PHONE=os.getenv('ADMIN_PHONE')
    IS_ADMIN=1

    MAIL_DEFAULT_SENDER=os.getenv('MAIL_DEFAULT_SENDER')
    MAIL_SERVER=os.getenv('MAIL_SERVER')
    MAIL_PORT=os.getenv('MAIL_PORT')
    # I will come back to this later
    # MAIL_USE_TLS=os.getenv('MAIL_USE_TLS')
    # MAIL_USE_SSL=os.getenv('MAIL_USE_SSL')
    MAIL_USERNAME=os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD=os.getenv('MAIL_PASSWORD')
    MAIL_DEBUG=os.getenv('MAIL_DEBUG')

    # Set SSL/TLS version explicitly
    MAIL_SSL_VERSION=os.getenv('MAIL_SSL_VERSION')
    
    # LANG
    BABEL_DEFAULT_LOCALE=os.getenv('BABEL_DEFAULT_LOCALE')
    BABEL_SUPPORTED_LOCALES=os.getenv('BABEL_SUPPORTED_LOCALES')

    CELERY_BROKER_URL=os.getenv('CELERY_BROKER_URL')
    CELERY_RESULT_BACKEND=os.getenv('CELERY_RESULT_BACKEND')



class ProductionConfig(Config):
    DEBUG = False
    MAIL_SERVER = os.getenv('MAIL_SERVER')
    MAIL_PORT = os.getenv('MAIL_PORT')
    MAIL_USE_TLS = str_to_bool(os.getenv('MAIL_USE_TLS'))
    MAIL_USE_SSL = str_to_bool(os.getenv('MAIL_USE_SSL'))
    MAIL_USERNAME=os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD=os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER=os.getenv('MAIL_DEFAULT_SENDER')
    MAIL_MAX_EMAILS=os.getenv('MAIL_MAX_EMAILS')
    MAIL_ASCII_ATTACHMENTS=os.getenv('MAIL_ASCII_ATTACHMENTS')
    MAIL_DEBUG=os.getenv('MAIL_DEBUG')


class StagingConfig(Config):
    DEVELOPMENT = True
    DEBUG = True
    MAIL_SERVER = os.getenv('MAIL_SERVER')
    MAIL_PORT = os.getenv('MAIL_PORT')
    MAIL_USE_TLS = str_to_bool(os.getenv('MAIL_USE_TLS'))
    MAIL_USE_SSL = str_to_bool(os.getenv('MAIL_USE_SSL'))
    MAIL_USERNAME=os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD=os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER=os.getenv('MAIL_DEFAULT_SENDER')
    MAIL_MAX_EMAILS=os.getenv('MAIL_MAX_EMAILS')
    MAIL_ASCII_ATTACHMENTS=os.getenv('MAIL_ASCII_ATTACHMENTS')
    MAIL_DEBUG=os.getenv('MAIL_DEBUG')


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True
    MAIL_SERVER = os.getenv('MAIL_SERVER')
    MAIL_PORT = os.getenv('MAIL_PORT')
    MAIL_USE_TLS = str_to_bool(os.getenv('MAIL_USE_TLS'))
    MAIL_USE_SSL = str_to_bool(os.getenv('MAIL_USE_SSL'))
    MAIL_USERNAME=os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD=os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER=os.getenv('MAIL_DEFAULT_SENDER')
    MAIL_MAX_EMAILS=os.getenv('MAIL_MAX_EMAILS')
    MAIL_ASCII_ATTACHMENTS=os.getenv('MAIL_ASCII_ATTACHMENTS')
    MAIL_DEBUG=os.getenv('MAIL_DEBUG')

# Set SSL/TLS version explicitly
# MAIL_SSL_VERSION='TLSv1_2'  # Adjust this based on server requirements




class TestingConfig(Config):
    TESTING = True
