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
    SECRET_KEY=os.getenv('SECRET_KEY')


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
