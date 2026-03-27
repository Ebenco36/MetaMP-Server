import os

from dotenv import load_dotenv


DEFAULT_ENV_FILES = {
    "development": ".env.development",
    "production": ".env.production",
    "testing": ".env.testing",
}


def configure_runtime_environment() -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/")
    os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
    load_environment_file()


def load_environment_file() -> None:
    flask_env = os.getenv("FLASK_ENV", "development").lower()
    env_file = DEFAULT_ENV_FILES.get(flask_env, ".env.development")
    load_dotenv(env_file, override=False)
