from __future__ import annotations

import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_CONNECT_TIMEOUT_SECONDS = int(
    os.getenv("INGESTION_HTTP_CONNECT_TIMEOUT_SECONDS", "10")
)
DEFAULT_READ_TIMEOUT_SECONDS = int(
    os.getenv("INGESTION_HTTP_READ_TIMEOUT_SECONDS", "120")
)
DEFAULT_MAX_RETRIES = int(os.getenv("INGESTION_HTTP_MAX_RETRIES", "5"))
DEFAULT_BACKOFF_FACTOR = float(
    os.getenv("INGESTION_HTTP_RETRY_BACKOFF_FACTOR", "2")
)
DEFAULT_POOL_CONNECTIONS = int(
    os.getenv("INGESTION_HTTP_POOL_CONNECTIONS", "20")
)
DEFAULT_POOL_MAXSIZE = int(os.getenv("INGESTION_HTTP_POOL_MAXSIZE", "20"))
RETRYABLE_STATUS_CODES = (408, 429, 500, 502, 503, 504)


def build_retrying_session(
    user_agent: str = "MetaMP-Server/1.0",
    headers: dict | None = None,
) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    if headers:
        session.headers.update(headers)

    retry_strategy = Retry(
        total=DEFAULT_MAX_RETRIES,
        connect=DEFAULT_MAX_RETRIES,
        read=DEFAULT_MAX_RETRIES,
        status=DEFAULT_MAX_RETRIES,
        allowed_methods=frozenset(["GET"]),
        status_forcelist=RETRYABLE_STATUS_CODES,
        backoff_factor=DEFAULT_BACKOFF_FACTOR,
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=DEFAULT_POOL_CONNECTIONS,
        pool_maxsize=DEFAULT_POOL_MAXSIZE,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def default_timeout() -> tuple[int, int]:
    return (
        DEFAULT_CONNECT_TIMEOUT_SECONDS,
        DEFAULT_READ_TIMEOUT_SECONDS,
    )
