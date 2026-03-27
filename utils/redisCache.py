import os
import redis
import json
import logging
import time
from flask_limiter import Limiter
from datetime import timedelta
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

env_debug = os.environ.get('FLASK_ENV', "development")
if env_debug == "development":
    load_dotenv('.env.development')
else:
    load_dotenv('.env.production')
class RedisCache:
    _instance = None
    _local_cache = {}
    host = os.getenv('REDIS_HOST', 'redis')
    max_value_bytes = int(os.getenv("REDIS_CACHE_MAX_VALUE_BYTES", "262144"))

    def __new__(cls, host=host, port=6379, db=0):
        """Create a new instance only if one doesn't already exist."""
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
            cls._instance.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
            )
        return cls._instance

    @staticmethod
    def _normalize_ttl(ttl):
        if ttl is None:
            return None
        return int(ttl.total_seconds()) if isinstance(ttl, timedelta) else int(ttl)

    @classmethod
    def _set_local_item(cls, key, value, ttl_seconds=None):
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        cls._local_cache[key] = (expires_at, value)

    @classmethod
    def _get_local_item(cls, key):
        cached = cls._local_cache.get(key)
        if cached is None:
            return None
        expires_at, value = cached
        if expires_at and expires_at <= time.time():
            cls._local_cache.pop(key, None)
            return None
        return value

    def set_item(self, key, value, ttl=None):
        """Set an item in the cache."""
        ttl_seconds = self._normalize_ttl(ttl)
        self._set_local_item(key, value, ttl_seconds=ttl_seconds)

        try:
            value_json = json.dumps(value, separators=(",", ":"))
            if len(value_json.encode("utf-8")) > self.max_value_bytes:
                logger.info(
                    "Skipping Redis store for oversized cache key %s (%s bytes)",
                    key,
                    len(value_json.encode("utf-8")),
                )
                return
            if ttl_seconds is not None:
                self.redis.setex(key, ttl_seconds, value_json)
            else:
                self.redis.set(key, value_json)
        except (TypeError, ValueError) as exc:
            logger.warning("Cache serialization failed for key %s: %s", key, exc)
        except redis.RedisError as exc:
            logger.warning("Redis unavailable while setting cache key %s: %s", key, exc)

    def get_item(self, key):
        """Get an item from the cache."""
        local_value = self._get_local_item(key)
        if local_value is not None:
            return local_value
        try:
            value_json = self.redis.get(key)
            if value_json:
                value = json.loads(value_json)
                self._set_local_item(key, value)
                return value
        except redis.RedisError as exc:
            logger.warning("Redis unavailable while reading cache key %s: %s", key, exc)
        return None

    def delete_item(self, key):
        """Delete an item from the cache."""
        self._local_cache.pop(key, None)
        try:
            self.redis.delete(key)
        except redis.RedisError as exc:
            logger.warning("Redis unavailable while deleting cache key %s: %s", key, exc)
