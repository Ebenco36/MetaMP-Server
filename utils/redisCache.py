import os
import redis
import json
from flask_limiter import Limiter
from datetime import timedelta
from dotenv import load_dotenv

env_debug = os.environ.get('FLASK_DEBUG', True)
if env_debug:
    load_dotenv('.env.development')
else:
    load_dotenv('.env.production')
class RedisCache:
    _instance = None
    host = os.getenv('REDIS_HOST')
    def __new__(cls, host=host, port=6379, db=0):
        """Create a new instance only if one doesn't already exist."""
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
            cls._instance.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        return cls._instance

    def set_item(self, key, value, ttl=None):
        """Set an item in the cache."""
        value_json = json.dumps(value)
        if ttl is not None:
            ttl_seconds = int(ttl.total_seconds()) if isinstance(ttl, timedelta) else int(ttl)
            self.redis.setex(key, ttl_seconds, value_json)
        else:
            self.redis.set(key, value_json)

    def get_item(self, key):
        """Get an item from the cache."""
        value_json = self.redis.get(key)
        if value_json:
            return json.loads(value_json)
        return None

    def delete_item(self, key):
        """Delete an item from the cache."""
        self.redis.delete(key)