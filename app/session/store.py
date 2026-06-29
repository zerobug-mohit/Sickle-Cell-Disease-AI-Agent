import json
from app.config import SESSION_BACKEND, REDIS_URL

_memory: dict[str, dict] = {}
_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def get_session(phone: str) -> dict:
    if SESSION_BACKEND == "redis":
        data = _get_redis().get(f"session:{phone}")
        return json.loads(data) if data else {"lang": "en"}
    if phone not in _memory:
        _memory[phone] = {"lang": "en"}
    return _memory[phone]


def save_session(phone: str, data: dict) -> None:
    if SESSION_BACKEND == "redis":
        _get_redis().set(f"session:{phone}", json.dumps(data), ex=86400)
        return
    _memory[phone] = data


def clear_session(phone: str) -> None:
    if SESSION_BACKEND == "redis":
        _get_redis().delete(f"session:{phone}")
        return
    _memory.pop(phone, None)
