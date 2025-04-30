import abc
import logging
from typing import Any

import backoff
import redis
from pydantic import BaseModel, ValidationError
from redis import Redis

from api.models.response_models import RecommendationsResponse

logger = logging.getLogger('uvicorn.error')

# Модели pydantic поддерживаемые для хранения в кеше
MODELS = [RecommendationsResponse, ]


class BaseAsyncCache(abc.ABC):
    """
    Абстрактный класс кеша с асинхронным доступом.
    Класс содержит контракт работы с кешем: как сохранять и получать данные моделей.
    """

    @abc.abstractmethod
    async def put_model(self, key: str, model: BaseModel, expire_in_seconds: int) -> None:
        """Асинхронно сохранить данные модели в кеше."""

    @abc.abstractmethod
    async def get_model(self, key: str) -> BaseModel | None:
        """Асинхронно получить данные модели из кеша."""


class RedisCache(BaseAsyncCache):

    def __init__(self, redis: Redis):
        self.redis = redis

    @backoff.on_exception(
        backoff.expo,
        exception=(redis.exceptions.ConnectionError, redis.exceptions.TimeoutError),
        max_time=60
    )
    async def put_model(self, key: str, model: BaseModel, expire_in_seconds: int) -> None:
        logger.debug("RedisCache put_model() starting...")
        await self.redis.set(name=key, value=model.model_dump_json(), ex=expire_in_seconds)

    @backoff.on_exception(
        backoff.expo,
        exception=(redis.exceptions.ConnectionError, redis.exceptions.TimeoutError),
        max_time=60
    )
    async def get_model(self, key: str) -> BaseModel | None:
        logger.debug("RedisCache get_model() starting...")
        data = await self.redis.get(key)
        if not data:
            return None

        result = self._check_model(data)

        return result

    @staticmethod
    def _check_model(data: Any) -> BaseModel | None:
        for klass in MODELS:
            try:
                model = klass.model_validate_json(data)
                logger.debug(f"RedisCache _check_model() found in cache {klass}")
                return model
            except ValidationError:
                pass
