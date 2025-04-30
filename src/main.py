import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi_limiter import FastAPILimiter
from redis.asyncio import Redis

from api.v1 import ml_recommender
from core import config_ml
from core.logger import LOGGING
from db import redis_client, session

logger = logging.getLogger('uvicorn.error')

SWAGGER_URL = config_ml.settings.swagger_url
REDOC_URL = config_ml.settings.redoc_url


# Подключение к базам при старте, закрытие подключений по завершению работы
@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("ML API - running")

    redis_client.redis = Redis(host=config_ml.settings.redis_host, port=config_ml.settings.redis_port)

    await FastAPILimiter.init(redis_client.redis)

    logger.info("ML API - started successfully")

    yield

    await redis_client.redis.close()

    await session.engine.dispose()

    logger.info("ML API - finished successfully")


app = FastAPI(
    title=config_ml.settings.project_name,
    description=config_ml.settings.project_description,
    version="1.0.0",
    openapi_tags=[
        {"name": "ML API", "description": "API for generates recommendations based on ML algorithms"},
    ],
    lifespan=lifespan
)

app.include_router(ml_recommender.router, prefix='/api/v1', tags=['ml_api'])


@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_schema():
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )


if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=config_ml.settings.service_port,
        log_config=LOGGING,
        log_level=logging.DEBUG,
    )
