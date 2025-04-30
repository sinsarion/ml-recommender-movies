from logging import config as logging_config

from pydantic import Field
from pydantic_settings import BaseSettings

from core.logger import LOGGING

logging_config.dictConfig(LOGGING)


class Settings(BaseSettings):
    project_name: str = Field('ML API')
    project_description: str = Field('API for generates recommendations based on ML algorithms')

    service_port: int = Field(8035)
    service_host: str = Field('0.0.0.0')
    service_base_url: str = Field('http://localhost:8035')

    swagger_url: str = Field('/docs')
    redoc_url: str = Field('/redoc')

    redis_host: str = Field('localhost')
    redis_port: int = Field(6379)
    redis_cache_expire_in_seconds: int = Field(3600)  # Срок хранения данных в кэше, в секундах

    postgres_db: str = Field('ml-db')
    postgres_user: str = Field('postgres')
    postgres_password: str = Field('secret')
    postgres_host: str = Field('ml-db')
    postgres_port: int = Field(5432)

    sqlalchemy_echo: bool = True
    sqlalchemy_url: str = Field('postgresql://postgres:secret@ml-db:5432/ml-db')
    database_url: str = Field('postgresql+asyncpg://postgres:secret@localhost:5432/ml-db')

    mongodb_name: str = Field('ml_database')
    mongodb_user: str = Field('root')
    mongodb_password: str = Field('secret')
    mongodb_host: str = Field('ml-mongodb')
    mongodb_port: int = Field(27017)


settings = Settings()
