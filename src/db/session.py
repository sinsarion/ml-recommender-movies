from core import config_ml
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker

# Создаем базовый класс для моделей
Base: DeclarativeMeta = declarative_base()

# Настраиваем движок SQLAlchemy
# Флаг echo=True означает что включено логирование SQL-запросов
engine = create_async_engine(config_ml.settings.database_url, echo=config_ml.settings.sqlalchemy_echo)

# Создаем фабрику сессий
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    # Флаг expire_on_commit=False означает, что мы не хотим, чтобы SQLAlchemy выдавал новые SQL-запросы
    # к базе данных при обращении к уже закоммиченным объектам
    expire_on_commit=False,
)


# Зависимость для получения сессии
async def get_session() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
