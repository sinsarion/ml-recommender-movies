import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import Depends, HTTPException
from redis.asyncio import Redis
from scipy.sparse import load_npz
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.response_models import RecommendationsResponse, FilmInfo
from core import config_ml
from core.redis_cache import BaseAsyncCache, RedisCache
from db.redis_client import get_redis
from db.session import get_session
from ml_lightfm.neu_mf_model import NeuMF

logger = logging.getLogger('uvicorn.error')

# Загрузка артефактов ML-модели LightFM
ARTIFACTS_DIR = Path("ml_lightfm/artifacts")  # папка с матрицами, маппингами индексов и обученной моделью LightFM
MODEL_PATH = ARTIFACTS_DIR / "lightfm_model.pkl"  # обученная модель LightFM
MODEL_NEU_MF_PATH = ARTIFACTS_DIR / "neumf_model_v2.pth"  # обученная модель NeuMF
USER_MAP_PATH = ARTIFACTS_DIR / "user_map.json"  # маппинг user_id -> index в матрице interactions.npz
MOVIE_MAP_PATH = ARTIFACTS_DIR / "movie_map.json"  # маппинг movie_id -> index в матрице interactions.npz
ITEM_FEATURES_PATH = ARTIFACTS_DIR / "item_features_all.npz"  # матрица item features (признаков фильмов)

MOVIELENS_32M_DATASET = "movielens_32m"
SUPPORTED_DATASET_LIST = [MOVIELENS_32M_DATASET]

ML_MODEL_NAME_LIGHTFM = "lightfm"
ML_MODEL_NAME_NEUMF = "neumf"
SUPPORTED_ML_MODEL_LIST = [ML_MODEL_NAME_LIGHTFM, ML_MODEL_NAME_NEUMF]

with open(MODEL_PATH, "rb") as f:
    # model — это обученная модель LightFM, которая помнит представления пользователей и объектов (фильмов),
    # а также, как они взаимодействовали
    model = pickle.load(f)

with open(USER_MAP_PATH, "r") as f:
    # словарь: user_id -> индекс пользователя (внутренний, числовой)
    user_map = json.load(f)

with open(MOVIE_MAP_PATH, "r") as f:
    # словарь: movie_id -> item_index (внутренний, числовой)
    movie_map = json.load(f)

# готовим словарь: item_index -> movie_id, чтобы потом из предсказания вернуть ID фильмов
reverse_movie_map = {v: int(k) for k, v in movie_map.items()}
item_features = load_npz(ITEM_FEATURES_PATH)

n_users = len(user_map)
n_items = len(movie_map)
neumf_model = NeuMF(num_users=n_users, num_items=n_items)
# Загружаем веса в neumf_model (обученная модель NeuMF)
neumf_model.load_state_dict(torch.load(MODEL_NEU_MF_PATH, map_location=torch.device("cpu")))
# Переводим модель в режим инференса (оценки, а не обучения)
# Отключается Dropout, BatchNorm ведёт себя по-другому и т.д.
neumf_model.eval()

def format_imdb_link(imdb_id: str | None) -> str | None:
    """Генерируем ссылку на сайт www.imdb.com по imdb_id"""
    if imdb_id and imdb_id.isdigit():
        return f"https://www.imdb.com/title/tt{imdb_id.zfill(7)}"
    return None


def format_tmdb_link(tmdb_id: str | None) -> str | None:
    """Генерируем ссылку на сайт www.themoviedb.org по tmdb_id"""
    if tmdb_id and tmdb_id.isdigit():
        return f"https://www.themoviedb.org/movie/{tmdb_id}"
    return None


class MLRecommenderService:

    def __init__(self, db_session: AsyncSession, cache: BaseAsyncCache):
        self.db_session = db_session
        self.cache = cache

    @staticmethod
    def neumf_predict_movielens_32m_movie_list(user_id: str, k: int = 10, batch_size: int = 1000) -> list[int]:
        """
        Делаем предсказание, генерируем список movies_id по user_id с использованием обученной NeuMF модели
        Возвращаем список movie_id — top-k фильмов, которые модель рекомендует пользователю
        """

        logger.info(f"[NeuMF] Запрошены рекомендации: user_id={user_id}, k={k}, batch_size={batch_size}")

        # Проверяем наличие пользователя в словаре
        if user_id not in user_map:
            logger.warning(f"[NeuMF] Пользователь {user_id} не найден в user_map. Отдаём пустой список.")
            return []

        # Получаем внутренний числовой индекс пользователя, с которым работает модель
        user_index = user_map[user_id]
        logger.debug(f"[NeuMF] Внутренний индекс пользователя: {user_index}")

        # Отображаем общее число фильмов (movie_map)
        # это количество необходимо, чтобы предсказать интерес пользователя ко всем фильмам
        logger.debug(f"[NeuMF] Количество фильмов в датасете: {n_items}")

        # Создаём массив оценок (одна оценка на каждый фильм) scores
        # размер = количеством фильмов, запишем в него предсказанные “оценки интереса” от модели
        scores = np.zeros(n_items, dtype=np.float32)

        # Отключаем подсчёт градиентов - автоградиенты PyTorch (ускоряет инференс и вычисления, экономит память)
        with torch.no_grad():
            # Будем обрабатывать фильмы батчами (по batch_size штук, например по 1000)
            # Это экономит память и ускоряет предсказания, особенно если фильмов тысячи или десятки тысяч
            for start in range(0, n_items, batch_size):
                end = min(start + batch_size, n_items)
                batch_size_actual = end - start

                # Создаём батч user_tensor
                # Например, если обрабатываем 1000 фильмов, создаём вектор из 1000 одинаковых user_index,
                # потому что хотим предсказать интерес одного пользователя ко всем этим фильмам
                user_tensor = torch.tensor([user_index] * batch_size_actual)

                # Создаём вектор item_tensor — индексы фильмов от start до end
                # Например: [0, 1, 2, ..., 999], [1000, 1001, ..., 1999] и т.д.
                item_tensor = torch.tensor(list(range(start, end)))

                # Прогоняем батч через модель:
                # Модель вернёт оценку интереса пользователя к каждому фильму в батче
                # model(user_tensor, item_tensor) выдаёт предсказанные оценки интереса для каждого пользователя-фильма в паре
                # .numpy() переводит из тензора в обычный NumPy-массив
                batch_scores = neumf_model(user_tensor, item_tensor).cpu().numpy()

                # Сохраняем оценки в итоговый массив scores
                scores[start:end] = batch_scores

                logger.debug(f"[NeuMF] Обработан батч: {start}–{end}, макс. оценка: {batch_scores.max():.4f}")

        # Сортируем фильмы по убыванию предсказанной оценки
        top_indices = np.argsort(-scores)[:k]
        logger.info(f"[NeuMF] Индексы top-{k} фильмов: {top_indices.tolist()}")

        # Преобразуем внутренние индексы обратно в movie_id
        movie_ids = [reverse_movie_map[i] for i in top_indices]

        logger.info(f"[NeuMF] movie_id рекомендаций: {movie_ids}")

        return movie_ids

    @staticmethod
    def lightfm_predict_movielens_32m_movie_list(user_id: str, k: int = 10) -> list[int]:
        """Делаем предсказание, генерируем список movies_id по user_id с помощью обученной ML-модели LightFM"""

        # Находим внутренний индекс пользователя по его внешнему user_id
        user_index = user_map[user_id]

        # Вычисляем количество фильмов
        # Это нам нужно, чтобы предсказать оценки для всех фильмов для одного пользователя
        n_items = item_features.shape[0]

        logger.info(f"Запрошены рекомендации для user_id={user_id} (внутренний индекс={user_index}).")
        logger.info(f"Количество фильмов (n_items): {n_items}")
        logger.info(f"Запуск прогноза ML-модели: model.predict...")

        # Запрос прогноза модели: насколько пользователю user_index понравится каждый из фильмов с индексами от 0 до n_items - 1
        # model.predict() возвращает вектор оценок (scores), по одному значению на каждый фильм
        # LightFM использует внутренние векторы пользователя + каждого фильма
        # и признаки из item_features, чтобы рассчитать скалярное значение — предсказанную оценку интереса
        scores = model.predict(
            user_ids=user_index,
            item_ids=np.arange(n_items),
            item_features=item_features
        )

        logger.info(f"model.predict завершён. Первые {k} предсказанных оценок: {scores[:k]}")

        # Сортируем оценки по убыванию (поэтому -scores) и берём топ-k индексов фильмов,
        # у которых наибольший прогноз интереса
        top_items = np.argsort(-scores)[:k]
        logger.info(f"Топ-{k} внутренних индексов фильмов: {top_items}")

        # Преобразуем внутренние индексы фильмов обратно в movie_id из БД
        # Это уже готовый список рекомендаций: фильмов, которые модель считает наиболее подходящими пользователю
        movie_ids = [reverse_movie_map[i] for i in top_items]
        logger.info(f"Топ-{k} movie_id: {movie_ids}")

        return movie_ids

    async def _save_to_cache(self, key: str, data: RecommendationsResponse,
                             ttl: int = config_ml.settings.redis_cache_expire_in_seconds) -> None:
        await self.cache.put_model(key=key, model=data, expire_in_seconds=ttl)

    @staticmethod
    def _generate_cache_key(user_id: str, ml_model: str, dataset: str, k: int) -> str:
        key = f"ml_service_cache_{user_id}_{ml_model}_{dataset}_{k}"
        logger.info(f"_generate_cache_key() key={key}")
        return key

    async def get_recommendations(self, user_id: str, ml_model: str, dataset: str,
                                  k: int = 10) -> RecommendationsResponse:
        """
        Генерируем список персонализированных рекомендаций фильмов

        Args:
            user_id: Идентификатор пользователя в датасете.
            ml_model: ML-модель, которая будет использоваться для генерации рекомендаций.
            dataset: Датасет, в рамках которого будет производиться поиск/генерация рекомендаций.
            k: Количество фильмов, которые должны войти в список рекомендаций.

        Returns:
            Список персонализированных рекомендаций фильмов
        """
        logger.info(f"Генерация персонализированных рекомендаций по user_id={user_id}, dataset={dataset}, k={k}...")

        if dataset not in SUPPORTED_DATASET_LIST:
            message = f"Unsupported dataset {dataset}"
            logger.info(message)
            raise HTTPException(status_code=400, detail=message)

        if ml_model not in SUPPORTED_ML_MODEL_LIST:
            message = f"Unsupported ml model {ml_model}"
            logger.info(message)
            raise HTTPException(status_code=400, detail=message)

        if user_id not in user_map:
            logger.warning(f"User {user_id} не найден в user_map.")
            # Если пользователь не найден - отдаём список не персонализированных рекомендаций
            return await self.get_non_personalized_recommendations(user_id, dataset, k)

        # Пробуем получить данные из кэша Redis
        cache_key = self._generate_cache_key(user_id, ml_model, dataset, k)
        cached_model = await self.cache.get_model(cache_key)
        if cached_model and isinstance(cached_model, RecommendationsResponse):
            logger.info(
                f"get_recommendations() cache_key = {cache_key} found in cache, return result")
            return cached_model

        # На данный момент поддерживается только один датасет, но всё равно делаем проверку
        if dataset == MOVIELENS_32M_DATASET:
            if ml_model == ML_MODEL_NAME_LIGHTFM:
                movie_ids = self.lightfm_predict_movielens_32m_movie_list(user_id, k)

            if ml_model == ML_MODEL_NAME_NEUMF:
                movie_ids = self.neumf_predict_movielens_32m_movie_list(user_id, k)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported dataset {dataset}")

        # Формируем запрос для загрузки из PostgreSQL всей доступной информации о фильмах
        query = text(f"""
            SELECT 
                m.id,
                m.title,
                r.rating AS avg_rating,
                l.imdb_id,
                l.tmdb_id,
                ARRAY_AGG(DISTINCT g.name) AS genres
            FROM movielens_32m.movies m
            LEFT JOIN {dataset}.movie_genres mg ON m.id = mg.movie_id
            LEFT JOIN {dataset}.genres g ON mg.genre_id = g.id
            LEFT JOIN {dataset}.links l ON m.id = l.movie_id
            LEFT JOIN LATERAL (
                SELECT movie_id, AVG(rating) AS rating
                FROM {dataset}.ratings
                WHERE movie_id = m.id
                GROUP BY movie_id
            ) r ON true
            WHERE m.id = ANY(:ids)
            GROUP BY m.id, r.rating, l.imdb_id, l.tmdb_id
        """)

        film_infos = []

        # Запрос данных в PostgreSQL
        rows = await self.db_session.execute(query, {"ids": movie_ids})

        # Формируем данные для ответа пользователю
        for row in rows:
            imdb_url = format_imdb_link(row.imdb_id)
            tmdb_url = format_tmdb_link(row.tmdb_id)
            film_info = FilmInfo(
                id=row.id,
                title=row.title,
                avg_rating=row.avg_rating,
                genres=row.genres or [],
                imdb_id=row.imdb_id,
                tmdb_id=row.tmdb_id,
                imdb_url=imdb_url,
                tmdb_url=tmdb_url
            )
            film_infos.append(film_info)

        data = RecommendationsResponse(user_id=user_id, has_personalization=True, recommendations=film_infos)

        # Сохраняем список рекомендаций в кеш
        await self._save_to_cache(key=cache_key, data=data)

        return data

    async def get_non_personalized_recommendations(self, user_id: str = '', dataset: str = MOVIELENS_32M_DATASET,
                                                   k: int = 10) -> RecommendationsResponse:
        """
        Генерируем список не персонализированных рекомендаций фильмов.
        Берём топ-k фильмов по среднему рейтингу.
        """

        logger.info("Генерация не персонализированных рекомендаций (топ по среднему рейтингу)...")

        # Пробуем получить данные из кэша Redis
        # user_id = '', т.к. отдаём кэш один для всех не персонализированных пользователей
        cache_key = self._generate_cache_key(user_id='', ml_model='', dataset=dataset, k=k)
        cached_model = await self.cache.get_model(cache_key)
        if cached_model and isinstance(cached_model, RecommendationsResponse):
            cached_model.user_id = user_id
            logger.info(
                f"get_non_personalized_recommendations() cache_key = {cache_key} found in cache, return result")
            return cached_model

        query = text(f"""
            SELECT 
                m.id,
                m.title,
                AVG(r.rating) AS avg_rating,
                ARRAY_AGG(DISTINCT g.name) AS genres,
                l.imdb_id,
                l.tmdb_id
            FROM (
                SELECT movie_id
                FROM {dataset}.ratings
                GROUP BY movie_id
                ORDER BY AVG(rating) DESC
                LIMIT :limit
            ) top_movies
            JOIN {dataset}.movies m ON m.id = top_movies.movie_id
            JOIN {dataset}.ratings r ON m.id = r.movie_id
            LEFT JOIN {dataset}.movie_genres mg ON m.id = mg.movie_id
            LEFT JOIN {dataset}.genres g ON mg.genre_id = g.id
            LEFT JOIN {dataset}.links l ON m.id = l.movie_id
            GROUP BY m.id, l.imdb_id, l.tmdb_id
            ORDER BY avg_rating DESC
        """)

        rows = await self.db_session.execute(query, {"limit": k})
        result = []
        for row in rows:
            imdb_url = format_imdb_link(row.imdb_id)
            tmdb_url = format_tmdb_link(row.tmdb_id)
            film_info = FilmInfo(
                id=row.id,
                title=row.title,
                avg_rating=row.avg_rating,
                genres=row.genres or [],
                imdb_id=row.imdb_id,
                tmdb_id=row.tmdb_id,
                imdb_url=imdb_url,
                tmdb_url=tmdb_url
            )
            result.append(film_info)

        data = RecommendationsResponse(user_id=user_id, has_personalization=False, recommendations=result)

        # Сохраняем список рекомендаций в кеш
        await self._save_to_cache(key=cache_key, data=data)

        return data


@lru_cache()
def get_ml_recommender_service(
        redis: Redis = Depends(get_redis),
        db_session: AsyncSession = Depends(get_session)
) -> MLRecommenderService:
    return MLRecommenderService(db_session=db_session, cache=RedisCache(redis))
