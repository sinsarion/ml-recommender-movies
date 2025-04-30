import logging
import random

import pandas as pd
from pymongo import MongoClient, ASCENDING

from review_templates import positive_reviews, negative_reviews, neutral_reviews

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройки MongoDB
DB_NAME = 'ml_database'
DB_USER = 'root'
DB_PASSWORD = 'secret'
DB_HOST = 'localhost'
DB_PORT = 27017

MONGO_URI = f"mongodb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"
CHUNK_SIZE = 500_000  # Количество строк на один чанк

# Путь к файлу ratings.csv
RATINGS_CSV = "./movielens_32m/ml-32m/ratings.csv"


def generate_review_text(rating):
    if rating >= 4.0:
        return random.choice(positive_reviews)
    elif rating <= 2.5:
        return random.choice(negative_reviews)
    else:
        return random.choice(neutral_reviews)


def create_indexes(collection):
    collection.create_index([("user_id", ASCENDING)])
    collection.create_index([("movie_id", ASCENDING)])
    collection.create_index([("user_id", ASCENDING), ("movie_id", ASCENDING)], unique=True)


def main():
    logger.info("Подключение к MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    likes_col = db.likes
    reviews_col = db.reviews

    logger.info("Очистка коллекций...")
    likes_col.drop()
    reviews_col.drop()

    create_indexes(likes_col)
    create_indexes(reviews_col)

    logger.info("Чтение файла по чанкам...")
    chunk_iter = pd.read_csv(RATINGS_CSV, chunksize=CHUNK_SIZE)

    total_likes = 0
    total_reviews = 0
    chunk_count = 0

    for chunk in chunk_iter:
        chunk_count += 1
        logger.info(f"Обработка чанка {chunk_count}...")

        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="s")

        likes_docs = []
        reviews_docs = []

        for row in chunk.itertuples():
            if row.rating >= 4.0:
                likes_docs.append({
                    "user_id": int(row.userId),
                    "movie_id": int(row.movieId),
                    "liked": True,
                    "timestamp": row.timestamp
                })
            if random.random() <= 0.3:
                reviews_docs.append({
                    "user_id": int(row.userId),
                    "movie_id": int(row.movieId),
                    "rating": float(row.rating),
                    "text": generate_review_text(row.rating),
                    "timestamp": row.timestamp
                })

        if likes_docs:
            try:
                likes_col.insert_many(likes_docs, ordered=False)
                total_likes += len(likes_docs)
            except Exception as e:
                logger.warning(f"Ошибка при вставке лайков: {e}")

        if reviews_docs:
            try:
                reviews_col.insert_many(reviews_docs, ordered=False)
                total_reviews += len(reviews_docs)
            except Exception as e:
                logger.warning(f"Ошибка при вставке рецензий: {e}")

        logger.info(f"Завершено: {chunk_count} чанков, {total_likes} лайков, {total_reviews} рецензий")

    logger.info("Загрузка завершена.")


if __name__ == "__main__":
    main()
