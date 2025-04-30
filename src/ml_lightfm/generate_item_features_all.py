import json
import re
from collections import defaultdict

import psycopg2
from lightfm.data import Dataset
from psycopg2.extras import DictCursor
from pymongo import MongoClient
from scipy.sparse import save_npz

# Настройки Postgres
POSTGRES_DB = 'ml-db'
POSTGRES_USER = 'postgres'
POSTGRES_PASSWORD = 'secret'
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = 5432
POSTGRES_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
POSTGRES_SCHEMA = "movielens_32m"

# Настройки MongoDB
MONGO_DB_USER = 'root'
MONGO_DB_PASSWORD = 'secret'
MONGO_DB_HOST = 'localhost'
MONGO_DB_PORT = 27017
MONGO_DB = "ml_database"
MONGO_COLLECTION = "reviews"
MONGO_URI = f"mongodb://{MONGO_DB_USER}:{MONGO_DB_PASSWORD}@{MONGO_DB_HOST}:{MONGO_DB_PORT}"


def normalize(s):
    return re.sub(r"[^a-z0-9_]", "", s.strip().lower().replace(" ", "_"))


def get_movie_genres(conn):
    print("Загружаем жанры из PostgreSQL...")

    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(f"""
            SELECT mg.movie_id, g.name as genre
            FROM {POSTGRES_SCHEMA}.movie_genres mg
            JOIN {POSTGRES_SCHEMA}.genres g ON mg.genre_id = g.id
        """)

        rows = cur.fetchall()

    genre_map = defaultdict(set)

    for row in rows:
        genre_map[int(row["movie_id"])].add(f"genre:{normalize(row['genre'])}")

    print(f"Жанровых тегов собрано для {len(genre_map)} фильмов.")
    return genre_map


def get_movie_tags(conn):
    print("Загружаем теги из PostgreSQL...")
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(f"""
            SELECT movie_id, tag
            FROM {POSTGRES_SCHEMA}.tags
        """)
        rows = cur.fetchall()
    tag_map = defaultdict(set)
    for row in rows:
        tag_map[int(row["movie_id"])].add(f"tag:{normalize(row['tag'])}")
    print(f"Тегов собрано для {len(tag_map)} фильмов.")
    return tag_map


def get_movie_desc_tokens(mongo_client):
    print("Загружаем токены описаний из MongoDB...")
    db = mongo_client[MONGO_DB]
    collection = db[MONGO_COLLECTION]
    desc_map = defaultdict(set)
    cursor = collection.find({}, {"movie_id": 1, "text_tokens": 1})
    count = 0
    for doc in cursor:
        if "movie_id" in doc and "text_tokens" in doc:
            movie_id = int(doc["movie_id"])
            tokens = [f"desc:{normalize(token)}" for token in doc["text_tokens"][:10]]
            desc_map[movie_id].update(tokens)
            count += 1
    print(f"Токены описаний загружены для {count} фильмов.")
    return desc_map


def combine_features(*dicts):
    combined = defaultdict(set)
    for d in dicts:
        for movie_id, features in d.items():
            combined[movie_id].update(features)
    return combined


def build_item_feature_matrix(movie_to_features, movie_map_path="artifacts/movie_map.json"):
    with open(movie_map_path, "r") as f:
        movie_map = {int(k): int(v) for k, v in json.load(f).items()}

    dataset = Dataset()
    dataset.fit_partial(
        items=movie_map.values(),
        item_features=set().union(*movie_to_features.values())
    )

    item_features = dataset.build_item_features(
        ((movie_map[movie_id], list(features)) for movie_id, features in movie_to_features.items() if
         movie_id in movie_map)
    )

    return item_features


def main():
    print("Подключаемся к PostgreSQL и MongoDB...")
    pg_conn = psycopg2.connect(POSTGRES_URI)
    mongo_client = MongoClient(MONGO_URI)

    genres = get_movie_genres(pg_conn)
    tags = get_movie_tags(pg_conn)
    desc_tokens = get_movie_desc_tokens(mongo_client)

    print("Объединяем признаки...")
    movie_to_features = combine_features(genres, tags, desc_tokens)
    print(f"Всего фильмов с фичами: {len(movie_to_features)}")

    print("Строим item_features матрицу...")
    item_features = build_item_feature_matrix(movie_to_features)

    print(f"item_features построены: shape = {item_features.shape}")
    save_npz("artifacts/item_features_all.npz", item_features)

    print("Сохранено в: artifacts/item_features_all.npz")


if __name__ == "__main__":
    main()
