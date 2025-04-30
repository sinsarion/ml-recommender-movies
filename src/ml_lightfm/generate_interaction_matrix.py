import json
import os

import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy.sparse import coo_matrix, save_npz

from src.core import config_ml

# Настройки MongoDB
DB_NAME = config_ml.settings.mongodb_name
DB_USER = config_ml.settings.mongodb_user
DB_PASSWORD = config_ml.settings.mongodb_password
DB_HOST = 'localhost'
DB_PORT = config_ml.settings.mongodb_port

MONGO_URI = f"mongodb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"

COLLECTION = "likes"


def load_likes():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    cursor = db[COLLECTION].find({"liked": True}, {"user_id": 1, "movie_id": 1, "_id": 0})
    return pd.DataFrame(list(cursor))


def generate_interaction_matrix(df):
    print("Создание словарей пользователей и фильмов...")

    unique_users = sorted(df["user_id"].unique())
    unique_movies = sorted(df["movie_id"].unique())

    user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}

    print(f"Пользователей: {len(user_map)}, фильмов: {len(movie_map)}")

    rows = df["user_id"].map(user_map)
    cols = df["movie_id"].map(movie_map)
    data = np.ones(len(df), dtype=np.float32)

    matrix = coo_matrix((data, (rows, cols)), shape=(len(user_map), len(movie_map)))

    # Конвертируем matrix в Compressed Sparse Row format и возвращаем
    return matrix.tocsr(), user_map, movie_map


def save_artifacts(matrix, user_map, movie_map, output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    save_npz(os.path.join(output_dir, "interactions.npz"), matrix)

    # Преобразуем ключи в обычные int
    user_map = {int(k): int(v) for k, v in user_map.items()}
    movie_map = {int(k): int(v) for k, v in movie_map.items()}

    with open(os.path.join(output_dir, "user_map.json"), "w") as f:
        json.dump(user_map, f)

    with open(os.path.join(output_dir, "movie_map.json"), "w") as f:
        json.dump(movie_map, f)

    print(f"Матрица и отображения сохранены в: {output_dir}/")


def main():
    print("Загрузка лайков из MongoDB...")
    df = load_likes()

    print("Генерация матрицы взаимодействий...")
    matrix, user_map, movie_map = generate_interaction_matrix(df)

    print("Сохраняем результаты...")
    save_artifacts(matrix, user_map, movie_map)


if __name__ == "__main__":
    main()
