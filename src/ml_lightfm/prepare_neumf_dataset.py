import pickle
import random
from pathlib import Path

from scipy.sparse import load_npz

# Пути
ARTIFACTS_DIR = Path("artifacts")
INTERACTIONS_PATH = ARTIFACTS_DIR / "interactions.npz"
NEUMF_DATASET_PATH = ARTIFACTS_DIR / "neumf_dataset.pkl"

# Загрузка sparse матрицы взаимодействий
print("Загружаем interactions.npz...")
interaction_matrix = load_npz(INTERACTIONS_PATH)
n_users, n_items = interaction_matrix.shape

print(f"Матрица взаимодействий: {n_users} пользователей, {n_items} фильмов")
positive_pairs = list(zip(*interaction_matrix.nonzero()))
positive_set = set(positive_pairs)

print(f"Положительных примеров: {len(positive_pairs)}")

# Генерация отрицательных примеров
print("Генерируем отрицательные примеры...")
negative_pairs = set()
while len(negative_pairs) < len(positive_pairs):
    u = random.randint(0, n_users - 1)
    i = random.randint(0, n_items - 1)
    if (u, i) not in positive_set:
        negative_pairs.add((u, i))

# Сборка датасета
print("Формируем датасет...")
dataset = [(u, i, 1) for u, i in positive_pairs] + [(u, i, 0) for u, i in negative_pairs]
random.shuffle(dataset)

# Сохраняем
with open(NEUMF_DATASET_PATH, "wb") as f:
    pickle.dump(dataset, f)

print(f"Сохранено в: {NEUMF_DATASET_PATH}")
print(f"Размер датасета: {len(dataset)} примеров (положительных и отрицательных)")
