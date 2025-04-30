import pickle
from pathlib import Path

from lightfm.evaluation import precision_at_k, auc_score
from scipy.sparse import load_npz

from lightfm import LightFM

ARTIFACTS_DIR = Path("artifacts")
INTERACTIONS_FILE = ARTIFACTS_DIR / "interactions.npz"
ITEM_FEATURES_FILE = ARTIFACTS_DIR / "item_features_all.npz"
MODEL_FILE = ARTIFACTS_DIR / "lightfm_model.pkl"


def main():
    print("Загружаем матрицы взаимодействий и признаков фильмов...")
    interactions = load_npz(INTERACTIONS_FILE)
    item_features = load_npz(ITEM_FEATURES_FILE)

    print(f"Interactions shape: {interactions.shape}")
    print(f"Item features shape: {item_features.shape}")

    print("Инициализируем модель LightFM (loss='warp')...")
    model = LightFM(loss='warp', no_components=64, random_state=42)

    print("Обучаем модель...")
    model.fit(interactions, item_features=item_features, epochs=30, num_threads=4)

    print("Сохраняем модель в файл...")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print(f"Модель сохранена: {MODEL_FILE}")


if __name__ == "__main__":
    main()
