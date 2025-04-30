import pickle
from pathlib import Path

from lightfm.evaluation import precision_at_k, auc_score, recall_at_k
from scipy.sparse import load_npz

from lightfm import LightFM

ARTIFACTS_DIR = Path("artifacts")
INTERACTIONS_FILE = ARTIFACTS_DIR / "interactions.npz"
ITEM_FEATURES_FILE = ARTIFACTS_DIR / "item_features_all.npz"
MODEL_FILE = ARTIFACTS_DIR / "lightfm_model.pkl"

# Задаём список метрик, которые хотим посчитать
METRICS = ["precision@10", "auc", "recall@10"]


def evaluate_model(model, interactions, item_features):
    print("Оцениваем качество модели...")
    results = {}

    if "precision@10" in METRICS:
        results["precision@10"] = precision_at_k(
            model, interactions, item_features=item_features, k=10
        ).mean()

    if "recall@10" in METRICS:
        results["recall@10"] = recall_at_k(
            model, interactions, item_features=item_features, k=10
        ).mean()

    if "auc" in METRICS:
        results["auc"] = auc_score(
            model, interactions, item_features=item_features
        ).mean()

    return results


def main():
    print("Загружаем модель и данные...")
    with open(MODEL_FILE, "rb") as f:
        model: LightFM = pickle.load(f)

    interactions = load_npz(INTERACTIONS_FILE)
    item_features = load_npz(ITEM_FEATURES_FILE)

    print("Выполняем оценку...")
    metrics = evaluate_model(model, interactions, item_features)

    print("Результаты:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
