import argparse
import json
import logging
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from neu_mf_model import NeuMF

logger = logging.getLogger('uvicorn.error')

ARTIFACTS_DIR = Path("artifacts")
DATASET_PATH = ARTIFACTS_DIR / "neumf_dataset.pkl"
MODEL_SAVE_PATH = ARTIFACTS_DIR / "neumf_model_v2.pth"
USER_MAP_PATH = ARTIFACTS_DIR / "user_map.json"
MOVIE_MAP_PATH = ARTIFACTS_DIR / "movie_map.json"


class InteractionDataset(Dataset):

    def __init__(self, data):
        self.user_ids = torch.tensor([u for u, _, _ in data], dtype=torch.long)
        self.item_ids = torch.tensor([i for _, i, _ in data], dtype=torch.long)
        self.labels = torch.tensor([l for _, _, l in data], dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


def train_model(
        _num_users: int,
        _num_items: int,
        embedding_dim_gmf: int = 32,
        embedding_dim_mlp: int = 64,
        mlp_hidden_layers: list[int] = [128, 64],
        dropout: float = 0.2,
        batch_size: int = 4096,
        epochs: int = 20,
        lr: float = 0.001,
        use_batchnorm: bool = True
):
    logger.info("Загружаем датасет...")
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    logger.info(f"Размер датасета: {len(dataset):,} примеров")
    dataloader = DataLoader(InteractionDataset(dataset), batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используем устройство: {device}")

    model = NeuMF(
        num_users=_num_users,
        num_items=_num_items,
        embedding_dim_gmf=embedding_dim_gmf,
        embedding_dim_mlp=embedding_dim_mlp,
        mlp_hidden_layers=mlp_hidden_layers,
        dropout=dropout,
        use_batchnorm=use_batchnorm
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for users, items, labels in progress:
            users, items, labels = users.to(device), items.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(users, items)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        logger.info(f"Epoch {epoch}, Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Модель NeuMF успешно обучена и сохранена в {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuMF model")
    parser.add_argument("--epochs", type=int, default=20, help="Количество эпох обучения")
    args = parser.parse_args()

    # Загружаем user_map и movie_map
    logger.info("Загружаем user_map.json и movie_map.json...")
    with open(USER_MAP_PATH, "r") as f:
        user_map = json.load(f)

    with open(MOVIE_MAP_PATH, "r") as f:
        movie_map = json.load(f)

    num_users = len(user_map)
    num_items = len(movie_map)

    logger.info(f"Кол-во пользователей: {num_users}, кол-во фильмов: {num_items}")

    train_model(_num_users=num_users, _num_items=num_items, epochs=args.epochs)
