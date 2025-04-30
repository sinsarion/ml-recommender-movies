import pickle

import torch
from torch.utils.data import Dataset, DataLoader


class NeuMFDataset(Dataset):

    def __init__(self, path: str):
        print(f"Загрузка датасета из {path}...")
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.users = torch.tensor([u for u, i, l in data], dtype=torch.long)
        self.items = torch.tensor([i for u, i, l in data], dtype=torch.long)
        self.labels = torch.tensor([l for u, i, l in data], dtype=torch.float32)

        print(f"Загружено {len(self.users)} примеров")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


def get_dataloaders(path: str, batch_size=1024, num_workers=2, test_split=0.1):
    dataset = NeuMFDataset(path)
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
