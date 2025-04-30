import torch
import torch.nn as nn


class NeuMF(nn.Module):

    def __init__(
            self,
            num_users: int,
            num_items: int,
            embedding_dim_gmf: int = 32,
            embedding_dim_mlp: int = 64,
            mlp_hidden_layers: list[int] = [128, 64],
            dropout: float = 0.2,
            use_batchnorm: bool = True
    ):
        super(NeuMF, self).__init__()

        # GMF embedding
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim_gmf)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim_gmf)

        # MLP embedding
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim_mlp)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim_mlp)

        # MLP layers
        mlp_layers = []
        input_size = embedding_dim_mlp * 2

        for hidden_size in mlp_hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            if use_batchnorm:
                mlp_layers.append(nn.BatchNorm1d(hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout))
            input_size = hidden_size

        self.mlp = nn.Sequential(*mlp_layers)

        # Output layer: объединяет GMF + MLP
        final_input_size = embedding_dim_gmf + mlp_hidden_layers[-1]
        self.output = nn.Linear(final_input_size, 1)

    def forward(self, user_ids, item_ids):
        # GMF поток
        user_gmf = self.user_embedding_gmf(user_ids)
        item_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_gmf * item_gmf  # элементное произведение

        # MLP поток
        user_mlp = self.user_embedding_mlp(user_ids)
        item_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Объединение потоков
        combined = torch.cat([gmf_output, mlp_output], dim=-1)

        # Прогноз и применение Sigmoid
        score = self.output(combined)
        return torch.sigmoid(score).squeeze()
