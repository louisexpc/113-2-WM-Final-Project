import torch
import torch.nn as nn

class NeuCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, layers=(64,32,16,8)):
        super().__init__()
        # GMF
        self.user_gmf = nn.Embedding(n_users, emb_dim)
        self.item_gmf = nn.Embedding(n_items, emb_dim)
        # MLP
        self.user_mlp = nn.Embedding(n_users, emb_dim)
        self.item_mlp = nn.Embedding(n_items, emb_dim)
        mlp_modules = []
        input_size = emb_dim * 2
        for l in layers:
            mlp_modules += [nn.Linear(input_size, l), nn.ReLU()]
            input_size = l
        self.mlp = nn.Sequential(*mlp_modules)
        # 融合 & 輸出
        self.predict = nn.Linear(emb_dim + layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u, i):
        gmf_out = self.user_gmf(u) * self.item_gmf(i)
        mlp_out = self.mlp(torch.cat([self.user_mlp(u), self.item_mlp(i)], dim=-1))
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.sigmoid(self.predict(concat)).squeeze()
