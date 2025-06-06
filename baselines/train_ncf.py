from __future__ import annotations

import os
import random
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ncf import NeuCF
from utils import ARGS, make_training_data, Evaluate
import argparse

# ---------------------------------------------------------------
# Dataset – point‑wise (user, item, label)
# ---------------------------------------------------------------
class NCFDataset(Dataset):
    """Generate positive + sampled negative pairs for point‑wise BCE training."""

    def __init__(self, data, item_num: int, user_sessions: dict[int, list[int]], neg_num: int = 4):
        """
        Args
        ----
        data: List[(u, input_seq, pos_item)] like make_training_data's output.
              We ONLY care about the final pos_item (recommendation target).
        item_num: total number of items (ID upper‑bound, exclusive).
        user_sessions: {user: [item ids ...]} for negative sampling.
        neg_num: how many negatives to sample *per positive* instance.
        """
        self.user_item_label: list[tuple[int, int, int]] = []  # (u, i, label)
        self.all_items = set(range(item_num))
        self.user_sessions = user_sessions

        for (u, _, pos_item) in data:
            # 1 × positive sample
            self.user_item_label.append((u, pos_item, 1))
            # neg_num × negatives
            for _ in range(neg_num):
                neg_item = random.randint(0, item_num - 1)
                while neg_item in user_sessions[u]:
                    neg_item = random.randint(0, item_num - 1)
                self.user_item_label.append((u, neg_item, 0))

    # -----------------------------------------------------------
    def __len__(self):
        return len(self.user_item_label)

    def __getitem__(self, idx):
        u, i, l = self.user_item_label[idx]
        return (
            torch.LongTensor([u]),
            torch.LongTensor([i]),
            torch.FloatTensor([l]),  # label as float for BCE
        )

# ---------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------

def train_one_epoch(model: NeuCF, dataloader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.BCELoss()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        users, items, labels = [x.squeeze().to(device) for x in batch]
        preds = model(users, items)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

# ---------------------------------------------------------------
# Predictor for your Evaluate wrapper
# ---------------------------------------------------------------

def build_predict_fn(model: NeuCF, device: torch.device):
    """Return a callable with the signature expected by Evaluate.run()."""

    # def _predict(user_tensor, input_tensor, pos_items, neg_items):  # keep API
    def _predict(user_tensor, pos_items):  # keep API
        model.eval()
        with torch.no_grad():
            user_tensor = user_tensor.squeeze().to(device)
            pos_items = pos_items.to(device)
            # Evaluate only sends *positive* items here, so we just compute those
            return model(user_tensor, pos_items)

    return _predict

# ---------------------------------------------------------------
# 6.  Main
# ---------------------------------------------------------------

def main(output_csv_path="result/ncf_result.csv") -> None:  # noqa: C901 – single entrypoint OK
    # ------------------------ Config -------------------------------------
    dataset_name = "filtered-h-and-m"
    batch_size = 4096*2
    epochs = 50
    lr = 1e-3
    latent_dim = 32
    mlp_layers = (64, 32, 16, 8)  # Feel free to tune
    window_size = 1
    neg_num = 10  # less negs because we use point‑wise BCE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------ Data ---------------------------------------
    base_path = Path(f"./dataset/{dataset_name}/processed_data")
    with open(base_path / "user_session.pkl", "rb") as f:
        user_session = pkl.load(f)
    with open(base_path / "testing_data.pkl", "rb") as f:
        neg_testing = pkl.load(f)

    num_user = max(user_session.keys()) + 1
    num_item = max(i for items in user_session.values() for i in items) + 1
    print(f"Dataset: {dataset_name}, Users: {num_user}, Items: {num_item}")

    train_data, testing_data = make_training_data(user_session, window_size=window_size)
    train_ds = NCFDataset(train_data, num_item, user_session, neg_num=neg_num)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=18,
        pin_memory=True,
        persistent_workers=True,
    )

    # ------------------------ Model --------------------------------------
    args = ARGS(num_user, num_item, latent_dim, learning_rate=lr)
    model = NeuCF(num_user, num_item, emb_dim=latent_dim, layers=mlp_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------------------------ Evaluator ----------------------------------
    evaluator = Evaluate(
        testing_data,
        neg_testing,
        isShow=False,
        device=device,
        output_csv_path=output_csv_path,
    )
    evaluator.set_eva_function(build_predict_fn(model, device))

    # ------------------------ Training loop ------------------------------
    best_hr10 = -1.0
    best_epoch = -1
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_ckpt_path = ckpt_dir / f"ncf_{dataset_name}_best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch}] Train BCE loss: {train_loss:.6f}")

        results, df = evaluator.run(batch_size=512, model='ncf', epochs=epoch)
        evaluator.print_result()
        hr10 = results["num_hit"][10]  # HR@10

        if hr10 > best_hr10:
            best_hr10 = hr10
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)
            df.to_csv(output_csv_path, index=False)
            print(f"📄 Saved batch eval result to {output_csv_path}")
            print(
                f"✅ [Epoch {epoch}] New best HR@10: {best_hr10:.4f} – model saved to {best_ckpt_path}",
                flush=True,
            )
        else:
            print(
                f"[Epoch {epoch}] HR@10 = {hr10:.4f}, best = {best_hr10:.4f} (epoch {best_epoch})",
                flush=True,
            )

    # ------------------------ Final save ---------------------------------
    torch.save(model.state_dict(), ckpt_dir / f"ncf_last_{dataset_name}.pt")
    print(f"Training completed. Last model saved to {ckpt_dir / f'ncf_last_{dataset_name}.pt'}")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_csv_path', type=str, default="result/ncf_result.csv", help='Path to save best result csv')
    args_cmd = parser.parse_args()
    output_csv_path = args_cmd.output_csv_path
    
    main(output_csv_path=output_csv_path)
