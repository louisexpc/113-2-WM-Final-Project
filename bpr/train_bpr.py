import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import pickle as pkl
import os

from bpr import BPR
from utils import ARGS, make_training_data, Evaluate

# ---------------------------------------------
# Dataset Class
class BPRDataset(Dataset):
    def __init__(self, data, item_num, user_sessions, neg_num=1):
        self.data = data
        self.item_num = item_num
        self.user_sessions = user_sessions
        self.neg_num = neg_num
        self.all_items = set(range(item_num))

    def __len__(self):
        return len(self.data) * self.neg_num

    def __getitem__(self, idx):
        user, pos_seq, pos_item = self.data[idx % len(self.data)]

        # 選一個負樣本
        neg_item = random.randint(0, self.item_num - 1)
        while neg_item in self.user_sessions[user]:
            neg_item = random.randint(0, self.item_num - 1)

        return torch.LongTensor([user]), torch.LongTensor([pos_item]), torch.LongTensor([neg_item])

# ---------------------------------------------
# Training Loop
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        user, pos_item, neg_item = [x.squeeze().to(device) for x in batch]

        pos_score, neg_score = model(user, pos_item, neg_item)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# ---------------------------------------------
# 評估封裝成 callable
def get_predict_function(model, device):
    def predict(user_tensor, input_tensor, pos_items, neg_items):
        model.eval()
        with torch.no_grad():
            user_tensor = user_tensor.squeeze().to(device)
            pos_items = pos_items.to(device)
            return model(user_tensor, pos_items, pos_items)
    return predict

# ---------------------------------------------
# Main Function
def main():
    # ---------- Config ----------
    dataset_name = "filtered-h-and-m"
    batch_size = 4096
    epochs = 1
    lr = 0.001
    latent_dim = 32
    window_size = 1
    neg_num = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load Data ----------ot
    base_path = f"./dataset/{dataset_name}/processed_data"
    with open(os.path.join(base_path, "user_session.pkl"), "rb") as f:
        user_session = pkl.load(f)

    with open(os.path.join(base_path, "testing_data.pkl"), "rb") as f:
        neg_testing = pkl.load(f)

    num_user = max(user_session.keys()) + 1
    num_item = max(i for items in user_session.values() for i in items) + 1

    print(f"Dataset: {dataset_name}, Users: {num_user}, Items: {num_item}")

    train_data, testing_data = make_training_data(user_session, window_size=window_size)
    dataset = BPRDataset(train_data, num_item, user_session, neg_num=neg_num)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    # ---------- Init Model ----------
    args = ARGS(num_user, num_item, latent_dim, learning_rate=lr)
    model = BPR(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---------- Evaluator ----------
    evaluator = Evaluate(testing_data, neg_testing, isShow=False, device=device, output_csv_path="result/best_result.csv")
    evaluator.set_eva_function(get_predict_function(model, device))

    # ---------- Train ----------
    # before training loop
    best_hr10 = -1
    best_epoch = -1
    model_save_path = f"checkpoints/bpr_{dataset_name}_best.pt"
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"[Epoch {epoch}] Loss: {loss:.4f}")

        # Evaluate
        results = evaluator.run(batch_size=512)
        evaluator.print_result()

        hr10 = results["num_hit"][10]  # ← HR@10

        if hr10 > best_hr10:
            best_hr10 = hr10
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ [Epoch {epoch}] New best HR@10: {best_hr10:.4f}, model saved to {model_save_path}")
        else:
            print(f"[Epoch {epoch}] HR@10 = {hr10:.4f}, best = {best_hr10:.4f} (epoch {best_epoch})")

    # ---------- Save Model ----------
    torch.save(model.state_dict(), f"checkpoints/bpr_last_{dataset_name}.pt")
    print(f"Model saved to checkpoints/bpr_last_{dataset_name}.pt")

if __name__ == "__main__":
    main()
