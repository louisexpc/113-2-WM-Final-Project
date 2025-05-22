import pandas as pd
import pickle as pkl
import numpy as np
import os
from collections import Counter

# --- 載入資料 ---
# 1. 命中資訊
df = pd.read_csv("./result/best_result.csv")

# 2. 使用者行為紀錄
with open("./dataset/filtered-h-and-m/processed_data/user_session.pkl", "rb") as f:
    user_session = pkl.load(f)

# --- 分類使用者 ---
control_hit = "hit@10"
hit_users = set(df[df[control_hit] == 1]["user_id"])
miss_users = set(df[df[control_hit] == 0]["user_id"])

# --- 計算 session 長度 ---
session_lengths_hit = [len(user_session[u]) for u in hit_users if u in user_session]
session_lengths_miss = [len(user_session[u]) for u in miss_users if u in user_session]

# --- 結果統計 ---
avg_hit = np.mean(session_lengths_hit)
avg_miss = np.mean(session_lengths_miss)

print(f"✅ 命中者數量: {len(session_lengths_hit)}")
print(f"❌ 未命中者數量: {len(session_lengths_miss)}")
print(f"\n🎯 命中者平均 session 長度: {avg_hit:.2f}")
print(f"💤 未命中者平均 session 長度: {avg_miss:.2f}")

# --- 所有使用者 session 長度資訊 ---
session_lens = [len(s) for s in user_session.values()]
print(f"\n📊 所有使用者最短 session 長度: {min(session_lens)}")
print(f"📊 所有使用者最長 session 長度: {max(session_lens)}")
print(f"📊 所有使用者平均 session 長度: {np.mean(session_lens):.2f}")

# --- 儲存清單 ---
hit_user_list = [u for u in hit_users if u in user_session]
miss_user_list = [u for u in miss_users if u in user_session]

output = {
    "hit_users": hit_user_list,
    "miss_users": miss_user_list,
    "hit_session_lengths": session_lengths_hit,
    "miss_session_lengths": session_lengths_miss,
    "avg_hit": avg_hit,
    "avg_miss": avg_miss,
}

with open("./result/hit_miss_user_analysis.pkl", "wb") as f:
    pkl.dump(output, f)

print("\n📦 已儲存命中與未命中使用者清單與統計至 hit_miss_user_analysis.pkl")


import matplotlib.pyplot as plt
import seaborn as sns

# --- 畫圖並儲存 ---
# sns.set(style="whitegrid")
# plt.figure(figsize=(12, 6))

# sns.kdeplot(session_lengths_hit, label="Hit", fill=True)
# sns.kdeplot(session_lengths_miss, label="Miss", fill=True)

# plt.title("Session Length Distribution: Hit vs Miss", fontsize=16)
# plt.xlabel("Session Length", fontsize=12)
# plt.ylabel("Density", fontsize=12)
# plt.legend()
# plt.tight_layout()

# os.makedirs("./result", exist_ok=True)
# plt.savefig("./result/session_length_distribution.png")
# plt.close()
# print("✅ 已儲存圖表至 ./result/session_length_distribution.png")

# # --- 畫圖 (focus: 0~14) ---
# sns.set(style="whitegrid")
# plt.figure(figsize=(12, 6))
# sns.kdeplot(session_lengths_hit, label="Hit", fill=True, clip=(0, 14))
# sns.kdeplot(session_lengths_miss, label="Miss", fill=True, clip=(0, 14))

# plt.title("Session Length Distribution (Zoomed in: 0–14)", fontsize=16)
# plt.xlabel("Session Length", fontsize=12)
# plt.ylabel("Density", fontsize=12)
# plt.xlim(0, 14)
# plt.legend()
# plt.tight_layout()

# os.makedirs("./result", exist_ok=True)
# plt.savefig("./result/session_length_distribution_zoomed.png")
# plt.close()

# print("✅ 已儲存圖表至 ./result/session_length_distribution_zoomed.png")


# --- 畫直方圖 ---
plt.figure(figsize=(12, 6))
bins = range(0, 30+1)

plt.hist(session_lengths_hit, bins=bins, alpha=0.6, label="Hit", color='tab:blue', edgecolor='black')
plt.hist(session_lengths_miss, bins=bins, alpha=0.6, label="Miss", color='tab:orange', edgecolor='black')

plt.title("Session Length Histogram: Hit vs Miss (0–30)", fontsize=16)
plt.xlabel("Session Length", fontsize=12)
plt.ylabel("Number of Users", fontsize=12)
plt.legend()
plt.tight_layout()

# --- 儲存圖檔 ---
os.makedirs("./result", exist_ok=True)
plt.savefig("./result/session_length_histogram.png")
plt.close()

print("✅ 已儲存直方圖至 ./result/session_length_histogram.png")



# 統計每個 session 長度下 hit 和 miss 用戶數
hit_count = Counter(session_lengths_hit)
miss_count = Counter(session_lengths_miss)
all_lengths = sorted(set(session_lengths_hit + session_lengths_miss))

all_lengths = all_lengths[:101]  # 只取前 15 個長度

hit_to_miss_ratio = []
hit_list = []
miss_list = []

for l in all_lengths:
    n_hit = hit_count.get(l, 0)
    n_miss = miss_count.get(l, 0)
    hit_list.append(n_hit)
    miss_list.append(n_miss)
    if n_miss > 0:
        ratio = n_hit / n_miss
    elif n_hit > 0:
        ratio = float('inf')  # 只命中沒 miss
    else:
        ratio = 0  # 這長度沒人
    hit_to_miss_ratio.append(ratio)
    
    
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(all_lengths, hit_to_miss_ratio, marker="o", label="Hit:Miss Ratio")
plt.title("Hit:Miss Ratio by Session Length", fontsize=16)
plt.xlabel("Session Length", fontsize=12)
plt.ylabel("Hit / Miss Ratio", fontsize=12)
plt.yscale('log')  # 若部分比例很大，可以用對數軸
plt.grid(True, which='both', axis='y')
plt.legend()
plt.tight_layout()
plt.savefig("./result/session_length_hit_to_miss_ratio.png")
plt.close()
print("✅ 已儲存 hit:miss 比例圖至 ./result/session_length_hit_to_miss_ratio.png")



plt.figure(figsize=(12, 6))
plt.bar(all_lengths, hit_list, alpha=0.6, label="Hit", color='tab:blue', edgecolor='black')
plt.bar(all_lengths, miss_list, alpha=0.6, label="Miss", color='tab:orange', edgecolor='black', bottom=hit_list)
plt.title("Hit vs Miss User Count by Session Length", fontsize=16)
plt.xlabel("Session Length", fontsize=12)
plt.ylabel("User Count", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("./result/session_length_hit_miss_usercount.png")
plt.close()
print("✅ 已儲存 hit/miss 用戶數柱狀圖至 ./result/session_length_hit_miss_usercount.png")

#---
import numpy as np
import matplotlib.pyplot as plt

# hit_list, miss_list, all_lengths 應來自你的前面程式

### 1. 過濾掉樣本數過少的 session length
min_count = 10
filtered_lengths = []
filtered_ratios = []
filtered_user_count = []

for l, h, m in zip(all_lengths, hit_list, miss_list):
    total = h + m
    if total >= min_count:
        filtered_lengths.append(l)
        filtered_ratios.append(h / m if m > 0 else float('inf'))
        filtered_user_count.append(total)

### 2. session length 分 bin 處理
bin_size = 5
max_len = max(all_lengths)
bins = list(range(0, max_len+bin_size, bin_size))

binned_hit = np.zeros(len(bins)-1)
binned_miss = np.zeros(len(bins)-1)
binned_user = np.zeros(len(bins)-1)

for l, h, m in zip(all_lengths, hit_list, miss_list):
    idx = np.digitize(l, bins) - 1
    if 0 <= idx < len(binned_hit):
        binned_hit[idx] += h
        binned_miss[idx] += m
        binned_user[idx] += h + m

binned_ratio = [h/m if m > 0 else float('inf') for h, m in zip(binned_hit, binned_miss)]
bin_centers = [0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)]

### 3. 畫圖
plt.figure(figsize=(14, 7))

# (A) 單一 session length 且人數 >= min_count
plt.plot(filtered_lengths, filtered_ratios, marker="o", linestyle="-", label=f"Per Length (n ≥ {min_count})", alpha=0.8)

# (B) Binned session length
plt.plot(bin_centers, binned_ratio, marker="s", linestyle="--", label=f"Binned (每{bin_size}個分組)", color="tab:orange", alpha=0.8)

plt.title("Hit:Miss Ratio by Session Length (Filtered & Binned)", fontsize=18)
plt.xlabel("Session Length", fontsize=14)
plt.ylabel("Hit / Miss Ratio", fontsize=14)
plt.yscale('log')
plt.grid(True, which='both', axis='y')
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("./result/session_length_hit_to_miss_ratio_combined.png")
plt.close()

print("✅ 已儲存合併圖於 ./result/session_length_hit_to_miss_ratio_combined.png")

### 可選：加碼存表格
import pandas as pd

df_ratio = pd.DataFrame({
    'session_length': filtered_lengths,
    'user_count': filtered_user_count,
    'hit_to_miss_ratio': filtered_ratios
})
df_ratio.to_csv("./result/session_length_hit_to_miss_ratio_filtered.csv", index=False)

df_binned = pd.DataFrame({
    'bin_center': bin_centers,
    'binned_user_count': binned_user,
    'binned_hit_to_miss_ratio': binned_ratio
})
df_binned.to_csv("./result/session_length_hit_to_miss_ratio_binned.csv", index=False)
print("✅ 已儲存細分與分 bin 統計表於 ./result/")
