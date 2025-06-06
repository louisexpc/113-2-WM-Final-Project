import pickle as pkl
import pandas as pd

# 1. 讀取 short session user id
with open("./result/origin_54_mapping_ncf/short_session_user_analysis.pkl", "rb") as f:
    short_user_data = pkl.load(f)

# 取出 <5, <10 的 user id
lt5_users = set(short_user_data['lt5']['user_ids'])
lt10_users = set(short_user_data['lt10']['user_ids'])

print(len(lt5_users))
print(len(lt10_users))

# 2. 讀取不同結果檔案
df_aug = pd.read_csv("./result/ncf_result_aug_short_54_dense.csv")

# 觀察 hit@10 狀況
control_hit = "hit@10"   # 修改為你表格的正確欄名

# 針對 <5
lt5_in_df = df_aug[df_aug['user_id'].isin(lt5_users)]
lt5_total = len(lt5_in_df)
lt5_hit = lt5_in_df[lt5_in_df[control_hit] == 1].shape[0]
lt5_ratio = lt5_hit / lt5_total if lt5_total > 0 else 0

# 針對 <10
lt10_in_df = df_aug[df_aug['user_id'].isin(lt10_users)]
lt10_total = len(lt10_in_df)
lt10_hit = lt10_in_df[lt10_in_df[control_hit] == 1].shape[0]
lt10_ratio = lt10_hit / lt10_total if lt10_total > 0 else 0

try:
    print(f"[AUG版本] session length < 5: hit={lt5_hit}, miss={lt5_total-lt5_hit}, ratio={lt5_hit/(lt5_total-lt5_hit):.4f}  命中率 = {lt5_hit} / {lt5_total} = {lt5_ratio:.4f}")
except ZeroDivisionError:
    print(f"[AUG版本] session length < 5: hit={lt5_hit}, miss={lt5_total-lt5_hit}, ratio={lt5_hit}/{lt5_total-lt5_hit} 命中率 = {lt5_hit} / {lt5_total} = {lt5_ratio:.4f}")
    
print(f"[AUG版本] session length < 10: hit={lt10_hit}, miss={lt10_total-lt10_hit}, ratio={lt10_hit/(lt10_total-lt10_hit):.4f} 命中率 = {lt10_hit} / {lt10_total} = {lt10_ratio:.4f}")

# 進一步可比較舊版本（沒aug那個csv），直接同法，讀入舊csv做對比即可
