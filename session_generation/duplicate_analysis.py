"""
duplicate_analysis.py
這份程式用以分析生成的 pickle 檔案中的重複數字和 None 值。
"""

import pickle
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # 設定後端為 Agg
import matplotlib.pyplot as plt

METHOD = 'bm25'  # 可以是 'tfidf', 'bm25', 'dense', 'hybrid'
LENGTH = "6"
# 讀取 pickle 檔案
with open(f'dataset/6/generated_real_name_{LENGTH}_{METHOD}.pkl', 'rb') as f:
    data = pickle.load(f)

# 檢查資料型態和筆數
print(f"資料型態: {type(data)}")

if isinstance(data, dict):
    print(f"字典總共有 {len(data)} 個鍵值對")
    print(f"鍵的範例: {list(data.keys())[:5]}")  # 顯示前5個鍵
elif isinstance(data, list):
    print(f"列表總共有 {len(data)} 筆資料")
else:
    print(f"資料長度: {len(data) if hasattr(data, '__len__') else '無法計算長度'}")

# 分析每個 list 中的重複數字和 None 值
duplicate_counts = []  # 儲存每個 list 中重複數字的總數
max_repeat_counts = []  # 儲存每個 list 中最大重複次數
none_counts = []  # 儲存每個 list 中 None 的數量
lists_with_none = 0  # 有 None 值的 list 數量
session_lengths = []  # 儲存每個 session 的長度

for key, value_list in data.items():
    if isinstance(value_list, list):
        # 檢查 session 長度
        session_length = len(value_list)
        session_lengths.append(session_length)
        # 檢查 None 值
        none_count = value_list.count(None)
        none_counts.append(none_count)
        if none_count > 0:
            lists_with_none += 1
        
        # 過濾掉 None 值後進行重複分析
        filtered_list = [x for x in value_list if x is not None]
        counter = Counter(filtered_list)
        
        # 計算重複的數字總數（出現次數 > 1 的數字）
        duplicate_count = sum(count - 1 for count in counter.values() if count > 1)
        duplicate_counts.append(duplicate_count)
        
        # 找出最大重複次數
        max_repeat = max(counter.values()) if counter.values() else 0
        if max_repeat > 1:
            max_repeat_counts.append(max_repeat)

if session_lengths:
    unique_lengths = set(session_lengths)
    all_same_length = len(unique_lengths) == 1
    
    print(f"\nSession 長度統計:")
    print(f"所有 session 長度是否都相同: {'是' if all_same_length else '否'}")
    
    if all_same_length:
        print(f"所有 session 長度都為: {session_lengths[0]}")
    else:
        min_length = min(session_lengths)
        max_length = max(session_lengths)
        avg_length = sum(session_lengths) / len(session_lengths)
        print(f"最短 session 長度: {min_length}")
        print(f"最長 session 長度: {max_length}")
        print(f"平均 session 長度: {avg_length:.2f}")
        
        # 顯示長度分布
        length_counter = Counter(session_lengths)
        print(f"長度分布:")
        for length, count in sorted(length_counter.items()):
            print(f"  長度 {length}: {count} 個 sessions")


# 計算統計資料
if duplicate_counts:
    avg_duplicates = sum(duplicate_counts) / len(duplicate_counts)
    print(f"\n重複數字統計:")
    print(f"平均重複字數: {avg_duplicates:.2f}")
    
    if max_repeat_counts:
        min_repeat = min(max_repeat_counts)
        max_repeat = max(max_repeat_counts)
        print(f"最少重複次數 (>1): {min_repeat}")
        print(f"最多重複次數: {max_repeat}")
    else:
        print("沒有發現重複的數字")
        
    # 顯示一些範例
    print(f"\n有重複數字的 list 數量: {sum(1 for count in duplicate_counts if count > 0)}")
    print(f"沒有重複數字的 list 數量: {sum(1 for count in duplicate_counts if count == 0)}")

# None 值統計 (只輸出到終端機)
if none_counts:
    total_none = sum(none_counts)
    avg_none = total_none / len(none_counts)
    print(f"\nNone 值統計:")
    print(f"總共有 {total_none} 個 None 值")
    print(f"平均每個 list 有 {avg_none:.2f} 個 None 值")
    print(f"有 None 值的 list 數量: {lists_with_none}")
    print(f"沒有 None 值的 list 數量: {len(none_counts) - lists_with_none}")
    
    if total_none > 0:
        max_none = max(none_counts)
        print(f"單一 list 中最多 None 值數量: {max_none}")

    # 畫重複字數分布的長條圖
    plt.figure(figsize=(12, 5))
    
    # 子圖1: 重複字數分布
    plt.subplot(1, 2, 1)
    duplicate_count_dist = Counter(duplicate_counts)
    x_values = sorted(duplicate_count_dist.keys())
    y_values = [duplicate_count_dist[x] for x in x_values]
    
    plt.bar(x_values, y_values, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Duplicate Count')
    plt.ylabel('Number of Lists')
    plt.title('Distribution of Duplicate Counts')
    plt.grid(True, alpha=0.3)
    
    # 子圖2: 最大重複次數分布
    if max_repeat_counts:
        plt.subplot(1, 2, 2)
        max_repeat_dist = Counter(max_repeat_counts)
        x_values2 = sorted(max_repeat_dist.keys())
        y_values2 = [max_repeat_dist[x] for x in x_values2]
        
        plt.bar(x_values2, y_values2, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Max Repeat Count')
        plt.ylabel('Number of Lists')
        plt.title('Distribution of Max Repeat Counts')
        plt.grid(True, alpha=0.3)
    
    # plt.title("tfidf")  # 或者 "bm25 重複分析" 或 "dense 重複分析"
    # plt.suptitle("TF-IDF Method - Duplicate Analysis", fontsize=16, fontweight='bold')
    # plt.suptitle("BM25 Method - Duplicate Analysis", fontsize=16, fontweight='bold')
    plt.suptitle(f"{METHOD} Method - Duplicate Analysis", fontsize=16, fontweight='bold')
    # plt.suptitle("Hybrid Method - Duplicate Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 為總標題留出空間
    # plt.savefig('duplicate_analysis_4_tfidf.png', dpi=300, bbox_inches='tight')  # 儲存圖片
    # plt.savefig('duplicate_analysis_4_bm25.png', dpi=300, bbox_inches='tight')  # 儲存圖片
    plt.savefig(f'duplicate_analysis_{LENGTH}_{METHOD}.png', dpi=300, bbox_inches='tight')  # 儲存圖片
    # plt.savefig('duplicate_analysis_6_hybrid.png', dpi=300, bbox_inches='tight')  # 儲存圖片
    print(f"圖表已儲存為 duplicate_analysis_{LENGTH}_{METHOD}.png")

    # 如果你想要顯示圖片，可以嘗試：
    # plt.show()