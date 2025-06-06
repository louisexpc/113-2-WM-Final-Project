import pickle
from collections import Counter

def check_list_lengths(pickle_file_path, expected_length_K):
    """
    讀取 pickle 檔案並檢查每個 value 中的 list 長度是否為給定的 K
    
    Args:
        pickle_file_path (str): pickle 檔案路徑
        expected_length_K (int): 預期的 list 長度
    
    Returns:
        dict: 檢查結果，包含統計資訊
    """
    try:
        # 讀取 pickle 檔案
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"成功讀取 pickle 檔案: {pickle_file_path}")
        print(f"資料類型: {type(data)}")
        print(f"總共有 {len(data)} 個項目")
        
        # 收集所有長度並統計
        lengths = []
        correct_length_count = 0
        incorrect_items = []
        correct_items = []  # 新增：收集長度正確的項目
        non_list_items = []
        
        for key, value in data.items():
            if isinstance(value, list):
                length = len(value)
                lengths.append(length)
                
                if length == expected_length_K:
                    correct_length_count += 1
                    correct_items.append({
                        'key': key,
                        'length': length,
                        'content_preview': value[:3] if len(value) >= 3 else value  # 顯示前3個元素作為預覽
                    })
                else:
                    incorrect_items.append({
                        'key': key,
                        'actual_length': length,
                        'expected_length': expected_length_K
                    })
            else:
                non_list_items.append({
                    'key': key,
                    'type': type(value).__name__
                })
        
        # 統計每個長度的個數
        length_counter = Counter(lengths)
        
        # 輸出長度統計
        print(f"\n=== 長度統計 ===")
        print(f"總共有 {len(lengths)} 個 list")
        print(f"不同長度的分布:")
        
        # 按長度排序顯示
        for length in sorted(length_counter.keys()):
            count = length_counter[length]
            percentage = count / len(lengths) * 100
            marker = " ← 目標長度" if length == expected_length_K else ""
            print(f"  長度 {length:3d}: {count:6d} 個 ({percentage:5.1f}%){marker}")
        
        # 輸出基本檢查結果
        print(f"\n=== 檢查結果 ===")
        print(f"預期長度: {expected_length_K}")
        print(f"長度正確的項目數量: {correct_length_count}")
        print(f"長度不正確的項目數量: {len(incorrect_items)}")
        if lengths:
            print(f"正確率: {correct_length_count / len(lengths) * 100:.2f}%")
        
        # 顯示長度正確的項目範例
        if correct_items:
            print(f"\n=== 長度正確的項目範例 ===")
            for item in correct_items[:5]:  # 只顯示前 5 個
                print(f"Key: {item['key']} - 長度: {item['length']}")
                print(f"  內容預覽: {item['content_preview']}")
            
            if len(correct_items) > 5:
                print(f"... 還有 {len(correct_items) - 5} 個項目長度正確")
        
        # 顯示非 list 項目
        if non_list_items:
            print(f"\n=== 非 list 類型的項目 ===")
            print(f"共 {len(non_list_items)} 個項目不是 list:")
            type_counter = Counter([item['type'] for item in non_list_items])
            for type_name, count in type_counter.items():
                print(f"  {type_name}: {count} 個")
        
        # 顯示長度不正確的項目範例
        if incorrect_items:
            print(f"\n=== 長度不正確的項目範例 ===")
            for item in incorrect_items[:5]:  # 只顯示前 5 個
                print(f"Key: {item['key']} - 實際長度: {item['actual_length']}")
            
            if len(incorrect_items) > 5:
                print(f"... 還有 {len(incorrect_items) - 5} 個項目長度不正確")
        
        return {
            'total_items': len(data),
            'list_items': len(lengths),
            'non_list_items': len(non_list_items),
            'length_distribution': dict(length_counter),
            'correct_count': correct_length_count,
            'incorrect_count': len(incorrect_items),
            'accuracy': correct_length_count / len(lengths) if lengths else 0,
            'correct_items': correct_items,  # 新增：返回正確項目
            'incorrect_items': incorrect_items,
            'non_list_items': non_list_items
        }
        
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {pickle_file_path}")
        return None
    except Exception as e:
        print(f"錯誤: {str(e)}")
        return None

# 使用範例
if __name__ == "__main__":
    # 請修改這些參數
    pickle_file_path = "dataset/generated_product_name_4_15.pkl"  # 你的 pickle 檔案路徑
    expected_K = 16  # 預期的 list 長度
    
    result = check_list_lengths(pickle_file_path, expected_K)
    
    if result:
        print(f"\n=== 最終統計摘要 ===")
        print(f"總項目數: {result['total_items']}")
        print(f"List 項目數: {result['list_items']}")
        print(f"目標長度 {expected_K} 的正確率: {result['accuracy']:.2%}")
        print(f"最常見的長度: {max(result['length_distribution'], key=result['length_distribution'].get) if result['length_distribution'] else 'N/A'}")