import re
import ast
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from data_loader import load_pickle


def extract_product_name(raw_output, fallback="Unknown Product"):
    """
    從 LLM 的輸出中提取商品名稱，並進行清理
    Args:
        raw_output (str): LLM 的原始輸出
        fallback (str): 如果無法提取商品名稱，則返回的後備值
    Returns:
        str: 提取的商品名稱
    """
    # Split the response into lines
    lines = raw_output.strip().split("\n")

    # Find the last non-empty line
    for line in reversed(lines):
        clean_line = line.strip()
        if clean_line:
            # Remove common prefixes
            for prefix in ["Sure,", "Here is", "Here are", "Of course,", "I'd be happy to help!"]:
                if clean_line.lower().startswith(prefix.lower()):
                    clean_line = clean_line[len(prefix):].strip()
            # Remove category hints if present
            clean_line = re.sub(r'for the ".+?" category', "", clean_line).strip()
            
            # Return the cleaned line if it looks like a product name
            if clean_line and len(clean_line.split()) >= 2:
                return clean_line

    # Fallback if no valid name is found
    print(f"⚠️ No valid product name found in: '{raw_output}'")
    return fallback


def extract_list_from_text(text, fallback=None):
    """
    從 LLM 的輸出中提取 Python 列表，並進行清理
    Args:
        text (str): LLM 的原始輸出
        fallback (list): 如果無法提取列表，則返回的後備值
    Returns:
        list: 提取的列表
    """
    if not text or not isinstance(text, str):
        print("⚠️ Invalid input. Using fallback.")
        return fallback or []

    # Strip leading instructions and whitespace
    cleaned_text = text.strip()

    # Extract the first valid list
    match = re.search(r'\[\s*(?:[^\[\]]|\[[^\[\]]*\])*\s*\]', cleaned_text, re.DOTALL)
    if match:
        try:
            extracted = match.group(0).replace("\n", "").replace("    ", "").strip()
            # Basic sanity check to avoid explanation leakage
            if extracted.startswith("[") and extracted.endswith("]"):
                return ast.literal_eval(extracted)
            else:
                print("⚠️ Unexpected non-list format. Using fallback.")
        except Exception as e:
            print(f"⚠️ Parsing failed: {e}")
    
    print("⚠️ No valid list found. Using fallback.")
    return fallback or []


def get_user_info(user_id: str, user_dataset: pd.DataFrame):
    """
    根據使用者 ID 從資料集中獲取使用者資訊
    Args:
        user_id (str): 使用者 ID
        user_dataset (pd.DataFrame): 包含使用者資訊的資料集，必須包含 "customer_id" 欄位
    Returns:
        pd.DataFrame: 包含該使用者資訊的資料集
    """
    user_info = user_dataset[user_dataset["customer_id"] == user_id]
    return user_info


def get_current_season():
    """
    Deprecated
    獲取當前季節
    Returns:
        str: 當前季節名稱
    """

    current_date = datetime.now()
    current_month = current_date.month
    if current_month in [12, 1, 2]:
        current_season = "Winter"
    elif current_month in [3, 4, 5]:
        current_season = "Spring"
    elif current_month in [6, 7, 8]:
        current_season = "Summer"
    else:
        current_season = "Autumn"
    
    return current_season


def get_product_name_examples(product_type):
    """
    從 CSV 檔案中讀取特定類別的商品名稱範例
    
    Args:
        product_type (str): 商品類別名稱

    Returns:
        str: 該類別的商品名稱範例，以逗號分隔
    """
    try:
        # 讀取 CSV 檔案，指定分隔符為 ':'
        product_examples = pd.read_csv("product_name_examples.csv", 
                                      sep=':', 
                                      names=['product_type', 'examples'])

        # 找到對應類別並回傳範例
        examples = product_examples[product_examples["product_type"] == product_type]["examples"].values

        if len(examples) == 0:
            print(f"⚠️ 找不到類別 '{product_type}' 的範例")
            return ""
            
        return examples[0]
        
    except Exception as e:
        print(f"⚠️ 讀取範例時發生錯誤: {e}")
        return ""
    
    
# === Pre-Filter Categories with Embeddings ===
def prefilter_categories(items_list, category_list, top_k=20):
    """
    使用 SentenceTransformer 模型預過濾類別
    Args:
        items_list (list): 商品名稱列表
        category_list (list): 類別名稱列表
        top_k (int): 返回的前 K 個類別數量
    Returns:
        list: 前 K 個類別名稱
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Encode the history as a single vector
    history_embedding = model.encode(items_list, convert_to_tensor=True)
    
    # Encode all categories
    category_embeddings = model.encode(category_list, convert_to_tensor=True)
    
    # Compute similarity
    cos_scores = util.cos_sim(history_embedding, category_embeddings)[0]
    top_results = cos_scores.topk(k=top_k)
    
    # Extract top-K categories
    top_categories = [category_list[i] for i in top_results.indices]
    top_scores = [cos_scores[i].item() for i in top_results.indices]
    
    # Return as a dictionary for better interpretability
    return top_categories


def article_to_product_type(article_id):
    """
    根據 articleID 獲取對應的商品類別
    Args:
        article_id (str): article ID
    Returns:
        str: 對應的商品類別
    """
    mapping = load_pickle("article_to_product_mapping.pkl")
    return mapping.get(article_id, "Unknown Product Type")
