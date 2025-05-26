from .utils import *

# === 主邏輯：補足使用者 Session 到 30 筆 ===
def enrich_user_sessions(tokenizer, model, user_sessions, category_list, total_len, top_k, short_model, prompt_path):
    enriched_sessions = {}
    for uid, items in user_sessions.items():
        needed = total_len - len(items)

        # Step 1: 歷史描述轉文字
        history_str = " ".join([article_to_product_type(item) for item in items])
        print(f"User: {uid}, History: {history_str}\n")

        # Step 2: 取得前 top_k 類別
        top_categories = prefilter_categories(history_str, category_list, top_k=top_k, model = short_model)
        print(f"Top {top_k} categories: {top_categories}\n")
        
        # Generate Categories
        raw_output = get_categories_from_history(tokenizer, model, history_str, category_list, total_len, prompt_path)
        print("LLM raw output:",raw_output, "\n")
        print(f"長度:{len(raw_output)}")
        print("End of the output")
        
    return raw_output