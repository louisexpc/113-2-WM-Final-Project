import pickle
import numpy as np
import pandas as pd
import torch

import re
import ast
from tqdm import tqdm
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from vllm import LLM, SamplingParams
# ---- Load data ----
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# --- Extract session from raw data (user-item mapping) ---
def extract_user_sessions_from_raw(raw_data):
    if not raw_data:
        return ValueError("Input data must contain value")
    
    simplified = {}
    for user_id, fields in raw_data.items():
        if 'article_id' not in fields:
            raise ValueError(f"User {user_id} does not contain 'artivle_id' key." )
        simplified[user_id] = fields["article_id"]

    return simplified

# --- Convert article_id to product type (mapping file needed) ---
## TODO
def article_to_product_type(article_id, path):
    # mapping = load_pickle("article_to_product_mapping.pkl")
    # return mapping.get(article_id, "Unknown Product Type")
    article_product_df = pd.read_csv(path)
    print(f"article_id: {article_id}")
    print(f"article_product_df['article_id']: {article_product_df['article_id']}")
    match = article_product_df[article_product_df['article_id'] == article_id]['product_type_name']
    if not match.empty:
        return match.iloc[0]  # 回傳第一個值的字串
    else:
        return "Unknown Product Type"


# === Pre-Filter Categories with Embeddings ===
def prefilter_categories(items_list, category_list, top_k, model):
    
    
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

# === 從描述中抽出乾淨名稱 ===
def extract_product_name(text, fallback="Unknown"):
    import re
    match = re.search(r"['\"]?([\w\s\-]+)['\"]?", text)
    return match.group(1).strip() if match else fallback


def extract_python_list(text, fallback=None):
    if fallback is None:
        fallback = []

    print(f"LLM generate: {text}")
    # 嘗試找出第一個類似 ["A", "B", ...] 的部分
    match = re.search(r"\[[^\[\]]+\]", text)
    if match:
        try:
            result = ast.literal_eval(match.group(0))
            # 確保是 list 且項目是 str
            if isinstance(result, list) and all(isinstance(item, str) for item in result):
                return result
        except Exception as e:
            print("⚠️ Eval error:", e)
    
    print("⚠️ 找不到 Python list 格式的回答，使用 fallback")
    return fallback

def get_categories_from_history(tokenizer, model, history_str, category_list, total_len, prompt_path):

    # Read prompt from file
    with open(prompt_path, "r", encoding='utf-8') as f:
        user_prompt_template = f.read()

    history_tokens = history_str
    

    # 少最後一筆
    k = len(history_tokens)
    trimmed_history = history_tokens[:-1]
    print(f"Trimmed history ({len(trimmed_history)}): {trimmed_history}")
    print(f"Ground truth (held out): {history_tokens[-1]}")


    history_counter = Counter(history_tokens)
    history_categories = list(history_counter.keys())
    print(f"history_categories: {history_categories}")
    print(f"frequency_count: {history_counter}")

    full_list = trimmed_history
    print(f"Initial history ({len(full_list)}): {full_list}")

    attempts = 0
    max_attempts = 5 # 避免無限loop

    while len(full_list) < total_len and attempts < max_attempts:


        # 只排除已出現的品類，不考慮次數
        remaining_categories = [c for c in category_list if c not in history_counter]

        num_to_add = total_len - len(trimmed_history)  # 注意這裡用 tokens 數量！

        print(f"num_to_add: {num_to_add}")
        
        if num_to_add <= 0:
            return history_tokens[:total_len]  # 若 history 太多，就截斷

        categories_str = ", ".join(remaining_categories)

        system_prompt = "You are a strict API for fashion recommendations. Do not respond with any explanation or formatting outside of a Python list."

        user_prompt = user_prompt_template.format(
            trimmed_history = trimmed_history,
            category_list = category_list,
            num_to_add = num_to_add    
        )

        # # llama2
        # prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        # llama3
        prompt = (
            "<|start_header_id|>system<|end_header_id|>\n" +
            system_prompt + "\n" +
            "<|start_header_id|>user<|end_header_id|>\n" +
            user_prompt + "\n" +
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

        ## 傳統方法
        # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # outputs = model.generate(**inputs, max_new_tokens=512)
        # response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        ## vllm
        # 定義sampling參數
        ## llama2
        # sampling_params = SamplingParams(temperature=0.7, max_tokens = 2048, stop=["</s>"])
        # llama3
        sampling_params = SamplingParams(temperature=0.7, max_tokens=1024, stop=["<|end_of_text|>", "<|start_header_id|>"])  # 讓它只輸出 Assistant 回應

        outputs = model.generate(prompt, sampling_params)
        response = outputs[0].outputs[0].text.strip()

        print(f"response: {response}, 長度:{len(response)}")

        # response = response.split("[/INST]")[-1].strip()
        
        print(f"response: {response}, 長度:{len(response)}")

        categories = extract_python_list(response, fallback=["tops", "shoes", "accessories"])
        print("✔️ Extracted:", categories)

        full_list += categories

        attempts += 1
    
    print(f"✅ Final list ({len(full_list)}): {full_list}")
    # full_list = history_tokens + categories
    return full_list[:total_len], k