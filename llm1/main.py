import pickle
import numpy as np
import pandas as pd
import torch
import os
os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID" #
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import re
import ast
from tqdm import tqdm
from collections import Counter
from utils import *
from module import *
import argparse

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import login
login("Token")

def parse_args():
    parser = argparse.ArgumentParser(description="Enrich user sessions with LLM + category embedding")
    
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="HuggingFace model name")
    parser.add_argument("--session_path", type=str, default="dataset/sessions_5_4.pkl", required=True, help="Path to user session .pkl file")
    parser.add_argument("--category_csv", type=str, default="dataset/product_types.csv", required=True, help="Path to product_types.csv")
    parser.add_argument("--prompt_path", type=str, default="get_product_type_from_history_prompt.txt", required=True, help="Path to LLM prompt .txt file")
    parser.add_argument("--save_path", type=str, default="enriched_sessions_checkpoint.pkl", help="Checkpoint save path")
    parser.add_argument("--final_save_path", type=str, default="enriched_sessions_final.pkl", help="Final output save path")
    parser.add_argument("--batch_size", type=int, default=10, help="Save every N users")
    parser.add_argument("--max_users", type=int, default=10000, help="Maximum number of users to process")
    parser.add_argument("--total_len", type=int, default=30, help="Total session length after enrichment")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top categories for similarity filtering")
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    # --- Load Models ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype = torch.float16)

    # === Pre-Filter Categories with Embeddings ===
    short_model = SentenceTransformer("all-MiniLM-L6-v2")


    raw_data = load_pickle(args.session_path)
    user_sessions = extract_user_sessions_from_raw(raw_data)

    categories_df = pd.read_csv(args.category_csv)
    category_list = categories_df["category_name"].tolist()
    print("Total categories loaded:", len(category_list))
    print("Sample categories:", category_list[:10])
    
    user_ids = list(user_sessions.keys())
    enriched = {}

    for i, user_id in enumerate(tqdm(user_ids)):
        if i >= args.max_users:  # ✅ 超過就停止
            print(f"⏹️ Stopped after processing {args.max_users} users")
            break

        try:
            user_session = {user_id: user_sessions[user_id]}
            enriched_session = enrich_user_sessions(tokenizer, model, user_session, category_list, total_len=args.total_len, top_k=args.top_k, short_model=short_model, prompt_path=args.prompt_path)
            enriched[user_id] = enriched_session
            print(f"✔️ Successfully enriched {user_id}")
        except Exception as e:
            print(f"❌ Failed for user {user_id}: {e}")

        # 每 BATCH_SIZE 筆就寫入檔案
        if (i + 1) % args.batch_size == 0:
            with open(args.save_path, "wb") as f:
                pickle.dump(enriched, f)
            print(f"💾 Saved checkpoint after {i + 1} users")

    # 最後再存一次（包含最後一批）
    with open(args.final_save_path, "wb") as f:
        pickle.dump(enriched, f)
    print("✅ Final file saved.")
