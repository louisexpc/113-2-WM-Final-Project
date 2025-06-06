import pickle
import numpy as np
import pandas as pd
import torch
import os
import re
import ast
from tqdm import tqdm
from collections import Counter

from .utils import *
from .module import *

def run_enrichment(cfg, logger):
    # === 設定 GPU ===
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device.cuda_visible_devices

    # === 載入模型 ===
    from sentence_transformers import SentenceTransformer, util
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import login
    from vllm import LLM, SamplingParams

    if cfg.hf.use_token:
        token = os.environ.get(cfg.hf.token_env_key, None)
        if token:
            login(token)

    # model = LLM(model=cfg.model.name, dtype='float16')
    logger.info(f"🚀 載入模型: {cfg.model.name}")
    model = LLM(
        model=cfg.model.name,
        gpu_memory_utilization=cfg.model.gpu_memory_utilization,
        max_model_len=cfg.model.max_model_len,
        max_num_seqs=cfg.model.max_num_seqs,
         dtype='float16'
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    # === Pre-Filter Categories with Embeddings ===
    # short_model = SentenceTransformer("all-MiniLM-L6-v2")
    short_model = SentenceTransformer(cfg.short_model.name)

    # === 載入資料 ===
    logger.info("🔍 Loading user session data...")
    raw_data = load_pickle(cfg.data.session_path)
    user_sessions = extract_user_sessions_from_raw(raw_data) #dict[int:List]

    logger.info("📂 Loading categories...")
    
    categories_df = pd.read_csv(cfg.data.category_csv)
    category_list = categories_df["category_name"].tolist()
    logger.info(f"Total categories: {len(category_list)}")

    enriched = {}
    user_ids = list(user_sessions.keys())

    for i, user_id in enumerate(tqdm(user_ids)):
        if i >= cfg.save.max_users:
            logger.info(f"⏹️ Reached max_users: {cfg.save.max_users}")
            break
        """Update: Max Length Limitation"""
        if len(user_sessions[user_id]) > cfg.model.total_len:
            enriched[user_id] = ([int(article_id) for article_id in user_sessions[user_id]], -1) #符合格式, K 用 -1 填補
        else:
            try:
                user_session = {user_id: [int(article_id) for article_id in user_sessions[user_id]]} #確保dtype 正確
                enriched_session, k = enrich_user_sessions( 
                    tokenizer, model, user_session, category_list, 
                    total_len=cfg.model.total_len, top_k=cfg.model.top_k, 
                    short_model=short_model, prompt_path=cfg.data.prompt_path, 
                    mapping_path= cfg.data.mapping_path)
                
                enriched_session[:k-1] = user_sessions[user_id][:-1]  # 保留原始 session 的前 k-1 個項目
                enriched_session.append(user_sessions[user_id][-1])  # 最後一個項目是 enriched 的結果
                enriched[user_id] = (enriched_session, k)

            except Exception as e:
                logger.warning(f"❌ Failed for user {user_id}: {e}")

        if (i + 1) % cfg.save.batch_size == 0:
            checkpoint_path = os.path.join(cfg.save.checkpoint_path)
            with open(checkpoint_path, "wb") as f:
                pickle.dump(enriched, f)
            logger.info(f"💾 Saved checkpoint at {i+1} users")

    # === 最終儲存 ===
    final_path = os.path.join(cfg.save.final_output_path)
    with open(final_path, "wb") as f:
        pickle.dump(enriched, f)
    logger.info(f"✅ Final output saved to {final_path}")

    return model, tokenizer
