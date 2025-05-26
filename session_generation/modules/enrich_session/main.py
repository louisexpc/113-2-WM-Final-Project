from .utils import load_pickle, extract_user_sessions_from_raw
from .module import enrich_user_sessions
import pickle
import os
from tqdm import tqdm
import torch

def run_enrichment(cfg, output_dir, logger):
    # === 設定 GPU ===
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device.cuda_visible_devices

    # === 載入模型 ===
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import login

    if cfg.hf.use_token:
        token = os.environ.get(cfg.hf.token_env_key, None)
        if token:
            login(token)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, device_map="auto", torch_dtype = torch.float16)
    short_model = SentenceTransformer(cfg.short_model.name)

    # === 載入資料 ===
    logger.info("🔍 Loading user session data...")
    raw_data = load_pickle(cfg.data.session_path)
    user_sessions = extract_user_sessions_from_raw(raw_data)

    logger.info("📂 Loading categories...")
    import pandas as pd
    categories_df = pd.read_csv(cfg.data.category_csv)
    category_list = categories_df["category_name"].tolist()
    logger.info(f"Total categories: {len(category_list)}")

    enriched = {}
    user_ids = list(user_sessions.keys())

    for i, user_id in enumerate(tqdm(user_ids)):
        if i >= cfg.save.max_users:
            logger.info(f"⏹️ Reached max_users: {cfg.save.max_users}")
            break
        try:
            user_session = {user_id: user_sessions[user_id]}
            enriched_session = enrich_user_sessions(
                tokenizer, model, user_session, category_list,
                total_len=cfg.model.total_len, top_k=cfg.model.top_k,
                short_model=short_model, prompt_path=cfg.data.prompt_path,mapping_file_path=cfg.data.article_to_product_mapping_path
            )
            enriched[user_id] = enriched_session
        except Exception as e:
            logger.warning(f"❌ Failed for user {user_id}: {e}")

        if (i + 1) % cfg.save.batch_size == 0:
            checkpoint_path = os.path.join(output_dir, "checkpoint.pkl")
            with open(checkpoint_path, "wb") as f:
                pickle.dump(enriched, f)
            logger.info(f"💾 Saved checkpoint at {i+1} users")

    # === 最終儲存 ===
    final_path = os.path.join(output_dir, "final.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(enriched, f)
    logger.info(f"✅ Final output saved to {final_path}")
