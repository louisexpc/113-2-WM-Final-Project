import os
import pickle
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from .utils.data_loader import load_pickle
from .llm2_vllm import get_product_name_from_product_type_vllm
from .llm2 import get_product_name_from_product_type



def save_result_incrementally(result_dict, output_path, logger):
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            existing = pickle.load(f)
    else:
        existing = {}
    existing.update(result_dict)
    with open(output_path, "wb") as f:
        pickle.dump(existing, f)
    logger.info(f"✅ 儲存 {len(result_dict)} 筆，總筆數：{len(existing)}")


def load_existing_user_ids(output_path):
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            existing = pickle.load(f)
        return set(existing.keys())
    return set()

def run_generate_product_names(cfg,output_dir,logger,model,tokenizer):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device.cuda_visible_devices)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    logger.info(f"🚀 載入模型: {cfg.model.model_name}")

    sessions_dict = load_pickle(cfg.data.input_path)
    """Update: 統一輸入資料格式"""
    customers_df = pd.read_csv(cfg.data.customers_path)
    output_path = os.path.join(output_dir, os.path.basename(cfg.data.output_path))
    processed_users = load_existing_user_ids(output_path)
    logger.info(f"🗂️ 已處理 {len(processed_users)} 位使用者")

    batch, count, skipped = {}, 0, 0
    for user_id, product_types in sessions_dict.items():
        if user_id in processed_users:
            skipped += 1
            continue
        batch[user_id] = product_types
        count += 1
        if count % cfg.data.batch_size == 0:
            logger.info(f"🚀 處理第 {count - cfg.data.batch_size + 1 + skipped} 到 {count + skipped} 筆")
            result = get_product_name_from_product_type(batch, customers_df, model, tokenizer)
            save_result_incrementally(result, output_path, logger)
            batch = {}

    if batch:
        logger.info(f"🚀 處理最後 {len(batch)} 筆")
        result = get_product_name_from_product_type(batch, customers_df, model, tokenizer)
        save_result_incrementally(result, output_path, logger)



def run_generate_product_names_vllm(cfg, output_dir, logger):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device.cuda_visible_devices)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    logger.info(f"🚀 載入模型: {cfg.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.local_model_dir)

    llm = LLM(
        model=cfg.model.local_model_dir,
        gpu_memory_utilization=cfg.model.gpu_memory_utilization,
        max_model_len=cfg.model.max_model_len,
        max_num_seqs=cfg.model.max_num_seqs,
    )
    
    sampling_params = SamplingParams(
        temperature=cfg.model.temperature,
        top_p=cfg.model.top_p,
        max_tokens=cfg.model.max_tokens,
    )

    sessions_dict = load_pickle(cfg.data.input_path)
    """Update: 統一輸入資料格式"""
    customers_df = pd.read_csv(cfg.data.customers_path)
    output_path = os.path.join(output_dir, os.path.basename(cfg.data.output_path))

    processed_users = load_existing_user_ids(output_path)
    logger.info(f"🗂️ 已處理 {len(processed_users)} 位使用者")

    batch, count, skipped = {}, 0, 0
    for user_id, product_tuple in sessions_dict.items():
        if user_id in processed_users:
            skipped += 1
            continue
        batch[user_id] = product_tuple
        count += 1
        if count % cfg.data.batch_size == 0:
            logger.info(f"🚀 處理第 {count - cfg.data.batch_size + 1 + skipped} 到 {count + skipped} 筆")
            result = get_product_name_from_product_type_vllm(batch, customers_df, llm, tokenizer, sampling_params)
            save_result_incrementally(result, output_path, logger)
            batch = {}

    if batch:
        logger.info(f"🚀 處理最後 {len(batch)} 筆")
        result = get_product_name_from_product_type_vllm(batch, customers_df, llm, tokenizer, sampling_params)
        save_result_incrementally(result, output_path, logger)
