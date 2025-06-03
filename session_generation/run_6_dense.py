# run.py
from dotenv import load_dotenv
import os
import time

from utils.config import load_config,load_huggingface_token
from utils.get_experiment_output_dir import get_experiment_output_dir
from modules.enrich_session.main import run_enrichment
from utils.logger import setup_logger
from modules.gen_product_names.main import run_generate_product_names,run_generate_product_names_vllm
from modules.get_real_item.main import find_best_items
load_dotenv()


def main(args):
    system_log = setup_logger(name="main",log_dir="system_output")


    if args.mapping:
        system_log.info("Only Execute Real Item Mapping")
        mapping_cfg = load_config(os.path.join("config","get_real_item_dense.yaml"))
        real_item_output_dir = get_experiment_output_dir(mapping_cfg.experiment)
        real_item_logger = setup_logger(name="real_item", log_dir=real_item_output_dir)

        start = time.time()
        find_best_items(mapping_cfg,real_item_logger)
        elapsed = (time.time() - start) / 3600
        system_log.info(f"🕒 Real Item Mapping 完成，耗時 {elapsed:.4f} 小時")

        system_log.info("Real Item Mapping Down")
        return

    """LLM 1"""
    system_log.info("Starting Enrich Session Module.")
    llm1_cfg = load_config(os.path.join("config","enrich_sessions_6.yaml"))
    llm1_output_dir = get_experiment_output_dir(llm1_cfg.experiment)
    logger = setup_logger(name="enrich", log_dir=llm1_output_dir)

    start = time.time()
    token = load_huggingface_token(llm1_cfg)
    model, tokenizer = run_enrichment(llm1_cfg, logger)
    elapsed = (time.time() - start) / 3600
    system_log.info(f"🕒 Enrich Session Module 完成，耗時 {elapsed:.4f} 小時")

    """LLM 2"""
    system_log.info("Starting Gen Product Name Module.")
    llm2_cfg = load_config(os.path.join("config","gen_product_names_6.yaml"))
    llm2_output_dir = get_experiment_output_dir(llm2_cfg.experiment)
    llm2_logger = setup_logger(name="gen", log_dir=llm2_output_dir)

    start = time.time()
    run_generate_product_names_vllm(llm2_cfg, llm2_output_dir, llm2_logger,model,tokenizer)
    elapsed = (time.time() - start) / 3600
    system_log.info(f"🕒 Gen Product Name Module 完成，耗時 {elapsed:.4f} 小時")

    """Get Real Items"""
    system_log.info("Starting Real Item Mapping Module")
    mapping_cfg = load_config(os.path.join("config","get_real_item.yaml"))
    real_item_output_dir = get_experiment_output_dir(mapping_cfg.experiment)

    start = time.time()
    real_item_logger = setup_logger(name="real_item", log_dir=real_item_output_dir)
    find_best_items(mapping_cfg,real_item_logger)
    elapsed = (time.time() - start) / 3600
    system_log.info(f"🕒 Real Item Mapping Module 完成，耗時 {elapsed:.4f} 小時")

    system_log.info("✅ All modules finished successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mapping', action="store_true", help='only execute real item mapping')

    args = parser.parse_args()

    main(args)