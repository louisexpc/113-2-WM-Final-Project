# run.py
from dotenv import load_dotenv
import os

from utils.config import load_config,load_huggingface_token
from utils.get_experiment_output_dir import get_experiment_output_dir
from modules.enrich_session.main import run_enrichment
from utils.logger import setup_logger
from modules.gen_product_names.main import run_generate_product_names,run_generate_product_names_vllm
load_dotenv()

llm1_cfg = load_config(os.path.join("config","enrich_sessions.yaml"))

llm1_output_dir = get_experiment_output_dir(llm1_cfg.experiment)
logger = setup_logger(name="enrich", log_dir=llm1_output_dir)


token = load_huggingface_token(llm1_cfg)

model, tokenizer = run_enrichment(llm1_cfg, logger)

llm2_cfg = load_config(os.path.join("config","gen_product_names.yaml"))
llm2_output_dir = get_experiment_output_dir(llm2_cfg.experiment)
llm2_logger = setup_logger(name="gen", log_dir=llm2_output_dir)
run_generate_product_names_vllm(llm2_cfg, llm2_output_dir, llm2_logger,model,tokenizer)
