# run.py
from dotenv import load_dotenv
import os

from utils.config import load_config,load_huggingface_token
from utils.get_experiment_output_dir import get_experiment_output_dir
from modules.enrich_session.main import run_enrichment
from utils.logger import setup_logger
from modules.gen_product_names.main import run_generate_product_names,run_generate_product_names_vllm
load_dotenv()

cfg = load_config(os.path.join("config","enrich_sessions.yaml"))

output_dir = get_experiment_output_dir(cfg.experiment)
logger = setup_logger(name="enrich", log_dir=output_dir)


token = load_huggingface_token(cfg)

model, tokenizer = run_enrichment(cfg, output_dir, logger)

llm2_cfg = load_config(os.path.join("config","gen_product_names.yaml"))
output_dir = get_experiment_output_dir(llm2_cfg.experiment)
logger = setup_logger(name="gen", log_dir=output_dir)
run_generate_product_names_vllm(llm2_cfg, output_dir, logger,model,tokenizer)
