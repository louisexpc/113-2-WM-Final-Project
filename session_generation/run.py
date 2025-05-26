# run.py
from dotenv import load_dotenv
import os

from utils.config import load_config,load_huggingface_token
from utils.get_experiment_output_dir import get_experiment_output_dir
from modules.enrich_session.main import run_enrichment
from utils.logger import setup_logger
load_dotenv()

cfg = load_config(os.path.join("config","enrich_sessions.yaml"))

output_dir = get_experiment_output_dir(cfg.experiment)
logger = setup_logger(name="enrich", log_dir=output_dir)


token = load_huggingface_token(cfg)

run_enrichment(cfg, output_dir, logger)

