import os
from datetime import datetime

def get_experiment_output_dir(exp_cfg):
    """
    根據 experiment config 建立 output 資料夾，支援 timestamp 選項。

    Args:
        exp_cfg (Config): 包含 id, output_base_dir, create_timestamp_dir 等設定

    Returns:
        str: 最終 output 路徑
    """
    base = exp_cfg.output_base_dir.rstrip('/')
    eid = exp_cfg.id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if exp_cfg.create_timestamp_dir else ""
    path = os.path.join(base, eid, timestamp) if timestamp else os.path.join(base, eid)

    os.makedirs(path, exist_ok=True)
    return path
