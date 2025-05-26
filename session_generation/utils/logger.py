import logging
import os

def setup_logger(name="main", log_dir=None, level=logging.INFO):
    """
    設定 logger，可同時輸出到 console 與檔案。

    Args:
        name (str): logger 名稱
        log_dir (str or None): 若提供則寫入 log 檔
        level (int): logging level，例如 logging.INFO
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # 避免重複添加 handler

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "run.log")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
