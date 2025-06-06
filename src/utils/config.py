import yaml
import os

class Config:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v)) 
            else:
                setattr(self, k, v)

    def to_dict(self):
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self.__dict__.items()}

    def as_flat_dict(self, prefix=""):
        flat = {}
        for k, v in self.__dict__.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, Config):
                flat.update(v.as_flat_dict(prefix=key))
            else:
                flat[key] = v
        return flat


def load_config(config_path):
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to YAML file.

    Returns:
        Config: Configuration dictionary-like object.
    """
    with open(config_path, 'r') as f:
        raw_dict = yaml.safe_load(f)
    return Config(raw_dict)

def load_huggingface_token(cfg):
    if not cfg.hf.use_token:
        return None
    key = cfg.hf.token_env_key
    if key not in os.environ:
        raise EnvironmentError(f"環境變數 {key} 不存在")
    return os.environ[key]
