from collections.abc import Mapping
import subprocess
from omegaconf import OmegaConf

def recursive_to_dict(obj):
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, (list, tuple)):
        return [recursive_to_dict(x) for x in obj]
    if isinstance(obj, Mapping):
        return {k: recursive_to_dict(v) for k, v in obj.items()}
    return obj

def get_commit_id():
    try:
        commit_id = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        commit_id = "unknown"
    return commit_id