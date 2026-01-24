import hashlib
import json
import subprocess
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


def to_stable_obj(obj: Any) -> Any:
    """
    あらゆる型を、JSONシリアライズ可能な標準的な型へ再帰的に変換する
    """
    # OmegaConf (DictConfig) の処理
    if isinstance(obj, DictConfig):
        return to_stable_obj(OmegaConf.to_container(obj, resolve=True))

    # 辞書型なら中身を再帰的に変換
    if isinstance(obj, dict):
        return {str(k): to_stable_obj(v) for k, v in obj.items()}

    # リストやタプルなら中身を再帰的に変換
    if isinstance(obj, (list, tuple)):
        return [to_stable_obj(i) for i in obj]

    # それ以外は文字列化（またはそのまま）
    if isinstance(obj, (int, float, bool, type(None), str)):
        return obj
    return str(obj)


def generate_complex_hash(*args, **kwargs) -> str:
    """
    *args と **kwargs をまとめて安定したハッシュ値を生成する
    """
    # 1. データを一つの構造にまとめる
    # argsは順番を維持、kwargsはキーをソートして正規化する準備
    combined_data = {"args": to_stable_obj(args), "kwargs": to_stable_obj(kwargs)}

    # 2. sort_keys=True で辞書の順序を固定してJSON化
    json_str = json.dumps(combined_data, sort_keys=True, ensure_ascii=True, default=str)

    # 3. ハッシュ化
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


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


def get_hydra_overrides():
    try:
        run_name_prefix = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
    except Exception:
        run_name_prefix = "default_run"
    return run_name_prefix
