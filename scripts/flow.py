import hashlib
import json
from typing import Any

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
from prefect import flow, get_run_logger, task

from neurosurrogate.modeling import (
    PCAPreProcessorWrapper,
    SINDySurrogateWrapper,
)
from neurosurrogate.modeling.numba_core import unified_simulater
from neurosurrogate.utils.plots import plot_simple


def generate_complex_hash(*args, **kwargs) -> str:
    """
    *args と **kwargs をまとめて安定したハッシュ値を生成する
    """

    def to_stable_obj(obj: Any) -> Any:
        """
        あらゆる型を、JSONシリアライズ可能な標準的な型へ再帰的に変換する
        """
        if isinstance(obj, DictConfig):  # OmegaConf (DictConfig) の処理
            return to_stable_obj(OmegaConf.to_container(obj, resolve=True))
        if isinstance(obj, dict):  # 辞書型なら中身を再帰的に変換
            return {str(k): to_stable_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):  # リストやタプルなら中身を再帰的に変換
            return [to_stable_obj(i) for i in obj]
        if isinstance(
            obj, (int, float, bool, type(None), str)
        ):  # それ以外は文字列化（またはそのまま）
            return obj
        return str(obj)

    combined_data = {
        "args": to_stable_obj(args),
        "kwargs": to_stable_obj(kwargs),
    }  # データを一つの構造にまとめる argsは順番を維持、kwargsはキーをソートして正規化する準備
    json_str = json.dumps(
        combined_data, sort_keys=True, ensure_ascii=True, default=str
    )  # 辞書の順序を固定してJSON化
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()  # hash化


@task(
    cache_key_fn=lambda context, params: str(params["task_seed"]), persist_result=True
)
def generate_single_dataset(dataset_cfg, neuron_cfg, task_seed, DT):
    """
    Simulates a neuron model based on configurations and preprocesses the result into a dataset.
    """
    # Configuration setup
    data_type = dataset_cfg["data_type"]
    i_ext = hydra.utils.instantiate(dataset_cfg["current"], task_seed=task_seed)
    return unified_simulater(
        dt=DT, u=i_ext, data_type=data_type, params_dict=neuron_cfg, mode="simulate"
    )


@task
def log_train_model(surrogate):
    summary = surrogate.get_loggable_summary()
    mlflow.log_dict(
        summary["equations"],
        artifact_file="sindy_equations.txt",
    )
    coef = summary["coefficients"]
    mlflow.log_text(
        np.array2string(coef, precision=3),
        artifact_file="coef.txt",
    )

    nonzero_term_num = np.count_nonzero(coef)
    mlflow.log_metrics(
        metrics={
            "nonzero_term_num": nonzero_term_num,
            "nonzero_term_ratio": nonzero_term_num / coef.size,
        }
    )

    mlflow.log_metrics(summary["static_calc_cost"])

    mlflow.log_text(
        "\n".join(summary["feature_names"]), artifact_file="feature_names.txt"
    )
    mlflow.log_text(
        "\n".join(summary["active_features"]), artifact_file="active_features.txt"
    )

    mlflow.log_param(
        "model_params",
        summary["model_params"],
    )
    mlflow.log_figure(summary["train_figure"], artifact_file="train.png")


def generate_dataset_flow(dataset_key, cfg):
    dataset_cfg = cfg.datasets[dataset_key]
    data_type = dataset_cfg.data_type
    neuron_cfg = cfg.neurons.get(data_type)

    task_seed = generate_complex_hash(
        dataset_cfg,
        neuron_cfg,
        cfg.seed,
    )
    ds = generate_single_dataset(
        dataset_cfg=dataset_cfg,
        neuron_cfg=neuron_cfg,
        task_seed=int(task_seed, 16) % (2**32),
        DT=cfg.simulater_dt,
    )
    fig = plot_simple(ds)
    mlflow.log_figure(fig, artifact_file=f"original/{data_type}/{dataset_key}.png")
    return ds


@task
def train_task(cfg, train_ds):
    import base

    # 2. Train Preprocessor
    preprocessor = PCAPreProcessorWrapper()
    preprocessor.fit(train_xr_dataset=train_ds)

    # 3. Train Model
    surrogate_model = SINDySurrogateWrapper(
        preprocessor=preprocessor,
        target_module=base,
        sindy_name="hh_sindy",
    )
    surrogate_model.fit(train_ds, direct=cfg.models.surrogate.direct)
    log_train_model(surrogate=surrogate_model)
    return surrogate_model


def eval_flow(
    name: str,
    surrogate_model,
    cfg,
):
    # generate_dataset
    ds = generate_dataset_flow(name, cfg)
    eval_result = surrogate_model.eval(ds)
    data_type = ds.attrs["model_type"]
    mlflow.log_figure(
        eval_result["preprocessed"],
        artifact_file=f"preprocessed/{data_type}/{name}.png",
    )
    mlflow.log_figure(
        eval_result["surrogate_figure"],
        artifact_file=f"surrogate/{data_type}/{name}.png",
    )
    mlflow.log_figure(
        eval_result["diff"], artifact_file=f"compare/{data_type}/{name}.png"
    )


@flow
def main_flow(cfg: DictConfig):
    logger = get_run_logger()
    logger.info("Start Flow")

    logger.info("start generate train data")
    train_ds = generate_dataset_flow("train", cfg)
    surrogate_model = train_task(cfg, train_ds)

    for name in cfg.datasets.keys():
        logger.info("start to eval_flow")
        eval_flow(
            name=name,
            surrogate_model=surrogate_model,
            cfg=cfg,
        )
