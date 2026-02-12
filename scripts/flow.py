import hashlib
import json
from typing import Any, Dict

import hydra
import mlflow
import numpy as np
from prefect import flow, get_run_logger, task

from neurosurrogate.modeling import (
    PCAPreProcessorWrapper,
    SINDySurrogateWrapper,
)
from neurosurrogate.modeling.numba_core import unified_simulater
from neurosurrogate.utils.plots import plot_simple


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


def log_eval_result(name, ds, eval_result):
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


def generate_complex_hash(*args, **kwargs) -> str:
    """
    *args と **kwargs をまとめて安定したハッシュ値を生成する
    """

    def to_stable_obj(obj: Any) -> Any:
        """
        あらゆる型を、JSONシリアライズ可能な標準的な型へ再帰的に変換する
        """
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
    cache_key_fn=lambda context, params: str(params["dataset_seed"]),
    persist_result=True,
)
def unified_simulater_wrapper(data_type, i_ext, DT, dataset_seed):
    """
    Simulates a neuron model based on configurations and preprocesses the result into a dataset.
    """
    return unified_simulater(dt=DT, u=i_ext, data_type=data_type, mode="simulate")


def generate_dataset_flow(dataset_key, datasets_cfg):
    dataset_cfg = datasets_cfg[dataset_key]
    data_type = dataset_cfg["data_type"]

    ds = unified_simulater_wrapper(
        data_type=data_type,
        i_ext=hydra.utils.instantiate(
            dataset_cfg["current"], current_seed=dataset_cfg["seed"]
        ),
        DT=dataset_cfg["dt"],
        dataset_seed=generate_complex_hash(dataset_cfg),
    )
    fig = plot_simple(ds)
    mlflow.log_figure(fig, artifact_file=f"original/{data_type}/{dataset_key}.png")
    return ds


@task
def train_task(train_ds):
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
    surrogate_model.fit(train_ds)
    return surrogate_model


@flow
def main_flow(datasets_cfg: Dict):
    logger = get_run_logger()
    logger.info("Start Flow")

    logger.info("start generate train data")
    train_ds = generate_dataset_flow("train", datasets_cfg)
    surrogate_model = train_task(train_ds)
    log_train_model(surrogate_model)

    for name in datasets_cfg.keys():
        logger.info(f"start {name}'s evaluation")
        ds = generate_dataset_flow(name, datasets_cfg)
        eval_result = surrogate_model.eval(ds)
        log_eval_result(name, ds, eval_result)
