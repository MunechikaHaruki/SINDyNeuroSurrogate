from functools import partial
from typing import cast

import mlflow
import pandas as pd
from io_handler import TARGET_EXP, RunInfo, load_surrogate_model

from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model.model_dataset import CurrentConfig, DatasetConfig
from neurosurrogate.model.model_neurosindy import transform_gate
from neurosurrogate.model.registry_neuron import MCMODELS
from neurosurrogate.profiler.profiler_wave import WaveformMetrics


def get_runs_df():
    experiment = mlflow.get_experiment_by_name(TARGET_EXP)
    if experiment is None:
        raise ValueError(
            f"Experiment '{TARGET_EXP}' が見つかりません。名前を確認してください。"
        )
    all_runs_df = cast(
        pd.DataFrame, mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    )
    if all_runs_df.empty:
        raise ValueError(f"Experiment '{TARGET_EXP}' にrunが存在しません。")
    runs_df = all_runs_df.copy()
    runs_df = runs_df.sort_values("start_time", ascending=False)
    runs_df["start_time"] = runs_df["start_time"].dt.strftime("%m-%d %H:%M:%S")
    runs_df = runs_df[
        ["tags.mlflow.runName", "run_id", "start_time"]
        + [c for c in runs_df.columns if "metrics" in c or "params" in c]
    ]
    return runs_df


def get_comp_names(model_name: str) -> list[str]:
    return MCMODELS[model_name].names


def build_dataset_cfg(
    current_type: str,
    run_id: str,
    current_params: dict | None,
    base_dataset_params: dict,
) -> DatasetConfig:
    if current_type == "train":
        return RunInfo.get_run_info(run_id).dataset
    assert current_params is not None
    return DatasetConfig.build_dataset(
        **base_dataset_params,
        pipeline=CurrentConfig.build_pipeline(current_type, current_params),
    )


def build_eval_result(
    dataset_cfg: DatasetConfig,
    run_id: str,
    surrogate_targets: list[str],
) -> dict:
    original_graph = dataset_cfg.net
    surrogate_model = load_surrogate_model(run_id)
    u = dataset_cfg.current.build()
    original_ds = unified_simulator(dt=dataset_cfg.dt, u=u, net=original_graph)

    surr_ds = unified_simulator(
        dt=dataset_cfg.dt,
        u=u,
        net=original_graph.with_surrogates(
            targets=set(surrogate_targets),
            make_surr=surrogate_model.make_surr_comp,
        ),
        surrogate_model=surrogate_model,
    )

    return {
        "original_ds": original_ds,
        "surr_ds": surr_ds,
        "dt": dataset_cfg.dt,
        "get_preprocessed": partial(transform_gate, surrogate_model.preprocessor, original_ds),
        "name_to_idx": MCMODELS[dataset_cfg.model_name].name_to_idx,
    }
