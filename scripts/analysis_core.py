from dataclasses import dataclass, field
from functools import partial
from typing import cast

import mlflow
import pandas as pd
from io_handler import TARGET_EXP, RunInfo, load_surrogate_model

from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model.model_dataset import CurrentConfig, DatasetConfig, NeuronGraph
from neurosurrogate.model.model_neurosindy import transform_gate
from neurosurrogate.model.registry_neuron import MCMODELS
from neurosurrogate.profiler.profiler_wave import calc_dynamic_metrics


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
    cols = [
        c for c in runs_df.columns if "metrics" in c or "params" in c or c == "run_id"
    ]
    runs_df = runs_df[
        ["tags.mlflow.runName", "run_id", "start_time"]
        + [c for c in cols if c != "run_id"]
    ]
    return runs_df


def get_comp_names(model_name: str) -> list[str]:
    return MCMODELS[model_name].names


@dataclass
class EvalInput:
    current_type: str
    run_id: str
    surrogate_targets: list[str]
    current_params: dict | None = field(default=None)
    base_dataset_params: dict | None = field(default=None)


def build_eval_result(params: EvalInput) -> dict:
    current_type = params.current_type
    if current_type == "train":
        dataset_cfg = RunInfo.get_run_info(params.run_id).dataset
        model_name = dataset_cfg.model_name
    else:
        assert (
            params.current_params is not None and params.base_dataset_params is not None
        )
        pipeline = CurrentConfig.build_pipeline(current_type, params.current_params)
        dataset_cfg = DatasetConfig.build_dataset(
            **params.base_dataset_params, pipeline=pipeline
        )
        model_name = params.base_dataset_params["model_name"]

    original_graph = dataset_cfg.net
    name_to_idx = MCMODELS[model_name].name_to_idx
    surrogate_model = load_surrogate_model(params.run_id)
    u = dataset_cfg.current.build()
    original_ds = unified_simulator(dt=dataset_cfg.dt, u=u, net=original_graph)

    surr_nodes = [
        surrogate_model.make_surr_comp(n.name)
        if n.name in params.surrogate_targets
        else n
        for n in original_graph.nodes
    ]
    surr_graph = NeuronGraph(
        nodes=surr_nodes, edges=original_graph.edges, stim=original_graph.stim
    )
    surr_ds = unified_simulator(
        dt=dataset_cfg.dt,
        u=u,
        net=surr_graph,
        surrogate_model=surrogate_model,
    )

    get_preprocessed = partial(
        transform_gate, surrogate_model.preprocessor, original_ds
    )
    get_metrics = partial(calc_dynamic_metrics, original_ds, surr_ds, dt=dataset_cfg.dt)
    return {
        "metrics": get_metrics,
        "get_preprocessed": get_preprocessed,
        "name_to_idx": name_to_idx,
        "datasets": {
            "orig": original_ds,
            "surr": surr_ds,
        },
    }
