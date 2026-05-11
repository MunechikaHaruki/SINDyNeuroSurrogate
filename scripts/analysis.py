import copy
import inspect
import os
import typing
from typing import Literal

import marimo as mo
import matplotlib.pyplot as plt
import mlflow
import yaml
from io_handler import TARGET_EXP, build_dataset, load_surrogate_model

from neurosurrogate.builder.build_current import (
    FUNC_MAP,
    build_current_pipeline,
    build_current_setting,
)
from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model.model_neurosindy import transform_gate
from neurosurrogate.profiler.profiler_view import draw_engine, spec_diff, view_model
from neurosurrogate.profiler.profiler_wave import calc_dynamic_metrics

CurrentList: list = ["train"] + list(FUNC_MAP.keys())

MplStyle = Literal["paper", "presentation"]


def init_cell():
    load_btn = mo.ui.button(
        label="ここをクリック！", value=False, on_click=lambda x: True
    )
    plt_options = list(typing.get_args(MplStyle))
    plt_btn = mo.ui.radio(options=plt_options, value=plt_options[0])

    current_dropdown = mo.ui.dropdown(CurrentList, value="steady")
    ui = mo.md(f"""
    ### MLflow データ解析
    - Reload MLflow: {load_btn}
    - matplotlib rendering setting: {plt_btn}
    - testCurrent Type: {current_dropdown}
    """)
    return ui, load_btn, plt_btn, current_dropdown


def _get_runs_df():
    experiment = mlflow.get_experiment_by_name(TARGET_EXP)
    if experiment is None:
        raise ValueError(
            f"Experiment '{TARGET_EXP}' が見つかりません。名前を確認してください。"
        )
    all_runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
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


def get_mlflow_runselector():
    runs_df = _get_runs_df()
    run_selector = mo.ui.table(
        runs_df[["tags.mlflow.runName", "run_id"]],
        label="比較・解析したいRunを複数選択",
        selection="multi",
    )
    return run_selector


def setup_matplotlib(matplotlib_style: MplStyle):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    STYLE_DIR = os.path.join(CURRENT_DIR, "./conf/style")
    plt.style.use(os.path.join(STYLE_DIR, "./base.mplstyle"))
    plt.style.use(os.path.join(STYLE_DIR, f"./{matplotlib_style}.mplstyle"))


def get_run_info(run_id: str) -> dict:
    client = mlflow.MlflowClient()

    def load_yaml(run_id: str, filename: str) -> dict:
        return yaml.safe_load(mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}"))

    def load_text(run_id: str, filename: str) -> str:
        return mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}")

    view_cfg = load_yaml(run_id, "view.json")

    return {
        "sindy_coef": view_model(**view_cfg),
        "dataset": load_yaml(run_id, "dataset.yaml"),  # 同じファイルなら参照共有でOK
        "runName": client.get_run(run_id).data.tags["mlflow.runName"],
        "run_id": run_id,
        "equations": load_text(run_id, "equations.txt"),
    }


def get_model_infos(run_ids):

    model_infos = {}
    for run_id in run_ids:
        run_info = get_run_info(run_id)
        model_infos[run_id] = {}
        model_infos[run_id]["runName"] = run_info["runName"]
        model_infos[run_id]["equations"] = run_info["equations"]
        model_infos[run_id]["dataset"] = run_info["dataset"]
        model_infos[run_id]["sindy_coef"] = run_info["sindy_coef"]
    return model_infos


def get_model_info_ui(run_ids):
    model_infos = get_model_infos(run_ids)
    return mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(
                        f"run_id:{run_id[:8]}.. &nbsp;&nbsp;　{model_infos[run_id]['runName']}"
                    ),
                    mo.md(f"{model_infos[run_id]['equations'][:40]}"),
                    mo.mpl.interactive(model_infos[run_id]["sindy_coef"]),
                ]
            )
            for run_id in run_ids
        ]
    )


def _make_ui_element(name: str, annotation: type, default):
    if annotation is int:
        return mo.ui.number(value=int(default), step=1, label=name)
    elif annotation is float:
        return mo.ui.number(value=float(default), step=0.1, label=name)
    elif annotation is bool:
        return mo.ui.checkbox(value=bool(default), label=name)
    elif annotation is list:
        return mo.ui.array([mo.ui.number(value=0.0, step=0.1)], label=name)

    else:
        raise NotImplementedError(f"{name}: {annotation} は未対応の型です")


def get_param_ui(current_type: str) -> mo.ui.dictionary:
    sig = inspect.signature(FUNC_MAP[current_type])
    return mo.ui.dictionary(
        {
            name: _make_ui_element(
                name,
                param.annotation,
                param.default if param.default is not inspect.Parameter.empty else 0,
            )
            for name, param in sig.parameters.items()
        }
    )


def resolve_config(run_id, current_type, params: dict):
    if current_type == "train":
        return get_run_info(run_id)["dataset"]
    pipeline = build_current_setting(current_type, params)
    return build_dataset(pipeline=pipeline)


def eval_dataset(run_id: str, dataset_cfg: dict):

    surrogate_model = load_surrogate_model(run_id)
    u = build_current_pipeline(dataset_cfg["current"])
    original_ds = unified_simulator(dt=dataset_cfg["dt"], u=u, net=dataset_cfg["net"])
    surr_net = copy.deepcopy(dataset_cfg["net"])

    target_comp_id = 0

    surr_net["nodes"][target_comp_id] = "surr"
    surr_ds = unified_simulator(
        dt=dataset_cfg["dt"],
        u=u,
        net=surr_net,
        surrogate_model=surrogate_model,
    )
    preprocessed_xr = transform_gate(
        surrogate_model.preprocessor, original_ds, target_comp_id=target_comp_id
    )
    return {
        "datasets": {
            "original": original_ds,
            "preprocessed": preprocessed_xr,
            "surrogate": surr_ds,
            "surr_id": target_comp_id,
        },
        "metrics": calc_dynamic_metrics(
            original_ds, surr_ds, target_comp_id, dataset_cfg["dt"]
        ),
    }


# draw_engine(spec_simple(result["datasets"]["preprocessed"]))
def view_dataset(result):
    return draw_engine(spec_diff(**result["datasets"]))
