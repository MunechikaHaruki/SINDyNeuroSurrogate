import inspect
import os
import typing
from dataclasses import dataclass
from functools import partial
from typing import Literal

import marimo as mo
import matplotlib.pyplot as plt
import mlflow
import yaml
from io_handler import TARGET_EXP, DatasetConfig, load_surrogate_model

from neurosurrogate.builder.build_current import FUNC_MAP, CurrentConfig
from neurosurrogate.calc_engine import unified_simulator
from neurosurrogate.model.model_neuron import MCMODELS, NeuronGraph, Node
from neurosurrogate.model.model_neurosindy import transform_gate
from neurosurrogate.profiler.draw_registry import DRAW_MAP
from neurosurrogate.profiler.profiler_view import view_model
from neurosurrogate.profiler.profiler_wave import calc_dynamic_metrics

CurrentList: list = ["train"] + list(FUNC_MAP.keys())
DRAW_LIST: list = list(DRAW_MAP.keys())
MplStyle = Literal["paper", "presentation"]
MCNameList = list(MCMODELS.keys())


get_comp_names = lambda base_btn: (
    MCMODELS[base_btn.base_dataset_ui.value["model_name"]].names
)


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


@dataclass
class BaseUI:
    plt_btn: mo.ui.button
    current_dropdown: mo.ui.dropdown
    base_dataset_ui: mo.ui.dictionary
    run_selector: mo.ui.table

    def render(self):
        return mo.md(f"""
        ### MLflow データ解析
        - matplotlib rendering setting: {self.plt_btn}
        - baseDatasetUI: {self.base_dataset_ui}
        {self.run_selector}
        """)

    def setup_mpl(self):
        matplotlib_style = self.plt_btn.value
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        STYLE_DIR = os.path.join(CURRENT_DIR, "./conf/style")
        plt.style.use(os.path.join(STYLE_DIR, "./base.mplstyle"))
        plt.style.use(os.path.join(STYLE_DIR, f"./{matplotlib_style}.mplstyle"))

    @property
    def run_ids(self):
        return self.run_selector.value["run_id"].tolist()

    @staticmethod
    def get_mlflow_runselector():
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
            c
            for c in runs_df.columns
            if "metrics" in c or "params" in c or c == "run_id"
        ]
        runs_df = runs_df[
            ["tags.mlflow.runName", "run_id", "start_time"]
            + [c for c in cols if c != "run_id"]
        ]

        return mo.ui.table(
            runs_df[["tags.mlflow.runName", "run_id"]],
            label="比較・解析したいRunを複数選択",
            selection="multi",
            initial_selection=[0],
        )

    @staticmethod
    def get_base_btn():
        plt_options = list(typing.get_args(MplStyle))
        return BaseUI(
            plt_btn=mo.ui.radio(options=plt_options, value=plt_options[0]),
            current_dropdown=mo.ui.dropdown(CurrentList, value="steady"),
            base_dataset_ui=mo.ui.dictionary(
                {
                    "dt": mo.ui.number(value=0.01, step=0.001, label="dt"),
                    "silence_duration": mo.ui.number(
                        value=80, step=1, label="silence_duration"
                    ),
                    "duration": mo.ui.number(value=800, step=100, label="duration"),
                    "model_name": mo.ui.dropdown(
                        options=list(MCMODELS.keys()), label="model_name", value="hh"
                    ),
                }
            ),
            run_selector=BaseUI.get_mlflow_runselector(),
        )

    def get_model_info_ui(self):
        run_ids = self.run_ids
        model_infos = {}
        for run_id in run_ids:
            run_info = get_run_info(run_id)
            model_infos[run_id] = {}
            model_infos[run_id]["runName"] = run_info["runName"]
            model_infos[run_id]["equations"] = run_info["equations"]
            model_infos[run_id]["dataset"] = run_info["dataset"]
            model_infos[run_id]["sindy_coef"] = run_info["sindy_coef"]
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


@dataclass
class ParamUI:
    current_ui: mo.ui.dictionary
    surrogate_target_ui: mo.ui.multiselect
    runid_dropdown: mo.ui.dropdown

    def render(self):

        return mo.md(f"""
        ### パラメタ設定
        - currentui: {self.current_ui}
        - surrogate target: {self.surrogate_target_ui}
        - runid_dropdown: {self.runid_dropdown}
        # """)

    @staticmethod
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

    @property
    def run_id(self):
        return self.runid_dropdown.value

    @staticmethod
    def get_detailed_btn(base_btn: BaseUI):

        current_sig = inspect.signature(FUNC_MAP[base_btn.current_dropdown.value])
        current_ui = mo.ui.dictionary(
            {
                name: ParamUI._make_ui_element(
                    name,
                    param.annotation,
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else 0,
                )
                for name, param in current_sig.parameters.items()
            }
        )
        surrogate_target_ui = mo.ui.multiselect(
            options=get_comp_names(base_btn), value=[get_comp_names(base_btn)[0]]
        )
        run_ids = base_btn.run_selector.value["run_id"].tolist()
        return ParamUI(
            current_ui=current_ui,
            surrogate_target_ui=surrogate_target_ui,
            runid_dropdown=mo.ui.dropdown(options=run_ids, value=run_ids[0]),
        )


def eval_dataset(base_btn: BaseUI, param_ui: ParamUI):
    current_type = base_btn.current_dropdown.value
    if current_type == "train":
        dataset_cfg = DatasetConfig.from_dict(get_run_info(param_ui.run_id)["dataset"])
        model_name = dataset_cfg["model_name"]
    else:
        pipeline = CurrentConfig.build_pipeline(current_type, param_ui.current_ui.value)
        dataset_cfg = DatasetConfig.build_dataset(
            **base_btn.base_dataset_ui.value, pipeline=pipeline
        )
        model_name = base_btn.base_dataset_ui.value["model_name"]

    original_graph = dataset_cfg.net

    name_to_idx = MCMODELS[model_name].name_to_idx
    surrogate_model = load_surrogate_model(param_ui.run_id)
    u = dataset_cfg.current.build()
    original_ds = unified_simulator(dt=dataset_cfg.dt, u=u, net=original_graph)

    surr_nodes = [
        Node(n.name, "surr") if n.name in param_ui.surrogate_target_ui.value else n
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


@dataclass
class EvalUI:
    eval_comp: mo.ui.dropdown
    draw_func: mo.ui.dropdown

    def render(self):
        return mo.md(f"評価対象のComp:{self.eval_comp},描画関数:{self.draw_func}")

    @staticmethod
    def get_eval_ui(param_ui: ParamUI):
        return EvalUI(
            eval_comp=mo.ui.dropdown(
                options=param_ui.surrogate_target_ui.value,
                value=param_ui.surrogate_target_ui.value[0],
            ),
            draw_func=mo.ui.dropdown(options=DRAW_LIST, value=DRAW_LIST[0]),
        )

    def view_result(self, result):
        target_comp_id = result["name_to_idx"](self.eval_comp.value)
        metrics = result["metrics"](target_comp_id)
        pre = result["get_preprocessed"](target_comp_id)
        print(pre.coords)
        print("vieowe")
        cards = mo.hstack(
            [
                mo.stat(label=k, value=f"{v:.4f}" if isinstance(v, float) else str(v))
                for k, v in metrics.items()
            ],
            wrap=True,
        )
        return mo.vstack(
            [
                cards,
                mo.mpl.interactive(
                    DRAW_MAP[self.draw_func.value](
                        result["datasets"]["orig"],
                        result["datasets"]["surr"],
                        pre,
                        target_comp_id,
                    )
                ),
            ]
        )
