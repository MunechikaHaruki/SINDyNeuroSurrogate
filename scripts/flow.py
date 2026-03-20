import logging
import tempfile
from pathlib import Path
from typing import Dict

import hydra
import mlflow

from neurosurrogate.modeling.calc_engine import unified_simulater
from neurosurrogate.modeling.profiler import calc_dynamic_metrics
from neurosurrogate.utils.plots import (
    draw_engine,
    plot_2d_attractor_comparison,
    plot_sindy_coefficients,
    spec_diff,
    spec_simple,
)

logger = logging.getLogger(__name__)


def save_xarray(ds, name, save_plotly=False):
    """
    plotlyでセーブされるhtmlは重い
    """
    datasets, spec = spec_simple(ds)
    fig = draw_engine(datasets, spec, engine="matplotlib")
    mlflow.log_figure(fig, artifact_file=f"{name}.png")
    if save_plotly:
        fig = draw_engine(datasets, spec, engine="plotly")
        mlflow.log_figure(fig, artifact_file=f"{name}.html")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / f"{name}.nc"
        if "features" in ds.indexes:
            ds_to_save = ds.reset_index("features")
        else:
            ds_to_save = ds
        ds_to_save.to_netcdf(local_path)
        mlflow.log_artifact(local_path, artifact_path="data")


@mlflow.trace
def train_model(surrogate, train_ds, target_comp_id):
    surrogate.fit(train_ds, target_comp_id)
    # surrogateモデルのロギング
    summary = surrogate.get_loggable_summary()

    mlflow.log_metrics(summary["metrics"])
    mlflow.log_params(summary["params"])

    for filename, content in summary["artifacts"]["texts"].items():
        mlflow.log_text(content, artifact_file=filename)

    for name, ds in summary["artifacts"]["xarray"].items():
        save_xarray(ds, name, save_plotly=True)

    model = summary["model"]
    fig = plot_sindy_coefficients(
        xi_matrix=model["xi"],
        feature_names=model["feature_names"],
        target_names=model["target_names"],
    )
    mlflow.log_figure(fig, artifact_file="sindy_coef.png")


@mlflow.trace
def generate_dataset_flow(dataset_key, datasets_cfg, models_arch):
    dataset_cfg = datasets_cfg[dataset_key]
    data_type = dataset_cfg["data_type"]

    ds = unified_simulater(
        u=hydra.utils.instantiate(dataset_cfg["current"]),
        dt=dataset_cfg["dt"],
        net=models_arch[data_type],
    )
    ds.attrs["model_type"] = data_type
    return ds


@mlflow.trace
def eval_diff(original_ds, name, datasets_cfg, surrogate_model, models_arch):
    data_type = original_ds.attrs["model_type"]
    target_comp_id = datasets_cfg[name]["target_comp_id"]
    dt = float(original_ds.attrs["dt"])
    predict_result = unified_simulater(
        dt=dt,
        u=original_ds["I_ext"].to_numpy(),
        net=models_arch[data_type],
        surrogate_target=target_comp_id,
        surrogate_model=surrogate_model,
    )

    preprocessed_xr = surrogate_model.preprocessor.transform(
        original_ds, target_comp_id=target_comp_id
    )

    # logging
    mlflow.log_metrics(
        calc_dynamic_metrics(original_ds, predict_result, target_comp_id, dt)
    )
    names = ["orig", "preprocessed", "surr"]
    datasets = [original_ds, preprocessed_xr, predict_result]
    for ds, name in zip(datasets, names):
        save_xarray(ds, name)

    datasets, spec = spec_diff(
        original_ds, preprocessed_xr, predict_result, surr_id=target_comp_id
    )
    mlflow.log_figure(
        draw_engine(datasets, spec, engine="matplotlib"),
        artifact_file="compare.png",
    )

    fig_phase = plot_2d_attractor_comparison(
        orig_ds=preprocessed_xr,
        surr_ds=predict_result,
        comp_id=target_comp_id,
        state_vars=["V", "latent1"],  # 実際のSINDyのターゲット変数名に合わせて変更
    )
    mlflow.log_figure(fig_phase, artifact_file="attractor_surr.png")


def main_flow(datasets_cfg: Dict, surrogate_model, models_arch, run_name):
    logger.info("Start Flow:start generate train data")
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"run_id:{run.info.run_id}")
        mlflow.log_dict(datasets_cfg, "datasets.yaml")
        train_ds = generate_dataset_flow("train", datasets_cfg, models_arch)
        target_comp_id = datasets_cfg["train"]["target_comp_id"]
        logger.info("Start Training")
        train_model(surrogate_model, train_ds, target_comp_id)
        for key in datasets_cfg.keys():
            logger.info(f"start {key}'s evaluation")
            try:
                with mlflow.start_run(run_name=f"Eval_{key}", nested=True):
                    mlflow.set_tag("eval_dataset", key)
                    mlflow.log_dict(datasets_cfg[key], "dataset.yaml")
                    ds = generate_dataset_flow(key, datasets_cfg, models_arch)
                    eval_diff(ds, key, datasets_cfg, surrogate_model, models_arch)
                    logger.info(f"Successfully finished evaluation for {key}")
            except Exception as e:
                logger.exception(f"Failed to evaluate {key}: {str(e)}")
                mlflow.set_tag("error_type", str(type(e).__name__))
                mlflow.set_tag("error_msg", str(e))
                continue
