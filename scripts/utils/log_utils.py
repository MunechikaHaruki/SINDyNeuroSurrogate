import logging

import hydra
import mlflow

from neurosurrogate.modeling.profiler import calc_dynamic_metrics
from neurosurrogate.utils.plots import (
    draw_engine,
    plot_2d_attractor_comparison,
    plot_sindy_coefficients,
    spec_diff,
    spec_simple,
)

logger = logging.getLogger(__name__)


def _save_xarray(ds, name):
    """
    plotlyでセーブされるhtmlは重い
    """
    datasets, spec = spec_simple(ds)
    fig = draw_engine(datasets, spec, engine="matplotlib")
    mlflow.log_figure(fig, artifact_file=f"{name}.png")


def get_hydra_overrides():
    try:
        hydra_overrides = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
    except Exception:
        hydra_overrides = "OverrideError"
    if hydra_overrides == "":
        hydra_overrides = "Default"
    return hydra_overrides


def log_dataset_cfg(dataset_cfg):
    mlflow.log_params(dataset_cfg)
    mlflow.log_params(dataset_cfg["current"]["pipeline"][0])
    mlflow.log_dict(dataset_cfg, "dataset.yaml")


def log_surrogate_summary(summary):
    mlflow.log_metrics(summary["metrics"])
    mlflow.log_params(summary["params"])

    for filename, content in summary["artifacts"]["texts"].items():
        mlflow.log_text(content, artifact_file=filename)

    for name, ds in summary["artifacts"]["xarray"].items():
        _save_xarray(ds, name)

    model = summary["model"]
    fig = plot_sindy_coefficients(
        xi_matrix=model["xi"],
        feature_names=model["feature_names"],
        target_names=model["target_names"],
    )
    mlflow.log_figure(fig, artifact_file="sindy_coef.png")


def log_eval_result(original_ds, surr_ds, preprocessed_xr, dataset_cfg):
    dt = dataset_cfg["dt"]
    target_comp_id = dataset_cfg["target_comp_id"]
    mlflow.log_metrics(calc_dynamic_metrics(original_ds, surr_ds, target_comp_id, dt))
    names = ["orig", "preprocessed", "surr"]
    datasets = [original_ds, preprocessed_xr, surr_ds]
    for ds, name in zip(datasets, names):
        _save_xarray(ds, name)

    datasets, spec = spec_diff(
        original_ds, preprocessed_xr, surr_ds, surr_id=target_comp_id
    )
    mlflow.log_figure(
        draw_engine(datasets, spec, engine="matplotlib"),
        artifact_file="compare.png",
    )

    fig_phase = plot_2d_attractor_comparison(
        orig_ds=preprocessed_xr,
        surr_ds=surr_ds,
        comp_id=target_comp_id,
        state_vars=["V", "latent1"],  # 実際のSINDyのターゲット変数名に合わせて変更
    )
    mlflow.log_figure(fig_phase, artifact_file="attractor_surr.png")


def log_target_metric(train_run_id, target_metric):
    # 指標として記録（グラフ描画やソート用）
    with mlflow.start_run(train_run_id):
        mlflow.log_metric("OPTUNA_TARGET_SCORE", target_metric)
        mlflow.set_tag("is_optuna_trial", "true")

        # ★ ここから追加：Run名を更新する ★
        # 現在のRun情報を取得して、元の名前を取り出す
        current_run = mlflow.get_run(train_run_id)
        original_name = current_run.data.tags.get("mlflow.runName", "Training_run")

        # スコアを小数点以下4桁などにフォーマットして名前に結合
        new_run_name = f"{original_name} | Score:{target_metric:.4f}"

        # mlflow.runName タグを上書きすることで、UI上の表示名が変わる
        mlflow.set_tag("mlflow.runName", new_run_name)
