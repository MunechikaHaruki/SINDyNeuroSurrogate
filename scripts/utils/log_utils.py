import logging

import mlflow

from neurosurrogate.modeling import SINDySummary
from neurosurrogate.utils.plots import (
    draw_engine,
    plot_sindy_coefficients,
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


def log_dataset_cfg(dataset_cfg):
    mlflow.log_params(dataset_cfg)
    mlflow.log_params(dataset_cfg["current"]["pipeline"][0])
    mlflow.log_dict(dataset_cfg, "dataset.yaml")


def log_surrogate_summary(summary: SINDySummary):
    mlflow.log_metrics(summary.metrics)
    mlflow.log_params(summary.params)

    for filename, content in summary.texts.items():
        mlflow.log_text(content, artifact_file=filename)

    for name, ds in summary.xarrays.items():
        _save_xarray(ds, name)

    fig = plot_sindy_coefficients(
        xi_matrix=summary.xi,
        feature_names=summary.feature_names,
        target_names=summary.target_names,
    )
    mlflow.log_figure(fig, artifact_file="sindy_coef.png")


def run_override(run_id, metric):
    current_run = mlflow.get_run(run_id)
    original_name = current_run.data.tags.get("mlflow.runName", "Run")
    mlflow.set_tag("mlflow.runName", f"{original_name} | Score:{metric:.4f}")
