import logging
import os

import hydra
import matplotlib
import mlflow
from builder import build_full_datasets, build_surrogate
from flow import main_flow
from neuron_models import MODEL_DEFINITIONS
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")


import matplotlib.pyplot as plt

# プロキシ設定を一時的に無効化
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1"


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Activate Script")
    # cfgの依存関係解決とビルド
    OmegaConf.resolve(cfg)
    dataset_cfg = build_full_datasets(cfg, MODEL_DEFINITIONS)
    logger.info(dataset_cfg)
    # mlflowの初期設定
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(cfg.experiment_name)
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"
    # matplotlibのstyle設定
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_STYLE_PATH = os.path.join(BASE_DIR, "./conf/style/base.mplstyle")
    plt.style.use(BASE_STYLE_PATH)
    STYLE_PATH = os.path.join(BASE_DIR, f"./conf/style/{cfg.matplotlib_style}.mplstyle")
    plt.style.use(STYLE_PATH)

    surrogate_model = build_surrogate(cfg.sindy.optimizer)
    # mlflow name
    try:
        hydra_overrides = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
    except Exception:
        hydra_overrides = "OverrideError"
    if hydra_overrides == "":
        hydra_overrides = "Default"
    # Prefect flow
    main_flow(dataset_cfg, surrogate_model, hydra_overrides)
    logger.info("Script ended")


if __name__ == "__main__":
    main()
