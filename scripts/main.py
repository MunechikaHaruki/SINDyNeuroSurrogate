import logging
import os

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from utils.flow import cli_flow

logger = logging.getLogger(__name__)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "../")


def setup_proxy():
    # プロキシ設定を一時的に無効化
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def setup_mlflow(is_multirun):
    MLRUN_DIR = os.path.join(PROJECT_ROOT, "mlruns")
    mlflow.set_tracking_uri(f"file://{MLRUN_DIR}")
    mlflow.enable_system_metrics_logging()
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"
    if is_multirun:
        mlflow.set_experiment("test_dynamic_datasets")
    else:
        mlflow.set_experiment("test_static_params")


def setup_matplotlib(matplotlib_style):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    STYLE_DIR = os.path.join(PROJECT_ROOT, "./scripts/conf/style")
    plt.style.use(os.path.join(STYLE_DIR, "./base.mplstyle"))
    plt.style.use(os.path.join(STYLE_DIR, f"./{matplotlib_style}.mplstyle"))


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Activate Script")
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    setup_proxy()
    is_multirun = HydraConfig.get().mode.name == "MULTIRUN"
    setup_mlflow(is_multirun)
    setup_matplotlib(cfg["matplotlib_style"])
    cli_flow(is_multirun, cfg["sindy"])


if __name__ == "__main__":
    main()
