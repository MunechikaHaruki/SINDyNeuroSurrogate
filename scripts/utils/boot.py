import os

import mlflow
from omegaconf import OmegaConf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "../../")


def setup_proxy():
    # プロキシ設定を一時的に無効化
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def setup_mlflow(experiment_name):
    MLRUN_DIR = os.path.join(PROJECT_ROOT, "mlruns")
    mlflow.set_tracking_uri(f"file://{MLRUN_DIR}")
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(experiment_name)
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"


def setup_matplotlib(matplotlib_style):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    STYLE_DIR = os.path.join(PROJECT_ROOT, "./scripts/conf/style")
    plt.style.use(os.path.join(STYLE_DIR, "./base.mplstyle"))
    plt.style.use(os.path.join(STYLE_DIR, f"./{matplotlib_style}.mplstyle"))


def setup_all(cfg):
    OmegaConf.resolve(cfg)
    setup_proxy()
    setup_mlflow(cfg.experiment_name)
    setup_matplotlib(cfg.matplotlib_style)
