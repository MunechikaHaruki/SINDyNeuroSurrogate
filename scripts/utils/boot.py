import os

import mlflow
from omegaconf import OmegaConf


def setup_proxy():
    # プロキシ設定を一時的に無効化
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def setup_mlflow(experiment_name):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(experiment_name)
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"


def setup_matplotlib(matplotlib_style):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.join(CURRENT_DIR, "../conf/style")
    BASE_STYLE_PATH = os.path.join(BASE_DIR, "./base.mplstyle")
    plt.style.use(BASE_STYLE_PATH)
    STYLE_PATH = os.path.join(BASE_DIR, f"./{matplotlib_style}.mplstyle")
    plt.style.use(STYLE_PATH)


def setup_all(cfg):
    OmegaConf.resolve(cfg)
    setup_proxy()
    setup_mlflow(cfg.experiment_name)
    setup_matplotlib(cfg.matplotlib_style)
