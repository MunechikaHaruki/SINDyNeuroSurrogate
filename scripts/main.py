import os

import matplotlib

matplotlib.use("Agg")
import logging
import subprocess

import hydra
import matplotlib.pyplot as plt
import mlflow
from flow import main_flow
from omegaconf import DictConfig, OmegaConf

# Prefectのインポートより前に環境変数を設定する
os.environ["PREFECT_LOGGING_EXTRA_LOGGERS"] = "neurosurrogate"

logger = logging.getLogger(__name__)


# プロキシ設定を一時的に無効化
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def get_commit_id():
    try:
        commit_id = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        commit_id = "unknown"
    return commit_id


def get_hydra_overrides():
    try:
        run_name_prefix = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
    except Exception:
        run_name_prefix = "default_run"
    return run_name_prefix


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Activate Script")
    OmegaConf.resolve(cfg)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    STYLE_PATH = os.path.join(BASE_DIR, f"./conf/style/{cfg.matplotlib_style}.mplstyle")
    plt.style.use(STYLE_PATH)

    run_name_prefix = get_hydra_overrides()
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(cfg.experiment_name)
    # Create run to generate ID
    with mlflow.start_run(run_name=run_name_prefix) as run:
        logger.info(f"run_id:{run.info.run_id}")
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")
        mlflow.set_tag("mlflow.runName", f"{run_name_prefix}_commit-{get_commit_id()}")
        # Prefect flow
        main_flow(cfg)
    logger.info("Script ended")


if __name__ == "__main__":
    main()
