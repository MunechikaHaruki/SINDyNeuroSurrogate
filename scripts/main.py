import os

import matplotlib

matplotlib.use("Agg")

import hydra
import matplotlib.pyplot as plt
import mlflow
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from prefect import flow
from tasks.data import (
    generate_dataset_flow,
)
from tasks.eval import (
    eval_flow,
)
from tasks.train import (
    train_flow,
)
from tasks.utils import get_commit_id, get_hydra_overrides

# プロキシ設定を一時的に無効化
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1"


@flow
def main_flow(cfg: DictConfig):
    train_ds = generate_dataset_flow("train", cfg)
    preprocessor, surrogate_model = train_flow(cfg, train_ds)

    for name in cfg.datasets.keys():
        eval_flow(
            name=name,
            preprocessor=preprocessor,
            surrogate_model=surrogate_model,
            cfg=cfg,
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
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


if __name__ == "__main__":
    main()
