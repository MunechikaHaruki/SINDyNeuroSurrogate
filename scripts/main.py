import json

import gokart
import hydra
import luigi
import mlflow
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from scripts.tasks.utils import RunAllLogging


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    try:
        run_name_prefix = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
    except Exception:
        run_name_prefix = "default_run"
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=run_name_prefix) as run:
        logger.info(f"run_id:{run.info.run_id}")

    luigi_config = luigi.configuration.get_config()
    luigi_config.set(
        "CommonConfig",
        "datasets_dict",
        json.dumps(OmegaConf.to_container(cfg.datasets, resolve=True)),
    )

    # set dataset_dict
    luigi_config.set(
        "CommonConfig",
        "neurons_dict",
        json.dumps(OmegaConf.to_container(cfg.neurons, resolve=True)),
    )
    luigi_config.set("CommonConfig", "model_cfg_yaml", OmegaConf.to_yaml(cfg.models))
    luigi_config.set("CommonConfig", "seed", str(cfg.seed))

    # set CommonConfig
    luigi_config.set("CommonConfig", "run_id", run.info.run_id)

    luigi_config.set("CommonConfig", "eval_cfg", json.dumps(dict(cfg.eval)))

    log_all_conf_task = RunAllLogging(
        cfg_yaml=OmegaConf.to_yaml(cfg), run_name_prefix=run_name_prefix
    )
    gokart.build(log_all_conf_task)


if __name__ == "__main__":
    main()
