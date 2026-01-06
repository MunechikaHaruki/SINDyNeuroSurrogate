import os
import subprocess
import sys

import gokart
import hydra
import luigi
import mlflow
from omegaconf import DictConfig, OmegaConf

from scripts.tasks.data import (
    LogMakeDatasetTask,
    LogPreprocessDataTask,
    MakeDatasetTask,
    PreProcessTask,
)
from scripts.tasks.eval import EvalTask, LogEvalTask
from scripts.tasks.train import TrainModelTask

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class LogAllConfTask(gokart.TaskOnKart):
    cfg_yaml = luigi.Parameter()
    log_dataset_task = gokart.TaskInstanceParameter()
    log_preprocess_task = gokart.TaskInstanceParameter()
    log_eval_task = gokart.TaskInstanceParameter()
    run_name_prefix = luigi.Parameter()

    def requires(self):
        return {
            "log_dataset_task": self.log_dataset_task,
            "log_preprocess_task": self.log_preprocess_task,
            "log_eval_task": self.log_eval_task,
        }

    def run(self):
        cfg = OmegaConf.create(self.cfg_yaml)
        dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        with mlflow.start_run(run_id=self.load()["log_eval_task"].get("run_id")):
            mlflow.log_dict(dict_cfg, "config.yaml")

            # --- Commit IDの取得 ---
            try:
                commit_id = (
                    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                    .decode("utf-8")
                    .strip()
                )
            except subprocess.CalledProcessError:
                commit_id = "unknown"  # gitリポジトリでない場合のフォールバック
            run_name = f"{self.run_name_prefix}_commit-{commit_id}"
            mlflow.set_tag("mlflow.runName", run_name)
            self.dump(True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    luigi_config = luigi.configuration.get_config()
    luigi_config.set(
        "CommonConfig", "datasets_cfg_yaml", OmegaConf.to_yaml(cfg.datasets)
    )
    luigi_config.set("CommonConfig", "neurons_cfg_yaml", OmegaConf.to_yaml(cfg.neurons))

    dataset_task = MakeDatasetTask(
        seed=cfg.seed,
        experiment_name=cfg.experiment_name,
    )
    train_task = TrainModelTask(
        model_cfg_yaml=OmegaConf.to_yaml(cfg.models),
        dataset_task=dataset_task,
    )
    preprocess_task = PreProcessTask(
        train_task=train_task,
    )

    eval_task = EvalTask(
        preprocess_task=preprocess_task, eval_cfg_yaml=OmegaConf.to_yaml(cfg.eval)
    )

    log_tasks = {
        "log_dataset_task": LogMakeDatasetTask(
            dataset_task=dataset_task,
        ),
        "log_preprocess_task": LogPreprocessDataTask(
            preprocess_task=preprocess_task,
        ),
        "log_eval_task": LogEvalTask(
            eval_task=eval_task,
            preprocess_task=preprocess_task,
        ),
    }

    try:
        run_name_prefix = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
    except Exception:
        run_name_prefix = "default_run"

    log_all_conf_task = LogAllConfTask(
        cfg_yaml=OmegaConf.to_yaml(cfg), run_name_prefix=run_name_prefix, **log_tasks
    )
    gokart.build(log_all_conf_task)


if __name__ == "__main__":
    main()
