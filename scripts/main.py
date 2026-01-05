import os
import subprocess
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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


class LogAllConfTask(gokart.TaskOnKart):
    cfg_yaml = luigi.Parameter()
    eval_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.eval_task

    def run(self):
        cfg = OmegaConf.create(self.cfg_yaml)
        dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        with mlflow.start_run(run_id=self.load()["run_id"]):
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
            overrides = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
            run_name = f"{overrides}_commit-{commit_id}"
            mlflow.set_tag("mlflow.runName", run_name)
            self.dump(True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    # gokartのタスクを実行
    dataset_task = MakeDatasetTask(
        datasets_cfg_yaml=OmegaConf.to_yaml(cfg.datasets),
        neurons_cfg_yaml=OmegaConf.to_yaml(cfg.neurons),
        seed=cfg.seed,
        experiment_name=cfg.experiment_name,
    )
    log_dataset_task = LogMakeDatasetTask(
        datasets_cfg_yaml=OmegaConf.to_yaml(cfg.datasets),
        neuron_cfg_yaml=OmegaConf.to_yaml(cfg.neurons),
        dataset_task=dataset_task,
    )
    train_task = TrainModelTask(
        model_cfg_yaml=OmegaConf.to_yaml(cfg.models),
        dataset_task=dataset_task,
    )
    preprocess_task = PreProcessTask(
        train_task=train_task,
    )
    log_preprocess_task = LogPreprocessDataTask(
        preprocess_task=preprocess_task,
        datasets_cfg_yaml=OmegaConf.to_yaml(cfg.datasets),
        neuron_cfg_yaml=OmegaConf.to_yaml(cfg.neurons),
    )
    eval_task = EvalTask(
        preprocess_task=preprocess_task,
        eval_cfg_yaml=OmegaConf.to_yaml(cfg.eval),
        neuron_cfg_yaml=OmegaConf.to_yaml(cfg.neurons),
        datasets_cfg_yaml=OmegaConf.to_yaml(cfg.datasets),
    )

    log_eval_task = LogEvalTask(
        eval_task=eval_task,
        preprocess_task=preprocess_task,
        datasets_cfg_yaml=OmegaConf.to_yaml(cfg.datasets),
    )

    log_all_conf_task = LogAllConfTask(
        cfg_yaml=OmegaConf.to_yaml(cfg), eval_task=eval_task
    )
    gokart.build(log_dataset_task)
    gokart.build(log_preprocess_task)
    gokart.build(log_eval_task)
    gokart.build(log_all_conf_task)


if __name__ == "__main__":
    main()
