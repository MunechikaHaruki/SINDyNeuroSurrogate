import subprocess
from collections.abc import Mapping

import gokart
import luigi
import mlflow
from omegaconf import OmegaConf


def recursive_to_dict(obj):
    if isinstance(obj, (list, tuple)):
        return [recursive_to_dict(x) for x in obj]
    if isinstance(obj, Mapping):
        return {k: recursive_to_dict(v) for k, v in obj.items()}
    return obj


class RunAllLogging(gokart.TaskOnKart):
    cfg_yaml = luigi.Parameter()
    run_name_prefix = luigi.Parameter()

    def requires(self):
        from scripts.tasks.data import (
            LogMakeDatasetTask,
        )
        from scripts.tasks.eval import LogEvalTask
        from scripts.tasks.train import LogPreprocessDataTask, LogTrainModelTask

        return {
            "log_dataset_task": LogMakeDatasetTask(),
            "log_preprocess_task": LogPreprocessDataTask(),
            "log_eval_task": LogEvalTask(),
            "log_trainmodel_task": LogTrainModelTask(),
        }

    def run(self):
        cfg = OmegaConf.create(self.cfg_yaml)
        dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        from tasks.utils import CommonConfig

        conf = CommonConfig()

        with mlflow.start_run(run_id=conf.run_id):
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


class CommonConfig(luigi.Config):
    datasets_dict = luigi.DictParameter()
    neurons_dict = luigi.DictParameter()
    model_cfg_dict = luigi.DictParameter()
    eval_cfg = luigi.DictParameter()
    seed = luigi.IntParameter()
    run_id = luigi.Parameter()
