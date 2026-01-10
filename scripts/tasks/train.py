import gokart
import hydra
import luigi
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from neurosurrogate.utils.data_processing import (
    _get_control_input,
    _prepare_train_data,
    get_gate_data,
    transform_dataset_with_preprocessor,
)

from .data import MakeDatasetTask
from .utils import CommonConfig, recursive_to_dict


class TrainPreprocessorTask(gokart.TaskOnKart):
    """前処理器（Preprocessor）の学習を行うタスク"""

    preprocessor_model_cfg = luigi.DictParameter(
        default=CommonConfig().model_cfg_dict["preprocessor"]
    )

    def requires(self):
        return MakeDatasetTask()

    def run(self):
        preprocessor_model_cfg = OmegaConf.create(
            recursive_to_dict(self.preprocessor_model_cfg)
        )
        preprocessor = hydra.utils.instantiate(preprocessor_model_cfg)

        # MakeDatasetTask returns the path dictionary
        train_xr_dataset = self.load()["train"].load()
        train_gate_data = get_gate_data(train_xr_dataset)

        logger.info("Fitting preprocessor...")
        preprocessor.fit(train_gate_data)
        self.dump(preprocessor)


class TrainModelTask(gokart.TaskOnKart):
    """モデルの学習を行うタスク"""

    surrogate_model_cfg = luigi.DictParameter(
        default=CommonConfig().model_cfg_dict["surrogate"]
    )
    train_dataset_type = luigi.DictParameter(
        default=CommonConfig().datasets_dict["train"]["data_type"]
    )

    def requires(self):
        return {
            "data": MakeDatasetTask(),
            "preprocessor": TrainPreprocessorTask(),
        }

    def run(self):
        surrogate_model_cfg = OmegaConf.create(
            recursive_to_dict(self.surrogate_model_cfg)
        )
        surrogate = hydra.utils.instantiate(surrogate_model_cfg)

        loaded_data = self.load()
        train_dataset = loaded_data["data"]["train"].load()
        preprocessor = loaded_data["preprocessor"]

        logger.debug(f"train_dataset {train_dataset}")

        train = _prepare_train_data(train_dataset, preprocessor)
        u = _get_control_input(train_dataset, data_type=self.train_dataset_type)

        logger.info("Fitting surrogate model...")
        surrogate.fit(
            train=train,
            u=u,
            t=train_dataset["time"].to_numpy(),
        )

        self.dump(surrogate)


class LogTrainModelTask(gokart.TaskOnKart):
    run_id = luigi.Parameter(default=CommonConfig().run_id)

    def requires(self):
        return TrainModelTask()

    def run(self):
        surrogate = self.load()
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_dict(
                surrogate.sindy.equations(precision=3),
                artifact_file="sindy_equations.txt",
            )
            mlflow.log_text(
                np.array2string(surrogate.sindy.optimizer.coef_, precision=3),
                artifact_file="coef.txt",
            )
            feature_names = surrogate.sindy.get_feature_names()
            mlflow.log_text("\n".join(feature_names), artifact_file="feature_names.txt")
            mlflow.log_param("sindy_params", str(surrogate.sindy.optimizer.get_params))
        self.dump(True)


class PreProcessDataTask(gokart.TaskOnKart):
    def requires(self):
        return {
            "preprocessor": TrainPreprocessorTask(),
            "data": MakeDatasetTask(),
        }

    def run(self):
        loaded_data = self.load()
        preprocessor = loaded_data["preprocessor"]
        dataset_targets = loaded_data["data"]

        preprocessed_datasets = {}
        # preprocess data
        for k, target in dataset_targets.items():
            xr_data = target.load()
            transformed_xr = transform_dataset_with_preprocessor(xr_data, preprocessor)
            logger.info(f"Transformed xr dataset: {k}")
            preprocessed_datasets[k] = transformed_xr
        self.dump(preprocessed_datasets)


class LogPreprocessDataTask(gokart.TaskOnKart):
    run_id = luigi.Parameter(default=CommonConfig().run_id)

    def requires(self):
        return PreProcessDataTask()

    def run(self):
        path_dict = self.load()

        conf = CommonConfig()
        datasets_cfg = OmegaConf.create(recursive_to_dict(conf.datasets_dict))
        neurons_cfg = OmegaConf.create(recursive_to_dict(conf.neurons_dict))

        with mlflow.start_run(run_id=self.run_id):
            for key, data in path_dict.items():
                self._process_and_log_dataset(key, data, datasets_cfg, neurons_cfg)

        self.dump(True)

    def _process_and_log_dataset(self, key, xr_data, datasets_cfg, neurons_cfg):
        """1つのデータセットに対して処理とログ出力を行う"""
        dataset_type = datasets_cfg[key].data_type
        u_cfg = neurons_cfg[dataset_type].transform.u
        data_vars = xr_data["vars"]
        external_input = xr_data[u_cfg.ind].sel(u_cfg.sel)
        fig = self._create_figure(data_vars, external_input)
        mlflow.log_figure(fig, f"preprocessed/{dataset_type}/{key}.png")
        plt.close(fig)

    @staticmethod
    def _create_figure(data_vars, external_input):
        """matplotlib の描画ロジックをカプセル化"""
        features = data_vars.features.values
        num_features = len(features)

        fig, axs = plt.subplots(
            nrows=1 + num_features,
            ncols=1,
            figsize=(10, 4 * (1 + num_features)),
            sharex=True,
            layout="constrained",  # タイトルの重なり防止
        )

        # 外部入力のプロット
        axs[0].plot(external_input.time, external_input, label="I_ext(t)")
        axs[0].set_ylabel("I_ext(t)")
        axs[0].legend()

        # 各特徴量のプロット
        for i, feature_name in enumerate(features):
            ax = axs[i + 1]
            ax.plot(
                data_vars.time, data_vars.sel(features=feature_name), label=feature_name
            )
            ax.set_ylabel(feature_name)
            ax.legend()

        axs[-1].set_xlabel("Time step")
        return fig
