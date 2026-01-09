import gokart
import hydra
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
from .utils import CommonConfig


class TrainPreprocessorTask(gokart.TaskOnKart):
    """前処理器（Preprocessor）の学習を行うタスク"""

    def requires(self):
        return MakeDatasetTask(seed=CommonConfig().seed)

    def run(self):
        conf = CommonConfig()
        model_cfg = OmegaConf.create(conf.model_cfg_yaml)
        preprocessor = hydra.utils.instantiate(model_cfg.preprocessor)

        # MakeDatasetTask returns the path dictionary
        dataset_paths = self.load()
        train_xr_dataset = dataset_paths["train"]
        train_gate_data = get_gate_data(train_xr_dataset)

        logger.info("Fitting preprocessor...")
        preprocessor.fit(train_gate_data)
        self.dump(preprocessor)


class TrainModelTask(gokart.TaskOnKart):
    """モデルの学習を行うタスク"""

    def requires(self):
        from scripts.tasks.data import MakeDatasetTask

        return {
            "data": MakeDatasetTask(seed=CommonConfig().seed),
            "preprocessor": TrainPreprocessorTask(),
        }

    def run(self):
        conf = CommonConfig()
        model_cfg = OmegaConf.create(conf.model_cfg_yaml)
        surrogate = hydra.utils.instantiate(model_cfg.surrogate)

        loaded_data = self.load()
        train_dataset = loaded_data["data"]["train"]
        preprocessor = loaded_data["preprocessor"]

        logger.trace(train_dataset)

        train = _prepare_train_data(train_dataset, preprocessor)
        u = _get_control_input(train_dataset, model_cfg)

        logger.info("Fitting surrogate model...")
        surrogate.fit(
            train=train,
            u=u,
            t=train_dataset["time"].to_numpy(),
        )

        self.dump(surrogate)


class LogTrainModelTask(gokart.TaskOnKart):
    def requires(self):
        return TrainModelTask()

    def run(self):
        conf = CommonConfig()
        surrogate = self.load()
        with mlflow.start_run(run_id=conf.run_id):
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
        from scripts.tasks.data import MakeDatasetTask

        return {
            "preprocessor": TrainPreprocessorTask(),
            "data": MakeDatasetTask(seed=CommonConfig().seed),
        }

    def run(self):
        loaded_data = self.load()
        preprocessor = loaded_data["preprocessor"]
        dataset_paths = loaded_data["data"]

        preprocessed_datasets = {}
        # preprocess data
        for k, xr_data in dataset_paths.items():
            transformed_xr = transform_dataset_with_preprocessor(xr_data, preprocessor)
            logger.info(f"Transformed xr dataset: {k}")
            preprocessed_datasets[k] = transformed_xr
        self.dump(preprocessed_datasets)


class LogPreprocessDataTask(gokart.TaskOnKart):
    def requires(self):
        return PreProcessDataTask()

    def run(self):
        path_dict = self.load()

        conf = CommonConfig()
        datasets_cfg = OmegaConf.create(conf.datasets_cfg_yaml)
        neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)

        with mlflow.start_run(run_id=conf.run_id):
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
