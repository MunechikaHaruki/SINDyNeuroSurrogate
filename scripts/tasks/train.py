import os

import gokart
import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf

from neurosurrogate.config import PROCESSED_DATA_DIR
from neurosurrogate.utils.data_processing import (
    GATE_VAR_SLICE,
    V_VAR_SLICE,
    _get_control_input,
    _prepare_train_data,
)

from .utils import CommonConfig


class TrainPreprocessorTask(gokart.TaskOnKart):
    """前処理器（Preprocessor）の学習を行うタスク"""

    def requires(self):
        from scripts.tasks.data import MakeDatasetTask

        return MakeDatasetTask(seed=CommonConfig().seed)

    def run(self):
        conf = CommonConfig()
        model_cfg = OmegaConf.create(conf.model_cfg_yaml)
        preprocessor = hydra.utils.instantiate(model_cfg.preprocessor)

        # MakeDatasetTask returns the path dictionary
        dataset_paths = self.load()
        train_xr_dataset = xr.open_dataset(dataset_paths["train"])
        train_gate_data = train_xr_dataset["vars"].to_numpy()[:, GATE_VAR_SLICE]

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
        dataset_paths = loaded_data["data"]
        preprocessor = loaded_data["preprocessor"]

        train_xr_dataset = xr.open_dataset(dataset_paths["train"])
        logger.trace(train_xr_dataset)

        train = _prepare_train_data(train_xr_dataset, preprocessor)
        u = _get_control_input(train_xr_dataset, model_cfg)

        logger.info("Fitting surrogate model...")
        surrogate.fit(
            train=train,
            u=u,
            t=train_xr_dataset["time"].to_numpy(),
        )

        self.dump(
            {
                "surrogate": surrogate,
                "preprocessor": preprocessor,
            }
        )


class LogTrainModelTask(gokart.TaskOnKart):
    def requires(self):
        return TrainModelTask()

    def run(self):
        conf = CommonConfig()
        surrogate = self.load()["surrogate"]
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


class PreProcessTask(gokart.TaskOnKart):
    def requires(self):
        from scripts.tasks.data import MakeDatasetTask
        from scripts.tasks.train import TrainModelTask

        return {
            "model": TrainModelTask(),
            "data": MakeDatasetTask(seed=CommonConfig().seed),
        }

    def run(self):
        loaded_data = self.load()
        model_data = loaded_data["model"]
        dataset_paths = loaded_data["data"]

        preprocessed_path_dict = {}
        # preprocess data
        for k, v in dataset_paths.items():
            xr_data = xr.open_dataset(v)
            xr_gate = xr_data["vars"].to_numpy()[:, GATE_VAR_SLICE]
            transformed_gate = model_data["preprocessor"].transform(xr_gate)
            V_data = xr_data["vars"][:, V_VAR_SLICE].to_numpy().reshape(-1, 1)
            new_vars = np.concatenate((V_data, transformed_gate), axis=1)
            new_feature_names = ["V"] + [
                f"latent{i + 1}" for i in range(transformed_gate.shape[1])
            ]
            transformed_xr = xr_data.copy().drop_vars("vars").drop_vars("features")
            transformed_xr["vars"] = xr.DataArray(
                new_vars,
                coords={
                    "time": xr_data.coords["time"],
                    "features": new_feature_names,  # 新しい次元と座標
                },
                dims=["time", "features"],  # 新しい次元名
            )
            logger.info(f"Transformed xr dataset: {k}")
            logger.info(transformed_gate.__repr__())
            preprocessed_path_dict[k] = PROCESSED_DATA_DIR / os.path.basename(v)
            transformed_xr.to_netcdf(preprocessed_path_dict[k])
        self.dump(preprocessed_path_dict)


class LogPreprocessDataTask(gokart.TaskOnKart):
    def requires(self):
        return PreProcessTask()

    def run(self):
        path_dict = self.load()

        conf = CommonConfig()
        datasets_cfg = OmegaConf.create(conf.datasets_cfg_yaml)
        neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)

        with mlflow.start_run(run_id=conf.run_id):
            for key, file_path in path_dict.items():
                self._process_and_log_dataset(key, file_path, datasets_cfg, neurons_cfg)

        self.dump(True)

    def _process_and_log_dataset(self, key, file_path, datasets_cfg, neurons_cfg):
        """1つのデータセットに対して処理とログ出力を行う"""
        with xr.open_dataset(file_path) as xr_data:
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
