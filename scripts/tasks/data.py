import os
import random
from pathlib import Path

import gokart
import gokart.file_processor
import h5py
import hydra
import luigi
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf

from neurosurrogate.config import PROCESSED_DATA_DIR
from neurosurrogate.dataset_utils._base import preprocess_dataset

from .utils import CommonConfig


class GenerateSingleDatasetTask(gokart.TaskOnKart):
    dataset_cfg_yaml = luigi.Parameter()
    neuron_cfg_yaml = luigi.Parameter()
    seed = luigi.IntParameter()

    def run(self):
        dataset_cfg = OmegaConf.create(self.dataset_cfg_yaml)
        data_type = dataset_cfg["data_type"]
        neuron_cfg = OmegaConf.create(self.neuron_cfg_yaml)
        params = hydra.utils.instantiate(neuron_cfg["params"])
        random.seed(self.seed)
        np.random.seed(self.seed)

        temp_h5 = Path(
            self.make_target(
                "data.h5", processor=gokart.file_processor.BinaryFileProcessor()
            ).path()
        )
        temp_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(temp_h5, "w") as fp:
            hydra.utils.instantiate(dataset_cfg["current"], fp=fp, dt=params.DT)
            hydra.utils.instantiate(neuron_cfg["simulator"], fp=fp, params=params)
        processed_info = preprocess_dataset(data_type, temp_h5, params)
        self.dump(processed_info)


class MakeDatasetTask(gokart.TaskOnKart):
    """データセットの生成と前処理を行うタスク"""

    seed = luigi.IntParameter()

    def requires(self):
        conf = CommonConfig()
        datasets = OmegaConf.create(conf.datasets_cfg_yaml)
        neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)
        return {
            name: GenerateSingleDatasetTask(
                dataset_cfg_yaml=OmegaConf.to_yaml(dataset_cfg),
                neuron_cfg_yaml=OmegaConf.to_yaml(neurons_cfg[dataset_cfg.data_type]),
                seed=self.seed + idx,
            )
            for idx, (name, dataset_cfg) in enumerate(datasets.items())
        }

    def run(self):
        self.dump({"path_dict": self.load()})


class LogMakeDatasetTask(gokart.TaskOnKart):
    def requires(self):
        return MakeDatasetTask(seed=CommonConfig().seed)

    def run(self):
        conf = CommonConfig()
        loaded_data = self.load()
        neurons_cfg = OmegaConf.create(conf.neurons_cfg_yaml)
        with mlflow.start_run(run_id=conf.run_id):
            for name, dataset_cfg in OmegaConf.create(conf.datasets_cfg_yaml).items():
                with xr.open_dataset(loaded_data["path_dict"][name]) as xr_data:
                    fig = hydra.utils.instantiate(
                        neurons_cfg[dataset_cfg.data_type].plot, xr=xr_data
                    )
                    mlflow.log_figure(
                        fig, f"original/{dataset_cfg.data_type}/{name}.png"
                    )
                    plt.close(fig)
                logger.info(f"Logged dataset: {name}")
        self.dump(True)


GATE_VAR_SLICE = slice(1, 4, None)
V_VAR_SLICE = slice(0, 1, None)


class PreProcessTask(gokart.TaskOnKart):
    def requires(self):
        from scripts.tasks.train import TrainModelTask

        return TrainModelTask()

    def run(self):
        loaded_data = self.load()
        preprocessed_path_dict = {}
        # preprocess data
        for k, v in loaded_data["path_dict"].items():
            xr_data = xr.open_dataset(v)
            xr_gate = xr_data["vars"].to_numpy()[:, GATE_VAR_SLICE]
            transformed_gate = loaded_data["preprocessor"].transform(xr_gate)
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
        self.dump({"path_dict": preprocessed_path_dict})


class LogPreprocessDataTask(gokart.TaskOnKart):
    def requires(self):
        return PreProcessTask()

    def run(self):
        loaded_data = self.load()
        path_dict = loaded_data.get("path_dict", {})

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
            # 設定の取得
            dataset_type = datasets_cfg[key].data_type
            u_cfg = neurons_cfg[dataset_type].transform.u

            # データの抽出
            data_vars = xr_data["vars"]
            external_input = xr_data[u_cfg.ind].sel(u_cfg.sel)

            # プロット作成
            fig = self._create_figure(data_vars, external_input)

            # MLflow への記録
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
