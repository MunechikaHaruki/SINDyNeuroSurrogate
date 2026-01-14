import gokart
import luigi
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from neurosurrogate.plots import _create_figure
from neurosurrogate.utils.data_processing import (
    _get_control_input,
    _prepare_train_data,
    get_gate_data,
    transform_dataset_with_preprocessor,
)

from .data import GenerateSingleDatasetTask, NetCDFProcessor
from .utils import CommonConfig, recursive_to_dict


class TrainPreprocessorTask(gokart.TaskOnKart):
    """前処理器（Preprocessor）の学習を行うタスク"""

    def requires(self):
        conf = CommonConfig()
        dataset_cfg = conf.datasets_dict["train"]
        neuron_cfg = conf.neurons_dict[dataset_cfg["data_type"]]
        return GenerateSingleDatasetTask(dataset_cfg=dataset_cfg, neuron_cfg=neuron_cfg)

    def run(self):
        from sklearn.decomposition import PCA

        preprocessor = PCA(n_components=1)

        # MakeDatasetTask returns the path dictionary
        train_xr_dataset = self.load()
        train_gate_data = get_gate_data(train_xr_dataset)

        logger.info("Fitting preprocessor...")
        preprocessor.fit(train_gate_data)
        self.dump(preprocessor)


class TrainModelTask(gokart.TaskOnKart):
    """モデルの学習を行うタスク"""

    surrogate_model_cfg = luigi.DictParameter(
        default=CommonConfig().model_cfg_dict["surrogate"]
    )
    train_dataset_type = luigi.Parameter(
        default=CommonConfig().datasets_dict["train"]["data_type"]
    )

    def requires(self):
        conf = CommonConfig()
        dataset_cfg = conf.datasets_dict["train"]
        neuron_cfg = conf.neurons_dict[dataset_cfg["data_type"]]
        return {
            "data": GenerateSingleDatasetTask(
                dataset_cfg=dataset_cfg, neuron_cfg=neuron_cfg
            ),
            "preprocessor": TrainPreprocessorTask(),
        }

    def run(self):
        surrogate_model_cfg = OmegaConf.create(
            recursive_to_dict(self.surrogate_model_cfg)
        )
        from neurosurrogate.modeling.surrogate import SINDySurrogate
        from neurosurrogate.utils.base_hh import hh_sindy, input_features

        loaded_data = self.load()
        train_dataset = loaded_data["data"]
        preprocessor = loaded_data["preprocessor"]
        train = _prepare_train_data(train_dataset, preprocessor)
        u = _get_control_input(train_dataset, data_type=self.train_dataset_type)
        hh_sindy.fit(
            train,
            u=u,
            t=train_dataset["time"].to_numpy(),
            feature_names=input_features,
        )
        surrogate = SINDySurrogate(hh_sindy, params=surrogate_model_cfg["params"])
        logger.debug(f"train_dataset {train_dataset}")
        logger.info("Fitting surrogate model...")
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


class PreProcessSingleDataTask(gokart.TaskOnKart):
    dataset_name = luigi.Parameter()

    def requires(self):
        conf = CommonConfig()
        dataset_cfg = conf.datasets_dict[self.dataset_name]
        neuron_cfg = conf.neurons_dict[dataset_cfg["data_type"]]

        return {
            "preprocessor": TrainPreprocessorTask(),
            "data": GenerateSingleDatasetTask(
                dataset_cfg=dataset_cfg, neuron_cfg=neuron_cfg
            ),
        }

    def output(self):
        return self.make_target(
            f"preprocessed_{self.dataset_name}.nc", processor=NetCDFProcessor()
        )

    def run(self):
        inputs = self.load()
        preprocessor = inputs["preprocessor"]
        xr_data = inputs["data"]

        transformed_xr = transform_dataset_with_preprocessor(xr_data, preprocessor)
        logger.info(f"Transformed xr dataset: {self.dataset_name}")
        self.dump(transformed_xr)


class PreProcessDataTask(gokart.TaskOnKart):
    def requires(self):
        conf = CommonConfig()
        return {
            name: PreProcessSingleDataTask(dataset_name=name)
            for name in conf.datasets_dict.keys()
        }

    def run(self):
        targets = self.input()
        preprocessed_datasets = {k: v.load() for k, v in targets.items()}
        self.dump(preprocessed_datasets)


class LogSinglePreprocessDataTask(gokart.TaskOnKart):
    dataset_key = luigi.Parameter()
    dataset_type = luigi.Parameter()
    run_id = luigi.Parameter(default=CommonConfig().run_id)

    def requires(self):
        return PreProcessSingleDataTask(dataset_name=self.dataset_key)

    def run(self):
        xr_data = self.load()
        with mlflow.start_run(run_id=self.run_id):
            """1つのデータセットに対して処理とログ出力を行う"""
            external_input = _get_control_input(xr_data, self.dataset_type)
            fig = _create_figure(xr_data["vars"], external_input)
            mlflow.log_figure(
                fig, f"preprocessed/{self.dataset_type}/{self.dataset_key}.png"
            )
            plt.close(fig)

        self.dump(True)


class LogPreprocessDataTask(gokart.TaskOnKart):
    def requires(self):
        conf = CommonConfig()

        return {
            name: LogSinglePreprocessDataTask(
                dataset_key=name,
                dataset_type=conf.datasets_dict[name]["data_type"],
            )
            for name in conf.datasets_dict.keys()
        }

    def run(self):
        self.dump(True)
