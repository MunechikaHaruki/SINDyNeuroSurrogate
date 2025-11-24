import os
import random
import subprocess
import time
from datetime import datetime

import gokart
import h5py
import hydra
import luigi
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from neurosurrogate.dataset_utils.hh.hh_simulator import hh_simulate, threecomp_simulate
from neurosurrogate.dataset_utils.traub.traub_simulator import traub_simulate
from omegaconf import DictConfig, OmegaConf

from neurosurrogate.config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SURROGATE_DATA_DIR,
)
from neurosurrogate.dataset import preprocess_dataset
from neurosurrogate.plots import plot_3comp_hh, plot_hh

GATE_VAR_SLICE = slice(1, 4, None)
V_VAR_SLICE = slice(0, 1, None)


class MakeDatasetTask(gokart.TaskOnKart):
    """データセットの生成と前処理を行うタスク"""

    datasets_cfg_yaml = luigi.Parameter()
    neurons_cfg_yaml = luigi.Parameter()
    experiment_name = luigi.Parameter()
    seed = luigi.IntParameter()

    def run(self):
        random.seed(self.seed)
        path_dict = {}
        neuron_cfg = OmegaConf.create(self.neurons_cfg_yaml)

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run.info.run_id}")

        for name, dataset_cfg in OmegaConf.create(self.datasets_cfg_yaml).items():
            file_name = f"{datetime.now()}_{name}.h5"
            params = hydra.utils.instantiate(neuron_cfg[dataset_cfg.data_type].params)
            output_path = RAW_DATA_DIR / dataset_cfg.data_type / file_name

            with h5py.File(output_path, "w") as fp:
                hydra.utils.instantiate(dataset_cfg.current, fp=fp, dt=params.DT)
            # Simulation
            SIMULATORS = {
                "hh": hh_simulate,
                "hh3": threecomp_simulate,
                "traub": traub_simulate,
            }
            with h5py.File(output_path, "a") as fp:
                start_time = time.perf_counter()
                SIMULATORS[dataset_cfg.data_type](fp, params)
                end_time = time.perf_counter()
                logger.info(f"Simulation time: {end_time - start_time:.4f}[s]")
            logger.success(f"Dataset generation complete: {output_path}")
            path_dict[name] = preprocess_dataset(
                dataset_cfg.data_type, file_name, params
            )

        self.dump({"path_dict": path_dict, "run_id": run_id})


class LogMakeDatasetTask(gokart.TaskOnKart):
    datasets_cfg_yaml = luigi.Parameter()
    dataset_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.dataset_task

    def run(self):
        path_dict = self.load()["path_dict"]
        with mlflow.start_run(run_id=self.load()["run_id"]):
            for name, dataset_cfg in OmegaConf.create(self.datasets_cfg_yaml).items():
                PLOTTERS = {
                    "hh": plot_hh,
                    "hh3": plot_3comp_hh,
                }
                xr_data = xr.open_dataset(path_dict[name])
                fig = PLOTTERS[dataset_cfg.data_type](xr_data)
                mlflow.log_figure(fig, f"oridginal/{name}.png")
                plt.close(fig)
            logger.info(f"Generated and preprocessed dataset: {name}")
        self.dump(True)


class SindySurrogateWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, surrogate_model):
        self.sindy_model = surrogate_model

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        init = model_input["init"][0]
        dt = model_input["dt"][0]
        iter = model_input["iter"][0]
        u = model_input.get("u", [None])[0]
        mode = model_input["mode"][0]
        return self.sindy_model.predict(init, dt, iter, u, mode)


class TrainModelTask(gokart.TaskOnKart):
    """モデルの学習を行うタスク"""

    model_cfg_yaml = luigi.Parameter()
    dataset_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.dataset_task

    def run(self):
        model_cfg = OmegaConf.create(self.model_cfg_yaml)
        preprocessor = hydra.utils.instantiate(model_cfg.preprocessor)
        surrogate = hydra.utils.instantiate(model_cfg.surrogate)

        train_xr_dataset = xr.open_dataset(self.load()["path_dict"]["train"])
        logger.trace(train_xr_dataset)
        train_gate_data = train_xr_dataset["vars"].to_numpy()[:, GATE_VAR_SLICE]
        V_data = train_xr_dataset["vars"].to_numpy()[:, V_VAR_SLICE]

        preprocessor.fit(train_gate_data)
        logger.info("Transforming training dataset...")
        transformed_gate = preprocessor.transform(train_gate_data)
        train = np.concatenate((V_data, transformed_gate), axis=1)
        logger.critical(train)
        logger.info("Fitting preprocessor...")

        if model_cfg.sel_train_u == "I_ext":
            u = train_xr_dataset["I_ext"].to_numpy()
        elif model_cfg.sel_train_u == "soma":
            u = train_xr_dataset["I_internal"].sel(direction="soma").to_numpy()

        logger.info("Fitting surrogate model...")
        surrogate.fit(
            train=train,
            u=u,
            t=train_xr_dataset["time"].to_numpy(),
        )

        with mlflow.start_run(run_id=self.load()["run_id"]):
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
            mlflow.pyfunc.log_model(
                name="model",
                python_model=SindySurrogateWrapper(surrogate),
                # input_example=input_data,
            )

        self.dump(
            {
                "preprocessor": preprocessor,
                "path_dict": self.load()["path_dict"],
                "run_id": self.load()["run_id"],
            }
        )


class PreProcessTask(gokart.TaskOnKart):
    train_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.train_task

    def run(self):
        preprocessed_path_dict = {}
        # preprocess data
        for k, v in self.load()["path_dict"].items():
            xr_data = xr.open_dataset(v)
            xr_gate = xr_data["vars"].to_numpy()[:, GATE_VAR_SLICE]
            transformed_gate = self.load()["preprocessor"].transform(xr_gate)
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
        self.dump(
            {"path_dict": preprocessed_path_dict, "run_id": self.load()["run_id"]}
        )


class LogPreprocessDataTask(gokart.TaskOnKart):
    preprocess_task = gokart.TaskInstanceParameter()
    datasets_cfg_yaml = luigi.Parameter()
    neuron_cfg_yaml = luigi.Parameter()

    def requires(self):
        return self.preprocess_task

    def run(self):
        datasets_cfg = OmegaConf.create(self.datasets_cfg_yaml)
        neurons_cfg = OmegaConf.create(self.neuron_cfg_yaml)

        with mlflow.start_run(run_id=self.load()["run_id"]):
            for k, v in self.load()["path_dict"].items():
                xr_data = xr.load_dataset(v)
                dataset_type = datasets_cfg[k].data_type
                u_dic = neurons_cfg[dataset_type].transform.u
                data = xr_data["vars"]
                external_input = xr_data[u_dic.ind].sel(u_dic.sel)
                num_features = len(data.features.values)

                fig, axs = plt.subplots(
                    1 + num_features,
                    1,
                    figsize=(10, 4 * (1 + num_features)),
                    sharex=True,
                )

                axs[0].plot(external_input.time, external_input, label="I_ext(t)")
                axs[0].set_ylabel("I_ext(t)")
                axs[0].legend()

                for i, feature in enumerate(data.features.values):
                    axs[i + 1].plot(
                        data.time, data.sel(features=feature), label=feature
                    )
                    axs[i + 1].set_ylabel(feature)
                    axs[i + 1].legend()
                axs[-1].set_xlabel("Time step")

                mlflow.log_figure(
                    fig,
                    f"preprocessed/{k}.png",
                )
                plt.close(fig)
        self.dump(True)

        # def log_psth(original_v, surrogate_v):
        #     def calc_psth(v: np.ndarray, delta: int):
        #         fire_indices, _ = find_peaks(v, height=0)
        #         psth_counts, bin_edges = np.histogram(
        #             fire_indices, bins=np.arange(0, len(v) + delta, delta)
        #         )
        #         return psth_counts, bin_edges

        #     def plot_psth(original_counts, surrogate_counts, bin_edges):
        #         bin_times = bin_edges * 0.01
        #         fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        #         axs[0].bar(
        #             bin_times[:-1],
        #             original_counts,
        #         )
        #         axs[1].bar(bin_times[:-1], surrogate_counts)
        #         return fig


class EvalTask(gokart.TaskOnKart):
    preprocess_task = gokart.TaskInstanceParameter()
    eval_cfg_yaml = luigi.Parameter()
    neuron_cfg_yaml = luigi.Parameter()
    datasets_cfg_yaml = luigi.Parameter()

    def requires(self):
        return self.preprocess_task

    def run(self):
        """
        Load a registered model from MLflow and make a prediction.
        """
        with mlflow.start_run(run_id=self.load()["run_id"]):
            model = mlflow.pyfunc.load_model(f"runs:/{self.load()['run_id']}/model")
        slicer_time = hydra.utils.instantiate(
            OmegaConf.create(self.eval_cfg_yaml).time_slice
        )
        datasets_cfg = OmegaConf.create(self.datasets_cfg_yaml)
        path_dict = {}
        for k, v in self.load()["path_dict"].items():
            logger.info(f"{v} started to process")
            ds = xr.open_dataset(v).isel(time=slicer_time)
            if datasets_cfg[k].data_type == "hh":
                mode = "SingleComp"
                u = ds["I_ext"].to_numpy()
                if OmegaConf.create(self.eval_cfg_yaml).onlyThreeComp is True:
                    logger.info(f"{k} is passed")
                    continue
            elif datasets_cfg[k].data_type == "hh3":
                mode = "ThreeComp"
                u = ds["I_ext"].to_numpy()
                if OmegaConf.create(self.eval_cfg_yaml).direct is True:
                    logger.info("Using direct ThreeComp mode")
                    neuron_cfg = OmegaConf.create(self.neuron_cfg_yaml)
                    u_dic = neuron_cfg[datasets_cfg[k].data_type].transform.u
                    u = ds[u_dic.ind].sel(u_dic.sel).to_numpy()
                    # u = ds["I_ext"].to_numpy() # 11/4のデータは,hh3,SingleComp,I_extでの予測　間違い
                    mode = "SingleComp"

            input_data = pd.DataFrame(
                {
                    "init": [ds["vars"][0]],
                    "dt": [0.01 * slicer_time.step],
                    "iter": [len(ds["time"].to_numpy())],
                    "u": [u],
                    "mode": [mode],
                }
            )
            logger.info(f"input:{input_data}")

            try:
                logger.critical(f"{k}")
                prediction = model.predict(input_data)
                logger.info(f"key:{k} prediction_result:{prediction}")
                if mode == "ThreeComp":
                    I_pre = 1 * (
                        prediction["vars"].sel(features="V_pre")
                        - prediction["vars"].sel(features="V")
                    )
                    I_post = 0.7 * (
                        prediction["vars"].sel(features="V")
                        - prediction["vars"].sel(features="V_post")
                    )
                    I_soma = I_pre - I_post
                    prediction["I_internal"] = xr.concat(
                        [I_pre, I_post, I_soma], dim="direction"
                    ).assign_coords(direction=["pre", "post", "soma"])
                logger.trace(prediction)

                file_path = SURROGATE_DATA_DIR / f"{datetime.now()}_{k}.npy"
                prediction.to_netcdf(file_path)
                path_dict[k] = file_path
            except ValueError as e:
                logger.error(f"Value Error: {e}")
        self.dump({"run_id": self.load()["run_id"], "path_dict": path_dict})


class LogEvalTask(gokart.TaskOnKart):
    datasets_cfg_yaml = luigi.Parameter()
    eval_task = gokart.TaskInstanceParameter()
    preprocess_task = gokart.TaskInstanceParameter()

    def requires(self):
        return {"eval_task": self.eval_task, "preprocess_task": self.preprocess_task}

    def run(self):
        def plot_diff(u: np.ndarray, original: xr.DataArray, surrogate: xr.DataArray):
            num_features = len(original.features.values)

            fig, axs = plt.subplots(
                1 + 2 * num_features,
                1,
                figsize=(10, 4 * (1 + num_features)),
                sharex=False,
            )

            # plot external_input (I_ext)
            axs[0].plot(u, label="I_ext(t)", color="gold")
            axs[0].set_ylabel("I_ext(t)")
            axs[0].legend()

            # 各 feature についてループ
            for i, feature in enumerate(original.features.values):
                # 1. 元のデータをプロット (引数 'oridginal' から)
                axs[2 * i + 1].plot(
                    original.time,
                    original.sel(features=feature),
                    color="blue",
                    label=f"Original {feature}",
                )
                axs[2 * i + 1].set_ylabel(feature)
                axs[2 * i + 1].legend()

                # 2. サロゲートモデルのデータをプロット (引数 'surrogate' から)
                #    surrogate も 'time' と 'features' の座標を持つと仮定
                axs[2 * i + 2].plot(
                    surrogate.time,
                    surrogate.sel(features=feature),
                    color="red",
                    label=f"Surrogate {feature}",
                )
                axs[2 * i + 2].set_ylabel(f"Surrogate {feature}")
                axs[2 * i + 2].legend()

            axs[-1].set_xlabel("Time step")
            fig.tight_layout()  # レイアウトを自動調整
            return fig

        datasets_cfg = OmegaConf.create(self.datasets_cfg_yaml)
        with mlflow.start_run(run_id=self.load()["eval_task"]["run_id"]):
            for k, v in self.load()["eval_task"]["path_dict"].items():
                if datasets_cfg[k].data_type == "hh":
                    plot_surrogate = plot_hh
                elif datasets_cfg[k].data_type == "hh3":
                    plot_surrogate = plot_3comp_hh
                surrogate_result = xr.open_dataset(v)
                preprocessed_result = xr.open_dataset(
                    self.load()["preprocess_task"]["path_dict"][k]
                )
                u = preprocessed_result["I_ext"].to_numpy()
                fig = plot_diff(
                    u, preprocessed_result["vars"], surrogate_result["vars"]
                )
                mlflow.log_figure(fig, f"compare/{k}.png")
                TMP = DATA_DIR / "show.png"
                fig.savefig(TMP)
                subprocess.run(["wezterm", "imgcat", TMP])

                mlflow.log_figure(
                    plot_surrogate(surrogate_result, surrogate=True),
                    f"surrogate_result/{k}.png",
                )
        self.dump(True)


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
        datasets_cfg_yaml=OmegaConf.to_yaml(cfg.datasets), dataset_task=dataset_task
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
