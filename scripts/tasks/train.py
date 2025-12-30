import gokart
import hydra
import luigi
import mlflow
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf

GATE_VAR_SLICE = slice(1, 4, None)
V_VAR_SLICE = slice(0, 1, None)


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
