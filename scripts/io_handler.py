import inspect
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import numpy as np
import yaml
from matplotlib.figure import Figure

from neurosurrogate.model.model_dataset import DatasetConfig
from neurosurrogate.model.model_neurosindy import SINDyNeuroSurrogate, make_surr_comp
from neurosurrogate.profiler.profiler_model import SINDyAnalyzer
from neurosurrogate.profiler.profiler_view import view_model

TARGET_EXP = "test_static_params"


logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent  # 階層に応じて調整
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
mlflow.enable_system_metrics_logging()
mlflow.set_experiment(TARGET_EXP)
os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = "1"


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    run_name: str
    sindy_coef: Figure
    dataset: DatasetConfig
    equations: str

    @staticmethod
    def get_run_info(run_id: str) -> "RunInfo":
        client = mlflow.MlflowClient()

        def load_yaml(filename: str) -> dict:
            return yaml.safe_load(
                mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}")
            )

        def load_text(filename: str) -> str:
            return mlflow.artifacts.load_text(f"runs:/{run_id}/{filename}")

        return RunInfo(
            run_id=run_id,
            run_name=client.get_run(run_id).data.tags["mlflow.runName"],
            sindy_coef=view_model(**load_yaml("view.json")),
            dataset=DatasetConfig.from_dict(load_yaml("dataset.yaml")),
            equations=load_text("equations.txt"),
        )


def log_surrogate_summary(summary: SINDyAnalyzer):
    mlflow.log_metrics(summary.metrics)
    mlflow.log_params(summary.params)
    mlflow.log_dict(
        summary.view,
        artifact_file="view.json",
    )

    for filename, content in summary.texts.items():
        mlflow.log_text(content, artifact_file=filename)


class SINDySurrogateMLflowModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "target_module", context.artifacts["target_module_path"]
        )
        target_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(target_module)
        source = open(context.artifacts["source_path"]).read()
        local_vars = {}
        exec(source, vars(target_module), local_vars)
        compute_theta = local_vars["dynamic_compute_theta"]
        xi_matrix = np.load(context.artifacts["xi_path"])
        self._gate_inits = np.load(context.artifacts["gate_inits_path"]).tolist()
        self.sindy_args = (xi_matrix, compute_theta)
        self.preprocessor = joblib.load(context.artifacts["preprocessor_path"])

    def make_surr_comp(self, name: str):
        return make_surr_comp(name, self._gate_inits)

    def predict(self, context, model_input):
        pass  # unified_simulatorに直接渡すので不要


def log_surrogate_model(surrogate: SINDyNeuroSurrogate):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        np.save(tmpdir / "xi.npy", surrogate.sindy.coefficients())
        np.save(tmpdir / "gate_inits.npy", np.array(surrogate._gate_inits))
        (tmpdir / "source.py").write_text(surrogate.source)
        joblib.dump(surrogate.preprocessor, tmpdir / "preprocessor.joblib")

        mlflow.pyfunc.log_model(
            artifact_path="surrogate_model",
            python_model=SINDySurrogateMLflowModel(),
            artifacts={
                "xi_path": str(tmpdir / "xi.npy"),
                "gate_inits_path": str(tmpdir / "gate_inits.npy"),
                "source_path": str(tmpdir / "source.py"),
                "target_module_path": inspect.getfile(surrogate.target_module),
                "preprocessor_path": str(tmpdir / "preprocessor.joblib"),
            },
        )


def load_surrogate_model(run_id: str):
    # log_surrogate_model で指定した artifact_path を使用
    model_uri = f"runs:/{run_id}/surrogate_model"
    logger.info(f"Loading custom MLflow model from: {model_uri}")

    try:
        # pyfuncとしてロード（内部で load_context が実行される）
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        # PythonModelの実体（SINDySurrogateMLflowModelのインスタンス）を取り出す
        surrogate = pyfunc_model._model_impl.python_model
        return surrogate

    except Exception as e:
        logger.error(f"Failed to load surrogate from MLflow: {e}")
        raise
