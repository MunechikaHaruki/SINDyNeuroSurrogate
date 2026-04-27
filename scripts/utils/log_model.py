import inspect
import logging
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np

from neurosurrogate.modeling import SINDySurrogateWrapper

logger = logging.getLogger(__name__)


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
        self.gate_init = np.load(context.artifacts["gate_init_path"])
        self.sindy_args = (xi_matrix, compute_theta)
        self.preprocessor = joblib.load(context.artifacts["preprocessor_path"])

    def predict(self, context, model_input):
        pass  # unified_simulatorに直接渡すので不要


def log_surrogate_model(surrogate: SINDySurrogateWrapper):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        np.save(tmpdir / "xi.npy", surrogate.sindy.coefficients())
        np.save(tmpdir / "gate_init.npy", surrogate.gate_init)
        (tmpdir / "source.py").write_text(surrogate.source)
        joblib.dump(surrogate.preprocessor, tmpdir / "preprocessor.joblib")

        mlflow.pyfunc.log_model(
            artifact_path="surrogate_model",
            python_model=SINDySurrogateMLflowModel(),
            artifacts={
                "xi_path": str(tmpdir / "xi.npy"),
                "gate_init_path": str(tmpdir / "gate_init.npy"),
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

        # ログ用に属性の存在確認（任意）
        if hasattr(surrogate, "xi_matrix"):
            logger.info(f"Model loaded. Xi matrix shape: {surrogate.xi_matrix.shape}")

        return surrogate

    except Exception as e:
        logger.error(f"Failed to load surrogate from MLflow: {e}")
        raise
