import hashlib

import numpy as np
from loguru import logger
from prefect import task
from prefect.tasks import task_input_hash

from neurosurrogate.dataset_utils._base import calc_ThreeComp_internal
from neurosurrogate.utils import PLOTTER_REGISTRY
from neurosurrogate.utils.data_processing import _get_control_input
from neurosurrogate.utils.plots import plot_diff

from .utils import fig_to_buff


@task
def single_eval(data_type, params, preprocessed_ds, surrogate_model):
    logger.info(f"{preprocessed_ds} started to process")

    input_data = {
        "init": preprocessed_ds["vars"][0],
        "dt": 0.01,
        "iter": len(preprocessed_ds["time"].to_numpy()),
        "u": _get_control_input(preprocessed_ds, data_type=data_type),
        "data_type": data_type,
    }
    logger.info(f"input:{input_data}")

    prediction = surrogate_model.predict(**input_data)
    logger.info(f"prediction_result:{prediction}")
    if data_type == "hh3":
        calc_ThreeComp_internal(
            prediction,
            params.get("G_12"),
            params.get("G_23"),
        )

    logger.trace(prediction)
    return prediction


def log_diff_eval_key_fn(context, params):
    def array_to_quick_hash(arr: np.ndarray) -> str:
        # 1. メタデータ（形状と型）
        meta = f"{arr.shape}-{arr.dtype}"

        # 2. 最初と最後、中央の数要素だけを抽出してハッシュ化
        # ※ データの中身が「端だけ同じで中身が違う」場合には衝突するので注意
        sample = arr.flat[:: max(1, arr.size // 10)]  # 全体の10要素程度をサンプリング

        combined = f"{meta}-{sample.tobytes()}"
        return hashlib.md5(combined.encode()).hexdigest()

    surrogate_hash = array_to_quick_hash(params["surrogate_result"]["vars"].to_numpy())
    preprocessed_hash = array_to_quick_hash(
        params["preprocessed_result"]["vars"].to_numpy()
    )
    return surrogate_hash + preprocessed_hash


@task(cache_key_fn=log_diff_eval_key_fn, persist_result=True)
def log_diff_eval(surrogate_result, preprocessed_result):
    u = preprocessed_result["I_ext"].to_numpy()
    fig = plot_diff(u, preprocessed_result["vars"], surrogate_result["vars"])
    return fig_to_buff(fig)


@task(cache_key_fn=task_input_hash, persist_result=True)
def log_single_eval(data_type, surrogate_result):
    fig = PLOTTER_REGISTRY[data_type](
        surrogate_result,
        surrogate=True,
    )
    return fig_to_buff(fig)
