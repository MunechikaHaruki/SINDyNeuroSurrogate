"""学習 experiment (`_TARGET_EXP`) の**成果物**: surrogate の pickle + spec.json と、
学習 run 1 本の中身を冊が読める形にしたもの (`description/`)。

    description/model.json               閉包項と前処理の中身、ネット、列の名前
    description/train/                   学習データ (`_files.write_waves`、点は 1 つ)
    description/fit/<comp_id>/x.f32      同定器へ渡した状態列 (x_names の順に連続)
    description/fit/<comp_id>/latent.f32 学習ゲートの潜在 (latent_names の順に連続)
    description/fit/<comp_id>/recon_rmse.f32  ゲート → 潜在 → ゲートの RMSE

答えるのは「その run の surrogate」だけ (`load_surrogate_runs`)。**どの run が居るか・
選べるかは知らない** (選ぶのは figures/ の図)。評価 (波形) も知らない。
**選んだ run がそのまま run 軸**で、選択を広げも縮めもしない (hydra の親子は
MLflow UI 上の grouping で、比較の単位ではない)。
"""

import tempfile
from functools import cache
from pathlib import Path

import mlflow
import mlflow.artifacts
from mlflow.tracking import MlflowClient

from neurosurrogate.surrogate.description import describe
from neurosurrogate.surrogate.model import Surrogate
from neurosurrogate.surrogate.runs import SurrogateRuns

from . import logger
from ._files import write_f32, write_json, write_waves

_SURR_ARTIFACT_DIR = "surrogate"
_DESCRIPTION_DIR = "description"


def log_surrogate_model(surrogate: Surrogate) -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        surrogate.save(tmp_str)
        mlflow.log_artifacts(tmp_str, artifact_path=_SURR_ARTIFACT_DIR)


def _write_description(root: Path, surrogate: Surrogate) -> None:
    desc = describe(surrogate)
    write_json(root / "model.json", desc.model)
    net = surrogate.spec.dataset.net
    names = {comp: net.name_to_idx(comp) for comp in net.names}
    write_waves(root / "train", [desc.training_data], names)
    for comp_id, fit in desc.fits.items():
        folder = root / "fit" / str(comp_id)
        folder.mkdir(parents=True)
        write_f32(folder / "x.f32", fit.x.T)
        write_f32(folder / "latent.f32", fit.latent.T)
        write_f32(folder / "recon_rmse.f32", fit.recon_rmse)


def log_surrogate_description(surrogate: Surrogate) -> None:
    """学習 run 1 本の中身を、開いている run の `description/` へ置く。学習 run だけで
    決まるので、学習のときに一度だけ書けば足りる。"""
    with tempfile.TemporaryDirectory() as tmp_str:
        _write_description(Path(tmp_str), surrogate)
        mlflow.log_artifacts(tmp_str, artifact_path=_DESCRIPTION_DIR)


def run_surrogate_description(run_id: str) -> None:
    """`description/` を持たない学習 run (書く前に学習した run) に、後から書き足す。"""
    client = MlflowClient()
    if client.list_artifacts(run_id, _DESCRIPTION_DIR):
        return
    logger.info("学習 run %s に %s/ を書き足す", run_id, _DESCRIPTION_DIR)
    with tempfile.TemporaryDirectory() as tmp_str:
        _write_description(Path(tmp_str), _load_surrogate_model(run_id))
        client.log_artifacts(run_id, tmp_str, _DESCRIPTION_DIR)


@cache
def _load_surrogate_model(run_id: str) -> Surrogate:
    """run_id → surrogate。**run_id ごとに 1 回だけ** DL + unpickle。artifact は run に
    対し不変なので使い回してよい。"""
    logger.debug(f"Loading surrogate from run {run_id}")
    with tempfile.TemporaryDirectory() as tmp_str:
        local = Path(
            mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/{_SURR_ARTIFACT_DIR}", dst_path=tmp_str
            )
        )
        return Surrogate.load(local)


def _load_run_names(run_ids: list[str]) -> tuple[str, ...]:
    """MLflowのrun ID列を一意なrun名列へ変換する。"""
    names = tuple(mlflow.get_run(run_id).info.run_name for run_id in run_ids)
    if None in names or len(set(names)) != len(names):
        raise ValueError(f"学習run名が欠けるか重複 {names}")
    return tuple(str(name) for name in names)


def load_surrogate_runs(run_ids: list[str]) -> SurrogateRuns:
    """MLflowのrun ID列から、一意なrun名を持つsurrogate列をロードする。"""
    return SurrogateRuns(
        tuple(
            (run_name, _load_surrogate_model(run_id))
            for run_id, run_name in zip(run_ids, _load_run_names(run_ids), strict=True)
        )
    )
