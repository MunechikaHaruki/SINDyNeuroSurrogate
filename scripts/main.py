import logging
import os
import tempfile
from pathlib import Path

import hydra
import mlflow
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow_io import log_surrogate_model, setup_mlflow
from omegaconf import DictConfig, OmegaConf

from neurosurrogate.surrogate.bundle import SurrogateBundle

logger = logging.getLogger(__name__)


def _disable_proxy() -> None:
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def _run_name(preset: str) -> str:
    extra = [
        o.rsplit(".", 1)[-1]
        for o in HydraConfig.get().overrides.task
        if not o.startswith("surrogate=")
    ]
    return " ".join([preset, *extra])


def _fit_and_log(cfg: DictConfig) -> None:
    """fit → 成果物を開いている run へ log (親/子で共有)。指標は marimo が
    surrogate から直接計算するので MLflow へは残さない (依存最小)。"""
    cfg_surr = OmegaConf.to_container(cfg, resolve=True)["surrogate"]
    assert isinstance(cfg_surr, dict)
    log_surrogate_model(SurrogateBundle.setup(cfg_surr))


def _log_config(cfg: DictConfig) -> None:
    """学習に使った合成 config を代表 run の artifact に yaml で残す (再現用)。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(OmegaConf.to_yaml(cfg))
        mlflow.log_artifact(str(path))


def _ensure_sweep_parent(preset: str) -> str | None:
    """--multirun 時のみ: sweep の親 run を返す。無ければ yaml本体 (sweeper 抜き config)
    を学習して親=代表を作る。子は parentRunId で紐付き run_selector は代表だけ出す。"""
    if HydraConfig.get().mode != RunMode.MULTIRUN:
        return None
    sweep_id = str(HydraConfig.get().sweep.dir)
    hits = mlflow.search_runs(
        filter_string=f"tags.sweep_id = '{sweep_id}'", output_format="list"
    )
    if hits:
        return hits[0].info.run_id
    parent_cfg = compose(config_name="config", overrides=[f"surrogate={preset}"])
    with mlflow.start_run(run_name=f"[parent]{preset}") as parent:
        mlflow.set_tag("sweep_id", sweep_id)
        mlflow.log_param("preset", preset)
        _log_config(parent_cfg)
        _fit_and_log(parent_cfg)
        return parent.info.run_id


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    _disable_proxy()
    setup_mlflow()
    preset = str(HydraConfig.get().runtime.choices["surrogate"])
    parent_id = _ensure_sweep_parent(preset)
    tags = {MLFLOW_PARENT_RUN_ID: parent_id} if parent_id else None
    with mlflow.start_run(run_name=_run_name(preset), tags=tags):
        # 出自 preset は学習に影響しない → pickle 外の MLflow param へ。
        mlflow.log_param("preset", preset)
        # 単発 (parent_id なし) はこの run 自身が代表 → config yaml を残す。
        # multirun の子は残さない (代表 = 親のみ)。
        if parent_id is None:
            _log_config(cfg)
        _fit_and_log(cfg)
    logger.info(f"[{_run_name(preset)}] 完了")


if __name__ == "__main__":
    main()
