"""サロゲート評価: dataset を原系/置換系で並走シミュし、comp 単位の指標
アクセサを束ねる。marimo/mlflow 非依存の純粋ドメイン層 (UI は analysis 側)。"""

import logging
from dataclasses import dataclass

import xarray as xr

from ..core import access
from ..core.coords import transform_gate
from ..core.network import DatasetConfig
from ..core.simulator import unified_simulator
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.replace import apply_surrogate, replaceable
from .wave import DynamicMetrics, WaveReport, diverged, wave_report

logger = logging.getLogger(__name__)


def preprocessed_latent(
    surrogate: SurrogateBundle,
    dataset: DatasetConfig,
    sim_ds: xr.Dataset,
    comp_id: int,
) -> xr.Dataset:
    """comp_id ノードのゲートを preprocessor で latent 圧縮した (V, latent...) xr を返す
    (診断用)。置換対象外 (学習ドメイン外) は latent 比較不可でエラー化。"""
    comp = dataset.net.nodes[comp_id]
    if not replaceable(surrogate.meta, comp):
        raise ValueError(
            f"comp {comp.name!r} は学習ドメイン外 → latent 比較不可 "
            f"(学習型 {surrogate.meta.comp_type.name!r})"
        )
    return transform_gate(surrogate.preprocessor, sim_ds, comp_id)


@dataclass(frozen=True)
class EvalResult:
    """置換シミュ結果 (original/surr) と comp 単位の指標アクセサ。"""

    surrogate: SurrogateBundle
    dataset: DatasetConfig
    original_ds: xr.Dataset
    surr_ds: xr.Dataset

    def preprocessed_latent(self, comp_id: int) -> xr.Dataset:
        """comp_id ノードの原系ゲートを surrogate の latent 空間へ射影 (診断用)。"""
        return preprocessed_latent(
            self.surrogate, self.dataset, self.original_ds, comp_id
        )

    def wave_report(
        self, comp_id: int, spike_orig: int = 0, spike_surr: int = 0
    ) -> WaveReport:
        dm = DynamicMetrics(self.original_ds, self.surr_ds, comp_id, self.dataset.dt)
        return wave_report(dm, spike_orig=spike_orig, spike_surr=spike_surr)


def _log_divergence(dataset: DatasetConfig, surr_ds: xr.Dataset) -> None:
    """置換系の発散を警告ログに出す。発散すると指標が nan/無意味になり図も潰れる
    → 図を読む前に原因側 (置換系) が壊れたと気付けるように。"""
    names = [
        dataset.net.nodes[int(i)].name
        for i in access.comp_ids(surr_ds)
        if diverged(access.potential(surr_ds, int(i)))
    ]
    if names:
        logger.warning("置換系の電位が発散: %s", ", ".join(names))


def evaluate(surrogate: SurrogateBundle, dataset: DatasetConfig) -> EvalResult:
    """dataset を原系とサロゲート置換系で並走シミュし EvalResult を返す。"""
    surr_ds = unified_simulator(apply_surrogate(surrogate, dataset))
    _log_divergence(dataset, surr_ds)
    return EvalResult(
        surrogate=surrogate,
        dataset=dataset,
        original_ds=unified_simulator(dataset),
        surr_ds=surr_ds,
    )
