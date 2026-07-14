"""サロゲート評価: dataset を原系/置換系で並走シミュし、comp 単位の指標
アクセサを束ねる。marimo/mlflow 非依存の純粋ドメイン層 (UI は analysis 側)。"""

from dataclasses import dataclass

import xarray as xr

from ..core.network import DatasetConfig
from ..core.simulator import unified_simulator
from ..surrogate import NeuroSurrogateBase, preprocessed_latent
from .wave import DynamicMetrics, WaveReport, wave_report


@dataclass(frozen=True)
class EvalResult:
    """置換シミュ結果 (original/surr) と comp 単位の指標アクセサ。"""

    surrogate: NeuroSurrogateBase
    dataset: DatasetConfig
    original_ds: xr.Dataset
    surr_ds: xr.Dataset

    @property
    def dt(self) -> float:
        return self.dataset.dt

    def name_to_idx(self, comp_name: str) -> int:
        return self.dataset.net.name_to_idx(comp_name)

    def dynamic_metrics(self, comp_id: int) -> DynamicMetrics:
        return DynamicMetrics(self.original_ds, self.surr_ds, comp_id, self.dt)

    def preprocessed_latent(self, comp_id: int) -> xr.Dataset:
        """comp_id ノードの原系ゲートを surrogate の latent 空間へ射影 (診断用)。"""
        return preprocessed_latent(
            self.surrogate, self.dataset, self.original_ds, comp_id
        )

    def wave_report(
        self, comp_id: int, spike_orig: int = 0, spike_surr: int = 0
    ) -> WaveReport:
        return wave_report(
            self.dynamic_metrics(comp_id),
            spike_orig=spike_orig,
            spike_surr=spike_surr,
        )


def evaluate(surrogate: NeuroSurrogateBase, dataset: DatasetConfig) -> EvalResult:
    """dataset を原系とサロゲート置換系で並走シミュし EvalResult を返す。"""
    return EvalResult(
        surrogate=surrogate,
        dataset=dataset,
        original_ds=unified_simulator(dataset),
        surr_ds=unified_simulator(surrogate.apply(dataset)),
    )
