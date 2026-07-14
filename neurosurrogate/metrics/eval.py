"""サロゲート評価: dataset を原系/置換系で並走シミュし、comp 単位の指標
アクセサを束ねる。marimo/mlflow 非依存の純粋ドメイン層 (UI は analysis 側)。"""

from dataclasses import dataclass

import xarray as xr

from ..core.network import DatasetConfig
from ..core.simulator import unified_simulator
from ..surrogate import NeuroSurrogateBase, transform_gate
from ..surrogate.replace import Verdict, verdict
from .wave import DynamicMetrics, WaveReport, wave_report


def preprocessed_latent(
    surrogate: NeuroSurrogateBase,
    dataset: DatasetConfig,
    sim_ds: xr.Dataset,
    comp_id: int,
) -> xr.Dataset:
    """comp_id ノードのゲートを preprocessor で latent 圧縮した (V, latent...) xr を返す
    (診断用)。学習ドメイン外 (verdict != REPLACE) は latent 比較不可でエラー化。"""
    comp = dataset.net.nodes[comp_id]
    if (v := verdict(surrogate.meta, comp)) is not Verdict.REPLACE:
        raise ValueError(
            f"comp {comp.name!r} は学習ドメイン外 ({v.name}) → latent 比較不可 "
            f"(学習型 {surrogate.meta.train_comp_type.name!r})"
        )
    return transform_gate(surrogate.preprocessor_bundle.preprocessor, sim_ds, comp_id)


@dataclass(frozen=True)
class EvalResult:
    """置換シミュ結果 (original/surr) と comp 単位の指標アクセサ。"""

    surrogate: NeuroSurrogateBase
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


def evaluate(surrogate: NeuroSurrogateBase, dataset: DatasetConfig) -> EvalResult:
    """dataset を原系とサロゲート置換系で並走シミュし EvalResult を返す。"""
    return EvalResult(
        surrogate=surrogate,
        dataset=dataset,
        original_ds=unified_simulator(dataset),
        surr_ds=unified_simulator(surrogate.apply(dataset)),
    )
