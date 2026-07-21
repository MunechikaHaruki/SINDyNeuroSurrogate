"""サロゲート amp 掃引評価: current 振幅を掃引して各サロゲートを原系/置換系で
並走シミュし、comp/metric 単位で掃引メトリクスを抽出。marimo/mlflow 非依存の
純粋ドメイン層 (UI/ラベル引き出しは analysis 側)。"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from ..core.network import DatasetConfig
from ..core.simulator import unified_simulator
from ..models import MCMODELS
from ..surrogate.ansatz import NeuroSurrogateBase
from ..surrogate.replace import apply_surrogate
from .wave import DynamicMetrics, extract_metric


@dataclass(frozen=True)
class CurrentSweepConfig:
    """current の sweep_param を amp 範囲で amp_steps 分割掃引する仕様。
    base_params は sweep_param 以外 (duration 等) の固定値、single 側 UI 値を
    引き継ぐための単一源。"""

    current_type: str
    sweep_param: str
    amp_start: float
    amp_stop: float
    amp_steps: int
    base_params: dict = field(default_factory=dict)

    @property
    def amp_values(self) -> np.ndarray:
        return np.linspace(self.amp_start, self.amp_stop, self.amp_steps)


@dataclass(frozen=True)
class SweepEval:
    """amp 掃引ごとの (orig, {rid: surr}) シミュ結果と comp/metric 抽出。"""

    amp_datasets: list[tuple[float, xr.Dataset, dict[str, xr.Dataset]]]
    model_name: str
    dt: float

    def metrics_df(self, eval_comp_name: str, metric_key: str) -> pd.DataFrame:
        """eval_comp × metric_key で amp 掃引メトリクスを DataFrame 化。"""
        eval_comp_id = MCMODELS[self.model_name].name_to_idx(eval_comp_name)
        rows: list[dict] = []
        for amp, orig_ds, surr_datasets in self.amp_datasets:
            extracted = {
                rid: extract_metric(
                    DynamicMetrics(orig_ds, surr_ds, eval_comp_id, self.dt), metric_key
                )
                for rid, surr_ds in surr_datasets.items()
            }
            orig_val = next(iter(extracted.values()))[0]
            row: dict = {"amplitude": amp}
            if orig_val is not None:
                row["original"] = orig_val
            row.update({rid: surr for rid, (_, surr) in extracted.items()})
            rows.append(row)
        return pd.DataFrame(rows)


def evaluate_sweep(
    surrogates: dict[str, NeuroSurrogateBase],
    *,
    model_name: str,
    dt: float,
    cfg: CurrentSweepConfig,
) -> SweepEval:
    """amp 掃引で各サロゲートを原系/置換系並走シミュし SweepEval を返す。"""
    net = MCMODELS[model_name]
    amp_datasets: list[tuple[float, xr.Dataset, dict[str, xr.Dataset]]] = []
    for amp in cfg.amp_values:
        dset = DatasetConfig(
            model_name=model_name,
            dt=dt,
            current_type=cfg.current_type,
            current_params={**cfg.base_params, cfg.sweep_param: float(amp)},
            net=net,
        )
        surr_datasets = {
            rid: unified_simulator(apply_surrogate(surrogate, dset))
            for rid, surrogate in surrogates.items()
        }
        amp_datasets.append((float(amp), unified_simulator(dset), surr_datasets))
    return SweepEval(amp_datasets=amp_datasets, model_name=model_name, dt=dt)
