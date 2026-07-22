"""サロゲート amp 掃引評価: current 振幅を掃引して各サロゲートを原系/置換系で
並走シミュし、comp/metric 単位で掃引メトリクスを抽出。marimo/mlflow 非依存の
純粋ドメイン層 (UI/ラベル引き出しは analysis 側)。"""

import inspect
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from ..core.network import DatasetConfig
from ..core.simulator import unified_simulator
from ..currents import CURRENT_MAP
from ..models import MCMODELS
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.replace import apply_surrogate
from .wave import DynamicMetrics, extract_metric


def sweepable_params(current_type: str) -> list[str]:
    """掃引対象候補となる numeric パラメータ名列。silence_duration/duration は
    掃引意図の対象でないため除外。0 件 = 掃引できない電流。"""
    return [
        name
        for name, p in inspect.signature(CURRENT_MAP[current_type]).parameters.items()
        if p.annotation in (int, float) and name not in ("silence_duration", "duration")
    ]


def sweep_labels(surrogates: list[SurrogateBundle]) -> list[str]:
    """掃引結果の識別キー列 (与えた順)。

    `meta.label` は学習構造 + 学習データまでしか区別しない → library_specs 違いや
    同 config の再実行は同じ label になる。掃引結果は label キーの dict なので、
    そのままだと silent に 1 run へ潰れ summary 表と掃引図が食い違う。衝突した
    label にだけ順序の連番を付けて潰れを防ぐ (選択を拒否せず全部見せる)。
    """
    counts = Counter(s.meta.label for s in surrogates)
    seen: Counter[str] = Counter()
    labels = []
    for s in surrogates:
        seen[s.meta.label] += 1
        n = seen[s.meta.label]
        labels.append(
            s.meta.label if counts[s.meta.label] == 1 else f"{s.meta.label}#{n}"
        )
    return labels


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
    surrogates: dict[str, SurrogateBundle],
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
