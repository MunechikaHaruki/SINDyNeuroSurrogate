"""サロゲート amp 掃引評価: current 振幅を掃引して各サロゲートを原系/置換系で
並走シミュし、comp/metric 単位で掃引メトリクスを抽出。marimo/mlflow 非依存の
純粋ドメイン層 (UI/ラベル引き出しは analysis 側)。"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from ..core.network import DatasetConfig
from ..core.simulator import unified_simulator
from ..neurons import MCMODELS
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.replace import apply_surrogate
from .wave import DynamicMetrics, extract_metric


def dedupe_labels(names: list[str]) -> list[str]:
    """衝突した名前にだけ順序の連番を付ける (与えた順)。結果 dict のキーが silent に
    潰れて表と図が食い違うのを防ぐ共通規約 (選択を拒否せず全部見せる)。"""
    counts = Counter(names)
    seen: Counter[str] = Counter()
    labels = []
    for name in names:
        seen[name] += 1
        labels.append(name if counts[name] == 1 else f"{name}#{seen[name]}")
    return labels


def sweep_labels(surrogates: list[SurrogateBundle]) -> list[str]:
    """掃引結果の run 軸の識別キー列 (与えた順)。

    `meta.label` は学習構造 + 学習データまでしか区別しない → library_specs 違いや
    同 config の再実行は同じ label になるため連番で潰れを防ぐ。
    """
    return dedupe_labels([s.meta.label for s in surrogates])


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
    """amp 掃引ごとの (orig, {rid: surr}) シミュ結果と comp/metric 抽出。掃引仕様
    (cfg) も持つ = 描画が軸名を結果から引ける (別引数で持ち回らない)。"""

    amp_datasets: list[tuple[float, xr.Dataset, dict[str, xr.Dataset]]]
    model_name: str
    dt: float
    cfg: CurrentSweepConfig

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
    return SweepEval(amp_datasets=amp_datasets, model_name=model_name, dt=dt, cfg=cfg)
