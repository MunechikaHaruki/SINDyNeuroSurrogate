"""サロゲート評価: dataset を原系/置換系で並走シミュし、comp 単位の指標
アクセサを束ねる。marimo/mlflow 非依存の純粋ドメイン層 (widget は scripts 側)。

単発 (`SimSpec` → `EvalResult`) と掃引 (`SweepSpec` → `SweepEval`) を並べて置く:
発散判定 (`log_divergence`) も置換系の作り方も同じで、違うのは軸を 1 本増やすかだけ。
"""

import logging
from dataclasses import dataclass

import pandas as pd
import xarray as xr

from ..core import access
from ..core.coords import transform_gate
from ..core.network import DatasetConfig, NeuronGraph
from ..core.simulator import unified_simulator
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.replace import apply_surrogate, replaceable
from .spec import SimSpec, SweepSpec, sweep_labels
from .wave import (
    DynamicMetrics,
    WaveReport,
    diverged,
    extract_metric,
    wave_report,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalResult:
    """置換シミュ結果 (original/surr) と comp 単位の指標アクセサ。"""

    surrogate: SurrogateBundle
    dataset: DatasetConfig
    original_ds: xr.Dataset
    surr_ds: xr.Dataset

    def preprocessed_latent(self, comp_id: int) -> xr.Dataset:
        """comp_id ノードの原系ゲートを surrogate の latent 空間へ射影した
        (V, latent...) xr (診断用)。置換対象外 (学習ドメイン外) は latent 比較不可。"""
        comp = self.dataset.net.nodes[comp_id]
        if not replaceable(self.surrogate.meta, comp):
            raise ValueError(
                f"comp {comp.name!r} は学習ドメイン外 → latent 比較不可 "
                f"(学習型 {self.surrogate.meta.comp_type.name!r})"
            )
        return transform_gate(self.surrogate.preprocessor, self.original_ds, comp_id)

    def wave_report(
        self, comp_id: int, spike_orig: int = 0, spike_surr: int = 0
    ) -> WaveReport:
        dm = DynamicMetrics(self.original_ds, self.surr_ds, comp_id, self.dataset.dt)
        return wave_report(dm, spike_orig=spike_orig, spike_surr=spike_surr)


def log_divergence(net: NeuronGraph, surr_ds: xr.Dataset, where: str) -> None:
    """置換系の発散を警告ログに出す。発散すると指標が nan/無意味になり図も潰れる
    → 図を読む前に原因側 (置換系) が壊れたと気付けるように。単発も掃引も同じ基準で
    通す (片方だけ黙る、が起きない)。"""
    names = [
        net.nodes[int(i)].name
        for i in access.comp_ids(surr_ds)
        if diverged(access.potential(surr_ds, int(i)))
    ]
    if names:
        logger.warning("置換系の電位が発散 (%s): %s", where, ", ".join(names))


def evaluate(surrogate: SurrogateBundle, dataset: DatasetConfig) -> EvalResult:
    """dataset を原系とサロゲート置換系で並走シミュし EvalResult を返す。"""
    surr_ds = unified_simulator(apply_surrogate(surrogate, dataset))
    log_divergence(dataset.net, surr_ds, dataset.model_name)
    return EvalResult(
        surrogate=surrogate,
        dataset=dataset,
        original_ds=unified_simulator(dataset),
        surr_ds=surr_ds,
    )


def run_sims(
    bundle: SurrogateBundle,
    specs: dict[str, SimSpec],
) -> dict[str, EvalResult]:
    """spec ごとに原系/置換系を並走シミュし label → EvalResult。置換できない spec は
    simulate しない (実行するかの判断は `spec.replaceable` = 描画側と同じ述語)。"""
    return {
        label: evaluate(bundle, s.dataset())
        for label, s in specs.items()
        if s.replaceable(bundle.meta)
    }


# --- 掃引 (run 軸 × 掃引軸) ---------------------------------------------------


@dataclass(frozen=True)
class SweepEval:
    """amp 掃引ごとの (orig, {rid: surr}) シミュ結果。**掃引仕様 (spec) をそのまま
    持つ** = 掃引軸名も適用先ネットも dt も描画が結果から引ける (別引数で持ち回らず、
    model_name/dt を結果側へ写し取らない)。"""

    spec: SweepSpec
    amp_datasets: list[tuple[float, xr.Dataset, dict[str, xr.Dataset]]]

    @property
    def run_labels(self) -> list[str]:
        """run 軸の識別キー (与えた順)。**描画はこれを結果から引く** = 掃引を回した
        ときのキーと図の行/列が構造的に一致する (呼び出し側で作り直さない)。"""
        return list(self.amp_datasets[0][2]) if self.amp_datasets else []

    def metrics_df(self, eval_comp_name: str, metric_key: str) -> pd.DataFrame:
        """eval_comp × metric_key で amp 掃引メトリクスを DataFrame 化 (列=run 軸)。
        原系の値は run に依らないので `original` 列 1 本へ畳む。"""
        eval_comp_id = self.spec.net.name_to_idx(eval_comp_name)
        rows: list[dict] = []
        for amp, orig_ds, surr_datasets in self.amp_datasets:
            row: dict = {"amplitude": amp}
            for rid, surr_ds in surr_datasets.items():
                orig, surr = extract_metric(
                    DynamicMetrics(orig_ds, surr_ds, eval_comp_id, self.spec.dt),
                    metric_key,
                )
                row[rid] = surr
                if orig is not None:
                    row["original"] = orig  # run に依らない = 同じ値の上書き
            rows.append(row)
        return pd.DataFrame(rows)


def evaluate_sweep(
    surrogates: dict[str, SurrogateBundle], spec: SweepSpec
) -> SweepEval:
    """掃引軸の各点で各サロゲートを原系/置換系並走シミュし SweepEval を返す。"""
    amp_datasets: list[tuple[float, xr.Dataset, dict[str, xr.Dataset]]] = []
    for amp in spec.amp_values:
        dset = spec.dataset_at(amp)
        surr_datasets = {}
        for rid, surrogate in surrogates.items():
            surr_datasets[rid] = unified_simulator(apply_surrogate(surrogate, dset))
            log_divergence(
                spec.net,
                surr_datasets[rid],
                f"{spec.label} {spec.sweep_param}={amp:.3g} / {rid}",
            )
        amp_datasets.append((float(amp), unified_simulator(dset), surr_datasets))
    return SweepEval(spec=spec, amp_datasets=amp_datasets)


def run_sweeps(
    bundles: list[SurrogateBundle],
    specs: dict[str, SweepSpec],
) -> dict[str, SweepEval]:
    """spec ごとに掃引評価し label → SweepEval。結果は **entry 軸 (この dict) ×
    run 軸 (`SweepEval.run_labels`)** の 2 段。"""
    surrogates = dict(zip(sweep_labels(bundles), bundles, strict=True))
    return {label: evaluate_sweep(surrogates, s) for label, s in specs.items()}
