"""サロゲート評価: spec の各点を原系/置換系で並走シミュし、指標アクセサを束ねる。
marimo/mlflow 非依存の純粋ドメイン層 (widget は scripts 側)。

**単発と掃引を分けない**: 結果は常に **点軸 × run 軸のグリッド** (`EvalGrid`) で、
単発は点が 1 つ・run が 1 本の退化形。かつて `EvalResult`/`SweepEval` に割れていた
ため「掃引だけ run 軸を持つ」「単発だけ置換不能を除外する」といった食い違いが生えた。
"""

import logging
from dataclasses import dataclass

import pandas as pd
import xarray as xr

from ..core import access
from ..core.coords import transform_gate
from ..core.network import NeuronGraph
from ..core.simulator import unified_simulator
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.replace import apply_surrogate, replaceable
from .spec import EvalSpec
from .wave import (
    DynamicMetrics,
    WaveReport,
    diverged,
    extract_metric,
    wave_report,
)

logger = logging.getLogger(__name__)

# --- 結果の型 (点軸 × run 軸のグリッド) --------------------------------------------


@dataclass(frozen=True)
class EvalPoint:
    """点 1 つ分の結果 = 原系 1 本 + run ごとの置換系。`value` は掃引点の値
    (掃引軸が無ければ None)。"""

    value: float | None
    original: xr.Dataset
    surrogates: dict[str, xr.Dataset]


@dataclass(frozen=True)
class EvalGrid:
    """1 spec の評価結果 = 点軸 × run 軸のグリッド。

    **spec をそのまま持つ** = 掃引軸名も適用先ネットも dt も描画が結果から引ける
    (model_name/dt を結果側へ写し取らない)。
    """

    spec: EvalSpec
    points: list[EvalPoint]

    @property
    def run_labels(self) -> list[str]:
        """run 軸の識別キー (与えた順)。**描画はこれを結果から引く** = 回したときの
        キーと図の行/列が構造的に一致する (呼び出し側で作り直さない)。"""
        return list(self.points[0].surrogates) if self.points else []

    @property
    def swept(self) -> bool:
        """点が 2 つ以上 = 点軸に沿った図 (メトリクス折れ線) が意味を持つか。"""
        return len(self.points) > 1

    def wave_report(
        self, index: int, run_label: str, comp_id: int, spike_orig: int, spike_surr: int
    ) -> WaveReport:
        """1 セル (点 × run) の波形/スパイク指標。"""
        point = self.points[index]
        dm = DynamicMetrics(
            point.original, point.surrogates[run_label], comp_id, self.spec.dt
        )
        return wave_report(dm, spike_orig=spike_orig, spike_surr=spike_surr)

    def metrics_df(self, comp_name: str, metric_key: str) -> pd.DataFrame:
        """点軸に沿った metric の DataFrame (列=run 軸)。原系の値は run に依らない
        ので `original` 列 1 本へ畳む。"""
        comp_id = self.spec.net.name_to_idx(comp_name)
        rows: list[dict] = []
        for point in self.points:
            row: dict = {"point": point.value}
            for run_label, surr_ds in point.surrogates.items():
                orig, surr = extract_metric(
                    DynamicMetrics(point.original, surr_ds, comp_id, self.spec.dt),
                    metric_key,
                )
                row[run_label] = surr
                if orig is not None:
                    row["original"] = orig  # run に依らない = 同じ値の上書き
            rows.append(row)
        return pd.DataFrame(rows)


# --- surrogate 側の診断 (結果でなく surrogate に属する自由関数) ---------------------


def preprocessed_latent(
    surrogate: SurrogateBundle, net: NeuronGraph, ds: xr.Dataset, comp_id: int
) -> xr.Dataset:
    """comp_id ノードの原系ゲートを surrogate の latent 空間へ射影した (V, latent...)
    xr (診断用)。置換対象外 (学習ドメイン外) は latent 比較不可。

    **結果でなく surrogate に属する操作**なので自由関数 (結果 artifact は surrogate を
    持たず run_id しか知らない = 呼び出し側が bundle を引いて渡す)。
    """
    comp = net.nodes[comp_id]
    if not replaceable(surrogate.meta, comp):
        # error_fig 経由で matplotlib テキストへ描かれる → CJK グリフ非対応で
        # 文字化けするため英語で書く。
        raise ValueError(
            f"comp {comp.name!r} is outside the trained domain -> latent comparison "
            f"not possible (trained type {surrogate.meta.comp_type.name!r})"
        )
    return transform_gate(surrogate.preprocessor, ds, comp_id)


def log_divergence(net: NeuronGraph, surr_ds: xr.Dataset, where: str) -> None:
    """置換系の発散を警告ログに出す。発散すると指標が nan/無意味になり図も潰れる
    → 図を読む前に原因側 (置換系) が壊れたと気付けるように。"""
    names = [
        net.nodes[int(i)].name
        for i in access.comp_ids(surr_ds)
        if diverged(access.potential(surr_ds, int(i)))
    ]
    if names:
        logger.warning("置換系の電位が発散 (%s): %s", where, ", ".join(names))


def _divergence_context(spec: EvalSpec, run_label: str, value: float | None) -> str:
    """発散ログの位置表示 (掃引点があればその値も)。"""
    at = "" if value is None or not spec.sweep else f" {spec.sweep.param}={value:.3g}"
    return f"{spec.label}{at} / {run_label}"


# --- 実行 (spec → 並走シミュ → EvalGrid) -------------------------------------------


def evaluate(surrogates: dict[str, SurrogateBundle], spec: EvalSpec) -> EvalGrid:
    """spec の各点を原系と各 run の置換系で並走シミュし `EvalGrid` を返す
    (点が 1 つなら単発・複数なら掃引で、経路は同じ)。"""
    points = []
    for value in spec.points:
        dset = spec.dataset_at(value)
        surr_datasets = {}
        for run_label, surrogate in surrogates.items():
            surr_ds = unified_simulator(apply_surrogate(surrogate, dset))
            log_divergence(
                spec.net, surr_ds, _divergence_context(spec, run_label, value)
            )
            surr_datasets[run_label] = surr_ds
        points.append(EvalPoint(value, unified_simulator(dset), surr_datasets))
    return EvalGrid(spec=spec, points=points)


def run_evals(
    surrogates: dict[str, SurrogateBundle], specs: dict[str, EvalSpec]
) -> dict[str, EvalGrid]:
    """spec ごとに評価し label → EvalGrid。

    **置換できない組み合わせは回さない**: run 軸を spec ごとに置換可能な run だけへ
    絞り (`spec.replaceable` = 描画側と同じ述語)、1 本も残らない spec は結果に入れない
    → 非互換が 1 つ混ざっても他の spec / 他の run は生き残る。
    """
    results = {}
    for label, spec in specs.items():
        # 変数名を spec.py の述語 `usable(meta, specs)` と揃えない: あちらは
        # 「1 本でも合う spec があるか」の bool、こちらは「この spec に合う run だけの
        # 部分集合」で意味が違う。
        compatible = {
            run_label: s
            for run_label, s in surrogates.items()
            if spec.replaceable(s.meta)
        }
        if compatible:
            results[label] = evaluate(compatible, spec)
    return results
