"""サロゲート評価: spec を原系/置換系で並走シミュする。marimo/mlflow 非依存の純粋
ドメイン層 (widget は scripts 側)。

**結果は 1 シミュ = 1 key = 1 Dataset のフラットな dict**: `SimKey = (label, run_id)`。
掃引点も run もどちらも `SimSpec` 自身のフィールドなので、束ねる型 (旧 `EvalGrid`) を
持たない — 軸を組み替える処理は呼び出し側 (`metrics.select`) が dict を舐めるだけ。
"""

import logging
from dataclasses import replace as dc_replace

import xarray as xr

from ..core import access
from ..core.coords import transform_gate
from ..core.diverge import diverged
from ..core.network import NeuronGraph
from ..core.simulator import unified_simulator
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.replace import apply_surrogate, replaceable
from .spec import SimSpec

logger = logging.getLogger(__name__)

SimKey = tuple[str, str | None]  # (label, run_id)。run_id=None は原系


# --- surrogate 側の診断 (結果でなく surrogate に属する自由関数) ---------------------


def preprocessed_latent(
    surrogate: SurrogateBundle, net: NeuronGraph, ds: xr.Dataset, comp_id: int
) -> xr.Dataset:
    """comp_id ノードの原系ゲートを surrogate の latent 空間へ射影した (V, latent...)
    xr (診断用)。置換対象外 (学習ドメイン外) は latent 比較不可。
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


# --- 実行 (spec → シミュ) -----------------------------------------------------------


def expand(
    specs: dict[str, SimSpec], surrogates: dict[str, SurrogateBundle]
) -> dict[SimKey, SimSpec]:
    """label × (原系 + 置換可能な run) の直積。原系は `run_id=None` として同じ経路
    (特別扱いしない)。1 本も置換できない label は落とす。返す spec は `run_id`
    フィールドを key と一致させて埋める (呼び出し側が key から surrogate を引ける)。
    """
    out: dict[SimKey, SimSpec] = {}
    for label, spec in specs.items():
        compatible = {
            run_id: s for run_id, s in surrogates.items() if spec.replaceable(s.meta)
        }
        if not compatible:
            continue
        out[(label, None)] = spec
        for run_id in compatible:
            out[(label, run_id)] = dc_replace(spec, run_id=run_id)
    return out


def simulate(spec: SimSpec, surrogate: SurrogateBundle | None) -> xr.Dataset:
    """1 シミュ。`surrogate=None` なら原系、あれば `apply_surrogate` してから回す。"""
    dset = spec.dataset()
    if surrogate is None:
        return unified_simulator(dset)
    surr_ds = unified_simulator(apply_surrogate(surrogate, dset))
    log_divergence(spec.net, surr_ds, f"{spec.name} / {surrogate.meta.label}")
    return surr_ds


def run_sims(
    specs: dict[str, SimSpec], surrogates: dict[str, SurrogateBundle]
) -> dict[SimKey, xr.Dataset]:
    """`expand` した各 key を並走シミュし `SimKey → Dataset` を返す。"""
    expanded = expand(specs, surrogates)
    return {
        key: simulate(spec, surrogates[key[1]] if key[1] is not None else None)
        for key, spec in expanded.items()
    }
