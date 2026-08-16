"""置換系の数値的破綻判定。eval (発散ログ) / metrics (発散セルの表示) の両方から
呼ばれる共通述語なのでどちらの層にも属させず core に置く。
"""

import logging

import numpy as np
import xarray as xr

from .network import NeuronGraph

_logger = logging.getLogger(__name__)

_DIVERGE_V = 1e3  # |V| [mV] の発散判定閾値 (生理的な範囲は ±200 程度)


def diverged(v: np.ndarray) -> bool:
    """電位系列が NaN/inf を含むか、生理的にあり得ない大きさへ振り切れたか。
    サロゲート置換系が数値的に破綻したかの共通基準 (評価ログ・波形図で共用)。"""
    return not bool(np.all(np.isfinite(v))) or float(np.abs(v).max()) > _DIVERGE_V


def log_divergence(net: NeuronGraph, surr_ds: xr.Dataset, where: str) -> None:
    """置換系の発散を警告ログに出す。発散すると指標が nan/無意味になり図も潰れる
    → 図を読む前に原因側 (置換系) が壊れたと気付けるように。"""
    from . import access

    names = [
        net.nodes[int(i)].name
        for i in access.comp_ids(surr_ds)
        if diverged(access.potential(surr_ds, int(i)))
    ]
    if names:
        _logger.warning("置換系の電位が発散 (%s): %s", where, ", ".join(names))
