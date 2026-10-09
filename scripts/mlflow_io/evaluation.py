"""評価: 1 系列を原系と、選んだ学習 run のそれぞれで回して波形 experiment に置く。

どの列を並べて見るかは MLflow に残さない (live-textbook の冊が波形 run から選ぶ)。
重い波形は `_series` が内容で再利用するので、同じ評価を回し直しても増えない。
"""

from catalog import SERIES

from . import logger
from ._series import run_series
from .surrogate import load_surrogate_runs


def evaluate(name: str, run_ids: tuple[str, ...]) -> None:
    """系列 `name` を原系と各学習 run で回す。系列を置換できない run は回さない。"""
    series = SERIES[name]
    runs = load_surrogate_runs(list(run_ids))
    surrs = runs.replacing(series)
    if not surrs:
        raise ValueError(f"{name}: 選択 run のどれでも置換できない (比較対象が無い)")
    run_series(name, series)
    for run_id, (run_name, bundle) in zip(run_ids, runs, strict=True):
        if run_name in surrs.names:
            run_series(name, series, run_id, bundle)
        else:
            logger.info("%s は %s を置換できないので回さない", run_name, name)
