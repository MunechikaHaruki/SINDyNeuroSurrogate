"""シミュレーション結果から出る成果物の集合。

3 つの集合が、そのまま保存の 3 段に対応する: run 横断 (`report_artifacts`) /
原系だけで決まるもの (`original_artifacts`) / 1 ペアの詳細 (`detail_artifacts`)。
**何を出すかはここが持ち**、合流点 (`artifact.bundle`) はどの段へ書くかだけを決める。
個々の Artifact は `report.py` (点軸 × run 軸) と `detail.py` (1 ペア) が返す。
再 export はしない。
"""

from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from ...artifact.model import Artifacts
from ...surrogate.runs import SurrogateRuns
from ..result import SeriesResults
from ..waveform import DynamicMetrics
from .detail import (
    attractor_artifact,
    diff_artifact,
    metrics_artifact,
    metrics_scalar_artifact,
    simple_artifact,
)
from .report import (
    current_preview_artifact,
    metric_artifact,
    summary_artifact,
    traces_artifact,
)


def report_artifacts(
    view: SeriesResults,
    runs: SurrogateRuns,
    comp_name: str,
    metric_key: str,
    ylim: tuple[float, float] | None,
) -> Artifacts:
    """**run 横断で出る成果物の全部**。点軸の折れ線は掃引した時だけ = 点が 1 つなら
    折れ線にする軸が無いので出さない (この判断もここが持つ)。"""
    artifacts = [summary_artifact(runs), traces_artifact(view, runs, comp_name)]
    if len(view.original_waves) > 1:
        artifacts.append(metric_artifact(view, runs, comp_name, metric_key, ylim))
    return Artifacts(tuple(artifacts))


def original_artifacts(view: SeriesResults) -> Artifacts:
    """原系の波形だけで決まる成果物。

    描くのは**先頭点の電流** (掃引値を埋めた複製)。`series.spec` は掃引値の入って
    いないカタログ既定なので、掃引系列では実際に回した波形と食い違う。
    """
    return Artifacts((current_preview_artifact(view.series.points[0]),))


def detail_artifacts(
    original: xr.Dataset,
    preprocessed: xr.Dataset,
    surrogate: xr.Dataset,
    comp_id: int,
    dt: float,
    comps: Sequence[int] | None,
    spike_orig: int,
    spike_surr: int,
) -> Artifacts:
    """**1 ペアから出る成果物の全部**。指標 (`DynamicMetrics`) の組み立てもここが
    持つ = 呼ぶ側は波形の対と、どのスパイクを比べるかだけ渡せばよい。"""
    dm = DynamicMetrics(original, surrogate, comp_id, dt)
    return Artifacts(
        (
            diff_artifact(original, preprocessed, surrogate, comp_id),
            simple_artifact(original, comps),
            attractor_artifact(preprocessed, surrogate, comp_id),
            metrics_artifact(dm, spike_orig, spike_surr),
            metrics_scalar_artifact(dm, spike_orig, spike_surr),
        )
    )
