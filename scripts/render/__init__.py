"""レポート run を**描く**パッケージ = 何をどの段に並べ、どこへ書くかの唯一の置き場所。

`mlflow_io` は run (experiment / 同一性 / 参照) だけを知り、成果物の組立と段構成は
ここに閉じる — 組立に `Tuning` と surrogate の中身が要り、MLflow の関心ではない。
`__init__` が組立 (どの図を作るか)、`save` が書き出し (段名と宛先)。
marimo の描画ボタンの入口。
"""

from pathlib import Path

from mlflow_io.report import load_report
from mlflow_io.surrogate import load_bundles
from tuning import Tuning

from neurosurrogate.artifact.bundle import (
    detail_artifacts,
    original_artifacts,
    report_artifacts,
    surrogate_artifacts,
)
from neurosurrogate.artifact.model import Artifacts
from neurosurrogate.sim.result import SeriesResults
from neurosurrogate.surrogate.bundle import SurrogateBundle

from .save import per_run, save_artifacts


def _series_artifacts(
    view: SeriesResults, bundles: dict[str, SurrogateBundle], tuning: Tuning
) -> dict[Path, Artifacts]:
    """波形 1 本ずつで決まる図 (`series/<run 名>/`) = 原系の入力電流と置換系の詳細図。
    段名は学習 run の MLflow run 名 (`models/` と同じ綴りで引ける)。"""
    return {
        Path("series/original"): original_artifacts(view),
        **per_run(
            "series",
            {
                run_id: detail_artifacts(
                    view,
                    run_id,
                    bundles[run_id],
                    tuning.eval_comp,
                    tuning.view_comps,
                    tuning.detail_point,
                    tuning.spike_orig,
                    tuning.spike_surr,
                )
                for run_id in view.run_ids
            },
        ),
    }


def render_report(report_run_id: str, tuning: Tuning) -> list[str]:
    """レポート run を描画し、全成果物を同じ run へ保存する唯一の interface。呼ぶ側が
    知るのは id と描画条件だけで、参照解決も surrogate ロードも区分もここに閉じる。"""
    view = load_report(report_run_id)
    bundles = load_bundles(view.run_ids)
    directories = per_run(
        "models",
        {
            run_id: surrogate_artifacts(bundle, tuning.view_comps)
            for run_id, bundle in bundles.items()
        },
    )
    directories.update(_series_artifacts(view, bundles, tuning))
    directories[Path()] = report_artifacts(
        view,
        bundles,
        tuning.eval_comp,
        tuning.metric,
        tuning.metric_ylim,
    )
    return save_artifacts(
        directories,
        report_run_id,
        tuning,
    )
