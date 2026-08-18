"""各ドメインの単一成果物を、保存側が扱う成果物列へ編成する。

**編成はここで閉じる** — どの図を作るか (`*_artifacts`) だけでなく、それを
どの段に置くか (`build_report`) まで。保存側に残るのは「段名を何と綴るか」
(MLflow の run 名を引く) と書き出しだけ。
"""

from __future__ import annotations

from pathlib import Path

from ..sim.artifacts import (
    metric_artifact,
    run_names,
    summary_artifact,
    traces_artifact,
)
from ..sim.result import SeriesResults
from ..surrogate.artifacts.model import (
    closure_artifact,
    neuron_graph_artifact,
    preprocessor_artifact,
)
from ..surrogate.artifacts.train import (
    train_manifold_artifact,
    train_preprocessed_artifact,
    train_raw_artifact,
    train_recon_artifact,
    train_v_coverage_artifact,
)
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.diagnostics import preprocessed_latent
from ..surrogate.replace import replaceable
from ..waveform.artifacts import (
    attractor_artifact,
    current_preview_artifact,
    diff_artifact,
    metrics_artifact,
    metrics_scalar_artifact,
    simple_artifact,
)
from ..waveform.dynamics import DynamicMetrics
from .model import Artifacts, Report, Tuning
from .plotting import use_style


def _check_eval_comp(view: SeriesResults, eval_comp: str) -> None:
    """評価対象の comp 名が束の net に居ることを確かめる (図の側で検出させない)。"""
    if eval_comp not in view.net.names:
        raise ValueError(f"eval_comp {eval_comp!r} not in {view.net.names!r}")


def surrogate_artifacts(
    bundle: SurrogateBundle, view_comps: tuple[str, ...] = ()
) -> Artifacts:
    """学習 run 1 本が自己記述できる成果物をまとめる。"""
    use_style()
    net = bundle.meta.dataset.net
    comps = [net.name_to_idx(comp) for comp in view_comps] or None
    return Artifacts(
        (
            *(
                artifact
                for artifact in (
                    closure_artifact(bundle.closure),
                    preprocessor_artifact(bundle.preprocessor),
                )
                if artifact is not None
            ),
            neuron_graph_artifact(
                net,
                {node.name for node in net.nodes if replaceable(bundle.meta, node)},
            ),
            train_raw_artifact(bundle, comps),
            train_preprocessed_artifact(bundle, comps),
            train_recon_artifact(bundle, comps),
            train_v_coverage_artifact(bundle, comps),
            train_manifold_artifact(bundle, comps),
        )
    )


def report_artifacts(
    view: SeriesResults,
    bundles: dict[str, SurrogateBundle],
    eval_comp: str,
    metric: str,
    metric_ylim: tuple[float, float] | None,
) -> Artifacts:
    """run 横断のサマリ・波形格子・点軸メトリクスをまとめる。"""
    use_style()
    _check_eval_comp(view, eval_comp)
    names = run_names(bundles)
    artifacts = [
        summary_artifact(bundles),
        traces_artifact(view, names, eval_comp),
    ]
    if len(view.original_waves) > 1:
        artifacts.append(metric_artifact(view, names, eval_comp, metric, metric_ylim))
    return Artifacts(tuple(artifacts))


def original_artifacts(view: SeriesResults) -> Artifacts:
    """原系の波形だけで決まる成果物をまとめる。"""
    use_style()
    # 描くのは**先頭点の電流** (掃引値を埋めた複製)。`series.spec` は掃引値の入って
    # いないカタログ既定なので、掃引系列では実際に回した波形と食い違う。
    return Artifacts((current_preview_artifact(view.series.points[0]),))


def detail_artifacts(
    view: SeriesResults,
    run_id: str,
    bundle: SurrogateBundle,
    eval_comp: str,
    view_comps: tuple[str, ...],
    detail_point: int,
    spike_orig: int,
    spike_surr: int,
) -> Artifacts:
    """選択した 1 点・1 モデルの波形成果物をまとめる。"""
    use_style()
    _check_eval_comp(view, eval_comp)
    comp_id = view.net.name_to_idx(eval_comp)
    original, surrogate = view.pair(
        min(detail_point, len(view.original_waves) - 1), view.column(run_id)
    )

    latent = preprocessed_latent(bundle, view.net, original, comp_id)
    dm = DynamicMetrics(original, surrogate, comp_id, view.dt)
    return Artifacts(
        (
            diff_artifact(original, latent, surrogate, comp_id),
            simple_artifact(
                original, [view.net.name_to_idx(comp) for comp in view_comps] or None
            ),
            attractor_artifact(latent, surrogate, comp_id),
            metrics_artifact(dm, spike_orig, spike_surr),
            metrics_scalar_artifact(dm, spike_orig, spike_surr),
        )
    )


def build_report(
    view: SeriesResults,
    bundles: dict[str, SurrogateBundle],
    tuning: Tuning,
    run_dirs: dict[str, str],
) -> Report:
    """1 レポート分の成果物を**段ごとに開く** = 描く側の唯一の入口。

    3 段: 直下が run 横断の産物、`models/<段名>/` が比べた 1 本ずつの自己記述図、
    `series/<段名>/` が波形 1 本で決まるもの (原系は `series/original/`)。`models/` と
    `series/` で同じ段名を使うので、1 本の run を 2 段から同じ綴りで辿れる。

    `run_dirs` は学習 run_id → 段名。**段名の決め方だけは持たない** — MLflow の
    run 名なので保存側 (`scripts/mlflow_io/artifacts.py`) が解いて渡す。
    """
    return Report(
        tuning,
        tuple(
            {
                Path(): report_artifacts(
                    view, bundles, tuning.eval_comp, tuning.metric, tuning.metric_ylim
                ),
                **{
                    Path("models") / run_dirs[run_id]: surrogate_artifacts(
                        bundle, tuning.view_comps
                    )
                    for run_id, bundle in bundles.items()
                },
                Path("series/original"): original_artifacts(view),
                **{
                    Path("series") / run_dirs[run_id]: detail_artifacts(
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
            }.items()
        ),
    )
