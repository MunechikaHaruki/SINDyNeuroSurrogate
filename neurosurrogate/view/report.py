"""評価結果 → 保存/表示できる `SaveEntry` 列の組立。marimo 非依存。

`metrics.spec` が「何をシミュしたか」(label → 結果)、`view/*` が「1 枚の図」を返す
のに対し、ここは **どの図をどの名前で並べるか** を決める層: model (置換シミュ不要の
静的図) / sim (spec ごとの波形+指標) / sweep (spec ごとの掃引図) の 3 グループを組み、
UI はそれを表示と保存へ流すだけ (表示と保存が同じ列 = 食い違わない)。

図の選択に要る表示設定は cfg の `draw` セクションから読む (widget 由来の値だが、
cfg の 1 セクションとして渡ってくるので UI 型には触れない)。
"""

import pandas as pd

from ..core.network import NeuronGraph
from ..metrics.eval import EvalResult
from ..metrics.eval_sweep import SweepEval, sweep_labels
from ..metrics.spec import cfg_targets, parse_sims
from ..neurons import MCMODELS
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.replace import replaced_names
from .engine import error_fig
from .model import closure_figs, preprocessor_figs, view_neuron_graph
from .preview import sweep_fig, sweep_trace_grid_fig
from .save import SaveEntry, entry
from .specs import draw_all
from .train import train_figs

# --- 表示設定 (cfg["draw"]) の読取 --------------------------------------------


def _draw(cfg: dict) -> dict:
    return dict(cfg.get("draw", {}))


def _eval_comp(cfg: dict) -> str:
    """評価対象 comp 名 (diff/指標の比較対象、1 件)。"""
    return str(cfg["draw"]["eval_comp"])


def _view_comps(net: NeuronGraph, cfg: dict) -> list[int] | None:
    """全 comp を並べる図に描く comp。UI では名前、view 層は comp_id で受ける。
    空選択 = 制限なし (None)。"""
    names = [str(c) for c in cfg["draw"]["view_comps"]]
    return [net.name_to_idx(c) for c in names] if names else None


# --- model (選択 run のロードのみ。置換シミュ不要) ------------------------------


def model_report(cfg: dict, bundle: SurrogateBundle | None) -> list[SaveEntry]:
    """静的モデル図。closure/preprocessor/train は適用先非依存で 1 回、neurograph は
    cfg が宣言する適用先ごと (置換ノードが違う。同じ target を複数電流で回しても
    図は 1 枚)。"""
    if bundle is None:
        return []
    entries = [
        entry(name, fig)
        for name, fig in [
            *closure_figs(bundle.closure),
            *preprocessor_figs(bundle.preprocessor),
        ]
    ]
    targets = cfg_targets(cfg)
    for tgt in targets:
        net = MCMODELS[tgt]
        entries.append(
            entry(
                f"{tgt}/neurograph",
                view_neuron_graph(net, replaced_names(bundle.meta, net)),
            )
        )
    # train データ図は適用先非依存 (学習データは meta から再生成)。comp 制限は
    # 代表 target で名前解決 (学習 comp 名は target を跨いで共通)。
    comps = _view_comps(MCMODELS[targets[0]], cfg) if targets else None
    entries += [entry(name, fig) for name, fig in train_figs(bundle, comps)]
    return entries


# --- sim (spec ごとの波形 + 指標) ----------------------------------------------


def _sim_report_one(label: str, res: EvalResult, cfg: dict) -> list[SaveEntry]:
    """1 sim spec 分の波形図 + メトリクス df (名前は spec ラベルで接頭)。eval_comp が
    その適用先に無ければ error 図に畳む。"""
    net = res.dataset.net
    eval_comp = _eval_comp(cfg)
    if eval_comp not in net.names:
        msg = f"{label}: eval_comp '{eval_comp}' が無い"
        return [entry(f"{label}/error", error_fig(msg))]
    comp_id = net.name_to_idx(eval_comp)
    draw = _draw(cfg)
    rep = res.wave_report(comp_id, int(draw["spike_orig"]), int(draw["spike_surr"]))
    return [
        *[
            entry(f"{label}/{name}", fig)
            for name, fig in draw_all(res, comp_id, _view_comps(net, cfg))
        ],
        entry(f"{label}/metrics", rep.df_metrics),
        entry(f"{label}/metrics_scalar", rep.df_scalar),
    ]


def sim_report(
    cfg: dict,
    bundle: SurrogateBundle | None,
    res: dict[str, EvalResult] | None,
) -> list[SaveEntry]:
    """全 sim spec の波形図 + メトリクス (置換シミュ結果 res が要る)。res に無い
    ラベルは非互換 (置換可能 comp 0 で `run_sims` が落とした) → error 図で明示。"""
    if bundle is None or res is None:
        return []
    entries: list[SaveEntry] = []
    for label in parse_sims(cfg):
        r = res.get(label)
        if r is None:
            entries.append(
                entry(f"{label}/error", error_fig(f"{label}: 置換可能な comp が無い"))
            )
            continue
        entries += _sim_report_one(label, r, cfg)
    return entries


# --- sweep (spec ごとの掃引図。run 軸 × amp 軸) ---------------------------------


def _eval_df(bundles: list[SurrogateBundle]) -> pd.DataFrame:
    """選択 run の学習側指標サマリ (掃引結果に依らないので res 無しでも出せる)。"""
    rows = [
        {"label": lbl, **s.metrics()}
        for lbl, s in zip(sweep_labels(bundles), bundles, strict=True)
    ]
    return pd.DataFrame(rows).set_index("label")


def _sweep_report_one(
    label: str,
    sweep: SweepEval,
    labels: list[str],
    cfg: dict,
) -> list[SaveEntry]:
    """1 sweep spec 分の波形格子 + メトリクス図 (名前は spec ラベルで接頭)。"""
    eval_comp = _eval_comp(cfg)
    if eval_comp not in MCMODELS[sweep.model_name].names:
        msg = f"{label}: eval_comp '{eval_comp}' が {sweep.model_name} に無い"
        return [entry(f"{label}/error", error_fig(msg))]
    draw = _draw(cfg)
    ylim = (
        None
        if draw["sweep_yauto"]
        else (float(draw["sweep_ymin"]), float(draw["sweep_ymax"]))
    )
    return [
        entry(f"{label}/sweep_traces", sweep_trace_grid_fig(sweep, eval_comp, labels)),
        entry(
            f"{label}/sweep",
            sweep_fig(sweep, eval_comp, draw["sweep_metric"], labels, ylim=ylim),
        ),
    ]


def sweep_report(
    cfg: dict,
    bundles: list[SurrogateBundle],
    res: dict[str, SweepEval] | None,
) -> list[SaveEntry]:
    """評価サマリ表 (選択 run) → sweep spec ごとの波形格子 + メトリクス図
    (res ゲート)。"""
    if not bundles:
        return []
    entries = [entry("eval_summary", _eval_df(bundles))]
    if res is None:
        return entries
    labels = sweep_labels(bundles)  # run_sweeps が run 軸のキーに使ったものと同一規則
    for label, sweep in res.items():
        entries += _sweep_report_one(label, sweep, labels, cfg)
    return entries


def result_groups(
    cfg: dict,
    bundle: SurrogateBundle | None,
    sweep_bundles: list[SurrogateBundle],
    res_sims: dict[str, EvalResult] | None,
    res_sweeps: dict[str, SweepEval] | None,
) -> dict[str, list[SaveEntry]]:
    """model / sim / sweep の 3 グループ (表示はタブ分け、保存は `flatten` で一括)。"""
    return {
        "model": model_report(cfg, bundle),
        "sim": sim_report(cfg, bundle, res_sims),
        "sweep": sweep_report(cfg, sweep_bundles, res_sweeps),
    }
