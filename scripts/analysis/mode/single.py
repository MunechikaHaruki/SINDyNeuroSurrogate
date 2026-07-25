from analysis.access import (
    comp_type_of,
    current_inputs,
    eval_comp_of,
    view_comps_of,
)
from analysis.save.panel import SaveEntry, entry
from analysis.targets import TARGET_MODEL

from neurosurrogate.core.network import DatasetConfig, NeuronGraph
from neurosurrogate.metrics.eval import EvalResult, evaluate
from neurosurrogate.neurons import MCMODELS
from neurosurrogate.surrogate.bundle import SurrogateBundle
from neurosurrogate.surrogate.meta import SurrogateMeta
from neurosurrogate.surrogate.replace import replaced_names
from neurosurrogate.view.engine import error_fig
from neurosurrogate.view.model import (
    closure_figs,
    preprocessor_figs,
    view_neuron_graph,
)
from neurosurrogate.view.specs import draw_all
from neurosurrogate.view.train import train_figs


def _targets(meta: SurrogateMeta) -> list[str]:
    """選択 run の comp_type の適用先 MC モデル全部 (single mode は全部を simulate)。"""
    return TARGET_MODEL[comp_type_of(meta)]


# ---------------------------------------------------------------------------
# Calc Eval
# ---------------------------------------------------------------------------


def calc_eval(
    cfg: dict,
    surrogate: SurrogateBundle,
) -> dict[str, EvalResult]:
    """選択 1 run (surrogate) を TARGET_MODEL[comp_type] の**各適用先へ置換して並走
    シミュ**、target 名 → EvalResult。comp_type は surrogate.meta から自動決定。
    置換可能な comp が無い target (非互換) は simulate せず落とす (eval_view が error
    図で明示)。シミュ入力 (dt/current) は cfg (base.json⊕meta.json) から。"""
    return {
        tgt: evaluate(
            surrogate,
            DatasetConfig.build_dataset(model_name=tgt, **current_inputs(cfg)),
        )
        for tgt in _targets(surrogate.meta)
        if replaced_names(surrogate.meta, MCMODELS[tgt])
    }


# ---------------------------------------------------------------------------
# View Result (save entry 列。表示は panel.render が担う)
# ---------------------------------------------------------------------------


def _view_comps(net: NeuronGraph, draw: dict) -> list[int] | None:
    """表示 comp は UI では名前、view 層は comp_id で受ける (access と同じ規約)。"""
    view_comps = view_comps_of(draw)
    return None if view_comps is None else [net.name_to_idx(c) for c in view_comps]


def model_view(
    surrogate: SurrogateBundle | None,
    draw: dict,
) -> list[SaveEntry]:
    """静的モデル図 (選択 run のロードのみ。置換シミュ不要)。closure/preprocessor/
    train は target 非依存で 1 回、neurograph は適用先ごと (置換ノードが違う)。"""
    if surrogate is None:
        return []
    entries = [
        entry(name, fig)
        for name, fig in [
            *closure_figs(surrogate.closure),
            *preprocessor_figs(surrogate.preprocessor),
        ]
    ]
    targets = _targets(surrogate.meta)
    for tgt in targets:
        net = MCMODELS[tgt]
        entries.append(
            entry(
                f"{tgt}/neurograph",
                view_neuron_graph(net, replaced_names(surrogate.meta, net)),
            )
        )
    # train データ図は target 非依存 (学習データは meta から再生成)。comp 制限は
    # 代表 target で名前解決 (学習 comp 名は target を跨いで共通)。
    comps = _view_comps(MCMODELS[targets[0]], draw)
    entries += [entry(name, fig) for name, fig in train_figs(surrogate, comps)]
    return entries


def _eval_view_one(tgt: str, res: EvalResult, draw: dict) -> list[SaveEntry]:
    """1 target 分の波形図 + メトリクス df (名前は target で接頭)。eval_comp が
    その target に無ければ error 図に畳む。"""
    net = res.dataset.net
    eval_comp = eval_comp_of(draw)
    if eval_comp not in net.names:
        msg = f"{tgt}: eval_comp '{eval_comp}' が無い"
        return [entry(f"{tgt}/error", error_fig(msg))]
    target_comp_id = net.name_to_idx(eval_comp)
    rep = res.wave_report(
        target_comp_id,
        int(draw["spike_orig"]),
        int(draw["spike_surr"]),
    )
    comps = _view_comps(net, draw)
    return [
        *[
            entry(f"{tgt}/{name}", fig)
            for name, fig in draw_all(res, target_comp_id, comps)
        ],
        entry(f"{tgt}/metrics", rep.df_metrics),
        entry(f"{tgt}/metrics_scalar", rep.df_scalar),
    ]


def eval_view(
    surrogate: SurrogateBundle | None,
    res: dict[str, EvalResult] | None,
    draw: dict,
) -> list[SaveEntry]:
    """全適用先の波形図 + メトリクス (置換シミュ結果 res が要る)。res に無い target
    は非互換 (置換可能 comp 0) → error 図で明示。"""
    if surrogate is None or res is None:
        return []
    entries: list[SaveEntry] = []
    for tgt in _targets(surrogate.meta):
        r = res.get(tgt)
        if r is None:
            entries.append(
                entry(f"{tgt}/error", error_fig(f"{tgt}: 置換可能な comp が無い"))
            )
            continue
        entries += _eval_view_one(tgt, r, draw)
    return entries
