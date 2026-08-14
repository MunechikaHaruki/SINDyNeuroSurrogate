"""**1 レポート = 1 系列 × N モデル**の組立と保存。marimo 非依存 (marimo は結果読込
+ surrogate ロードだけ持ち、組立/保存はここへ委譲する)。

`eval` が「何を回して何が出たか」を持つのに対し、ここは **どの図をどの名前で
並べるか**。レポートの単位は「ある系列の電流たちで N 本の surrogate を比べる」の 1 問
= 系列を跨ぐ図は無い。名前は **MLflow の experiment がそのまま 2 段**で、
`models/<学習 run>/` (run 自身について描けるもの) と `report/<レポート run>/`
(この 1 レポートの産物) に割れる → `dest` は 1 つで足り、レポートを描き足しても
学習 run の図が複製されない。

**何を描くかを宣言しない**のが不変条件: モデル側は run 自身が描けるもの
(`surrogate_figs` が bundle の型から解く)、評価側は結果の形 (点が 2 つ以上なら
折れ線が出る) で決まる。**単発と掃引で経路を分けない** — 点が 1 つなら格子が 1 列に
なり点軸の折れ線が出ないだけ。

**描く対象は結果 (`SeriesView`) 自身**で、評価条件の宣言 (カタログの `SERIES`) は
受け取らない: 結果 artifact は入力仕様を自分で持つので、設定ファイルと無関係に
(別セッションで回した結果でも) 描ける。

**ドメインを横断する唯一の層**: 波形 (`neurosurrogate.waveform`) も surrogate の
自己記述 (`neurosurrogate.surrogate.figures`) も互いを知らず、両者を 1 つの報告へ
束ねるのはここだけ。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import access
from ..core.network import NeuronGraph
from ..plotting import error_fig, use_style
from ..surrogate.bundle import SurrogateBundle
from ..surrogate.diagnostics import preprocessed_latent
from ..surrogate.figures import summary_df, surrogate_figs
from ..waveform import cell_figs, current_preview_fig, dm_of, wave_report
from .grid import metric_fig, trace_grid_fig
from .results import SeriesView, run_names
from .save import SaveEntry, slug


@dataclass(frozen=True)
class Tuning:
    """**1 レポートの描画条件**。既定値と「どのキーがあるか」の単一源はここで、値を
    与える場所は `scripts/marimo.py` の widget 1 箇所だけ (カタログは「何を回すか」
    しか持たない = 描き方は図を見て決め直すものなので寿命が違う)。

    先頭の `eval_comp` だけ既定値が無い — 適用先が変われば comp 名も変わるので、
    系列を選んだ後に決まる。残りは既定のままでもレポートが出る。

    **何を描くかは宣言しない**: モデル側の図はその run が自分について描けるもの
    (`surrogate.figures.surrogate_figs` が bundle の型から解く)、評価側の図は結果の形
    (点が 2 つ以上なら折れ線が出る) で決まる。図の種類名がこの型に出てこないのが
    不変条件。
    """

    eval_comp: str  # 比較対象 comp (1 件)
    view_comps: tuple[str, ...] = ()  # 全 comp を並べる図の表示制限 (空=全部)
    metric: str = "spike_count"  # 点軸の折れ線に使う指標
    detail_point: int = 0  # 詳細図 (diff/attractor/指標) を描く点の index
    spike_orig: int = 0  # 特徴量比較に使う原系の何本目のスパイクか
    spike_surr: int = 0  # 同じく置換系
    metric_ylim: tuple[float, float] | None = None  # 折れ線の y レンジ (None=auto)


def _comp_ids(comps: tuple[str, ...], net: NeuronGraph) -> list[int] | None:
    """全 comp を並べる図に描く comp。宣言では名前、描画側は comp_id で受ける。
    空選択 = 制限なし (None)。"""
    return [net.name_to_idx(c) for c in comps] or None


def _i_ext_ylim(
    bundles: dict[str, SurrogateBundle], view: SeriesView
) -> tuple[float, float] | None:
    """train_raw.png の I_ext パネルで揃える共通 y レンジ (発表用、5% パディング)。
    レポート内の学習軌道 (train_xr) と系列の原系軌道すべてを見て決める。"""
    arrays = [access.i_ext_values(b.train_xr) for b in bundles.values()] + [
        access.i_ext_values(r.dataset) for r in view.points
    ]
    if not arrays:
        return None
    lo = min(float(a.min()) for a in arrays)
    hi = max(float(a.max()) for a in arrays)
    pad = (hi - lo) * 0.05 or 1.0
    return (lo - pad, hi + pad)


# --- model (run のロードのみ。置換シミュ不要) -----------------------------------


def model_entries(
    bundles: dict[str, SurrogateBundle],
    dirs: dict[str, str],
    net: NeuronGraph | None = None,
    comps: list[int] | None = None,
    i_ext_ylim: tuple[float, float] | None = None,
) -> list[SaveEntry]:
    """比べる N 本それぞれの自己記述図 (`models/<学習 run>/...`)。

    **run 横断のサマリ表はここに無い** — 中身が「今 何本を比べているか」で変わるので
    レポート側 (`eval_entries`) の産物 (置いてしまうと別の選択で描き替わる)。

    **レポートの外に置く** — 描く対象は学習 run そのもの (置換シミュの結果を受け
    取らない) なので、レポートの下に置くとレポートの数だけ同じ図が複製される。
    `dirs` (run_id → 保存段の名前) を受けるのは、その段が **MLflow の学習 run** に
    対応するから = 名前を解けるのは mlflow を知る側だけ (表示名 `run_names` は凡例と
    表の見出し用で、保存段には使わない)。ただし `net`/`comps`/`i_ext_ylim` は表示の
    揃えとしてレポート側から来る = 別のレポートを描き足すと見た目だけ後勝ちで
    更新される (描いた対象は変わらない)。

    **全 run 分描く** — レポートの単位が 1 系列 × N モデルなので N は比べたい本数
    そのもの (代表 1 本で済ませる必要がない)。学習データ図は `train_xr` の再生成を
    伴うが、それは N 本を比べると決めた分のコスト。
    """
    entries: list[SaveEntry] = []
    for run_id, bundle in bundles.items():
        entries += [
            SaveEntry(f"models/{slug(dirs[run_id])}/{name}", fig, sources=(run_id,))
            for name, fig in surrogate_figs(bundle, net, comps, i_ext_ylim)
        ]
    return entries


# --- eval (系列の結果: 格子 + 選択点の詳細図 + 点軸メトリクス) ------------------


def _detail_entries(
    view: SeriesView,
    bundles: dict[str, SurrogateBundle],
    names: dict[str, str],
    tuning: Tuning,
    root: str,
) -> list[SaveEntry]:
    """選択した 1 点 × 各モデルの詳細図 + メトリクス表
    (`report/<レポート run>/<model>/p<点>/...`)。
    点 index を名前に入れるので、つまみを動かしても前の点を上書きしない。

    潜在射影は run ごとの surrogate が要るので bundles から引く (結果 artifact は
    surrogate を持たない = 描画側が run_id で対応付ける)。
    """
    index = view.clamp(tuning.detail_point)
    net = view.net
    comp_id = net.name_to_idx(tuning.eval_comp)
    entries: list[SaveEntry] = []
    for run_id in view.run_ids:
        orig, surr = view.pair(index, run_id)
        figs = cell_figs(
            orig.dataset,
            surr.dataset,
            comp_id,
            # 潜在射影は原系だけで決まるが、どの surrogate で射影するかは run ごと
            # → run_id と原系をこの反復の値で束縛する (lazy 参照)。
            lambda rid=run_id, o=orig: preprocessed_latent(  # type: ignore[misc]
                bundles[rid], net, o.dataset, comp_id
            ),
            _comp_ids(tuning.view_comps, net),
        )
        metrics = wave_report(
            dm_of(orig, surr, comp_id), tuning.spike_orig, tuning.spike_surr
        )
        # run 表示名は凡例向けに改行/`/` を含む → 名前に混ぜる分だけ slug 化
        run = slug(names[run_id])
        entries += [
            SaveEntry(
                f"{root}/{run}/p{index}/{fname}",
                artifact,
                sources=view.sources,
                draw=tuning,
            )
            for fname, artifact in (*figs, *metrics)
        ]
    return entries


def eval_entries(
    view: SeriesView,
    bundles: dict[str, SurrogateBundle],
    tuning: Tuning,
    dir: str,
) -> list[SaveEntry]:
    """系列の結果: 入力電流プレビュー → 波形格子 (点 × モデル) → 選択点の詳細図 →
    点軸メトリクス折れ線。折れ線は**点が 2 つ以上のときだけ** (単発で 1 点の折れ線を
    出さない)。

    どれも 1 レポート (= 選んだ run 群 × 系列 1 つ) の産物で run 単位に割れない
    (格子も折れ線も run 横断) → **レポート run 1 つが保存段**。`dir` を受けるのは
    その名前を解けるのが mlflow を知る側だけだから。
    """
    names = run_names(bundles)
    root = f"report/{slug(dir)}"
    entries = [
        # 学習側サマリ表も**選択した N 本**の産物 (比べる本数が変われば表が変わる)
        SaveEntry(f"{root}/{name}", df, sources=tuple(bundles), draw=tuning)
        for name, df in summary_df(
            {names[run_id]: bundle for run_id, bundle in bundles.items()}
        )
    ]
    entries += [
        SaveEntry(
            f"{root}/current",
            current_preview_fig(view.points[0].spec),
            sources=view.sources,
            draw=tuning,
        ),
        SaveEntry(
            f"{root}/traces",
            trace_grid_fig(view, names, tuning.eval_comp),
            sources=view.sources,
            draw=tuning,
        ),
    ]
    if len(view.points) > 1:
        entries.append(
            SaveEntry(
                f"{root}/metric",
                metric_fig(
                    view, names, tuning.eval_comp, tuning.metric, tuning.metric_ylim
                ),
                sources=view.sources,
                draw=tuning,
            )
        )
    return entries + _detail_entries(view, bundles, names, tuning, root)


# --- 入口 ----------------------------------------------------------------------


def report_entries(
    view: SeriesView,
    bundles: dict[str, SurrogateBundle],
    tuning: Tuning,
    model_dirs: dict[str, str],
    report_dir: str,
) -> list[SaveEntry]:
    """1 レポート (1 系列 × N モデル) を組み立てる唯一の入口。**返すのは成果物の列で
    保存はしない** — どこへ書くかは呼び出し側の関心で、この層は「何がどの名前で
    並ぶか」だけを決める (`save_entries` に渡せばそのまま落ちる)。成果物ごとの由来
    (`sources`/`draw`) は各 `SaveEntry` が持つ。

    名前は **MLflow の experiment がそのまま 2 段**になる: `models/<学習 run>/` (run
    自身について描けるもの) と `report/<レポート run>/` (この 1 レポートの産物)。
    段の名前 (`model_dirs` / `report_dir`) は run の同一性そのものなので、解けるのは
    mlflow を知る側だけ → 引数で受ける。

    `eval_comp` が適用先に無い = **手元の結果に対する表示設定の誤り**なので、黙って
    何かを描かずエラー図 1 枚を返して気付けるようにする。
    """
    use_style()
    if tuning.eval_comp not in view.net.names:
        # matplotlib テキストとして描かれる (CJK グリフ非対応) → 英語で書く。
        msg = f"{view.name}: eval_comp {tuning.eval_comp!r} not in {view.target!r}"
        # レポート配下に置く — 誤りは選んだレポートに紐づくので、別レポートの
        # エラー図と潰し合わない (root に置くとどれの話か分からなくなる)。
        return [SaveEntry(f"report/{slug(report_dir)}/error", error_fig(msg))]
    # 学習側 (train_raw) は全軌道を覆うレンジ、評価側 (diff) は panels_diff の
    # 発表用既定に任せる (共有すると学習パルスの最大値で評価 step が潰れる)。
    return model_entries(
        bundles,
        model_dirs,
        view.net,
        _comp_ids(tuning.view_comps, view.net),
        _i_ext_ylim(bundles, view),
    ) + eval_entries(view, bundles, tuning, report_dir)
