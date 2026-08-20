"""**この研究で回したい条件**を並べた 1 枚のカタログ。

ドメイン層 (`neurosurrogate/`) は「どう回すか」の型と手続きだけを持ち、
**何を回すか**はここに集まる。中身は 2 つ:

- `_EVALS` — 素材 (1 条件 = `SimSpec`)
- `SERIES` — 掃引 (`EvalSeries`。素材を電流パラメータで振ったもの。単発も「点 1 つ
  の系列」として同じ経路を通る)。**置換対象ノード名もここに書く** (`replace_targets`)
  = どこを置換する実験かは条件の一部で、「互換なノードを全部」という暗黙の選定に
  任せない。互換かどうかの検証は surrogate 側が名前ごとに行う。載るのは
  surrogate を持たない素の系列 =
  カタログは原系の掃引そのもので、回す側が run ごとに置換器を掛けて
  run 軸を張る
**描き方 (つまみ) はここに持たない**: 比較対象 comp も指標も
図を見て
決め直すもので、カタログに置くと「回す条件」と同じ寿命に見えてしまう。置き場所は
marimo の widget 1 箇所 (`SimSpec.net` が解いた comp 名から選択肢が出るので、
適用先と噛み合わない comp を書けない)。

設定ファイルは持たない。実験条件を型で書けば綴り間違いは import 時に落ち、
スキーマという型の弱い写しを二重に管理せずに済む。条件を変えたら別の実験 =
コードの差分に出るのが正しい。
"""

import numpy as np

from neurosurrogate.neurons.traub19 import DEND_STIM_IDX, name_at
from neurosurrogate.sim.spec import EvalSeries, SimSpec

# 掃引つき評価の共通電流パラメータ (刺激前の静穏 + 本体長)。掃引軸の値は入らない
# (`EvalSeries` が点ごとに埋める)。
_STIM = {"silence_duration": 10.0, "duration": 300.0}
_DT = 0.01
# traub19 の全 comp (soma は index 8)。`replace_targets` に綴るための名前列で、
# 形態から導く (`net.names`) のとは違う — 実験がどこを置換すると宣言したかは
# 適用先の変化に黙って追随してはいけない。
_TRAUB19_ALL = (
    "c00", "c01", "c02", "c03", "c04", "c05", "c06", "c07",
    "soma",
    "c09", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18",
)  # fmt: skip

_EVALS: dict[str, SimSpec] = {
    # 単体 traub の素の応答 (置換の足場が動くかを最短で見る)。掃引なしで完結。
    "traub_soma_dc": SimSpec(
        target="traub",
        current_type="lin&steady",
        dt=_DT,
        current_params={"silence_duration": 10.0, "duration": 40.0, "value": 3.0},
    ),
    # 刺激部位だけを変えた対照ペア (soma / dend)。同じ電流軸で比べる。
    "traub19_somastim": SimSpec(
        target="traub19",
        current_type="lin&steady",
        dt=_DT,
        current_params=_STIM,
    ),
    "traub19_dendstim": SimSpec(
        target="traub19",
        stim=name_at(DEND_STIM_IDX),
        current_type="lin&steady",
        dt=_DT,
        current_params=_STIM,
    ),
    # 入力の速さに対する追従 (パルス周波数掃引)。
    "traub19_pulse_freq": SimSpec(
        target="traub19",
        current_type="periodic&pulse",
        dt=_DT,
        current_params={**_STIM, "amplitude": 20, "baseline": 0.0},
    ),
}

# **系列名の単一源**。素材を `_EVALS` から名前で引き、ここで軸と点を与える。
SERIES: dict[str, EvalSeries] = {
    "traub_soma_dc": EvalSeries(
        spec=_EVALS["traub_soma_dc"], replace_targets=("soma",)
    ),
    "traub19_somastim": EvalSeries(
        spec=_EVALS["traub19_somastim"],
        replace_targets=("soma",),
        param="value",
        values=np.linspace(0.0, 10.0, 5).tolist(),
    ),
    "traub19_dendstim": EvalSeries(
        spec=_EVALS["traub19_dendstim"],
        replace_targets=("soma",),
        param="value",
        values=np.linspace(0.0, 10.0, 5).tolist(),
    ),
    "traub19_pulse_freq": EvalSeries(
        spec=_EVALS["traub19_pulse_freq"],
        replace_targets=("soma",),
        param="frequency",
        values=np.linspace(10.0, 50.0, 5).tolist(),
    ),
    # 上の somastim と**置換範囲だけ**が違う対照: soma 1 つでなく 19 comp 全部を
    # 置換する。19 comp 教師の preset (traub_useMC19CompForTeachingData / poster_*)
    # が本来主張しているのはこちら = 教師データの範囲と置換範囲を揃えた条件。
    # 対象は net から導かず綴る — 形態が変われば「この実験が何を主張しているか」も
    # 変わるので、差分に出ないまま追随されては困る。
    "traub19_somastim_allcomp": EvalSeries(
        spec=_EVALS["traub19_somastim"],
        replace_targets=_TRAUB19_ALL,
        param="value",
        values=np.linspace(0.0, 10.0, 5).tolist(),
    ),
}


def comp_names(series_name: str | None) -> list[str]:
    """系列名 → その系列の適用先に在る comp 名 (未選択は空)。comp のつまみ
    (`eval_comp` / `view_comps`) の選択肢はこれ = 適用先と噛み合わない comp を
    選べない。名前の解決は適用先を知る `SimSpec.net` に任せる。"""
    return sorted(SERIES[series_name].spec.net.names) if series_name else []
