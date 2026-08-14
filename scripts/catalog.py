"""**この研究で回したい条件と、その描き方**を並べた 1 枚のカタログ。

ドメイン層 (`neurosurrogate/`) は「どう回すか」の型と手続きだけを持ち、
**何を回すか**はここに集まる。中身は 3 つ:

- `EVALS` — 素材 (1 条件 = `SimSpec`)
- `SERIES` — 掃引 (`EvalSeries`。素材を電流パラメータで振ったもの。単発も「点 1 つ
  の系列」として同じ経路を通る)。載るのは surrogate を持たない素の系列 =
  カタログは原系の掃引そのもので、回す側が run ごとに `with_surrogate` して
  run 軸を張る
- `REPORT` — 描画宣言 (系列名 → `Report`)。**1 系列 = 1 レポート**なので `SERIES` と
  同じキー空間。同じファイルに置いて名前の対応が目で追えるようにしてある

設定ファイルは持たない。実験条件も描画宣言も型で書けば綴り間違いは import 時に
落ち、スキーマという型の弱い写しを二重に管理せずに済む。条件を変えたら別の実験 =
コードの差分に出るのが正しい。
"""

import numpy as np

from neurosurrogate.report.spec import Report
from neurosurrogate.sim.eval import EvalSeries
from neurosurrogate.sim.spec import SimSpec

# 掃引つき評価の共通電流パラメータ (刺激前の静穏 + 本体長)。掃引軸の値は入らない
# (`EvalSeries` が点ごとに埋める)。
_STIM = {"silence_duration": 10.0, "duration": 300.0}
_DT = 0.01

EVALS: dict[str, SimSpec] = {
    # 単体 traub の素の応答 (置換の足場が動くかを最短で見る)。掃引なしで完結。
    "traub_soma_dc": SimSpec(
        target="traub",
        current_type="lin&steady",
        dt=_DT,
        current_params={"silence_duration": 10.0, "duration": 40.0, "value": 3.0},
    ),
    # 刺激部位だけを変えた対照ペア (soma / dend)。同じ電流軸で比べる。
    "traub19_somastim": SimSpec(
        target="traub19_soma",
        current_type="lin&steady",
        dt=_DT,
        current_params=_STIM,
    ),
    "traub19_dendstim": SimSpec(
        target="traub19_soma_dendstim",
        current_type="lin&steady",
        dt=_DT,
        current_params=_STIM,
    ),
    # 入力の速さに対する追従 (パルス周波数掃引)。
    "traub19_pulse_freq": SimSpec(
        target="traub19_soma",
        current_type="periodic&pulse",
        dt=_DT,
        current_params={**_STIM, "amplitude": 20, "baseline": 0.0},
    ),
}

# **系列名の単一源**。素材を `EVALS` から名前で引き、ここで軸と点を与える。
SERIES: dict[str, EvalSeries] = {
    "traub_soma_dc": EvalSeries(spec=EVALS["traub_soma_dc"]),
    "traub19_somastim": EvalSeries(
        spec=EVALS["traub19_somastim"],
        param="value",
        values=np.linspace(0.0, 10.0, 5).tolist(),
    ),
    "traub19_dendstim": EvalSeries(
        spec=EVALS["traub19_dendstim"],
        param="value",
        values=np.linspace(0.0, 10.0, 5).tolist(),
    ),
    "traub19_pulse_freq": EvalSeries(
        spec=EVALS["traub19_pulse_freq"],
        param="frequency",
        values=np.linspace(10.0, 50.0, 5).tolist(),
    ),
}

# 描画宣言。**1 系列 = 1 レポート** (その系列の電流たちで N 本の surrogate を比べる)
# なので `SERIES` と同じキー空間を張る。**図を調整するたびに書き換える対象**だが
# 設定ファイルにはしない: 型で書けば既定値・使えるキー・系列名が 1 箇所で解決する。
#
# **何の図を出すかは書かない** — モデル側は run 自身が描けるもの、評価側は結果の形
# (点数) で決まる。ここに書くのは「どの comp を比較の主役に据えるか」だけ。
# 描きながら回すつまみ (詳細図の点・スパイク番号・y レンジ) は `report.Tuning` で、
# marimo の widget が持つ (カタログには残らない)。
REPORT: dict[str, Report] = {
    "traub_soma_dc": Report(eval_comp="soma", view_comps=("soma",)),
    "traub19_somastim": Report(eval_comp="soma", view_comps=("soma",)),
    "traub19_dendstim": Report(eval_comp="soma", view_comps=("soma",)),
    "traub19_pulse_freq": Report(eval_comp="soma", view_comps=("soma",)),
}
