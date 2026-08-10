"""**何をどう回すか**を扱う層: 1 シミュの仕様 (`SimSpec`) と実行 (`simulate`)、
それを電流パラメータで振った掃引実験 (`EvalSeries`)、そして回したい条件の倉庫
(`EVALS` / `SERIES`)。marimo/mlflow 非依存の純粋ドメイン層。

**軸は点軸 (電流パラメータ) 1 本だけ**: `EvalSeries` が持つ surrogate は 1 つで、
run_id という識別子はこのモジュールに一切現れない。run ごとに系列を作って回し、
直積 (`metrics.select.SimKey` = `(系列名, 点 index, run_id)`) を組むのは結果を扱う層。
2 つの軸を 1 箇所で同時に扱わないことが、この分割の目的。

**評価したい条件は型で宣言する** (設定ファイルを持たない = スキーマという型の弱い
写しを二重に管理しない)。描画の宣言だけは設定ファイル `scripts/conf/draw.json` に
残る — あちらは図を調整するたびに書き換える対象で性格が違う。

**表示名も永続化も関心でない**: 凡例や図の見出しは描画層 (`metrics`)、結果の保存/
読込は MLflow の評価 experiment (`scripts/mlflow_io.py`) が持つ。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Self

import numpy as np
import xarray as xr

from .core.diverge import log_divergence
from .core.network import DatasetConfig, NeuronGraph
from .core.simulator import unified_simulator
from .neurons import MCMODELS
from .surrogate.bundle import SurrogateBundle
from .surrogate.meta import SurrogateMeta
from .surrogate.replace import apply_surrogate
from .surrogate.replace import replaceable as node_replaceable

# --- 仕様 --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SimSpec:
    """1 回のシミュレーションの仕様 = **純粋な計算入力**: 適用先 target × 電流
    (掃引点は `current_params` に確定済み)。これだけで波形が決まる。

    **識別は一切持たない** — 系列名は `SERIES` のキー、どの surrogate で回すかは
    `simulate` の引数、出所は結果側 (`SimResult`)。おかげで `hash()` が「同じ波形を
    出す入力か」と正確に一致する。
    """

    target: str
    current_type: str
    dt: float
    current_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """永続化 (MLflow の評価 run) が入力仕様として持ち回る形へ。"""
        return {
            "target": self.target,
            "current_type": self.current_type,
            "dt": self.dt,
            "current_params": self.current_params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            target=str(d["target"]),
            current_type=str(d["current_type"]),
            dt=float(d["dt"]),
            current_params=dict(d.get("current_params") or {}),
        )

    def key(self) -> str:
        """**同じ入力とみなす単位**の正規化文字列 (dict 順序に依らず一致する)。
        「この spec と同じか」を問う側はここを経由する。"""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    def hash(self) -> str:
        """`key()` の短縮ハッシュ。保存側が「同じ入力を既に回したか」を引くための
        キー (完全な仕様は別に持つので、衝突しない長さがあれば足りる)。"""
        return hashlib.sha1(self.key().encode()).hexdigest()[:8]

    @property
    def net(self) -> NeuronGraph:
        """適用先の MC モデル。**target 名 → ネットの解決はここだけ**が行い、結果型も
        描画も spec 経由で引く (文字列キーを持ち回らない)。"""
        return MCMODELS[self.target]

    def dataset(self) -> DatasetConfig:
        """入力 (原系)。置換系は `apply_surrogate` が非破壊で作る。"""
        return DatasetConfig.build_dataset(
            model_name=self.target,
            dt=self.dt,
            current_type=self.current_type,
            current_params=self.current_params,
        )


# --- 条件の倉庫 (この研究で回したい素材) ----------------------------------------------

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


# --- 実行 (1 シミュ) ---------------------------------------------------------------


@dataclass(frozen=True)
class SimResult:
    """1 シミュの結果 = 入力 (`SimSpec`) + 波形。`simulate` の返り値。

    **どこの何だったか (系列名 / 点 index / どの run / どこに保存されたか) は
    持たない**: 系列の中の位置は `EvalSeries.simulate` が返す並び順、run はそれを
    呼んだ側が知っている。結果を集めて軸を張るのは結果を扱う層
    (`metrics.results.SeriesView`)、保存先の id は永続化層の関心。"""

    spec: SimSpec
    dataset: xr.Dataset
    # 系列の中で振られていた電流パラメータ名 (単発 / 系列の外で回した = None)。1 シミュ
    # には無い情報なので `EvalSeries.simulate` が書き足す欄で、図の x 軸に使う。
    axis: str | None = None

    @property
    def point(self) -> float | None:
        """軸上の位置 (単発なら None)。`current_params` に確定済みの値を読むだけ
        — 二重に持たない。"""
        return float(self.spec.current_params[self.axis]) if self.axis else None


def simulate(spec: SimSpec, surrogate: SurrogateBundle | None) -> SimResult:
    """1 シミュ。`surrogate=None` なら原系、あれば `apply_surrogate` してから回す。"""
    dset = spec.dataset()
    if surrogate is None:
        return SimResult(spec, unified_simulator(dset))
    surr_ds = unified_simulator(apply_surrogate(surrogate, dset))
    # 系列名は spec が持たない (SERIES のキーが単一源) → 入力そのもので名乗る。
    where = f"{spec.target}/{spec.current_type} / {surrogate.meta.label}"
    log_divergence(spec.net, surr_ds, where)
    return SimResult(spec, surr_ds)


# --- 掃引 (点軸) ---------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSeries:
    """**1 回の掃引実験そのもの**: 何を (`spec`)・どの電流パラメータで振り
    (`param`/`values`)・どの surrogate で回すか (`surrogate`)。これだけで
    `simulate()` が引数なしに走る = 実験の記述と実行が 1 つの型に閉じる。

    `param` を渡さなければ単発 (点 1 つ) で、以降は掃引と同じ経路を通る。
    `surrogate=None` は原系。

    **run 軸は畳み込まない**: 持つのは surrogate 1 つだけで、run_id という識別子は
    この型にも このモジュールにも無い。複数 run を回すのは呼び出し側が run ごとに
    `dataclasses.replace(series, surrogate=...)` した系列を回すこと。
    """

    spec: SimSpec
    param: str | None = None  # 掃引する電流パラメータ名 (None=単発)。図の x 軸
    values: Sequence[float] = ()  # 掃引点の値列 (等間隔でなくてもよい)
    surrogate: SurrogateBundle | None = None  # 置換器 (None=原系)

    @property
    def points(self) -> list[SimSpec]:
        """点ごとの計算入力 (値順)。`current_params[param]` に値を埋めた複製 =
        `spec` と `values` からの派生で、二重に持たない。"""
        if self.param is None:
            return [self.spec]
        return [
            dc_replace(
                self.spec,
                current_params={**self.spec.current_params, self.param: float(v)},
            )
            for v in self.values
        ]

    def replaceable(self, meta: SurrogateMeta) -> bool:
        """この系列を置換できる surrogate か = 適用先に置換されるノードが 1 つでも
        あるか (点は適用先を変えないので `spec` で判定)。"""
        return any(node_replaceable(meta, n) for n in self.spec.net.nodes)

    def simulate(self) -> list[SimResult]:
        """点列を順に回す (**系列 → 結果の唯一の入口**)。返りは点の並び順で、素の
        結果に掃引軸だけ書き足す (1 シミュは自分が何の軸の上に居るかを知らない)。"""
        return [
            dc_replace(simulate(spec, self.surrogate), axis=self.param)
            for spec in self.points
        ]


# --- カタログ (この研究で回したい系列) ------------------------------------------------

# **系列名の単一源**。素材は `eval.EVALS` から名前で引き、ここで軸と点を与える
# (単発も「点 1 つの系列」として同じ経路を通る)。中身は `EvalSeries` の構築引数
# そのもので、回す側が surrogate を足して `EvalSeries(**SERIES[name],
# surrogate=...)` と組む = カタログ自体は surrogate を知らない素材のまま。
SERIES: dict[str, dict] = {
    "traub_soma_dc": {"spec": EVALS["traub_soma_dc"]},
    "traub19_somastim": {
        "spec": EVALS["traub19_somastim"],
        "param": "value",
        "values": np.linspace(0.0, 10.0, 5),
    },
    "traub19_dendstim": {
        "spec": EVALS["traub19_dendstim"],
        "param": "value",
        "values": np.linspace(0.0, 10.0, 5),
    },
    "traub19_pulse_freq": {
        "spec": EVALS["traub19_pulse_freq"],
        "param": "frequency",
        "values": np.linspace(10.0, 50.0, 5),
    },
}
