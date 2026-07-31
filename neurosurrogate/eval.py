"""評価の 3 点セット: **1 シミュの仕様 (`SimSpec`) / 回したい条件 (`EVALS`) /
1 シミュの実行 (`simulate`)**。marimo/mlflow 非依存の純粋ドメイン層。

この 3 つは同じ問い「何を回すか」の表と裏なので 1 枚に置く。run 軸 (surrogate) を
掛けて複数本を束ねるのは別の関心 → `runs.py`。

**基本単位は 1 回のシミュレーション**: 掃引点も run も `SimSpec` 自身のフィールドで、
「どのシミュを指すか」を決める同格のパラメータ。掃引は `sweep` が点ごとの `SimSpec`
へ展開し尽くす — 展開後に「掃引軸」という型は残らず、点が自分の軸名 (`sweep_param`)
を持つだけ。run は `run_id` (`None` = 原系)。

**評価したい条件は `EVALS` が型で宣言する** (設定ファイルを持たない = スキーマという
型の弱い写しを二重に管理しない)。形は **系列名 → `EvalSeries` (軸 + 点列)** で、
系列名は dict のキーが単一源 (`SimSpec` は識別も軸も持たない)。描画の宣言だけは
設定ファイル `scripts/conf/draw.json` に残る — あちらは図を調整するたびに
書き換える対象で性格が違う。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
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
from .surrogate.replace import apply_surrogate

# --- 仕様 --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SimSpec:
    """1 回のシミュレーションの仕様 = **純粋な計算入力**: 適用先 target × 電流
    (掃引点は `current_params` に確定済み)。これだけで波形が決まる。

    **識別は一切持たない** — 系列名/label は `EVALS` の構造と `runs` の規約、
    どの surrogate で回すかは `simulate` の引数、出所は結果側 (`runs.SimResult`)。
    おかげで `hash()` が「同じ波形を出す入力か」と正確に一致する。
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


@dataclass(frozen=True)
class EvalSeries:
    """1 系列 = 軸 + その上の点列。**軸は系列の性質**なので点 (`SimSpec`) は持たない
    (単発は軸なし = 点 1 つ)。点の x 座標は `current_params[axis]` に確定済み。"""

    points: list[SimSpec]
    axis: str | None = None  # 掃引した電流パラメータ名 (None=単発)。図の x 軸


def sweep(spec: SimSpec, param: str, values: Iterable[float]) -> EvalSeries:
    """1 spec を電流パラメータ `param` の値ごとに振った系列。値列は呼び出し側が
    そのまま渡す (等間隔なら `np.linspace`、そうでなくてもよい)。"""
    return EvalSeries(
        [
            dc_replace(spec, current_params={**spec.current_params, param: float(v)})
            for v in values
        ],
        axis=param,
    )


# --- カタログ (この研究で回したい条件) ------------------------------------------------

# 掃引つき評価の共通電流パラメータ (刺激前の静穏 + 本体長)。
_STIM = {"silence_duration": 10.0, "duration": 300.0}
_DT = 0.01

EVALS: dict[str, EvalSeries] = {
    # 単体 traub の素の応答 (置換の足場が動くかを最短で見る)。
    "traub_soma_dc": EvalSeries(
        [
            SimSpec(
                target="traub",
                current_type="lin&steady",
                dt=_DT,
                current_params={
                    "silence_duration": 10.0,
                    "duration": 40.0,
                    "value": 3.0,
                },
            )
        ]
    ),
    # 刺激部位だけを変えた対照ペア (soma / dend)。同じ電流軸で比べる。
    "traub19_somastim": sweep(
        SimSpec(
            target="traub19_soma",
            current_type="lin&steady",
            dt=_DT,
            current_params=_STIM,
        ),
        "value",
        np.linspace(0.0, 10.0, 5),
    ),
    "traub19_dendstim": sweep(
        SimSpec(
            target="traub19_soma_dendstim",
            current_type="lin&steady",
            dt=_DT,
            current_params=_STIM,
        ),
        "value",
        np.linspace(0.0, 10.0, 5),
    ),
    # 入力の速さに対する追従 (パルス周波数掃引)。
    "traub19_pulse_freq": sweep(
        SimSpec(
            target="traub19_soma",
            current_type="periodic&pulse",
            dt=_DT,
            current_params={**_STIM, "amplitude": 20, "baseline": 0.0},
        ),
        "frequency",
        np.linspace(10.0, 50.0, 5),
    ),
}


# --- 実行 (1 シミュ) ---------------------------------------------------------------


def simulate(spec: SimSpec, surrogate: SurrogateBundle | None) -> xr.Dataset:
    """1 シミュ。`surrogate=None` なら原系、あれば `apply_surrogate` してから回す。"""
    dset = spec.dataset()
    if surrogate is None:
        return unified_simulator(dset)
    surr_ds = unified_simulator(apply_surrogate(surrogate, dset))
    # 系列名は spec が持たない (EVALS のキーが単一源) → 入力そのもので名乗る。
    where = f"{spec.target}/{spec.current_type} / {surrogate.meta.label}"
    log_divergence(spec.net, surr_ds, where)
    return surr_ds
