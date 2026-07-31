"""**1 回のシミュレーション**だけを扱う層: 仕様 (`SimSpec`) / 名前付きの条件倉庫
(`EVALS`) / 実行 (`simulate`)。marimo/mlflow 非依存の純粋ドメイン層。

ここには「複数シミュ」の概念が一切ない — 掃引 (点列) も run 軸 (surrogate) も
`runs.py` の関心。`EVALS` は 1 名前 = 1 条件の素材倉庫で、それを軸で振って系列に
組み立てるのは `runs.SERIES`。

**評価したい条件は型で宣言する** (設定ファイルを持たない = スキーマという型の弱い
写しを二重に管理しない)。描画の宣言だけは設定ファイル `scripts/conf/draw.json` に
残る — あちらは図を調整するたびに書き換える対象で性格が違う。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Self

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

    **識別は一切持たない** — 系列名/label は `runs.SERIES` の構造と規約、
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


# --- 条件の倉庫 (この研究で回したい素材) ----------------------------------------------

# 掃引つき評価の共通電流パラメータ (刺激前の静穏 + 本体長)。掃引軸の値は入らない
# (`runs.sweep` が点ごとに埋める)。
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


def simulate(spec: SimSpec, surrogate: SurrogateBundle | None) -> xr.Dataset:
    """1 シミュ。`surrogate=None` なら原系、あれば `apply_surrogate` してから回す。"""
    dset = spec.dataset()
    if surrogate is None:
        return unified_simulator(dset)
    surr_ds = unified_simulator(apply_surrogate(surrogate, dset))
    # 系列名は spec が持たない (SERIES のキーが単一源) → 入力そのもので名乗る。
    where = f"{spec.target}/{spec.current_type} / {surrogate.meta.label}"
    log_divergence(spec.net, surr_ds, where)
    return surr_ds
