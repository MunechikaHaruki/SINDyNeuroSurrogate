"""評価の 3 点セット: **1 シミュの仕様 (`SimSpec`) / 回したい条件 (`EVALS`) /
1 シミュの実行 (`simulate`)**。marimo/mlflow 非依存の純粋ドメイン層。

この 3 つは同じ問い「何を回すか」の表と裏なので 1 枚に置く。run 軸 (surrogate) を
掛けて複数本を束ねるのは別の関心 → `runs.py`。

**基本単位は 1 回のシミュレーション**: 掃引点も run も `SimSpec` 自身のフィールドで、
「どのシミュを指すか」を決める同格のパラメータ。掃引は `sweep` が点ごとの `SimSpec`
へ展開し尽くす — 展開後に「掃引軸」という型は残らず、点が自分の軸名 (`sweep_param`)
を持つだけ。run は `run_id` (`None` = 原系)。

**評価したい条件は `EVALS` が型で宣言する** (設定ファイルを持たない = スキーマという
型の弱い写しを二重に管理しない)。描画の宣言だけは設定ファイル
`scripts/conf/draw.json` に残る — あちらは図を調整するたびに書き換える対象で
性格が違う。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
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
    """1 回のシミュレーションの仕様 = 適用先 target × 電流 (掃引点は
    `current_params` に確定済み) × run。原系/置換系を非対称に扱わない —
    `run_id=None` が原系、それ以外が surrogate の MLflow run_id で、経路は共通。
    """

    name: str  # 掃引しても不変な系列名 (図の系列識別に使う。label の導出元)
    target: str
    current_type: str
    dt: float
    current_params: dict = field(default_factory=dict)
    sweep_param: str | None = None  # 掃引軸名 (None=掃引なし)。図の x 軸ラベル用
    run_id: str | None = None  # None=原系、str=surrogate の MLflow run_id

    @property
    def sweep_value(self) -> float | None:
        """掃引軸の値 (掃引なしなら None)。`current_params` に確定済みの値を読むだけ
        — 二重に持たない。"""
        return (
            float(self.current_params[self.sweep_param]) if self.sweep_param else None
        )

    def to_dict(self) -> dict:
        """永続化 (MLflow の評価 run) が入力仕様として持ち回る形へ。"""
        return {
            "name": self.name,
            "target": self.target,
            "current_type": self.current_type,
            "dt": self.dt,
            "current_params": self.current_params,
            "sweep_param": self.sweep_param,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            name=str(d["name"]),
            target=str(d["target"]),
            current_type=str(d["current_type"]),
            dt=float(d["dt"]),
            current_params=dict(d.get("current_params") or {}),
            sweep_param=d.get("sweep_param"),
            run_id=d.get("run_id"),
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


def sweep(spec: SimSpec, param: str, values: Iterable[float]) -> list[SimSpec]:
    """1 spec を電流パラメータ `param` の値ごとに展開する。値列は呼び出し側が
    そのまま渡す (等間隔なら `np.linspace`、そうでなくてもよい)。展開後の各点は
    自分がどの軸上に居るかを `sweep_param` として持つ。"""
    return [
        dc_replace(
            spec,
            current_params={**spec.current_params, param: float(v)},
            sweep_param=param,
        )
        for v in values
    ]


def labeled(*groups: SimSpec | Sequence[SimSpec]) -> dict[str, SimSpec]:
    """宣言 → label → SimSpec。**1 引数 = 1 系列**で、単発 (spec 1 つ) は `name`、
    掃引 (`sweep` の返り値) は `name#i` が label になる。label を宣言側が書かない
    = dict のキーと `SimSpec.name` がズレようがない。"""
    out: dict[str, SimSpec] = {}
    for group in groups:
        points = [group] if isinstance(group, SimSpec) else list(group)
        if len(points) == 1:
            out[points[0].name] = points[0]
        else:
            out.update({f"{p.name}#{i}": p for i, p in enumerate(points)})
    return out


# --- カタログ (この研究で回したい条件) ------------------------------------------------

# 掃引つき評価の共通電流パラメータ (刺激前の静穏 + 本体長)。
_STIM = {"silence_duration": 10.0, "duration": 300.0}
_DT = 0.01

EVALS: dict[str, SimSpec] = labeled(
    # 単体 traub の素の応答 (置換の足場が動くかを最短で見る)。
    SimSpec(
        name="traub_soma_dc",
        target="traub",
        current_type="lin&steady",
        dt=_DT,
        current_params={"silence_duration": 10.0, "duration": 40.0, "value": 3.0},
    ),
    # 刺激部位だけを変えた対照ペア (soma / dend)。同じ電流軸で比べる。
    sweep(
        SimSpec(
            name="traub19_somastim",
            target="traub19_soma",
            current_type="lin&steady",
            dt=_DT,
            current_params=_STIM,
        ),
        "value",
        np.linspace(0.0, 10.0, 5),
    ),
    sweep(
        SimSpec(
            name="traub19_dendstim",
            target="traub19_soma_dendstim",
            current_type="lin&steady",
            dt=_DT,
            current_params=_STIM,
        ),
        "value",
        np.linspace(0.0, 10.0, 5),
    ),
    # 入力の速さに対する追従 (パルス周波数掃引)。
    sweep(
        SimSpec(
            name="traub19_pulse_freq",
            target="traub19_soma",
            current_type="periodic&pulse",
            dt=_DT,
            current_params={**_STIM, "amplitude": 20, "baseline": 0.0},
        ),
        "frequency",
        np.linspace(10.0, 50.0, 5),
    ),
)


# --- 実行 (1 シミュ) ---------------------------------------------------------------


def simulate(spec: SimSpec, surrogate: SurrogateBundle | None) -> xr.Dataset:
    """1 シミュ。`surrogate=None` なら原系、あれば `apply_surrogate` してから回す。"""
    dset = spec.dataset()
    if surrogate is None:
        return unified_simulator(dset)
    surr_ds = unified_simulator(apply_surrogate(surrogate, dset))
    log_divergence(spec.net, surr_ds, f"{spec.name} / {surrogate.meta.label}")
    return surr_ds
