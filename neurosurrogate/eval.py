"""**どう回すか**を扱う層: 1 シミュの仕様 (`SimSpec`) と実行 (`simulate`)、それを
電流パラメータで振った掃引実験 (`EvalSeries`)。marimo/mlflow 非依存の純粋ドメイン層。

**何を回すかは持たない**: 実際に回したい条件の一覧は `scripts/catalog.py`
(`EVALS` / `SERIES`)。ここは型と手続きだけで、この研究に固有の値は入らない。

**軸は点軸 (電流パラメータ) 1 本だけ**: `EvalSeries` が持つ surrogate は 1 つで、
run_id という識別子はこのモジュールに一切現れない。run ごとに系列を作って回し、
run 軸を掛けるのは結果を扱う層 (`report.ResultSet`)。2 つの軸を 1 箇所で同時に
扱わないことが、この分割の目的。

**`EvalSeries` は保存の単位でもある** (1 系列 = 1 評価 run)。点は `points` から
決まる派生なので、点ごとの識別子 (どの系列の何番目か) はどこにも要らない:
保存側は波形を点の順に並べて置き、読む側は `attach` で系列に貼り直す。

**条件は設定ファイルでなく型で宣言する** (スキーマという型の弱い写しを二重に
管理しない)。描画の宣言 (`report.ReportSpec`) も同じで、どちらも実体は
`scripts/catalog.py` に型のまま並ぶ。

**表示名も関心でない**: 凡例や図の見出しは結果を扱う層 (`report`)、結果の保存/
読込は MLflow の評価 experiment (`scripts/mlflow_io.py`) が持つ。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Self

import xarray as xr

from .core.diverge import log_divergence
from .core.simulator import unified_simulator
from .spec import SimSpec, short_hash
from .surrogate.bundle import SurrogateBundle
from .surrogate.meta import SurrogateMeta
from .surrogate.replace import apply_surrogate
from .surrogate.replace import replaceable as node_replaceable

# --- 実行 (1 シミュ) ---------------------------------------------------------------


@dataclass(frozen=True)
class SimResult:
    """1 シミュの結果 = 入力 (`SimSpec`) + 波形。`simulate` の返り値。

    **どこの何だったか (系列名 / 点 index / どの run / どこに保存されたか) は
    持たない**: 系列の中の位置は `EvalSeries.simulate` が返す並び順、run はそれを
    呼んだ側が知っている。結果を集めて軸を張るのは結果を扱う層
    (`report.SeriesView`)、保存先の id は永続化層の関心。"""

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
    dset = spec.materialize()
    if surrogate is None:
        return SimResult(spec, unified_simulator(dset))
    surr_ds = unified_simulator(apply_surrogate(surrogate, dset))
    # 系列名は spec が持たない (カタログのキーが単一源) → 入力そのもので名乗る。
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
    `with_surrogate` した系列を回すこと。

    **保存の単位でもある**: 1 系列 = 1 評価 run (点列を丸ごと 1 artifact に持つ)
    なので、「同じ掃引を既に回したか」を引く鍵 (`hash`) と往復の形 (`to_dict` /
    `from_dict`) をこの型が持つ。surrogate は往復に含まない (置換器は学習 run 側の
    成果物で、系列はどれで回したかを run_id で名乗る)。
    """

    spec: SimSpec
    param: str | None = None  # 掃引する電流パラメータ名 (None=単発)。図の x 軸
    values: Sequence[float] = ()  # 掃引点の値列 (等間隔でなくてもよい)
    surrogate: SurrogateBundle | None = None  # 置換器 (None=原系)

    def with_surrogate(self, surrogate: SurrogateBundle | None) -> EvalSeries:
        """置換器だけ差し替えた同じ掃引 (カタログの素材 → run 軸の 1 本)。"""
        return dc_replace(self, surrogate=surrogate)

    def to_dict(self) -> dict:
        """永続化 (評価 run の param) が持ち回る形 = 掃引の定義そのもの。"""
        return {
            "spec": self.spec.to_dict(),
            "param": self.param,
            "values": [float(v) for v in self.values],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            spec=SimSpec.from_dict(d["spec"]),
            param=d["param"] or None,
            values=[float(v) for v in d["values"]],
        )

    def hash(self) -> str:
        """**同じ掃引を既に回したか**の鍵 (surrogate は含まない = 原系の再利用が
        これ 1 本で効く)。置換系は呼び出し側がここに run_id を組む。"""
        return short_hash(json.dumps(self.to_dict(), sort_keys=True, default=str))

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

    def attach(self, datasets: Sequence[xr.Dataset]) -> list[SimResult]:
        """保存済みの波形列 → 点列の `SimResult` (**再シミュ無しの `simulate`**)。

        点の並びと各点の計算入力は `points` が単一源 = 波形さえ順に保存してあれば
        点ごとの識別子を持ち回らずに復元できる。"""
        return [
            SimResult(spec, ds, axis=self.param)
            for spec, ds in zip(self.points, datasets, strict=True)
        ]
