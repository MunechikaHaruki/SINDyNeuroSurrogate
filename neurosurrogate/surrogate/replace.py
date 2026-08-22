"""**学習済みのものを別のネットへ当てる**ことの語彙一式。

`model.py` が「学習済みサロゲートを持つ・保存する・読む」だけを担うのに対し、ここは
**適用先 (`NeuronGraph`) を知る唯一の場所**: どこへ当てられるか (`_rejected_targets`
/ `applicable`)、当てられないなら何がどう食い違ったか (`_Absent` / `_TypeMismatch` /
`_ParamsMismatch`)、そして当てること (`replace`) が 1 モジュールに閉じる。

**判定と適用を分けない**のは、両者が同じ規則の裏表だから — `replace` が断る条件は
`applicable` が偽になる条件そのもので、離して置くと 2 通りの「置換してよい」が育つ。

型のメソッドでなくモジュール関数なのは、置換が `Surrogate` の関心でないから
(適用先はサロゲートの持ち物ではない)。`in_train_domain` は spec 自身の性質
(学習ドメイン) なので `model.py` 側に残り、ここはそれを名前ごとに問うだけ。

**外から呼べるのは `applicable` と `replace` の 2 つだけ** — 他は全部 `_` 付きで、
理由の型も `_rejected_targets` も内部の綴り。呼ぶ側に要るのは「当てられるか」と
「当てる」だけで、なぜ当てられないかは**例外の文言としてしか出ない**
(`_NotReplaceable` も `_` 付き = 型名で catch する経路は無く、外からは `ValueError`)。
構造のまま読みたくなったらそのとき `_` を外して呼ぶ側を足す。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace

from ..core.network import DatasetConfig, NeuronGraph
from ..sim.spec import EvalSeries
from .model import Surrogate, SurrogateSpec

# --- 置換できない理由 -------------------------------------------------------
# 3 つで**理由の全部** (`_rejected_targets` がこれ以外を返さない = 網羅が型で見える)。
# どれも「何がどう食い違ったか」の**事実だけ**を持ち、文にはしない — 文にするのは
# `_NotReplaceable.__str__` 1 箇所。事実と文を分ける利得は**この module の中で閉じて
# いる**: 理由の種類が増えたとき `_describe` の match が網羅を検査してくれる。
# (`.rejections` を外から構造のまま読む経路はまだ無い。要るようになったら `_` を
# 外して呼ぶ側を足す = そのとき初めて公開になる。)


@dataclass(frozen=True)
class _Absent:
    """その名前が適用先に無い。"""

    name: str
    available: tuple[str, ...]


@dataclass(frozen=True)
class _TypeMismatch:
    """comp の種類が学習した種類と違う。"""

    name: str
    node_type: str
    train_type: str


@dataclass(frozen=True)
class _ParamsMismatch:
    """種類は合うが回路 params が学習ドメイン外 (params 一致を要求する定式化の場合)。"""

    name: str
    node_params: tuple | None
    train_name: str
    train_params: tuple | None


# 3 つの直和 = `_describe` の match が網羅を検査される単位。
_Rejection = _Absent | _TypeMismatch | _ParamsMismatch


def _describe(rejection: _Rejection) -> str:
    """理由 1 件を人の読む 1 件へ。**事実 → 文の変換はここだけ**。"""
    match rejection:
        case _Absent(name, available):
            return f"{name!r}: 適用先 {list(available)} に存在しない"
        case _TypeMismatch(name, node_type, train_type):
            return f"{name!r}: 種類 {node_type!r} ≠ 学習した種類 {train_type!r}"
        case _ParamsMismatch(name, node_params, train_name, train_params):
            return (
                f"{name!r}: params 非両立 = 学習ドメイン外\n"
                f"    train({train_name}): {train_params}\n"
                f"    node({name}): {node_params}"
            )


class _NotReplaceable(ValueError):
    """`replace` が置換を断ったこと。

    理由は**構造のまま** `rejections` に持ち、文にするのは `__str__` = 例外の提示。
    握って部分適用に落とさないので、送出は常に「1 つも置換していない」を意味する。
    """

    def __init__(self, surr_type_name: str, rejections: Sequence[_Rejection]) -> None:
        self.rejections = tuple(rejections)
        super().__init__(surr_type_name)

    def __str__(self) -> str:
        return f"{self.args[0]} で置換できない対象:\n  " + "\n  ".join(
            _describe(rejection) for rejection in self.rejections
        )


# --- どこへ当てられるか -----------------------------------------------------


def _rejected_targets(
    spec: SurrogateSpec, net: NeuronGraph, targets: Sequence[str]
) -> list[_Rejection]:
    """明示指定のうち net 上で置換できないもの (空なら全部置換可)。

    通るかどうかは `spec.in_train_domain` が答え、ここはそれが偽だった理由を名前ごとに
    **事実の形で**挙げるだけ = どの名前がなぜ通らなかったかを必ず名指しできる
    (「何個か落ちた」で終わらせない)。**文言は持たない** — 分かるのは食い違った値まで
    で、それをどう見せるかは提示側 (`_NotReplaceable`) の関心。

    読み手は 2 つ: `replace` が例外へ載せ、run 選択 (`SurrogateRuns.replacing`) は
    `applicable` 経由で真偽だけを見て絞る (落ちた run の理由はどこにも出ない —
    出したくなったらこの返り値と理由の型を公開にして UI まで運ぶ)。
    """
    rejections: list[_Rejection] = []
    for name in targets:
        if name not in net.names:
            rejections.append(_Absent(name, tuple(net.names)))
            continue
        comp = net.nodes[net.name_to_idx(name)]
        if spec.in_train_domain(comp):
            continue
        if comp.type != spec.comp_type:
            rejections.append(_TypeMismatch(name, comp.type.name, spec.comp_type.name))
        else:
            rejections.append(
                _ParamsMismatch(
                    name,
                    comp.resolved_params,
                    spec.train_comp().name,
                    spec.train_comp().resolved_params,
                )
            )
    return rejections


def applicable(spec: SurrogateSpec, series: EvalSeries) -> bool:
    """系列が置換対象に挙げたノードを**全部**置換できるか。

    部分一致は不可 = 「指定したうち通ったものだけ静かに置換」を作らない。学習結果を
    見ないので (spec だけで決まる) 、run を pickle ごとロードせず一覧から絞れる。
    """
    return bool(series.replace_targets) and not _rejected_targets(
        spec, series.spec.net, series.replace_targets
    )


# --- 当てる -----------------------------------------------------------------


def replace(
    surrogate: Surrogate, dataset: DatasetConfig, targets: Sequence[str]
) -> DatasetConfig:
    """**明示指定された** targets を surrogate へ置換した実行入力を返す。

    「互換なノードを全部」置換しない = 適用先の形態が変わっても置換範囲は動かない。
    1 つでも置換できなければ何も置換せず `_NotReplaceable` (部分適用を作らない)。
    """
    if not targets:
        raise ValueError("置換対象が空: 置換するノード名を明示指定すること")
    rejected = _rejected_targets(surrogate.spec, dataset.net, targets)
    if rejected:
        raise _NotReplaceable(surrogate.spec.surr_type_name(), rejected)
    names = set(targets)
    surr_comp_type = surrogate.ansatz.surr_comp_type(
        surrogate.spec, surrogate.preprocessor, surrogate.closure
    )
    return dc_replace(
        dataset,
        net=dc_replace(
            dataset.net,
            nodes=[
                dc_replace(node, type=surr_comp_type) if node.name in names else node
                for node in dataset.net.nodes
            ],
        ),
    )
