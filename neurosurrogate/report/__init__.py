"""結果ドメイン: 1 系列を軸に開いて報告へ畳む (**1 レポート = 1 系列 × N モデル**)。
marimo/MLflow 非依存。

package 直下が持つのは 3 つの描画 module (`model` / `series` / `report`) が共有する
**run 軸**だけ (成果物の運搬形は図を出す全層の共通型 `plotting.Artifact`):

- 張る: `series_matrix` (カタログ × 学習 run → 回す組合せ)
- 開く: `SeriesView` (1 系列を点軸 (電流パラメータ) × run 軸 (どの surrogate) に開いた
  並び = 1 レポートの単位。複数系列は素の dict で持つだけ = 束ねる型を作らない)
- 回す: `simulate_views` (張って開くまでをその場で。保存を経由しない経路)
- 名乗る: `run_names` (run_id → 表示名。凡例と行見出し)

`run_names` が結果を引数に取らないのに同居するのは、**衝突した label に連番を振る**
= 「今 run 軸に何本並ぶか」を知らないと決まらないため (run 軸の関心)。

**run 軸を持ち込むのはここ**: `eval.EvalSeries` が持つ surrogate は 1 つで run_id を
知らない。カタログ (`scripts/catalog.py` の `SERIES`) に run ごとの surrogate を
載せた系列を組むのは `series_matrix` ただ 1 つで、その場で回す経路
(`simulate_views`) も永続化を経由する経路 (`scripts/mlflow_io`) も同じ組合せを通る。

**点は識別子を持たない**: 保存の単位が 1 系列 = 1 評価 run なので、点の並びは常に
`EvalSeries.points` が単一源。結果を「並べ直す」処理はどこにも無い。

図そのものは 3 module に分かれ、どれも `list[Artifact]` を返すだけ = **どの関数を
呼んだかが保存段を決める** (図は属する run も由来も名乗らない。段を解くのは MLflow を
知る `scripts/mlflow_io`)。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..core.network import NeuronGraph
from ..sim.eval import EvalSeries, SimResult
from ..surrogate.bundle import SurrogateBundle


def series_matrix(
    catalog: dict[str, EvalSeries], bundles: dict[str, SurrogateBundle]
) -> list[tuple[str, EvalSeries, dict[str, EvalSeries]]]:
    """カタログ (系列名 → 素の `EvalSeries`) × run 軸 →
    (系列名, 原系, run_id → 置換系) の列。**run 軸を掛ける唯一の場所**。

    1 本も置換できない系列は落とす (回しても比較対象が無い)。回すのも保存するのも
    呼び出し側の関心で、ここは「どの組合せを回すか」だけを決める。
    """
    out = []
    for name, original in catalog.items():
        surrs = {
            run_id: original.with_surrogate(bundle)
            for run_id, bundle in bundles.items()
            if original.replaceable(bundle.meta)
        }
        if surrs:
            out.append((name, original, surrs))
    return out


def run_names(bundles: dict[str, SurrogateBundle]) -> dict[str, str]:
    """run_id → 表示名 (凡例/行見出し)。**表示名は結果でなく surrogate 側から解く**
    (結果は run_id という同一性だけを持つ)。

    `meta.label` は学習構造 + 学習データまでしか区別しない → library_specs 違いや
    同 config の再実行は同じ label になるため、衝突したものにだけ与えた順の連番を
    付けて潰れを防ぐ (選択を拒否せず全部見せる)。
    """
    labels = [b.meta.label for b in bundles.values()]
    counts = Counter(labels)
    seen: Counter[str] = Counter()
    out: dict[str, str] = {}
    for run_id, label in zip(bundles, labels, strict=True):
        seen[label] += 1
        out[run_id] = label if counts[label] == 1 else f"{label}#{seen[label]}"
    return out


@dataclass(frozen=True)
class SeriesView:
    """**1 系列を両軸に開いたもの**: 点軸に値順で並んだ原系 (`points`) と、run ごとに
    同じ並びの置換系 (`surrs`)。列の長さは常に揃う。

    描画/指標はここだけを見る = 「点を値で並べ直す」「run 軸を数える」を各図が
    再実装しない。点の位置は list の添字そのもので、平坦キーは出てこない。
    """

    name: str
    points: list[SimResult]  # 原系 (掃引点の値順。単発なら 1 件)
    surrs: dict[str, list[SimResult]]  # run_id → `points` と同じ並びの置換系
    original_id: str = ""  # 原系を読んだ評価 run の id (回した直後は空)
    surr_ids: dict[str, str] = field(default_factory=dict)  # 学習 run → 評価 run

    def __post_init__(self) -> None:
        bad = [rid for rid, col in self.surrs.items() if len(col) != len(self.points)]
        if bad:
            raise ValueError(f"{self.name}: 点数が原系と揃わない run {bad}")
        # 評価 run の id は **run 軸と揃うか、丸ごと無いか**の二択 (読んだ結果か、
        # その場で回した結果か)。半端に欠けた id は保存段が別の run の名前へ落ちる
        # ことを意味するので、図を描く前に構築時点で弾く。
        if self.surr_ids and set(self.surr_ids) != set(self.surrs):
            raise ValueError(
                f"{self.name}: 評価 run の id が run 軸と揃わない "
                f"{sorted(set(self.surr_ids) ^ set(self.surrs))}"
            )
        # 原系と置換系も同じ二択 (片方だけの id は `sources` が黙って短くなる =
        # 由来の欠けた成果物になる)。空文字の id も「無い」と同じなので弾く。
        if bool(self.original_id) != bool(self.surr_ids) or not all(
            self.surr_ids.values()
        ):
            raise ValueError(f"{self.name}: 評価 run の id が半端に欠けている")

    @property
    def sources(self) -> tuple[str, ...]:
        """読んだ評価 run の id (原系が先、置換系は run 軸の順)。回した直後は空。
        成果物の由来 (`meta.json`) にそのまま落ちる。"""
        return tuple(i for i in (self.original_id, *self.surr_ids.values()) if i)

    def series_id(self, run_id: str) -> str:
        """学習 run → その置換系を持つ評価 run の id (回した直後は空)。保存段が
        評価 run 1 本になる成果物 (詳細図) の宛先を解くのに使う。"""
        return self.surr_ids.get(run_id, "")

    @property
    def run_ids(self) -> list[str]:
        """置換系の run_id (与えた順。原系は含まない)。"""
        return list(self.surrs)

    @property
    def net(self) -> NeuronGraph:
        """適用先 (点は適用先を変えないので先頭から引く)。"""
        return self.points[0].spec.net

    @property
    def target(self) -> str:
        return self.points[0].spec.target

    @property
    def axis(self) -> str | None:
        """掃引した電流パラメータ名 (単発なら None)。図の x 軸。"""
        return self.points[0].axis

    @property
    def values(self) -> list[float | None]:
        """点軸の値 (単発なら `[None]`)。列見出しと折れ線の x に使う。"""
        return [r.point for r in self.points]

    def pair(self, index: int, run_id: str) -> tuple[SimResult, SimResult]:
        """点 `index` の (原系, 置換系)。"""
        return self.points[index], self.surrs[run_id][index]

    def clamp(self, index: int) -> int:
        """点 index を手元の点数へ収める (設定が実際の点数を超えていても描く)。"""
        return min(index, len(self.points) - 1)


def simulate_views(
    catalog: dict[str, EvalSeries], bundles: dict[str, SurrogateBundle]
) -> dict[str, SeriesView]:
    """カタログ × run 軸を**その場で回して**系列名 → `SeriesView` を集める
    (保存を経由しない経路)。永続化した結果を読む経路は `scripts/mlflow_io.py`。

    束ねる型は持たない: 描画の単位が 1 系列 (= 1 `SeriesView`) なので、複数系列は
    ただの dict で足りる。

    原系は掃引の内容 (`EvalSeries.hash`) で共有する = 同じ掃引を宣言した系列が
    2 つあっても原系のシミュは 1 度だけ (保存側の再利用と同じ鍵で効く)。"""
    originals: dict[str, list[SimResult]] = {}
    out: dict[str, SeriesView] = {}
    for name, original, surrs in series_matrix(catalog, bundles):
        if original.hash() not in originals:
            originals[original.hash()] = original.simulate()
        out[name] = SeriesView(
            name,
            originals[original.hash()],
            {rid: s.simulate() for rid, s in surrs.items()},
        )
    return out
