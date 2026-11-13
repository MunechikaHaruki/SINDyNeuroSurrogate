// 実験の共通の部品。宣言の形（系列・学習 run・GPU の実測・解決済みの実験）と、図が使う物差し（Γ の線と許容幅）と読み方。
// 実験そのものは実験群ごとの実験層 config/<群>.yaml（形は schema.ts）が持ち、experiments.ts が並べる。
import type { Input, SurrogateConfig } from '../../shared/spec.ts';

/** 1 回の掃引実験（Python の EvalSeries）の宣言。送る spec では、入力（Input）と、置き換えたモデルの側の置換範囲に分かれる。 */
export type Series = Input & {
  /** 置換するノード名。互換かどうかは Python が名前ごとに検証する。 */
  replace_targets: string[];
};

/** 学習 run 1 本の宣言。name は学習 run の名前（実験の中の `<仕方の名前>.<群分け>` かテストの名前）で、同一性には入らない。 */
export type Train = { name: string; config: SurrogateConfig };

/** GPU の実測 1 つ。model は学習 run の名前（null なら元のモデル）、cells は並べる細胞の数（128 の倍数）、ncu は Nsight Compute で 1 刻みのカーネルを測るか。 */
export type Bench = {
  name: string;
  model: string | null;
  series: string;
  cells: number[];
  ncu: boolean;
};


/** Γ で「発火の時刻まで合う」とみなす線。 */
export const GAMMA_LINE = 0.9;

/** Γ の許容幅 Δ [ms]（2026-10-27 に決めた物差し）。計算する機械には series の頼み方で渡す。 */
export const COINCIDENCE_WINDOW = 2;

/** 学習 run 1 本。name は実験の中の名前（`<仕方の名前>.<群分け>`）、method と grouping は図が列を引く名前。 */
export type Model = { name: string; method: string; grouping: string; train: Train };

/** 解決済みの実験。description は何を回して何を見るかの一文（Markdown）。学習 run と系列を値で持ち、pairs が回す組（学習 run の名前 × 系列の名前）。
 * groupings と methods は画面が絞り込みに使う名前の並び。 */
export type Plan = {
  id: string;
  description: string;
  groupings: string[];
  methods: string[];
  models: Model[];
  series: Record<string, Series>;
  pairs: { model: string; series: string }[];
  /** GPU の実測。無ければ空。 */
  benches: Bench[];
  /** 群分けを全部回すときの、群分けを組むゲート（画面が群分けの掃引の節を出す）。 */
  gates?: string[];
};

/** ゲートの群分けを全部並べる。2 つ以上のゲートを入れた群だけを、群の中は `+`、群の間は `_` でつないで名前にし（例 h+n_a+c）、全部が 1 つずつの分け方は除く。減らす状態の少ない順、同じなら名前の順。 */
export function groupings(gates: string[]): string[] {
  let partitions: string[][][] = [[]];
  for (const gate of gates) {
    const next: string[][][] = [];
    for (const blocks of partitions) {
      for (let i = 0; i < blocks.length; i++) next.push(blocks.map((b, j) => (j === i ? [...b, gate] : b)));
      next.push([...blocks, [gate]]);
    }
    partitions = next;
  }
  const list: { name: string; reduced: number }[] = [];
  for (const blocks of partitions) {
    const groups = blocks.filter(b => b.length > 1);
    if (groups.length === 0) continue;
    let reduced = 0;
    for (const g of groups) reduced += g.length - 1;
    list.push({ name: groups.map(g => g.join('+')).join('_'), reduced });
  }
  list.sort((x, y) => x.reduced - y.reduced || x.name.localeCompare(y.name));
  return list.map(g => g.name);
}

/** 決まる点（どちらかが発火した点）での Γ の平均。決まる点が無ければ null。 */
export function meanGamma(gammas: (number | null)[]): number | null {
  let sum = 0;
  let n = 0;
  for (const g of gammas) {
    if (g === null) continue;
    sum += g;
    n++;
  }
  return n > 0 ? sum / n : null;
}

/** 群分けの名前（`h+n_a+c`）を、群ごとのゲートの小文字の名前（[['h','n'],['a','c']]）にする。 */
export function blocksOf(grouping: string): string[][] {
  return grouping.split('_').map(block => block.split('+'));
}

/** 発火の始まり（rheobase）。掃引の値を小さい順に見て、初めて発火した値。どこでも発火しなければ null。 */
export function rheobase(values: number[], counts: number[]): number | null {
  const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const first = order.find((i) => counts[i] > 0);
  return first === undefined ? null : values[first];
}
