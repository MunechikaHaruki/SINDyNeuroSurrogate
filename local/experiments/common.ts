// 実験の共通の部品。実験の形（回す組と、回す前に書いた主張と判定）と、判定が使う値の読み方・線・三値の論理。
// 実験そのものは実験群ごとの実験層 config/<群>.yaml（形は schema.ts）と判定層 config/<群>.checks.ts が持ち、experiments.ts が並べる。
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

/** 判定 1 行。ok が null なら判定できない。 */
export type Judgement = { label: string; value: string; line: string; ok: boolean | null };

/** 列 1 本の値。grouping と method は学習 run の名前の部分（`h+n_a+c`、`kam`）、values は掃引の値、gammas と counts（元, モデル）は掃引の点の順。 */
export type Row = {
  grouping: string;
  method: string;
  values: number[];
  gammas: (number | null)[];
  counts: [number, number][];
};

/** 判定。系列ごとの列（主張の series の順）と、主張が絞る群分けで判定する。 */
export type Judge = (tables: Row[][], groupings?: string[]) => Judgement[];

/** 判定層（config/<群>.checks.ts）の主張 1 つ。主張の一文はこの項目の docstring に書く。run は読む run（実験層の runs の名前）、series は判定が読む系列（区画は系列の置き換える最初の区画）、groupings は判定を絞る群分け。 */
export type Check = { run: string; series: string[]; groupings?: string[]; judge: Judge };
/** 判定層。主張の名前 → 主張。 */
export type Checks = Record<string, Check>;

/** 主張 1 つ。name は判定と図が引く名前、series は判定が読む系列（区画は系列の置き換える最初の区画）、groupings は判定を絞る群分け。 */
export type Claim = {
  name: string;
  claim: string;
  series: string[];
  groupings?: string[];
  judge: Judge;
};

/** 学習 run 1 本。name は実験の中の名前（`<仕方の名前>.<群分け>`）、method と grouping は判定と図が列を引く名前。 */
export type Model = { name: string; method: string; grouping: string; train: Train };

/** 解決済みの実験。description は何を回して何を見るかの一文（Markdown）。学習 run と系列を値で持ち、pairs が回す組（学習 run の名前 × 系列の名前）。
 * groupings と methods は画面が絞り込みに使う名前の並び。判定の無い実験は claims が空。 */
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
  claims: Claim[];
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

/** 仕方の読み方。 */
export const NAMES: Record<string, string> = {
  kam: 'KAM',
  'kam-absorb-m': 'KAM（m を吸収）',
  'pca-sindy': 'PCA＋SINDy',
  'ae-sindy': 'AE＋SINDy',
  'ae-latent5-sindy': 'NEURO2026',
};

/** 群分けと仕方の列。無いか 2 本以上あれば undefined（どれを読むか決まらない）。 */
export function one(rows: Row[], grouping: string, method: string): Row | undefined {
  const found = rows.filter(r => r.grouping === grouping && r.method === method);
  return found.length === 1 ? found[0] : undefined;
}
export const mean = (row: Row | undefined) => (row ? meanGamma(row.gammas) : null);
export const shown = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v.toFixed(2));
export const above = (v: number | null | undefined) => (v === null || v === undefined ? null : v >= GAMMA_LINE);
export const below = (v: number | null | undefined) => (v === null || v === undefined ? null : v < GAMMA_LINE);
/** 三値の「すべて」。一つでも偽なら偽、偽が無く欠けがあれば null。 */
export const all = (xs: (boolean | null)[]) => (xs.includes(false) ? false : xs.includes(null) ? null : true);
/** 「当てはまる組が 0 個」。当てはまる組があれば偽、無くても決まらない組があれば null。 */
export function none(found: string[], unknown: string[]): Pick<Judgement, 'value' | 'ok'> {
  const value =
    found.length > 0 ? found.join('・') : unknown.length > 0 ? `なし（判定できない組 ${unknown.join('・')}）` : 'なし';
  return { value, ok: found.length > 0 ? false : unknown.length > 0 ? null : true };
}
export const groupingsOf = (rows: Row[]) => [...new Set(rows.map(r => r.grouping))];

/** 置き換えた列 1 本。run は置き換えた run、method と grouping は学習 run の仕方と群分けの名前。 */
export type Column = { run: string; method: string; grouping: string };

/** 系列の列を判定の値にする。values は掃引の値、crossings は列の run ごとの点ごとの Γ と発火の回数（元, モデル）。 */
export function rowsOf(
  columns: Column[],
  values: number[],
  crossings: Record<string, { gamma: number | null; count: [number, number] }[]>,
): Row[] {
  const rows: Row[] = [];
  for (const column of columns) {
    const points = crossings[column.run] ?? [];
    const gammas: (number | null)[] = [];
    const counts: [number, number][] = [];
    for (const p of points) {
      gammas.push(p.gamma);
      counts.push(p.count);
    }
    rows.push({ method: column.method, grouping: column.grouping, values, gammas, counts });
  }
  return rows;
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
