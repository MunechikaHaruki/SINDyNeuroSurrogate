// 実験の共通の部品。実験の形（回す組と、回す前に書いた主張と判定）と、判定が使う値の読み方・線・三値の論理。
// 実験そのものは主題ごとの宣言 config/<主題>.yaml（sweep.yaml 改良の探索、property.yaml 性質の発見、mc.yaml 拡張）と、形 schema.ts・判定 checks.ts が持ち、experiments.ts が並べる。
import type { Bench, Series, Train } from '../orchestrate/spec.ts';

/** Γ で「発火の時刻まで合う」とみなす線。 */
export const GAMMA_LINE = 0.9;

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
