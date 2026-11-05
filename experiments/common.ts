// 実験の共通の部品。実験の形（回す組と、回す前に書いた主張と判定）と、判定が使う値の読み方・線・三値の論理。
// 実験そのものは分類ごとのディレクトリ（mc/ 拡張、sweep/ 改良の探索、property/ 性質の発見）の experiments.json と checks.ts が持ち、experiments.ts が並べる。
import { partsOf } from '../evaluation/orchestrate/conf.ts';

/** Γ で「発火の時刻まで合う」とみなす線。 */
export const GAMMA_LINE = 0.9;

/** 判定 1 行。ok が null なら判定できない。 */
export type Judgement = { label: string; value: string; line: string; ok: boolean | null };

/** 列 1 本の値。grouping と method は学習 run の名前の部分（`hn_ac`、`kam`）、values は掃引の値、gammas と counts（元, モデル）は掃引の点の順。 */
export type Row = {
  grouping: string;
  method: string;
  values: number[];
  gammas: (number | null)[];
  counts: [number, number][];
  summary: Record<string, number | null>;
};

/** 判定。系列ごとの列（主張の series の順）と、主張が絞る群分けで判定する。 */
export type Judge = (tables: Row[][], groupings?: string[]) => Judgement[];

/** 主張 1 つ。name は判定と図が引く名前、series は判定が読む系列（comp を書かなければ系列の置き換える最初の区画）、groupings は判定を絞る群分け。 */
export type Claim = {
  name: string;
  claim: string;
  series: { name: string; comp?: string }[];
  groupings?: string[];
  judge: Judge;
};

/** 実験。groupings × methods × series が回す run の直積で、series は主張が読む系列を合わせたもの。 */
export type Plan = {
  id: string;
  groupings: string[];
  methods: string[];
  series: string[];
  claims: Claim[];
};

/** ゲートの群分けを全部並べる。2 つ以上のゲートを入れた群だけを名前にし（例 hn_ac）、全部が 1 つずつの分け方は除く。減らす状態の少ない順、同じなら名前の順。 */
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
    list.push({ name: groups.map(g => g.join('')).join('_'), reduced });
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
  kamm: 'KAM（m を吸収）',
  pca: 'PCA',
  ae: 'AE',
  poster: 'NEURO2026',
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

/** 置き換えた列 1 本。run は置き換えた run、model は学習 run の名前、summary はその学習の summary.json。 */
export type Column = { run: string; model: string; summary: Record<string, number | null> };

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
    rows.push({ ...partsOf(column.model), values, gammas, counts, summary: column.summary });
  }
  return rows;
}
