// 主張の判定。判定ごとに読む系列を宣言し、点ごとの Γ と発火の回数、学習の summary から、値・線・成り立つかを出す。
// 成り立つかは三値で、値が欠けていれば null（判定できない）。系列を読んで列にするのは API（serve.ts の claims）。
import { partsOf } from '../orchestrate/conf.ts';

/** Γ で「発火の時刻まで合う」とみなす線。 */
export const GAMMA_LINE = 0.9;

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

/** 判定。series は読む系列（comp を書かなければ系列の置き換える最初の区画）。judge は系列ごとの列と、図が絞った群分けで判定する。 */
export type Check = {
  series: { name: string; comp?: string }[];
  judge: (tables: Row[][], groupings?: string[]) => Judgement[];
};

/** 比べる縮約の仕方。 */
const METHODS = ['kam', 'pca', 'ae'];
const NAMES: Record<string, string> = {
  kam: 'KAM',
  kamm: 'KAM（m を吸収）',
  pca: 'PCA',
  ae: 'AE',
  poster: 'NEURO2026',
};

/** 群分けと仕方の列。無いか 2 本以上あれば undefined（どれを読むか決まらない）。 */
function one(rows: Row[], grouping: string, method: string): Row | undefined {
  const found = rows.filter(r => r.grouping === grouping && r.method === method);
  return found.length === 1 ? found[0] : undefined;
}
const mean = (row: Row | undefined) => (row ? meanGamma(row.gammas) : null);
const shown = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v.toFixed(2));
const above = (v: number | null | undefined) => (v === null || v === undefined ? null : v >= GAMMA_LINE);
const below = (v: number | null | undefined) => (v === null || v === undefined ? null : v < GAMMA_LINE);
/** 三値の「すべて」。一つでも偽なら偽、偽が無く欠けがあれば null。 */
const all = (xs: (boolean | null)[]) => (xs.includes(false) ? false : xs.includes(null) ? null : true);
/** 「当てはまる組が 0 個」。当てはまる組があれば偽、無くても決まらない組があれば null。 */
function none(found: string[], unknown: string[]): Pick<Judgement, 'value' | 'ok'> {
  const value =
    found.length > 0 ? found.join('・') : unknown.length > 0 ? `なし（判定できない組 ${unknown.join('・')}）` : 'なし';
  return { value, ok: found.length > 0 ? false : unknown.length > 0 ? null : true };
}
const groupingsOf = (rows: Row[]) => [...new Set(rows.map(r => r.grouping))];

/** ゲートの組と再現：ac は KAM・PCA・AE のどれでも線以上で、ほかにそういう組は無い。KAM で n を h 以外のゲートとまとめた組は線を下回る。 */
function acOnly([rows]: Row[][]): Judgement[] {
  const allAbove = (g: string) => all(METHODS.map(m => above(mean(one(rows, g, m)))));
  const others = groupingsOf(rows).filter(g => g !== 'ac');
  const nMixed = groupingsOf(rows).filter(g =>
    g.split('_').some(block => block.includes('n') && [...block].some(x => x !== 'n' && x !== 'h')),
  );
  const nGammas = nMixed.map(g => mean(one(rows, g, 'kam')));
  return [
    {
      label: 'ac の平均 Γ（KAM・PCA・AE）',
      value: METHODS.map(m => `${NAMES[m]} ${shown(mean(one(rows, 'ac', m)))}`).join('・'),
      line: `3 つとも ≥ ${GAMMA_LINE}`,
      ok: allAbove('ac'),
    },
    {
      label: 'ac 以外で 3 つとも線以上の組',
      line: '0 個',
      ...none(
        others.filter(g => allAbove(g) === true),
        others.filter(g => allAbove(g) === null),
      ),
    },
    {
      label: 'KAM で n を h 以外とまとめた組の平均 Γ の最大',
      value: `${shown(Math.max(...nGammas.map(v => v ?? -Infinity)))}（${nMixed.length} 組）`,
      line: `どれも < ${GAMMA_LINE}`,
      ok: nMixed.length === 0 ? null : all(nGammas.map(below)),
    },
  ];
}

/** 入力と再現：hn は 3 つの仕方のどれでも Kistler の揺らぐ電流の方が平均 Γ が高く、最も強い揺らぎで線以上なのは KAM だけ。 */
function fluctuatingInput([steady, kistler]: Row[][]): Judgement[] {
  const rises = METHODS.map(m => ({ m, from: mean(one(steady, 'hn', m)), to: mean(one(kistler, 'hn', m)) }));
  const values = kistler[0]?.values ?? [];
  const strongest = values.indexOf(Math.max(...values));
  const atStrongest = METHODS.map(m => ({ m, ok: above(one(kistler, 'hn', m)?.gammas[strongest]) }));
  const passing = atStrongest.filter(a => a.ok === true).map(a => a.m);
  return [
    {
      label: 'hn の平均 Γ（定常電流 → 揺らぐ電流）',
      value: rises.map(r => `${NAMES[r.m]} ${shown(r.from)} → ${shown(r.to)}`).join('・'),
      line: '3 つとも上がる',
      ok: all(rises.map(r => (r.from === null || r.to === null ? null : r.to > r.from))),
    },
    {
      label: `最も強い揺らぎ（${values[strongest] ?? '—'}）で線以上の仕方`,
      value: passing.length > 0 ? passing.map(m => NAMES[m]).join('・') : 'なし',
      line: 'KAM だけ',
      ok: strongest < 0 || atStrongest.some(a => a.ok === null) ? null : passing.length === 1 && passing[0] === 'kam',
    },
  ];
}

/** m の吸収：図の組は吸収しない KAM で線以上で、吸収すると発火の回数の差はどの点でも一つ以内のまま、平均 Γ が線を下回る。 */
function absorbM([rows]: Row[][], groupings: string[] = groupingsOf(rows)): Judgement[] {
  const pairs = groupings.map(g => ({ g, kam: one(rows, g, 'kam'), kamm: one(rows, g, 'kamm') }));
  return [
    {
      label: '吸収しない KAM の平均 Γ',
      value: pairs.map(p => `${p.g} ${shown(mean(p.kam))}`).join('・'),
      line: `どれも ≥ ${GAMMA_LINE}`,
      ok: all(pairs.map(p => above(mean(p.kam)))),
    },
    {
      label: '吸収した KAM の発火の回数の差の最大',
      value: pairs
        .map(p => `${p.g} ${p.kamm ? Math.max(...p.kamm.counts.map(([o, m]) => Math.abs(o - m))) : '—'}`)
        .join('・'),
      line: 'どれも ≤ 1',
      ok: all(pairs.map(p => (p.kamm ? p.kamm.counts.every(([o, m]) => Math.abs(o - m) <= 1) : null))),
    },
    {
      label: '吸収した KAM の平均 Γ',
      value: pairs.map(p => `${p.g} ${shown(mean(p.kamm))}`).join('・'),
      line: `どれも < ${GAMMA_LINE}`,
      ok: all(pairs.map(p => below(mean(p.kamm)))),
    },
  ];
}

/** 多区画への当てはめ：hn の KAM の soma の平均 Γ は、Na と DR を持つ 5 区画を置き換えた方が soma だけより高い。 */
function naDrBetter([soma, nadr]: Row[][]): Judgement[] {
  const from = mean(one(soma, 'hn', 'kam'));
  const to = mean(one(nadr, 'hn', 'kam'));
  return [
    {
      label: 'soma の平均 Γ（soma だけ → 5 区画）',
      value: `${shown(from)} → ${shown(to)}`,
      line: '上がる',
      ok: from === null || to === null ? null : to > from,
    },
  ];
}

/** 全ゲートを一つにまとめる形：係数の半分超が非零で、1 刻みの演算がどの種類も元より減らずどれかが増え、刺激 0 で元は発火しサロゲートの時刻は合わない。 */
function posterShape([rows]: Row[][]): Judgement[] {
  const row = one(rows, 'mhnabc', 'poster');
  const nnz = row?.summary['nnz_ratio'] ?? null;
  const costs = Object.entries(row?.summary ?? {}).filter(([k]) => k.startsWith('cost/surr-orig/'));
  const zero = row ? row.values.indexOf(0) : -1;
  const [original] = zero >= 0 ? (row?.counts[zero] ?? [0, 0]) : [0, 0];
  const gamma = zero >= 0 ? (row?.gammas[zero] ?? null) : null;
  return [
    { label: '非零の係数の割合', value: shown(nnz), line: '> 0.5', ok: nnz === null ? null : nnz > 0.5 },
    {
      label: '1 刻みの演算の、元との差（種類ごと）',
      value:
        costs
          .map(([k, v]) => `${k.slice('cost/surr-orig/'.length)} ${v === null ? '—' : (v > 0 ? '+' : '') + v}`)
          .join('・') || '—',
      line: 'どれも ≥ 0、どれかが > 0',
      ok:
        costs.length === 0 || costs.some(([, v]) => v === null)
          ? null
          : costs.every(([, v]) => v! >= 0) && costs.some(([, v]) => v! > 0),
    },
    {
      label: '刺激 0 の、元の発火の回数と Γ',
      value: zero < 0 ? '刺激 0 の点が無い' : `${original}、Γ ${shown(gamma)}`,
      line: `元 > 0、Γ < ${GAMMA_LINE}`,
      ok: zero < 0 ? null : original > 0 ? below(gamma) : false,
    },
  ];
}

/** 判定の名前 → 判定。conf.json の実験の主張の check が名前で引く。 */
export const CHECKS: Record<string, Check> = {
  'ac-only': { series: [{ name: 'traub_somastim' }], judge: acOnly },
  'fluctuating-input': { series: [{ name: 'traub_somastim' }, { name: 'traub_kistler' }], judge: fluctuatingInput },
  'absorb-m': { series: [{ name: 'traub_somastim' }], judge: absorbM },
  'nadr-better': {
    series: [
      { name: 'traub19_somastim', comp: 'soma' },
      { name: 'traub19_somastim_nadr', comp: 'soma' },
    ],
    judge: naDrBetter,
  },
  'poster-shape': { series: [{ name: 'traub19_somastim', comp: 'soma' }], judge: posterShape },
};

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
