import { expect, test } from 'bun:test';
import { groupings, type Row } from './common.ts';
import { CLAIMS, EXPERIMENTS, pairsOf } from './experiments.ts';

const row = (grouping: string, method: string, gammas: (number | null)[], extra: Partial<Row> = {}): Row => ({
  grouping,
  method,
  values: gammas.map((_, k) => k),
  gammas,
  counts: gammas.map(() => [5, 5]),
  summary: {},
  ...extra,
});
const oks = (name: string, tables: Row[][], groupings?: string[]) =>
  CLAIMS[name].judge(tables, groupings).map(j => j.ok);

const ac = [row('ac', 'kam', [0.98]), row('ac', 'pca', [0.91]), row('ac', 'ae', [1])];
const hn = [row('hn', 'kam', [1]), row('hn', 'pca', [0.1]), row('hn', 'ae', [1])];
const na = [row('na', 'kam', [0.04]), row('na', 'pca', [0]), row('na', 'ae', [0])];

test('ac-only holds only when ac alone reaches the line and KAM breaks when n joins a gate other than h', () => {
  expect(oks('ac-only', [[...ac, ...hn, ...na]])).toEqual([true, true, true]);
  expect(oks('ac-only', [[...ac, ...hn.slice(0, 1), row('hn', 'pca', [0.95]), hn[2], ...na]])).toEqual([
    true,
    false,
    true,
  ]);
  expect(oks('ac-only', [[...ac, ...hn, row('na', 'kam', [0.95]), ...na.slice(1)]])).toEqual([true, true, false]);
});

test('a missing run cannot be judged, so a negative claim does not pass by default', () => {
  expect(oks('ac-only', [[...ac, hn[0], hn[2], ...na]])).toEqual([true, null, true]);
  expect(oks('ac-only', [[...ac, ...hn]])).toEqual([true, true, null]);
  expect(oks('ac-only', [[...ac, ...ac, ...hn, ...na]])[0]).toBeNull();
});

test('fluctuating-input reads the strongest fluctuation by its value, not its position', () => {
  const steady = [row('hn', 'kam', [0.8]), row('hn', 'pca', [0.1]), row('hn', 'ae', [0.5])];
  const values = [30, 10];
  const kistler = [
    row('hn', 'kam', [0.97, 1], { values }),
    row('hn', 'pca', [0.2, 0.9], { values }),
    row('hn', 'ae', [0.3, 0.9], { values }),
  ];
  expect(oks('fluctuating-input', [steady, kistler])).toEqual([true, true]);
});

test('absorb-m judges the groupings of the figure', () => {
  const rows = [
    row('ac', 'kam', [1]),
    row('ac', 'kamm', [0.3], { counts: [[6, 7]] }),
    row('hn', 'kam', [0.5]),
    row('hn', 'kamm', [0.3], { counts: [[6, 9]] }),
  ];
  expect(oks('absorb-m', [rows], ['ac'])).toEqual([true, true, true]);
  expect(oks('absorb-m', [rows], ['ac', 'hn'])).toEqual([false, false, true]);
});

test('poster-shape compares every kind of operation and finds the unstimulated point by its value', () => {
  const poster = (summary: Record<string, number>) =>
    row('mhnabc', 'poster', [1, 0], {
      values: [10, 0],
      counts: [
        [3, 3],
        [6, 0],
      ],
      summary: { nnz_ratio: 0.9, ...summary },
    });
  expect(oks('poster-shape', [[poster({ 'cost/surr-orig/pm': 700, 'cost/surr-orig/mul': 0 })]])).toEqual([
    true,
    true,
    true,
  ]);
  expect(oks('poster-shape', [[poster({ 'cost/surr-orig/pm': 700, 'cost/surr-orig/mul': -1 })]])[1]).toBe(false);
  expect(oks('nadr-better', [[row('hn', 'kam', [0.6])], [row('hn', 'kam', [0.9])]])).toEqual([true]);
});

/** 測った数値の書き方。説明と主張には書かず、判定が runs から出す。 */
const MEASURED = /\d\s*(mV|ms|µA|%|回)/;

test('every claim reads only series and groupings its experiment runs', () => {
  for (const e of EXPERIMENTS)
    for (const c of e.claims) {
      for (const { name } of c.series)
        expect(e.series, `実験 ${e.id} の主張 ${c.name} が読む系列 ${name} を実験が回さない`).toContain(name);
      for (const g of c.groupings ?? [])
        expect(e.groupings, `実験 ${e.id} の主張 ${c.name} の群分け ${g} を回さない`).toContain(g);
    }
});

test('descriptions and claims carry no measured numbers', () => {
  for (const e of EXPERIMENTS) {
    const text = [e.description, ...e.claims.map(c => c.claim)].join(' ');
    expect(text, `実験 ${e.id} の説明か主張に測った数値がある（判定が出す）`).not.toMatch(MEASURED);
  }
});

test('an experiment runs the product of its groupings, methods and series', () => {
  const { pairs } = pairsOf({ groupings: ['hn', 'ac'], methods: ['kam', 'pca'], series: ['s'] });
  expect(pairs.map(p => `${p.model}×${p.series}`)).toEqual(['kam_hn×s', 'kam_ac×s', 'pca_hn×s', 'pca_ac×s']);
});

test('a sweep spans every grouping of its gates', () => {
  const names = groupings(['h', 'n', 'a', 'c']);
  expect(names.length).toBe(14);
  expect(names).toContain('hn_ac');
  expect(names).toContain('hc_na');
  expect(names[names.length - 1]).toBe('hnac');
});
