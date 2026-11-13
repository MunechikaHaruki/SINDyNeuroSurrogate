import { expect, test } from 'bun:test';
import { groupings, type Row } from './common.ts';
import { build, groupPlans } from './schema.ts';
import { claimTexts } from '../orchestrate/resolve.ts';
import { CLAIMS, EXPERIMENTS } from './experiments.ts';

const row = (grouping: string, method: string, gammas: (number | null)[], extra: Partial<Row> = {}): Row => ({
  grouping,
  method,
  values: gammas.map((_, k) => k),
  gammas,
  counts: gammas.map(() => [5, 5]),
  ...extra,
});
const oks = (name: string, tables: Row[][], groupings?: string[]) =>
  CLAIMS[name].judge(tables, groupings).map(j => j.ok);

const ac = [row('a+c', 'kam', [0.98]), row('a+c', 'pca-sindy', [0.91]), row('a+c', 'ae-sindy', [1])];
const hn = [row('h+n', 'kam', [1]), row('h+n', 'pca-sindy', [0.1]), row('h+n', 'ae-sindy', [1])];
const na = [row('n+a', 'kam', [0.04]), row('n+a', 'pca-sindy', [0]), row('n+a', 'ae-sindy', [0])];

test('only-a+c-keeps-spike-timing holds only when ac alone reaches the line and KAM breaks when n joins a gate other than h', () => {
  expect(oks('only-a+c-keeps-spike-timing', [[...ac, ...hn, ...na]])).toEqual([true, true, true]);
  expect(oks('only-a+c-keeps-spike-timing', [[...ac, ...hn.slice(0, 1), row('h+n', 'pca-sindy', [0.95]), hn[2], ...na]])).toEqual([
    true,
    false,
    true,
  ]);
  expect(oks('only-a+c-keeps-spike-timing', [[...ac, ...hn, row('n+a', 'kam', [0.95]), ...na.slice(1)]])).toEqual([true, true, false]);
});

test('a missing run cannot be judged, so a negative claim does not pass by default', () => {
  expect(oks('only-a+c-keeps-spike-timing', [[...ac, hn[0], hn[2], ...na]])).toEqual([true, null, true]);
  expect(oks('only-a+c-keeps-spike-timing', [[...ac, ...hn]])).toEqual([true, true, null]);
  expect(oks('only-a+c-keeps-spike-timing', [[...ac, ...ac, ...hn, ...na]])[0]).toBeNull();
});

test('fluctuating-input-keeps-timing-better reads the strongest fluctuation by its value, not its position', () => {
  const steady = [row('h+n', 'kam', [0.8]), row('h+n', 'pca-sindy', [0.1]), row('h+n', 'ae-sindy', [0.5])];
  const values = [30, 10];
  const kistler = [
    row('h+n', 'kam', [0.97, 1], { values }),
    row('h+n', 'pca-sindy', [0.2, 0.9], { values }),
    row('h+n', 'ae-sindy', [0.3, 0.9], { values }),
  ];
  expect(oks('fluctuating-input-keeps-timing-better', [steady, kistler])).toEqual([true, true]);
});

test('absorbing-m-loses-spike-timing judges the groupings of the figure', () => {
  const rows = [
    row('a+c', 'kam', [1]),
    row('a+c', 'kam-absorb-m', [0.3], { counts: [[6, 7]] }),
    row('h+n', 'kam', [0.5]),
    row('h+n', 'kam-absorb-m', [0.3], { counts: [[6, 9]] }),
  ];
  expect(oks('absorbing-m-loses-spike-timing', [rows], ['a+c'])).toEqual([true, true, true]);
  expect(oks('absorbing-m-loses-spike-timing', [rows], ['a+c', 'h+n'])).toEqual([false, false, true]);
});

/** 測った数値の書き方。説明と主張には書かず、判定が runs から出す。 */
const MEASURED = /\d\s*(mV|ms|µA|%|回)/;

test('every claim reads only series and groupings its experiment runs', () => {
  for (const e of EXPERIMENTS)
    for (const c of e.claims) {
      for (const name of c.series)
        expect(Object.keys(e.series), `実験 ${e.id} の主張 ${c.name} が読む系列 ${name} を実験が回さない`).toContain(
          name,
        );
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

test('an experiment runs the product of its methods, groupings and series, and a method can override its preset', () => {
  const plan = build(
    'test',
    't',
    {
      description: 'test',
      groupings: ['h+n', 'a+c'],
      methods: ['kam', { name: 'pca2', from: 'pca-sindy', d: 2 }],
      series: ['traub-1comp-steady'],
    },
    [],
  );
  expect(plan.pairs.map(p => `${p.model}×${p.series}`)).toEqual([
    'kam.h+n×traub-1comp-steady',
    'kam.a+c×traub-1comp-steady',
    'pca2.h+n×traub-1comp-steady',
    'pca2.a+c×traub-1comp-steady',
  ]);
  expect(plan.models.map(m => [m.method, m.grouping, m.train.config.groups[0].d])).toEqual([
    ['kam', 'h+n', 1],
    ['kam', 'a+c', 1],
    ['pca2', 'h+n', 2],
    ['pca2', 'a+c', 2],
  ]);
});

test('a sweep spans every grouping of its gates', () => {
  const names = groupings(['h', 'n', 'a', 'c']);
  expect(names.length).toBe(14);
  expect(names).toContain('h+n_a+c');
  expect(names).toContain('h+c_n+a');
  expect(names[names.length - 1]).toBe('h+n+a+c');
});

test('a run takes the setup and replaces only the fields it writes, groupings as a whole', () => {
  const doc = [
    { description: 'test' },
    {
      setup: { methods: ['kam'], gates: ['h', 'n'], series: ['traub-1comp-steady'] },
      runs: {
        all: { description: 'all' },
        one: { description: 'one', groupings: ['h+n'], series: ['traub-1comp-kistler'] },
      },
    },
  ];
  const [all, one] = groupPlans('t', doc, {}, {}).plans;
  expect([all.groupings, all.gates, Object.keys(all.series)]).toEqual([['h+n'], ['h', 'n'], ['traub-1comp-steady']]);
  expect([one.groupings, one.gates, Object.keys(one.series)]).toEqual([['h+n'], undefined, ['traub-1comp-kistler']]);
  expect(() => groupPlans('t', [{ description: 'x', extra: 1 }, doc[1]], {}, {})).toThrow();
});

test('every claim carries the docstring of its judgement as its sentence', () => {
  const texts = claimTexts();
  for (const c of Object.values(CLAIMS)) {
    expect(texts[c.name], `判定層の主張 ${c.name} に docstring が無い`).toBeTruthy();
    expect(c.claim).toBe(texts[c.name]);
  }
});
