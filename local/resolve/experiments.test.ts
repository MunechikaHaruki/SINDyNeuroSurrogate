import { expect, test } from 'bun:test';
import { groupings } from './presets.ts';
import { build, groupPlans } from './schema.ts';
import { EXPERIMENTS } from './experiments.ts';

/** 測った数値の書き方。説明には書かず、結果は図と表が runs から出す。 */
const MEASURED = /\d\s*(mV|ms|µA|%|回)/;

test('descriptions carry no measured numbers', () => {
  for (const e of EXPERIMENTS)
    expect(e.description, `実験 ${e.id} の説明に測った数値がある（結果は図と表が出す）`).not.toMatch(MEASURED);
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
  const [all, one] = groupPlans('t', doc).plans;
  expect([all.groupings, all.gates, Object.keys(all.series)]).toEqual([['h+n'], ['h', 'n'], ['traub-1comp-steady']]);
  expect([one.groupings, one.gates, Object.keys(one.series)]).toEqual([['h+n'], undefined, ['traub-1comp-kistler']]);
  expect(() => groupPlans('t', [{ description: 'x', extra: 1 }, doc[1]])).toThrow();
});
