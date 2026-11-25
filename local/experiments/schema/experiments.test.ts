import { expect, test } from 'bun:test';
import { groupings } from './presets.ts';
import { build, groupPlans } from './schema.ts';

test('an experiment runs the product of its methods, groupings and series, and a method can be a named diff of a part', () => {
  const plan = build('test', 't', {
    groupings: ['h+n', 'a+c'],
    methods: ['kam', { name: 'pca2', from: 'pca-sindy', d: 2 }],
    series: ['traub-1comp-steady'],
  });
  expect(plan.pairs.map(p => `${p.model}×${p.series}`)).toEqual([
    'kam.h+n×traub-1comp-steady',
    'kam.a+c×traub-1comp-steady',
    'pca2.h+n×traub-1comp-steady',
    'pca2.a+c×traub-1comp-steady',
  ]);
  expect(plan.models.map(m => [m.method, m.grouping, m.config.groups[0].d])).toEqual([
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

test('a run takes the group base and may replace only the fields the group varies', () => {
  const doc = (vary: string[], run: Record<string, unknown>) => [
    { description: 'test' },
    {
      base: { methods: ['kam'], groupings: { all: ['h', 'n'] }, series: ['traub-1comp-steady'] },
      vary,
      runs: { base: null, run },
    },
  ];
  const [base, one] = groupPlans(
    't',
    doc(['groupings', 'series'], { groupings: ['h+n'], series: ['traub-1comp-kistler'] }),
  ).plans;
  expect([base.groupings, Object.keys(base.series)]).toEqual([['h+n'], ['traub-1comp-steady']]);
  expect([one.groupings, Object.keys(one.series)]).toEqual([['h+n'], ['traub-1comp-kistler']]);
  expect(() => groupPlans('t', doc(['series'], { groupings: ['h+n'] }))).toThrow(/groupings/);
  expect(() => groupPlans('t', doc(['nothing'], {}))).toThrow();
  expect(() => groupPlans('t', [{ description: 'x', extra: 1 }, doc([], {})[1]])).toThrow();
  const old = doc([], {});
  expect(() => groupPlans('t', [old[0], { ...old[1], setup: {} }])).toThrow();
});

test('a run merges its training input field by field over the group base', () => {
  const [plan] = groupPlans('t', [
    { description: 'test' },
    {
      base: {
        methods: ['kam'],
        groupings: ['h+n'],
        series: ['traub-1comp-steady'],
        train: { target: 'traub19', duration: 1000 },
      },
      vary: ['train'],
      runs: { seeded: { train: { seed: 1 } } },
    },
  ]).plans;
  expect(plan.models[0].config.data).toMatchObject({ target: 'traub19', current_params: { duration: 1000, seed: 1 } });
});

test('the training input is a setting of its own, shared by every model of a run', () => {
  const plan = build('test', 't', {
    methods: ['kam', 'pca-sindy'],
    groupings: ['h+n'],
    series: ['traub-1comp-steady'],
    train: { duration: 1000, seed: 3 },
  });
  expect(plan.models.map(m => m.config.data)).toEqual(
    Array(2).fill({ target: 'traub', current_type: 'train', dt: 0.01, current_params: { duration: 1000, seed: 3 } }),
  );
  const nineteen = build('test', 't', {
    methods: ['kam'],
    groupings: ['h+n'],
    series: ['traub-1comp-steady'],
    train: { target: 'traub19' },
  });
  expect(nineteen.models[0].config.data.target).toBe('traub19');
  expect(nineteen.models[0].config.data.current_params).toEqual({ duration: 9000 });
});

test('an input is laid over each replacement range, unless a series pins its range with @', () => {
  const plan = build('test', 't', {
    methods: ['kam'],
    groupings: ['h+n'],
    series: ['traub-19comp-steady', 'traub-1comp-steady@soma'],
    replace: ['soma', 'na-dr'],
  });
  expect(Object.entries(plan.series).map(([name, s]) => [name, s.replace_targets.length])).toEqual([
    ['traub-19comp-steady@soma', 1],
    ['traub-19comp-steady@na-dr', 5],
    ['traub-1comp-steady@soma', 1],
  ]);
  expect(plan.series['traub-19comp-steady@soma'].spec).toEqual(plan.series['traub-19comp-steady@na-dr'].spec);
  const single = build('test', 't', { methods: ['kam'], groupings: ['h+n'], series: ['traub-1comp-steady'] });
  expect(Object.keys(single.series)).toEqual(['traub-1comp-steady']);
  expect(() =>
    build('test', 't', { methods: ['kam'], groupings: ['h+n'], series: ['traub-1comp-steady@nowhere'] }),
  ).toThrow();
});
