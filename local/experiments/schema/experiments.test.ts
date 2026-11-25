import { expect, test } from 'bun:test';
import { groupings, inputOf, partOf, patched } from './base.ts';
import { build, groupPlans } from './schema.ts';

test('an experiment runs the product of its learners, groupings and inputs, and a learner can be a named diff of a base file', () => {
  const plan = build('test', 't', {
    groupings: ['h+n', 'a+c'],
    parts: ['kam', { name: 'pca2', from: 'pca-sindy', group: { d: 2 } }],
    inputs: ['traub-1comp-steady'],
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

test('a diff is a JSON merge patch: objects merge field by field, null removes a field, arrays and values replace', () => {
  expect(patched({ a: { b: 1, c: [1, 2] }, d: 1 }, { a: { c: [3] }, d: null, e: 2 })).toEqual({
    a: { b: 1, c: [3] },
    e: 2,
  });
  const ude = partOf({
    name: 'u',
    from: 'ae-sindy',
    group: { rhs: 'ude', rhs_params: { library_specs: null, epochs: 3 } },
  });
  expect(ude.group.rhs_params).toEqual({ epochs: 3 });
});

test("a diff that applies to every learner is laid after the learner's own diff, and errors name where they come from", () => {
  const part = partOf({ name: 'x', from: 'kam', data: { current_params: { duration: 500 } } }, [
    { data: { current_params: { duration: 1000 } } },
  ]);
  expect(part.data.current_params.duration).toBe(1000);
  expect(() => partOf({ name: 'x', from: 'kam', grop: 1 }, [], 'ここ：')).toThrow(/^ここ：学習器 kam に差分/);
  expect(() => partOf('kam', [{ data: { traget: 'traub' } }])).toThrow();
});

test('a diff may only write what the sent shape has', () => {
  expect(() => partOf({ name: 'x', from: 'pca-sindy', grop: { d: 2 } })).toThrow();
  expect(() => partOf({ name: 'x', from: 'pca-sindy', group: { compresor: 'ae' } })).toThrow();
  expect(() => inputOf({ name: 'x', from: 'traub-1comp-steady', replace: ['soma'] })).toThrow();
  expect(() => partOf('nowhere')).toThrow(/config\/parts/);
  expect(
    inputOf({ name: 'x', from: 'traub-1comp-steady', spec: { current_params: { seed: 3 } } }).spec.current_params,
  ).toMatchObject({
    duration: 300,
    seed: 3,
  });
});

test('a run takes the group base, replaces the fields it writes, and lays its diffs over the group diffs', () => {
  const [base, run] = groupPlans('t', [
    { description: 'test' },
    {
      base: {
        parts: ['kam'],
        part: { data: { target: 'traub19', current_params: { duration: 1000 } } },
        groupings: { all: ['h', 'n'] },
        inputs: ['traub-1comp-steady'],
      },
      runs: { base: null, run: { part: { data: { current_params: { seed: 1 } } }, inputs: ['traub-1comp-kistler'] } },
    },
  ]).plans;
  expect([base.groupings, Object.keys(base.series)]).toEqual([['h+n'], ['traub-1comp-steady']]);
  expect(Object.keys(run.series)).toEqual(['traub-1comp-kistler']);
  expect(run.models[0].config.data).toMatchObject({ target: 'traub19', current_params: { duration: 1000, seed: 1 } });
  const [cleared] = groupPlans('t', [
    { description: 'test' },
    {
      base: {
        parts: ['kam-absorb-m'],
        part: { data: { current_params: { duration: 1000 } } },
        groupings: ['h+n'],
        inputs: ['traub-1comp-steady'],
      },
      runs: { cleared: { part: { absorbed: null }, input: { values: [0] } } },
    },
  ]).plans;
  expect('absorbed' in cleared.models[0].config).toBe(false);
  expect(cleared.series['traub-1comp-steady'].values).toEqual([0]);
  expect(() => groupPlans('t', [{ description: 'x' }, { base: {}, runs: { r: { methods: ['kam'] } } }])).toThrow();
  expect(() => groupPlans('t', [{ description: 'x' }, { base: {}, vary: [], runs: { r: null } }])).toThrow();
  expect(() =>
    groupPlans('t', [
      { description: 'x', extra: 1 },
      { base: {}, runs: { r: null } },
    ]),
  ).toThrow();
});

test('an input is laid over each replacement range, named <input>@<range>', () => {
  const plan = build('test', 't', {
    parts: ['kam'],
    groupings: ['h+n'],
    inputs: ['traub-19comp-steady'],
    replace: { soma: ['soma'], 'na-dr': ['c05', 'c07', 'soma', 'c09', 'c11'] },
  });
  expect(Object.entries(plan.series).map(([name, s]) => [name, s.replace_targets.length])).toEqual([
    ['traub-19comp-steady@soma', 1],
    ['traub-19comp-steady@na-dr', 5],
  ]);
  expect(plan.series['traub-19comp-steady@soma'].spec).toEqual(plan.series['traub-19comp-steady@na-dr'].spec);
  const one = (inputs: unknown[]) =>
    build('test', 't', { parts: ['kam'], groupings: ['h+n'], inputs: inputs as never });
  expect(Object.keys(one(['traub-1comp-steady']).series)).toEqual(['traub-1comp-steady']);
  expect(() => one(['traub-1comp-steady', 'traub-1comp-steady'])).toThrow(/二つ/);
  const pinned = build('test', 't', {
    parts: ['kam'],
    groupings: ['h+n'],
    inputs: [{ name: 'x', from: 'traub-19comp-steady', replace_targets: ['c05'] }],
    replace: { soma: ['soma'] },
  });
  expect(pinned.series['x@soma'].replace_targets).toEqual(['soma']);
  expect(() =>
    build('test', 't', {
      parts: ['kam'],
      groupings: ['h+n'],
      inputs: [{ name: 'a@b', from: 'traub-1comp-steady' }],
      replace: { soma: ['soma'] },
    }),
  ).toThrow(/@/);
});
