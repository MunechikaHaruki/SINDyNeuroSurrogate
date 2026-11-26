import { expect, test } from 'bun:test';
import { groupings, inputOf, partOf, patched } from './base.ts';
import { build, groupPlans } from './schema.ts';

/** 基準 traub からの差分（実験群の YAML と同じ書き方）。 */
const KAM = { name: 'kam', from: 'traub', group: { compressor: 'kam', rhs: 'kam', rhs_params: { library_specs: null } } };
const STEADY = { name: 'steady', from: 'traub', sweep: { value: [0, 5] } };
const STEADY_19 = { name: 'steady-19', from: 'traub', spec: { target: 'traub19' } };

test('an experiment runs the product of its learners, groupings and inputs, and a learner can be a named diff of a base file', () => {
  const plan = build('test', 't', {
    groupings: ['h+n', 'a+c'],
    parts: [KAM, { name: 'pca2', from: 'traub', group: { d: 2 } }],
    inputs: [STEADY],
  });
  expect(plan.pairs.map(p => `${p.model}×${p.series}`)).toEqual([
    'kam.h+n×steady',
    'kam.a+c×steady',
    'pca2.h+n×steady',
    'pca2.a+c×steady',
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
    from: 'traub',
    group: { compressor: 'ae', rhs: 'ude', rhs_params: { library_specs: null, epochs: 3 } },
  });
  expect(ude.group.rhs_params).toEqual({ epochs: 3 });
});

test("a diff that applies to every learner is laid after the learner's own diff, and errors name where they come from", () => {
  const part = partOf({ ...KAM, name: 'x', data: { current_params: { duration: 500 } } }, [
    { data: { current_params: { duration: 1000 } } },
  ]);
  expect(part.data.current_params.duration).toBe(1000);
  expect(() => partOf({ name: 'x', from: 'traub', grop: 1 }, [], 'ここ：')).toThrow(/^ここ：学習器 traub に差分/);
  expect(() => partOf('traub', [{ data: { traget: 'traub' } }])).toThrow();
});

test('a diff may only write what the sent shape has', () => {
  expect(() => partOf({ name: 'x', from: 'traub', grop: { d: 2 } })).toThrow();
  expect(() => partOf({ name: 'x', from: 'traub', group: { compresor: 'ae' } })).toThrow();
  expect(() => inputOf({ name: 'x', from: 'traub', replace: ['soma'] })).toThrow();
  expect(() => partOf('nowhere')).toThrow(/config\/parts/);
  expect(
    inputOf({ name: 'x', from: 'traub', spec: { current_params: { seed: 3 } } }).spec.current_params,
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
        parts: [KAM],
        part: { data: { target: 'traub19', current_params: { duration: 1000 } } },
        groupings: { all: ['h', 'n'] },
        inputs: [STEADY],
      },
      runs: {
        base: null,
        run: {
          part: { data: { current_params: { seed: 1 } } },
          inputs: [{ name: 'kistler', from: 'traub', spec: { current_type: 'random&kistler' }, sweep: { sigma: [10] } }],
        },
      },
    },
  ]).plans;
  expect([base.groupings, Object.keys(base.series)]).toEqual([['h+n'], ['steady']]);
  expect(base.series.steady.sweep).toEqual({ param: 'value', values: [0, 5] });
  expect(Object.keys(run.series)).toEqual(['kistler']);
  expect(run.models[0].config.data).toMatchObject({ target: 'traub19', current_params: { duration: 1000, seed: 1 } });
  const [cleared] = groupPlans('t', [
    { description: 'test' },
    {
      base: {
        parts: [{ ...KAM, absorbed: ['M'] }],
        part: { data: { current_params: { duration: 1000 } } },
        groupings: ['h+n'],
        inputs: [STEADY],
        input: { spec: { current_params: { seed: 1 } } },
      },
      runs: { cleared: { part: { absorbed: null }, input: { spec: { current_params: { seed: null } } } } },
    },
  ]).plans;
  expect('absorbed' in cleared.models[0].config).toBe(false);
  expect('seed' in cleared.series.steady.spec.current_params).toBe(false);
  expect(() => groupPlans('t', [{ description: 'x' }, { base: {}, runs: { r: { methods: [KAM] } } }])).toThrow();
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
    parts: [KAM],
    groupings: ['h+n'],
    inputs: [STEADY_19],
    replace: { soma: ['soma'], 'na-dr': ['c05', 'c07', 'soma', 'c09', 'c11'] },
  });
  expect(Object.entries(plan.series).map(([name, s]) => [name, s.replace_targets.length])).toEqual([
    ['steady-19@soma', 1],
    ['steady-19@na-dr', 5],
  ]);
  expect(plan.series['steady-19@soma'].spec).toEqual(plan.series['steady-19@na-dr'].spec);
  const one = (inputs: unknown[]) =>
    build('test', 't', { parts: [KAM], groupings: ['h+n'], inputs: inputs as never });
  expect(Object.keys(one([STEADY]).series)).toEqual(['steady']);
  expect(() => one([STEADY, STEADY])).toThrow(/二つ/);
  const pinned = build('test', 't', {
    parts: [KAM],
    groupings: ['h+n'],
    inputs: [{ name: 'x', from: 'traub', spec: { target: 'traub19' }, replace_targets: ['c05'] }],
    replace: { soma: ['soma'] },
  });
  expect(pinned.series['x@soma'].replace_targets).toEqual(['soma']);
  expect(() =>
    build('test', 't', {
      parts: [KAM],
      groupings: ['h+n'],
      inputs: [{ name: 'a@b', from: 'traub' }],
      replace: { soma: ['soma'] },
    }),
  ).toThrow(/@/);
});
