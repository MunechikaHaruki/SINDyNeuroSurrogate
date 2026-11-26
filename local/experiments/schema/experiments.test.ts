import { expect, test } from 'bun:test';
import { groupings, inputOf, partOf, patched } from './base.ts';
import { build, groupPlans } from './schema.ts';

/** 系列の基準からの差分（宣言の YAML と同じ書き方）。名前は traub-steady@soma と中身から付く。 */
const STEADY = { sweep: { value: [0, 5] } };

test('an experiment runs the product of its learners, groupings and inputs', () => {
  const plan = build('test', 't', {
    learners: ['kam', 'pca-sindy-d2'],
    groupings: ['h+n', 'a+c'],
    inputs: [STEADY],
  });
  expect(plan.pairs.map(p => `${p.model}×${p.series}`)).toEqual([
    'kam.h+n×traub-steady@soma',
    'kam.a+c×traub-steady@soma',
    'pca-sindy-d2.h+n×traub-steady@soma',
    'pca-sindy-d2.a+c×traub-steady@soma',
  ]);
  expect(plan.models.map(m => [m.method, m.grouping, m.config.groups[0].d])).toEqual([
    ['kam', 'h+n', 1],
    ['kam', 'a+c', 1],
    ['pca-sindy-d2', 'h+n', 2],
    ['pca-sindy-d2', 'a+c', 2],
  ]);
  expect(plan.series['traub-steady@soma'].sweep).toEqual({ param: 'value', values: [0, 5] });
  const single = build('test', 't', { learners: ['pca-sindy'], groupings: ['h+n'], inputs: [{}] });
  expect(single.series['traub-steady@soma'].sweep).toBeNull();
});

test('every grouping of the gates can be spanned', () => {
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
  const ude = partOf({ group: { compressor: 'ae', rhs: 'ude', rhs_params: { library_specs: null, epochs: 3 } } });
  expect(ude.group.rhs_params).toEqual({ epochs: 3 });
});

test("the experiment's train is laid over every learner's training current", () => {
  const plan = build('test', 't', {
    learners: ['kam', 'pca-sindy'],
    train: { target: 'traub19', current_params: { duration: 1000, seed: 2 } },
    groupings: ['h+n'],
    inputs: [STEADY],
  });
  for (const m of plan.models)
    expect(m.config.data).toMatchObject({ target: 'traub19', current_params: { duration: 1000, seed: 2 } });
});

test('a diff may only write what the sent shape has, and errors name where they come from', () => {
  expect(() => partOf({ grop: { d: 2 } }, [], 'ここ：', 'x')).toThrow(/^ここ：学習器 x に差分/);
  expect(() => partOf({ group: { compresor: 'ae' } })).toThrow();
  expect(() => partOf({}, [{ data: { traget: 'traub' } }])).toThrow();
  expect(() => inputOf({ replace: ['soma'] })).toThrow();
  expect(inputOf({ spec: { current_params: { seed: 3 } } }).series.spec.current_params).toMatchObject({
    duration: 300,
    seed: 3,
  });
});

test('the declaration file lists groups of whole experiments in the written order, and anything else is refused', () => {
  const doc = (groups: unknown) => [{ description: 'test' }, { groups }];
  const one = { learners: ['kam'], groupings: ['h+n'], inputs: [STEADY] };
  const groups = groupPlans(doc({ b: { description: 'B', experiments: { e: one } }, a: { description: 'A', experiments: { f: one } } }));
  expect(groups.map(g => [g.name, g.description, g.plans.map(p => p.id)])).toEqual([
    ['b', 'B', ['e']],
    ['a', 'A', ['f']],
  ]);
  const wrong = [
    doc({}),
    doc({ g: { description: 'x', experiments: {} } }),
    doc({ g: { experiments: { e: one } } }),
    doc({ g: { description: 'x', experiments: { e: { learners: ['kam'], groupings: ['h+n'] } } } }),
    doc({ g: { description: 'x', experiments: { e: { ...one, part: {} } } } }),
    doc({ g: { description: 'x', experiments: { e: { ...one, learners: ['kam-sindy'] } } } }),
    [{ description: 'x', extra: 1 }, { groups: { g: { description: 'x', experiments: { e: one } } } }],
  ];
  for (const w of wrong) expect(() => groupPlans(w)).toThrow();
});

test('series that differ only in what they replace share the current, and series names are unique', () => {
  const plan = build('test', 't', {
    learners: ['kam'],
    groupings: ['h+n'],
    inputs: [
      { spec: { target: 'traub19' } },
      { spec: { target: 'traub19' }, replace_targets: ['c05', 'c07', 'soma', 'c09', 'c11'] },
    ],
  });
  expect(Object.keys(plan.series)).toEqual(['traub19-steady@soma', 'traub19-steady@c05+c07+soma+c09+c11']);
  expect(plan.series['traub19-steady@soma'].spec).toEqual(plan.series['traub19-steady@c05+c07+soma+c09+c11'].spec);
  expect(() => build('test', 't', { learners: ['kam'], groupings: ['h+n'], inputs: [STEADY, STEADY] })).toThrow(/二つ/);
});

test('a bench measures the original and every model on each of its series', () => {
  const plan = build('test', 't', {
    learners: ['kam'],
    groupings: ['h+n'],
    inputs: [STEADY],
    bench: { series: ['traub-steady@soma'], cells: [1024], ncu: false },
  });
  expect(plan.benches.map(b => b.name)).toEqual(['original@traub-steady@soma', 'kam.h+n@traub-steady@soma']);
  const missing = { series: ['nowhere'], cells: [1024] as [number], ncu: false };
  expect(() => build('test', 't', { learners: ['kam'], groupings: ['h+n'], inputs: [STEADY], bench: missing })).toThrow();
});
