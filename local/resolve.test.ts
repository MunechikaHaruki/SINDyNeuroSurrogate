import { expect, test } from 'bun:test';
import { TESTS, groupsOf, inputOf, modelFrom, partOf } from './experiments/schema/base.ts';
import { hashOf, inputKey } from '../shared/hash.ts';
import { pointsOf } from '../shared/spec.ts';
import { modelSpec, partSpec, resolved, stale, simSpec, trainKey } from './resolve.ts';
import { EXPERIMENTS, withTraining } from './experiments/schema/experiments.ts';

/** 学習器の基準 traub からの差分（実験群の YAML と同じ書き方）。 */
const KAM = { name: 'kam', from: 'traub', group: { compressor: 'kam', rhs: 'kam', rhs_params: { library_specs: null } } };
const KAM_M = { ...KAM, name: 'kam-absorb-m', absorbed: ['M'] };
const AE = { name: 'ae-sindy', from: 'traub', group: { compressor: 'ae' } };

test('the model hash does not depend on the order the config was written in', () => {
  const train = TESTS._test_hh_sindy;
  const { config } = train;
  const reordered = {
    train_comp: config.train_comp,
    groups: config.groups,
    data: config.data,
    form: config.form,
    cell: config.cell,
  };
  expect(hashOf(modelSpec({ name: train.name, config: reordered }))).toBe(hashOf(modelSpec(train)));
});

test('a learner is shared by every model that has its group, whatever the other groups are', () => {
  const alone = modelFrom('pca-sindy.h+n', partOf('traub'), 'h+n');
  const paired = modelFrom('pca-sindy.h+n_a+c', partOf('traub'), 'h+n_a+c');
  const absorbed = modelFrom('kam-absorb-m.h+n', partOf(KAM_M), 'h+n');
  const kam = modelFrom('kam.h+n', partOf(KAM), 'h+n');
  expect(hashOf(partSpec(paired, paired.config.groups[0]))).toBe(hashOf(partSpec(alone, alone.config.groups[0])));
  expect(hashOf(modelSpec(absorbed))).not.toBe(hashOf(modelSpec(kam)));
  const model = modelSpec(paired);
  if (model.kind !== 'model') throw new Error('model でない');
  expect(model.parts).toHaveLength(2);
});

test('a KAM model has no learners and holds its groups and training input itself, and KAM is not mixed with learners', () => {
  const kam = modelFrom('kam.h+n_a+c', partOf(KAM), 'h+n_a+c');
  const model = modelSpec(kam);
  if (model.kind !== 'model') throw new Error('model でない');
  expect(model.parts).toEqual([]);
  expect(model.kam?.groups.map(g => g.gates)).toEqual([
    ['H', 'N'],
    ['A', 'C'],
  ]);
  expect(() => partSpec(kam, kam.config.groups[0])).toThrow();
  const plan = EXPERIMENTS.find(e => e.models.some(m => m.method === 'kam'))!;
  const paths = Object.keys(resolved(plan));
  expect(paths.some(path => path.startsWith(`${plan.id}/parts/kam`))).toBe(false);
  expect(model.kam?.input).toBe(trainKey(kam));
});

test('the training data is the original run of a training series the experiment runs, and every model is also run on it', () => {
  for (const plan of EXPERIMENTS) {
    const files = resolved(plan);
    const originals = new Map<string, string>();
    for (const [path, spec] of Object.entries(files))
      if (spec.kind === 'sim' && spec.model === null) originals.set(inputKey(spec), path);
    for (const m of plan.models) {
      const original = originals.get(trainKey(m));
      expect(original, `${plan.id} の ${m.name} の学習の入力を回さない`).toBeDefined();
      expect(files[original!.replace('/original/', `/${m.name}/`)]).toBeDefined();
    }
  }
});

test('an experiment holds one file per distinct learner', () => {
  for (const plan of EXPERIMENTS) {
    const files = Object.entries(resolved(plan));
    const hashes = files.filter(([path]) => path.startsWith(`${plan.id}/parts/`)).map(([, spec]) => hashOf(spec));
    expect(new Set(hashes).size, `${plan.id} の parts/ に同じ中身のファイルがある`).toBe(hashes.length);
  }
});

test('series that differ only in what they replace send the same input', () => {
  const soma = inputOf({ name: 'soma', from: 'traub', spec: { target: 'traub19' } });
  const all = inputOf({ name: 'all', from: 'traub', spec: { target: 'traub19' }, replace_targets: ['soma', 'c00'] });
  const [a, b] = [soma, all].map(s => simSpec(pointsOf(s)[0].input, s.replace_targets, 't'));
  if (a.kind !== 'sim' || b.kind !== 'sim') throw new Error('sim でない');
  expect(a.input).toEqual(b.input);
  expect(a.model).not.toEqual(b.model);
});

test('a sweep opens into one input per value, named <param>=<value>, and a series without one is the single point', () => {
  const swept = inputOf({ name: 's', from: 'traub', sweep: { value: [0, 2.5] } });
  expect(pointsOf(swept).map(p => [p.name, p.input.spec.current_params.value])).toEqual([
    ['value=0', 0],
    ['value=2.5', 2.5],
  ]);
  expect(pointsOf(inputOf('traub')).map(p => p.name)).toEqual(['single']);
  const plan = EXPERIMENTS.find(e => e.id === 'test-1comp')!;
  const paths = Object.keys(resolved(plan)).filter(p => p.includes('/sim/traub-1comp-steady-test/'));
  expect(paths.filter(p => p.includes('/original/'))).toEqual([
    'test-1comp/sim/traub-1comp-steady-test/original/value=0.json',
    'test-1comp/sim/traub-1comp-steady-test/original/value=10.json',
  ]);
  expect(() => inputOf({ name: 's', from: 'traub', sweep: { value: [0], sigma: [1] } } as never)).toThrow();
  expect(() => inputOf({ name: 's', from: 'traub', sweep: { duration: [100] } })).toThrow(/current_params/);
  expect(() => partOf({ name: 'p', from: 'traub', sweep: { value: [0] } })).toThrow(/系列だけ/);
});

test('a GPU bench assigns the points of its series to the cells in the declared order', () => {
  const plan = EXPERIMENTS.find(e => e.id === 'test-gpu')!;
  const bench = resolved(plan)['test-gpu/bench/original@traub-1comp-steady-test.json'];
  if (bench?.kind !== 'bench') throw new Error('bench でない');
  expect(bench.inputs).toEqual(pointsOf(plan.series['traub-1comp-steady-test']).map(p => p.input));
});

test('a series with a point equal to a training input is refused', () => {
  const plan = EXPERIMENTS.find(e => e.id === 'test-1comp')!;
  const train = plan.series['traub-train-duration-1000'];
  const { 'traub-train-duration-1000': _, ...declared } = plan.series;
  const clash = { ...train, spec: { ...train.spec, current_params: {} }, sweep: { param: 'duration', values: [1000] } };
  const pairs = plan.pairs.filter(p => p.series !== 'traub-train-duration-1000');
  expect(() => withTraining({ ...plan, series: { ...declared, clash }, pairs })).toThrow(/学習の入力と同じ/);
});

test('a model lays one learner over each group of a grouping', () => {
  expect(groupsOf('h+n_a+c')).toEqual([
    ['H', 'N'],
    ['A', 'C'],
  ]);
  const kam = modelFrom('kam.h+n_a+c', partOf(KAM), 'h+n_a+c').config;
  expect(kam.groups.map(g => [g.gates, g.compressor, g.rhs, g.d])).toEqual([
    [['H', 'N'], 'kam', 'kam', 1],
    [['A', 'C'], 'kam', 'kam', 1],
  ]);
  expect(
    modelFrom('pca-sindy.h+n', partOf('traub'), 'h+n').config.groups[0].rhs_params.library_specs?.length,
  ).toBeGreaterThan(0);
  const pca = modelFrom('pca-sindy.h+n', partOf('traub'), 'h+n');
  const ae = modelFrom('ae-sindy.h+n', partOf(AE), 'h+n');
  expect(hashOf(modelSpec(pca))).not.toBe(hashOf(modelSpec(ae)));
  expect([
    partSpec(ae, ae.config.groups[0]).device,
    partSpec(TESTS._test_hh_ae, TESTS._test_hh_ae.config.groups[0]).device,
  ]).toEqual(['gpu', 'cpu']);
  expect(() => partOf('xyz')).toThrow();
  expect(modelFrom('d2', partOf({ name: 'd2', from: 'traub', group: { d: 2 } }), 'h+n').config.groups[0].d).toBe(2);
  expect(() => partOf({ name: 'd2', from: 'traub', group: { d: 'two' } })).toThrow();
  expect(() => groupsOf('h+n_a')).toThrow();
  expect(modelFrom('kam-absorb-m.h+n', partOf(KAM_M), 'h+n').config.absorbed).toEqual(['M']);
  expect('absorbed' in modelFrom('kam.h+n', partOf(KAM), 'h+n').config).toBe(false);
  expect(() => modelFrom('kam-absorb-m.m+h', partOf(KAM_M), 'm+h')).toThrow();
});

test('resolved/ holds exactly the spec.json files the declarations send, so a change to them shows in the diff', () => {
  expect(stale(), 'resolved/ が宣言と違う（bun local/resolve.ts write で書き直す）').toEqual([]);
});
