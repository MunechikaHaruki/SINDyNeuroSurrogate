import { expect, test } from 'bun:test';
import { PRESET_SERIES as SERIES, TESTS, groupsOf, methodOf, modelFrom } from './experiments/schema/presets.ts';
import { hashOf } from '../shared/hash.ts';
import { modelSpec, partSpec, resolved, stale, simSpec } from './resolve.ts';
import { EXPERIMENTS } from './experiments/schema/experiments.ts';

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
  const alone = modelFrom('pca-sindy.h+n', methodOf('pca-sindy'), 'h+n');
  const paired = modelFrom('pca-sindy.h+n_a+c', methodOf('pca-sindy'), 'h+n_a+c');
  const absorbed = modelFrom('kam-absorb-m.h+n', methodOf('kam-absorb-m'), 'h+n');
  const kam = modelFrom('kam.h+n', methodOf('kam'), 'h+n');
  expect(hashOf(partSpec(paired, paired.config.groups[0]))).toBe(hashOf(partSpec(alone, alone.config.groups[0])));
  expect(hashOf(modelSpec(absorbed))).not.toBe(hashOf(modelSpec(kam)));
  const model = modelSpec(paired);
  if (model.kind !== 'model') throw new Error('model でない');
  expect(model.parts).toHaveLength(2);
});

test('a KAM model has no learners and holds its groups and training data itself, and KAM is not mixed with learners', () => {
  const kam = modelFrom('kam.h+n_a+c', methodOf('kam'), 'h+n_a+c');
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
  expect(paths.some(path => path.startsWith(`${plan.id}/data/`))).toBe(true);
});

test('an experiment holds one file per distinct training data and learner', () => {
  for (const plan of EXPERIMENTS) {
    const files = Object.entries(resolved(plan));
    for (const dir of ['data', 'parts']) {
      const hashes = files.filter(([path]) => path.startsWith(`${plan.id}/${dir}/`)).map(([, spec]) => hashOf(spec));
      expect(new Set(hashes).size, `${plan.id} の ${dir}/ に同じ中身のファイルがある`).toBe(hashes.length);
    }
  }
});

test('series that differ only in what they replace send the same input', () => {
  const soma = simSpec(SERIES['traub-19comp-steady-replace-soma'], 't');
  const all = simSpec(SERIES['traub-19comp-steady-replace-all'], 't');
  if (soma.kind !== 'sim' || all.kind !== 'sim') throw new Error('sim でない');
  expect(soma.input).toEqual(all.input);
  expect(soma.model).not.toEqual(all.model);
});

test('a model is a method laid over a grouping', () => {
  expect(groupsOf('h+n_a+c')).toEqual([
    ['H', 'N'],
    ['A', 'C'],
  ]);
  const kam = modelFrom('kam.h+n_a+c', methodOf('kam'), 'h+n_a+c').config;
  expect(kam.groups.map(g => [g.gates, g.compressor, g.rhs, g.d])).toEqual([
    [['H', 'N'], 'kam', 'kam', 1],
    [['A', 'C'], 'kam', 'kam', 1],
  ]);
  expect(
    modelFrom('pca-sindy.h+n', methodOf('pca-sindy'), 'h+n').config.groups[0].rhs_params.library_specs?.length,
  ).toBeGreaterThan(0);
  const pca = modelFrom('pca-sindy.h+n', methodOf('pca-sindy'), 'h+n');
  const ae = modelFrom('ae-sindy.h+n', methodOf('ae-sindy'), 'h+n');
  expect(hashOf(modelSpec(pca))).not.toBe(hashOf(modelSpec(ae)));
  expect([
    partSpec(ae, ae.config.groups[0]).device,
    partSpec(TESTS._test_hh_ae, TESTS._test_hh_ae.config.groups[0]).device,
  ]).toEqual(['gpu', 'cpu']);
  expect(() => methodOf('xyz')).toThrow();
  expect(modelFrom('d2', methodOf({ from: 'pca-sindy', d: 2 }), 'h+n').config.groups[0].d).toBe(2);
  expect(() => methodOf({ from: 'pca-sindy', d: 'two' })).toThrow();
  expect(() => groupsOf('h+n_a')).toThrow();
  expect(modelFrom('kam-absorb-m.h+n', methodOf('kam-absorb-m'), 'h+n').config.absorbed).toEqual(['M']);
  expect('absorbed' in modelFrom('kam.h+n', methodOf('kam'), 'h+n').config).toBe(false);
  expect(() => modelFrom('kam-absorb-m.m+h', methodOf('kam-absorb-m'), 'm+h')).toThrow();
});

test('resolved/ holds exactly the spec.json files the declarations send, so a change to them shows in the diff', () => {
  expect(stale(), 'resolved/ が宣言と違う（bun local/resolve.ts write で書き直す）').toEqual([]);
});
