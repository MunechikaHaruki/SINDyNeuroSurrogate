import { expect, test } from 'bun:test';
import { PRESET_SERIES as SERIES, TESTS, groupsOf, methodOf, trainFrom } from './presets.ts';
import { hashOf } from '../../shared/hash.ts';
import { stale, simSpec, trainSpec } from './resolve.ts';

test('the train hash does not depend on the order the config was written in', () => {
  const train = TESTS._test_hh_sindy;
  const { config } = train;
  const reordered = {
    train_comp: config.train_comp,
    groups: config.groups,
    data: config.data,
    form: config.form,
    cell: config.cell,
  };
  expect(hashOf(trainSpec({ name: train.name, config: reordered }))).toBe(hashOf(trainSpec(train)));
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
  const kam = trainFrom('kam.h+n_a+c', methodOf('kam'), 'h+n_a+c').config;
  expect(kam.groups.map(g => [g.gates, g.compressor, g.rhs, g.d])).toEqual([
    [['H', 'N'], 'kam', 'kam', 1],
    [['A', 'C'], 'kam', 'kam', 1],
  ]);
  expect(trainFrom('pca-sindy.h+n', methodOf('pca-sindy'), 'h+n').config.groups[0].rhs_params.library_specs?.length).toBeGreaterThan(
    0,
  );
  expect(hashOf(trainSpec(trainFrom('pca-sindy.h+n', methodOf('pca-sindy'), 'h+n')))).not.toBe(
    hashOf(trainSpec(trainFrom('ae-sindy.h+n', methodOf('ae-sindy'), 'h+n'))),
  );
  expect([trainSpec(trainFrom('ae-sindy.h+n', methodOf('ae-sindy'), 'h+n')).device, trainSpec(TESTS._test_hh_ae).device]).toEqual([
    'gpu',
    'cpu',
  ]);
  expect(() => methodOf('xyz')).toThrow();
  expect(trainFrom('d2', methodOf({ from: 'pca-sindy', d: 2 }), 'h+n').config.groups[0].d).toBe(2);
  expect(() => methodOf({ from: 'pca-sindy', d: 'two' })).toThrow();
  expect(() => groupsOf('h+n_a')).toThrow();
  expect(trainFrom('kam-absorb-m.h+n', methodOf('kam-absorb-m'), 'h+n').config.absorbed).toEqual(['M']);
  expect('absorbed' in trainFrom('kam.h+n', methodOf('kam'), 'h+n').config).toBe(false);
  expect(() => trainFrom('kam-absorb-m.m+h', methodOf('kam-absorb-m'), 'm+h')).toThrow();
});

test('resolved/ holds exactly the spec.json files the declarations send, so a change to them shows in the diff', () => {
  expect(stale(), 'resolved/ が宣言と違う（bun local/resolve/resolve.ts write で書き直す）').toEqual([]);
});
