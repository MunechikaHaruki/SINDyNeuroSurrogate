import { expect, test } from 'bun:test';
import { merge, PRESETS, SERIES } from './conf.ts';
import { canonical, hashOf, originalSpec, replacedSpec, trainSpec } from './orchestrate.ts';

test('merge mixes nested objects and replaces arrays and values', () => {
  const base = { a: { b: 1, c: [1, 2] }, d: 'x' };
  expect(merge(base, { a: { c: [3] } })).toEqual({ a: { b: 1, c: [3] }, d: 'x' });
  expect(base.a.c).toEqual([1, 2]);
});

test('the key does not depend on the order the spec was written in', () => {
  expect(canonical({ b: 1, a: { d: [1, { f: 2, e: 3 }], c: null } })).toBe(
    '{"a":{"c":null,"d":[1,{"e":3,"f":2}]},"b":1}',
  );
  const config = PRESETS._test_hh_sindy[0].config;
  const reordered = { closure: config.closure, ansatz: config.ansatz, preprocessor: config.preprocessor, spec: config.spec };
  expect(hashOf(trainSpec(reordered))).toBe(hashOf(trainSpec(config)));
});

test('series that differ only in what they replace share the original run but not the replaced one', () => {
  const soma = SERIES.traub19_somastim;
  const all = SERIES.traub19_somastim_allcomp;
  expect(hashOf(originalSpec(soma))).toBe(hashOf(originalSpec(all)));
  expect(hashOf(replacedSpec(soma, 't'))).not.toBe(hashOf(replacedSpec(all, 't')));
  expect(hashOf(replacedSpec(soma, 't'))).not.toBe(hashOf(replacedSpec(soma, 'u')));
});

test('a sweep gives one differently configured run per point', () => {
  const runs = PRESETS.traub_useMC19CompForTeachingData;
  expect(runs.map(r => r.config.spec.n_components)).toEqual([6, 7, 8]);
  expect(new Set(runs.map(r => hashOf(trainSpec(r.config)))).size).toBe(3);
});
