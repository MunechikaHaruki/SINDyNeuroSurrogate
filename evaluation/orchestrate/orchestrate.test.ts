import { expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PRESET_SERIES as SERIES, TESTS, groupsOf, methodOf, trainFrom } from '../experiments/presets.ts';
import { canonical, dirOf, hashOf, jobs, simSpec, trainSpec } from './orchestrate.ts';
import { stale } from './resolve.ts';

test('the key does not depend on the order the spec was written in', () => {
  expect(canonical({ b: 1, a: { d: [1, { f: 2, e: 3 }], c: null } })).toBe(
    '{"a":{"c":null,"d":[1,{"e":3,"f":2}]},"b":1}',
  );
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

test('series that differ only in what they replace share the input directory but not the model', () => {
  const soma = SERIES['traub-19comp-steady-replace-soma'];
  const all = SERIES['traub-19comp-steady-replace-all'];
  const original = dirOf(simSpec(soma, null));
  expect(original).toMatch(/^inputs\/[0-9a-f]{12}\/original$/);
  expect(dirOf(simSpec(all, null))).toBe(original);
  const input = original.slice(0, -'original'.length);
  for (const dir of [dirOf(simSpec(soma, 't')), dirOf(simSpec(all, 't')), dirOf(simSpec(soma, 'u'))])
    expect(dir.startsWith(input)).toBe(true);
  expect(dirOf(simSpec(soma, 't'))).not.toBe(dirOf(simSpec(all, 't')));
  expect(dirOf(simSpec(soma, 't'))).not.toBe(dirOf(simSpec(soma, 'u')));
  expect(dirOf(trainSpec(TESTS._test_hh_sindy))).toMatch(/^models\/[0-9a-f]{12}$/);
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

test('a job is running while its pid lives, done on exit code 0, and failed otherwise', () => {
  const dir = join(mkdtempSync(join(tmpdir(), 'runs-')), '.jobs');
  mkdirSync(dir);
  function put(key: string, model: string, started: string, files: Record<string, string>): void {
    writeFileSync(join(dir, `${key}.json`), JSON.stringify({ job: { spec: model }, started }));
    for (const [ext, text] of Object.entries(files)) writeFileSync(join(dir, `${key}.${ext}`), text);
  }
  put('a', 'ok', '2026-01-01', { exit: '0', log: 'one\ntwo\n' });
  put('b', 'bad', '2026-01-02', { exit: '1' });
  put('c', 'gone', '2026-01-03', {});
  const found = jobs(join(dir, '..'));
  expect(found.map(j => [j.started, j.state])).toEqual([
    ['2026-01-03', 'failed'],
    ['2026-01-02', 'failed'],
    ['2026-01-01', 'done'],
  ]);
  expect(found[2].log).toEqual(['one', 'two']);
});

test('resolved/ holds exactly the spec.json files the declarations send, so a change to them shows in the diff', () => {
  expect(stale(), 'resolved/ が宣言と違う（bun evaluation/orchestrate/resolve.ts write で書き直す）').toEqual([]);
});
