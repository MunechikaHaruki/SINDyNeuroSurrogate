import { expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SERIES, groupsOf, trainOf } from './conf.ts';
import { canonical, dirOf, hashOf, jobs, simSpec, trainSpec } from './orchestrate.ts';

test('the key does not depend on the order the spec was written in', () => {
  expect(canonical({ b: 1, a: { d: [1, { f: 2, e: 3 }], c: null } })).toBe(
    '{"a":{"c":null,"d":[1,{"e":3,"f":2}]},"b":1}',
  );
  const config = trainOf('_test_hh_sindy').config;
  const reordered = { train_comp: config.train_comp, groups: config.groups, data: config.data, form: config.form, cell: config.cell };
  expect(hashOf(trainSpec(reordered))).toBe(hashOf(trainSpec(config)));
});

test('series that differ only in what they replace share the input directory but not the model', () => {
  const soma = SERIES.traub19_somastim;
  const all = SERIES.traub19_somastim_allcomp;
  const original = dirOf(simSpec(soma, null));
  expect(original).toMatch(/^inputs\/[0-9a-f]{12}\/original$/);
  expect(dirOf(simSpec(all, null))).toBe(original);
  const input = original.slice(0, -'original'.length);
  for (const dir of [dirOf(simSpec(soma, 't')), dirOf(simSpec(all, 't')), dirOf(simSpec(soma, 'u'))]) expect(dir.startsWith(input)).toBe(true);
  expect(dirOf(simSpec(soma, 't'))).not.toBe(dirOf(simSpec(all, 't')));
  expect(dirOf(simSpec(soma, 't'))).not.toBe(dirOf(simSpec(soma, 'u')));
  expect(dirOf(trainSpec(trainOf('_test_hh_sindy').config))).toMatch(/^models\/[0-9a-f]{12}$/);
});

test('a model is a method laid over a grouping', () => {
  expect(groupsOf('hn_ac')).toEqual([['H', 'N'], ['A', 'C']]);
  const kam = trainOf('kam_hn_ac').config;
  expect(kam.groups.map(g => [g.gates, g.compressor, g.rhs, g.d])).toEqual([[['H', 'N'], 'kam', 'kam', 1], [['A', 'C'], 'kam', 'kam', 1]]);
  expect(trainOf('pca_hn').config.groups[0].rhs_params.library_specs?.length).toBeGreaterThan(0);
  expect(hashOf(trainSpec(trainOf('pca_hn').config))).not.toBe(hashOf(trainSpec(trainOf('ae_hn').config)));
  expect(() => trainOf('xyz_hn')).toThrow();
  expect(() => groupsOf('hn_a')).toThrow();
});

test('a job is running while its pid lives, done on exit code 0, and failed otherwise', () => {
  const dir = join(mkdtempSync(join(tmpdir(), 'runs-')), '.jobs');
  mkdirSync(dir);
  function put(key: string, model: string, started: string, files: Record<string, string>): void {
    writeFileSync(join(dir, `${key}.json`), JSON.stringify({ job: { action: 'train', model }, started }));
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
