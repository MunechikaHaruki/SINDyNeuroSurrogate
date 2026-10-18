import { expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { EXPERIMENTS, PRESETS, SERIES } from './conf.ts';
import { canonical, hashOf, jobs, originalSpec, replacedSpec, trainSpec } from './orchestrate.ts';

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

test('every experiment names declared presets and series', () => {
  for (const experiment of Object.values(EXPERIMENTS)) {
    for (const preset of experiment.presets) expect(PRESETS).toHaveProperty(preset);
    for (const series of experiment.series) expect(SERIES).toHaveProperty(series);
  }
});

test('a job is running while its pid lives, done on exit code 0, and failed otherwise', () => {
  const dir = join(mkdtempSync(join(tmpdir(), 'runs-')), '.jobs');
  mkdirSync(dir);
  function put(key: string, preset: string, started: string, files: Record<string, string>): void {
    writeFileSync(join(dir, `${key}.json`), JSON.stringify({ job: { action: 'train', preset }, started }));
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
