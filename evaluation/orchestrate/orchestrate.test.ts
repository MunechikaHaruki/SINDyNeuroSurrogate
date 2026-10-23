import { expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { EXPERIMENTS, PRESETS, SERIES } from './conf.ts';
import { canonical, dirOf, hashOf, jobs, simSpec, trainSpec } from './orchestrate.ts';

test('the key does not depend on the order the spec was written in', () => {
  expect(canonical({ b: 1, a: { d: [1, { f: 2, e: 3 }], c: null } })).toBe(
    '{"a":{"c":null,"d":[1,{"e":3,"f":2}]},"b":1}',
  );
  const config = PRESETS._test_hh_sindy[0].config;
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
  expect(dirOf(trainSpec(PRESETS._test_hh_sindy[0].config))).toMatch(/^models\/[0-9a-f]{12}$/);
});

test('a sweep gives one differently configured run per point', () => {
  const runs = PRESETS.traub_useMC19CompForTeachingData;
  expect(runs.map(r => r.config.groups[0].d)).toEqual([6, 7, 8]);
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
