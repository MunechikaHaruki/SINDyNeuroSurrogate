import { expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { FORMAT, type Spec } from '../../shared/spec.ts';
import { dirOf, jobs } from './orchestrate.ts';

const input = { spec: { target: 'traub-19comp', current_type: 'steady', dt: 0.01, current_params: { amplitude: 1 } }, param: null, values: [0] };
const sim = (model: { train: string; replace_targets: string[] } | null): Spec => ({ format: FORMAT, device: 'cpu', kind: 'sim', input, model });

test('specs that differ only in what they replace share the input directory but not the model', () => {
  const original = dirOf(sim(null));
  expect(original).toMatch(/^inputs\/[0-9a-f]{12}\/original$/);
  const prefix = original.slice(0, -'original'.length);
  const soma = dirOf(sim({ train: 't', replace_targets: ['soma'] }));
  const all = dirOf(sim({ train: 't', replace_targets: ['soma', 'axon'] }));
  const other = dirOf(sim({ train: 'u', replace_targets: ['soma'] }));
  for (const dir of [soma, all, other]) expect(dir.startsWith(prefix)).toBe(true);
  expect(soma).not.toBe(all);
  expect(soma).not.toBe(other);
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
