import { expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { Status } from '../../shared/spec.ts';
import type { WaveMeta } from '../../shared/runs.ts';
import { RESOLVED } from '../run/places.ts';
import { server } from './serve.ts';

const runs = mkdtempSync(join(tmpdir(), 'runs-'));
const handle = server(runs);

const N = 1000;
const DT = 0.1;
const META: WaveMeta = {
  t0: 0,
  dt: DT,
  n: [N],
  features: [
    [0, 'V', false],
    [0, 'H', true],
  ],
  nodes: null,
  names: { soma: 0 },
};

/** 2 ms ごとに 0 mV を越える電位と、ゲートの列を持つ run。shift [点] だけ発火を遅らせる。 */
function put(run: string, shift: number): void {
  const dir = join(runs, run);
  mkdirSync(join(dir, 'waves/0'), { recursive: true });
  const values = new Float64Array(2 * N);
  for (let i = 0; i < N; i++) {
    values[i] = (i - shift) % 20 === 0 ? 30 : -65;
    values[N + i] = i / N;
  }
  writeFileSync(join(dir, 'waves/meta.json'), JSON.stringify(META));
  writeFileSync(join(dir, 'waves/0/comp0.f64'), Buffer.from(values.buffer));
  writeFileSync(join(dir, 'spikes.json'), JSON.stringify({ soma: [{ peak_voltage: [30] }] }));
}
put('inputs/i/original', 10);
put('inputs/i/m', 11);

async function post(op: string, body: unknown): Promise<Response> {
  return handle(
    new Request(`http://x/${op}`, {
      method: 'POST',
      body: JSON.stringify(body),
      headers: { 'content-type': 'application/json' },
    }),
  );
}

test('a trace comes back decimated to at most two points per bucket', async () => {
  const array = { run: 'inputs/i/original', file: 'waves/0/comp0.f64', column: 0, n: N };
  const [points] = await (
    await post('traces', { requests: [{ array, t0: 0, dt: DT, start: 0, end: N * DT, buckets: 10 }] })
  ).json();
  expect(points.length).toBeLessThanOrEqual(20);
  expect(Math.max(...points.map((p: { y: number }) => p.y))).toBe(30);
});

test('the series metrics count crossings and coincidence against the original', async () => {
  const reply = await (
    await post('series', { original: 'inputs/i/original', columns: ['inputs/i/m'], comp: 'soma', delta: 2 })
  ).json();
  expect(reply.times.original[0].length).toBe(reply.times.columns['inputs/i/m'][0].length);
  expect(reply.crossings['inputs/i/m'][0].count[0]).toBe(reply.crossings['inputs/i/m'][0].count[1]);
  expect(reply.crossings['inputs/i/m'][0].gamma).toBeGreaterThan(0.9);
  expect(reply.pairs['inputs/i/m']).toHaveLength(1);
});

test('only json and f64 inside runs can be read, and unknown ops are refused', async () => {
  expect((await post('json', { run: 'inputs/i/original', path: 'waves/meta.json' })).status).toBe(200);
  expect((await post('json', { run: '..', path: 'etc/passwd.json' })).status).toBe(400);
  expect((await post('json', { run: 'inputs/i/original', path: 'waves/0/comp0.f64' })).status).toBe(400);
  expect((await post('shell', {})).status).toBe(404);
  expect((await post('launch', { job: { action: 'rm' } })).status).toBe(400);
});

test('the status lists the experiments under resolved/ without reading the declarations', async () => {
  const reply = (await (await post('status', {})).json()) as Status;
  const dirs = readdirSync(RESOLVED, { withFileTypes: true })
    .filter(e => e.isDirectory())
    .map(e => e.name)
    .sort();
  expect(reply.experiments.map(e => e.name)).toEqual(dirs);
});
