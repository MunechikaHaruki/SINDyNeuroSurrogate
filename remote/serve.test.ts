import { expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { hashOf } from '../shared/hash.ts';
import { FORMATS, type Plans, type Spec } from '../shared/spec.ts';
import type { WaveMeta } from '../shared/runs.ts';
import type { ExperimentView, Experiments, InputsReply, ModelsReply, RegistryReply } from '../shared/api.ts';
import { dirOf } from './run/places.ts';
import { server } from './serve.ts';

const runs = mkdtempSync(join(tmpdir(), 'runs-'));
const resolved = mkdtempSync(join(tmpdir(), 'resolved-'));
process.env.SINDY_RESOLVED = resolved;
const handle = server(runs);

const N = 1000;
const DT = 0.1;
const SIM = { target: 'hh', current_type: 'steady', dt: DT, current_params: { duration: N * DT } };
const GROUP = {
  gates: ['H'],
  compressor: 'pca' as const,
  d: 1,
  rhs: 'sindy' as const,
  compressor_params: {},
  rhs_params: {},
};
const DATA: Spec = { format: FORMATS.data, device: 'cpu', kind: 'data', data: SIM };
const PART: Spec = {
  format: FORMATS.part,
  device: 'cpu',
  kind: 'part',
  data: hashOf(DATA),
  cell: 'hh',
  form: 'hybrid',
  train_comp: null,
  group: GROUP,
};
const MODEL: Spec = {
  format: FORMATS.model,
  device: 'cpu',
  kind: 'model',
  cell: 'hh',
  form: 'hybrid',
  parts: [hashOf(PART)],
};
const INPUT = { spec: { ...SIM, stim: 'soma' }, param: null, values: [] };
const ORIGINAL: Spec = { format: FORMATS.sim, device: 'cpu', kind: 'sim', input: INPUT, model: null };
const REPLACED: Spec = { ...ORIGINAL, model: { surrogate: hashOf(MODEL), replace_targets: ['soma'] } };
const PLANS: Plans = {
  groups: [{ name: 'g', description: '群' }],
  experiments: [
    {
      id: 'e',
      group: 'g',
      description: '実験',
      groupings: ['h'],
      methods: ['pca'],
      models: [
        {
          name: 'm',
          method: 'pca',
          grouping: 'h',
          config: { cell: 'hh', form: 'hybrid', data: SIM, groups: [GROUP], train_comp: null },
        },
      ],
      series: { s: { ...INPUT, replace_targets: ['soma'] } },
      pairs: [{ model: 'm', series: 's' }],
      benches: [],
    },
  ],
  measures: { coincidence_window: 2 },
};

/** ファイルを書く（ディレクトリも作る）。 */
function write(file: string, content: string | Buffer): void {
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, content);
}

/** 目録 meta の 2 変数の波形。1 つ目は 2 ms ごとに 0 mV を越える電位で、shift [点] だけ発火を遅らせる。 */
function waves(dir: string, meta: WaveMeta, shift: number): void {
  const values = new Float64Array(2 * N);
  for (let i = 0; i < N; i++) {
    values[i] = (i - shift) % 20 === 0 ? 30 : -65;
    values[N + i] = i / N;
  }
  write(join(dir, 'meta.json'), JSON.stringify(meta));
  write(join(dir, '0/comp0.f64'), Buffer.from(values.buffer));
  write(join(dir, '0/I_ext.f64'), Buffer.from(new Float64Array(N).fill(1).buffer));
}

/** spec の run を置く。 */
function put(spec: Spec, kind: string, files: Record<string, unknown> = {}): string {
  const dir = join(runs, dirOf(spec));
  write(join(dir, 'spec.json'), JSON.stringify(spec));
  write(
    join(dir, 'meta.json'),
    JSON.stringify({ kind, name: kind, created: '2026-11-16T00:00:00', elapsed: 1, unit: 'μA/cm²' }),
  );
  for (const [name, value] of Object.entries(files)) write(join(dir, name), JSON.stringify(value));
  return dir;
}

const meta = (second: string): WaveMeta => ({
  t0: 0,
  dt: DT,
  n: [N],
  features: [
    [0, 'V', false],
    [0, second, second === 'H'],
  ],
  nodes: null,
  names: { soma: 0 },
});

for (const [path, spec] of Object.entries({
  'e/data/d': DATA,
  'e/parts/p': PART,
  'e/models/m': MODEL,
  'e/sim/s/original': ORIGINAL,
  'e/sim/s/m': REPLACED,
}))
  write(join(resolved, `${path}.json`), JSON.stringify(spec));
write(join(resolved, 'plans.json'), JSON.stringify(PLANS));
waves(join(put(DATA, 'data'), 'waves'), meta('H'), 10);
put(PART, 'part', {
  'part.json': {
    gates: ['H'],
    latents: ['h.z1'],
    comps: ['soma'],
    closure: { kind: 'sindy', targets: ['z'], features: ['1'], xi: [[0.5]] },
    preprocessor: { kind: 'pca', explained_variance_ratio: [0.9, 0.1], n_components: 1 },
    metrics: { nnz: 1 },
  },
});
for (const file of ['latent', 'recon_rmse'])
  write(join(runs, dirOf(PART), `fit/0/${file}.f64`), Buffer.from(new Float64Array(N).fill(0.5).buffer));
put(MODEL, 'model', { 'summary.json': { 'cost/surr-orig/mul': -3 } });
waves(join(put(ORIGINAL, 'original'), 'waves'), meta('H'), 10);
const replaced = put(REPLACED, 'replaced');
waves(join(replaced, 'waves'), meta('h.z1'), 11);
waves(join(replaced, 'projected/soma'), meta('h.z1'), 10);

async function post<T>(op: string, body: unknown): Promise<{ status: number; reply: T }> {
  const response = await handle(
    new Request(`http://x/${op}`, {
      method: 'POST',
      body: JSON.stringify(body),
      headers: { 'content-type': 'application/json' },
    }),
  );
  return { status: response.status, reply: response.ok ? ((await response.json()) as T) : (undefined as T) };
}

test('the experiments come in the declared order with their progress and run names', async () => {
  const { reply } = await post<Experiments>('experiments', {});
  expect(reply.groups.map(g => g.name)).toEqual(['g']);
  expect(reply.groups[0].experiments[0]).toMatchObject({ id: 'e', progress: '結果あり', runs: { models: ['m'], inputs: ['s'], bench: [] } });
  expect(reply.groups[0].experiments[0].runs.data).toHaveLength(1);
  expect(reply.groups[0].experiments[0].runs.parts).toHaveLength(1);
  expect(reply.measures.coincidenceWindow).toBe(2);
});

test('an experiment gives its setup, the training data and learners of each model and the state of each pair', async () => {
  const { reply } = await post<ExperimentView>('experiment', { experiment: 'e' });
  expect(reply.setup.map(r => r.label)).toContain('系列');
  expect(reply.models[0].name).toBe('m');
  expect(reply.data.map(d => d.name)).toEqual([reply.models[0].data]);
  expect(reply.parts.map(p => p.name)).toEqual(reply.models[0].parts);
  expect(reply.pairs[0]).toEqual({ input: 's', model: 'm', state: 'done' });
  expect(reply.rerun.done).toBe(reply.rerun.total);
});

test('an input compares the original with each model, keyed by model name', async () => {
  const { reply } = await post<InputsReply>('inputs', { experiment: 'e', input: 's' });
  expect(reply.input.points).toEqual(['単発']);
  expect(reply.columns.map(c => c.model)).toEqual(['m']);
  expect(reply.points[0].voltages.map(v => v.model)).toEqual([null, 'm']);
  expect(reply.points[0].voltages[0].points.length).toBeLessThanOrEqual(800);
  expect(reply.summaries.m.gamma).toBeGreaterThan(0.9);
  expect(reply.spikes.original[0].times.length).toBe(reply.spikes.columns.m[0].times.length);
  expect((await post('inputs', { experiment: 'e', input: 's', comp: 'dend' })).status).toBe(400);
});

test('a model gives the comparison at one point, with the original projected onto its latents', async () => {
  const { reply } = await post<ModelsReply>('models', { experiment: 'e', model: 'm', comps: ['soma'] });
  expect(reply.compare?.voltage.map(l => l.label)).toEqual(['原系', '置換系']);
  expect(Object.keys(reply.compare?.latents ?? {})).toEqual(['h.z1']);
  expect(reply.compare?.axes).toEqual({ x: 'V', y: 'h.z1' });
});

test('the registry names each run by its place under runs/', async () => {
  const { reply } = await post<RegistryReply>('registry', {});
  expect(new Set(reply.runs.map(r => r.place))).toEqual(new Set(['data', 'parts', 'models', 'inputs']));
  expect(reply.runs.every(r => r.id.startsWith(`${r.place}/`))).toBe(true);
});

test('the training data and learner pages read their runs by declared name', async () => {
  const view = (await post<ExperimentView>('experiment', { experiment: 'e' })).reply;
  const names = { data: view.models[0].data, part: view.models[0].parts[0] };
  const data = await post<{ coverage: unknown[]; stim: string }>('data', { experiment: 'e', data: names.data, comps: ['soma'] });
  expect(data.reply.stim).toBe('soma');
  expect(data.reply.coverage).toHaveLength(1);
  const part = await post<{ pca: { kept: number }; path: unknown[] }>('parts', { experiment: 'e', part: names.part, comps: ['soma'] });
  expect(part.reply.pca.kept).toBe(1);
  expect(part.reply.path).toHaveLength(1);
  const shown = await post<{ comps: string[]; shown: string[] }>('data', { experiment: 'e', data: names.data });
  expect(shown.reply.shown).toEqual(['soma']);
  expect((await post<{ shown: string[] }>('parts', { experiment: 'e', part: names.part })).reply.shown).toEqual(['soma']);
});

test('unknown ops and unknown jobs are refused', async () => {
  expect((await post('shell', {})).status).toBe(404);
  expect((await post('json', { run: '..', path: 'etc/passwd.json' })).status).toBe(404);
  expect((await post('launch', { job: { action: 'rm' } })).status).toBe(400);
  expect((await post('experiment', { experiment: 'x' })).status).toBe(400);
});
