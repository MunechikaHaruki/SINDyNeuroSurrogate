import { expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { hashOf, inputKey } from '../shared/hash.ts';
import { FORMATS, type Plans, type Spec } from '../shared/spec.ts';
import type { WaveMeta } from '../shared/runs.ts';
import type {
  ExperimentView,
  Experiments,
  InputsReply,
  ModelsReply,
  PartsReply,
  RegistryReply,
} from '../shared/api.ts';
import { seriesComparison } from './read/comparison.ts';
import type { Context } from './read/experiments.ts';
import { inputs } from './read/inputs.ts';
import { models } from './read/models.ts';
import { parts } from './read/parts.ts';
import { runFiles, type RunFiles } from './read/waves.ts';
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
const INPUT = { spec: { ...SIM, stim: 'soma' }, param: null, values: [] };
/** 系列 s の元のモデル。単発の入力で、学習器と KAM のモデルの学習データも兼ねる（学習の系列）。 */
const ORIGINAL: Spec = { format: FORMATS.sim, device: 'cpu', kind: 'sim', input: INPUT, model: null };
const TRAIN = inputKey({ format: FORMATS.sim, device: 'cpu', input: INPUT });
const PART: Spec = {
  format: FORMATS.part,
  device: 'cpu',
  kind: 'part',
  input: TRAIN,
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
/** KAM のモデル。学習器を持たず、群と学習の入力（図を描く軌道）を spec に直に持つ。 */
const KAM_GROUP = {
  gates: ['M', 'H'],
  compressor: 'kam' as const,
  d: 1,
  rhs: 'kam' as const,
  compressor_params: {},
  rhs_params: {},
};
const KAM_MODEL: Spec = {
  ...MODEL,
  parts: [],
  kam: { groups: [KAM_GROUP], input: TRAIN, train_comp: null },
} as Spec;
const REPLACED: Spec = { ...ORIGINAL, model: { surrogate: hashOf(MODEL), replace_targets: ['soma'] } };
const PLANS: Plans = {
  groups: [{ name: 'g', description: '群' }],
  experiments: [
    {
      id: 'e',
      group: 'g',
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
    {
      id: 'k',
      group: 'g',
      groupings: ['m+h'],
      methods: ['kam'],
      models: [
        {
          name: 'n',
          method: 'kam',
          grouping: 'm+h',
          config: { cell: 'hh', form: 'hybrid', data: SIM, groups: [KAM_GROUP], train_comp: null },
        },
      ],
      series: { s: { ...INPUT, replace_targets: ['soma'] } },
      pairs: [],
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

/** 変数 variables（先頭は膜電位）の波形。膜電位は 2 ms ごとに 0 mV を越え、ほかは i / N。 */
function manyWaves(dir: string, variables: string[]): void {
  const values = new Float64Array(variables.length * N);
  for (let i = 0; i < N; i++) {
    values[i] = i % 20 === 0 ? 30 : -65;
    for (let k = 1; k < variables.length; k++) values[k * N + i] = i / N;
  }
  const features = variables.map((v, k) => [0, v, k > 0] as [number, string, boolean]);
  write(join(dir, 'meta.json'), JSON.stringify({ t0: 0, dt: DT, n: [N], features, nodes: null, names: { soma: 0 } }));
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
  'e/parts/p': PART,
  'e/models/m': MODEL,
  'e/sim/s/original': ORIGINAL,
  'e/sim/s/m': REPLACED,
  'k/sim/s/original': ORIGINAL,
  'k/models/n': KAM_MODEL,
}))
  write(join(resolved, `${path}.json`), JSON.stringify(spec));
write(join(resolved, 'plans.json'), JSON.stringify(PLANS));
const curve = { latent: 'h.z1', z: [-1, 1], gates: [[0, 1]] };
put(PART, 'part', {
  'compressor/decode.json': { gates: ['H'], order: ['M', 'H', 'N'], axes: [curve] },
  'compressor/own.json': {
    kind: 'pca',
    explained_variance_ratio: [0.9, 0.1],
    n_components: 1,
    components: [[1]],
    mean: [0.5],
  },
  'rhs/own.json': { kind: 'sindy', targets: ['z'], features: ['1'], xi: [[0.5]] },
  'rhs/field.json': {
    V: [-60, -40],
    latents: [
      {
        latent: 'h.z1',
        z: [0, 1, 2],
        learned: [
          [1, 1],
          [0, 0],
          [-1, -1],
        ],
        original: [
          [2, 2],
          [1, 1],
          [0, 0],
        ],
      },
    ],
  },
  'fit/0/summary.json': { reconstruction_mse: 0.01, reconstruction_mse_ratio: 0.1 },
});
const kamRun = put(KAM_MODEL, 'model', {
  'summary.json': {},
  'kam.json': {
    comp: 'soma',
    absorbed: [],
    groups: [{ gates: ['M', 'H'], latent: 'm+h.z1', weights: [0.4, 0.6], criterion: 0.2 }],
  },
});
manyWaves(join(kamRun, 'equivalent'), ['V', 'M', 'H', 'm+h.z1']);
manyWaves(join(kamRun, 'reduced'), ['V', 'm+h.z1']);
for (const file of ['latent', 'recon_rmse'])
  write(join(runs, dirOf(PART), `fit/0/${file}.f64`), Buffer.from(new Float64Array(N).fill(0.5).buffer));
const cost = Object.fromEntries(
  ['exp', 'div', 'pm', 'mul'].flatMap(op => [
    [`cost/surrogate/${op}`, 2],
    [`cost/original/${op}`, 5],
    [`cost/surr-orig/${op}`, -3],
  ]),
);
put(MODEL, 'model', { 'summary.json': cost });
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
  expect(reply.groups[0].experiments[0]).toMatchObject({
    id: 'e',
    progress: '結果あり',
    runs: { models: ['m'], inputs: ['s'], bench: [] },
  });
  expect(reply.groups[0].experiments[0].runs.parts).toHaveLength(1);
  expect(reply.measures.coincidenceWindow).toBe(2);
});

test('an experiment gives its setup, the training input and learners of each model and the state of each pair', async () => {
  const { reply } = await post<ExperimentView>('experiment', { experiment: 'e' });
  expect(reply.setup.map(r => r.label)).toContain('系列');
  expect(reply.models[0].name).toBe('m');
  expect(reply.models[0].train).toBe('s');
  expect(reply.parts.map(p => p.name)).toEqual(reply.models[0].parts);
  expect(reply.pairs[0]).toEqual({ input: 's', model: 'm', state: 'done' });
  expect(reply.rerun.done).toBe(reply.rerun.total);
});

test('an input compares the original with each model, keyed by model name', async () => {
  const { reply } = await post<InputsReply>('inputs', { experiment: 'e', input: 's' });
  expect(reply.input.points).toEqual(['単発']);
  expect(reply.input.trained).toEqual(['m']);
  expect(reply.columns.map(c => c.model)).toEqual(['m']);
  expect(reply.points[0].voltages.map(v => v.model)).toEqual([null, 'm']);
  expect(reply.points[0].voltages[0].points.length).toBeLessThanOrEqual(800);
  expect(reply.summaries.m.gamma).toBeGreaterThan(0.9);
  expect(reply.spikes.original[0].times.length).toBe(reply.spikes.columns.m[0].times.length);
  expect((await post('inputs', { experiment: 'e', input: 's', comp: 'dend' })).status).toBe(400);
});

test('a model gives only what its run holds: how it is built and the op cost', async () => {
  const { reply } = await post<ModelsReply>('models', { experiment: 'e', model: 'm' });
  expect(reply).toMatchObject({ model: 'm', cell: 'hh', form: 'hybrid', absorbed: [], run: { state: 'done' } });
  expect(reply.parts).toHaveLength(1);
  expect(reply.cost?.find(r => r.op === 'mul')).toEqual({ op: 'mul', surrogate: 2, original: 5, diff: -3 });
});

test('an input gives one pair in latent space, with the original projected, and the whole original on request', async () => {
  const { reply } = await post<InputsReply>('inputs', {
    experiment: 'e',
    input: 's',
    waves: false,
    pair: { model: 'm', point: 0 },
    whole: { point: 0, comps: ['soma'] },
  });
  expect(Object.keys(reply.pair?.latents ?? {})).toEqual(['h.z1']);
  expect(reply.pair?.latents['h.z1'].map(l => l.label)).toEqual(['原系', '置換系']);
  expect(Object.keys(reply.pair?.phases ?? {})).toEqual(['h.z1']);
  expect(reply.pair?.phases['h.z1'].map(l => l.label)).toEqual(['原系（射影）', '置換系']);
  expect(reply.pair?.sameScale).toBe(false);
  expect(Object.keys(reply.crossings)).toEqual(['m']);
  expect(reply.crossings.m[0]).toMatchObject({ diverged: false });
  expect(reply.pair?.rebuilt.map(r => [r.gate, r.comp])).toEqual([['H', 'soma']]);
  expect(reply.pair?.rebuilt[0].rebuilt.length).toBeGreaterThan(0);
  expect(reply.pair?.order).toEqual(['M', 'H', 'N']);
  expect(reply.pair?.errors.map(e => e.gate)).toEqual(['H']);
  expect(reply.whole?.voltage.map(l => l.label)).toEqual(['soma']);
  expect(reply.whole?.gates.map(l => l.label)).toEqual(['H']);
  expect((await post('inputs', { experiment: 'e', input: 's', pair: { model: 'x', point: 0 } })).status).toBe(400);
});

test('the registry names each run by its place under runs/', async () => {
  const { reply } = await post<RegistryReply>('registry', {});
  expect(new Set(reply.runs.map(r => r.place))).toEqual(new Set(['parts', 'models', 'inputs']));
  expect(reply.runs.every(r => r.id.startsWith(`${r.place}/`))).toBe(true);
});

test('the learner page reads its run by declared name, with the training trail in gate space', async () => {
  const view = (await post<ExperimentView>('experiment', { experiment: 'e' })).reply;
  const name = view.models[0].parts[0];
  const part = await post<PartsReply>('parts', { experiment: 'e', part: name, comps: ['soma'], window: [10, 50] });
  expect(part.reply.setup.find(r => r.label === '群のゲート')?.value).toBe('H');
  expect(part.reply.setup.find(r => r.label === '圧縮器')?.value).toBe('pca');
  const reply = part.reply;
  expect(reply.train).toBe('s');
  expect(reply.trail).toEqual({ comps: ['soma'], shown: ['soma'], duration: N * DT, window: [10, 50] });
  expect(reply.gateOrder).toEqual(['M', 'H', 'N']);
  expect(reply.gates).toEqual(['H']);
  expect(reply.field?.latents[0].steady.map(l => l.label)).toEqual(['元のモデル', '学習器']);
  expect(reply.field?.latents[0].steady[1].points).toEqual([
    { x: -60, y: 1 },
    { x: -40, y: 1 },
  ]);
  expect(reply.space?.axes).toEqual([{ latent: 'h.z1', points: [[0], [1]], z: [-1, 1] }]);
  expect(reply.space?.data.map(d => d.label)).toEqual(['soma']);
  expect(reply.space?.levels).toEqual([]);
  expect(reply.space?.data[0].points.length).toBeGreaterThan(0);
  expect(reply.space?.data[0].points.every(p => p.length === 1)).toBe(true);
  expect(reply.compressor).toEqual({
    kind: 'pca',
    ratios: [0.9, 0.1],
    kept: 1,
    explained: 0.9,
    components: [[1]],
    mean: [0.5],
  });
  if (reply.rhs?.kind !== 'sindy') throw new Error('SINDy の右辺のはず');
  expect(reply.rhs.nnz).toBe(1);
  expect(reply.rhs.equations).toEqual([{ target: 'z', groups: [{ rate: '', terms: [{ monomial: '1', xi: 0.5 }] }] }]);
  expect((await post<PartsReply>('parts', { experiment: 'e', part: name })).reply.trail.shown).toEqual(['soma']);
});

test('a KAM model has no learners, points to its training input and answers its conditions and Kepler-style figures', async () => {
  const view = (await post<ExperimentView>('experiment', { experiment: 'k' })).reply;
  expect(view.models[0].parts).toEqual([]);
  expect(view.models[0].train).toBe('s');
  const { reply } = await post<ModelsReply>('models', { experiment: 'k', model: 'n', window: [0, 50] });
  expect(reply.parts).toEqual([]);
  expect(reply.kam?.groups).toEqual([{ gates: ['M', 'H'], latent: 'm+h.z1', weights: [0.4, 0.6], criterion: 0.2 }]);
  const figures = reply.kam?.figures;
  expect(figures?.window).toEqual([0, 50]);
  expect(figures?.train).toBe('s');
  expect(figures?.current.map(l => l.label)).toEqual(['I']);
  expect(figures?.membrane.map(l => l.label)).toEqual(['V']);
  expect(figures?.potentials.map(p => p.lines.map(l => l.label))).toEqual([['v_m', 'v_h', 'ψ']]);
  expect(figures?.voltage.map(l => l.label)).toEqual(['元のモデル', '縮約']);
  expect(figures?.phase[0].lines.every(l => l.points.length > 0)).toBe(true);
  const plain = (await post<ModelsReply>('models', { experiment: 'e', model: 'm' })).reply;
  expect(plain.kam).toBeNull();
});

test('unknown ops and unknown jobs are refused', async () => {
  expect((await post('shell', {})).status).toBe(404);
  expect((await post('json', { run: '..', path: 'etc/passwd.json' })).status).toBe(404);
  expect((await post('launch', { job: { action: 'rm' } })).status).toBe(400);
  expect((await post('experiment', { experiment: 'x' })).status).toBe(400);
});

/** 開いた run を数える読み口。口ごとに、開いた run の集合を返す。 */
function recording(): { ctx: Context; opened: Set<string> } {
  const files = runFiles(runs);
  const opened = new Set<string>();
  const note = (run: string) => opened.add(run);
  const watched: RunFiles = {
    column: ref => (note(ref.run), files.column(ref)),
    json: <T>(run: string, file: string): T => (note(run), files.json<T>(run, file)),
    text: (run, file) => (note(run), files.text(run, file)),
    dirs: (run, dir) => (note(run), files.dirs(run, dir)),
  };
  return { ctx: { runs, files: watched, metrics: seriesComparison(watched) }, opened };
}

test('each place opens only its own run and the runs its spec points to by hash', async () => {
  const view = (await post<ExperimentView>('experiment', { experiment: 'e' })).reply;
  const read = (answer: (ctx: Context) => unknown) => {
    const { ctx, opened } = recording();
    answer(ctx);
    return [...opened].sort();
  };
  expect(read(ctx => parts(ctx, { experiment: 'e', part: view.models[0].parts[0] }))).toEqual(
    [dirOf(ORIGINAL), dirOf(PART)].sort(),
  );
  expect(read(ctx => models(ctx, { experiment: 'k', model: 'n' }))).toEqual([dirOf(KAM_MODEL)]);
  expect(read(ctx => models(ctx, { experiment: 'e', model: 'm' }))).toEqual([dirOf(MODEL)]);
  const pair = { input: 's', pair: { model: 'm', point: 0 }, whole: { point: 0, comps: ['soma'] } };
  // 1 組の潜在空間は、置換系の spec が指すモデルと、その spec が指す学習器の decoder も読む（入力の上の圧縮）
  expect(read(ctx => inputs(ctx, { experiment: 'e', ...pair }))).toEqual(
    [dirOf(ORIGINAL), dirOf(REPLACED), dirOf(MODEL), dirOf(PART)].sort(),
  );
});
