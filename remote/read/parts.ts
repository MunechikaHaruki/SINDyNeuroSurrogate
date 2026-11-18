// 学習器（runs/parts/<学習器>/）の答え。学習器の run と、その spec が指す学習データの run（data/<spec.data>/）だけを読む。
// どの学習器にもある中身（組み立ての表、学習データの上の膜電位・群のゲート・潜在・再構成誤差の時系列、ゲートと潜在の軌跡）と、
// 型ごとの中身（KAM の条件と再構成、SINDy の学習した右辺と圧縮器の再構成・圧縮器ごとの中身）を分けて返す。
import type { Learner, Line, PartsReply, PartsRequest, Reconstruction, SindyEquation } from '../../shared/api.ts';
import { compArray, fitArray, gatesOf, RUN, type ArrayRef, type PartJson } from '../../shared/runs.ts';
import type { Spec } from '../../shared/spec.ts';
import { experimentStatus } from '../run/status.ts';
import { learnedRun, type Context } from './experiments.ts';
import { spanOf, wavesOf } from './inputs.ts';
import { line, trajectory } from './waves.ts';

/** 潜在の軌跡の軸。群の潜在が 2 次元以上なら先頭の 2 つ、1 次元なら V とその潜在。 */
export function pathAxes(part: PartJson): PartsReply['axes'] {
  const names = part.latents;
  if (names.length >= 2) return { x: names[0], y: names[1], xIsVoltage: false };
  return { x: 'V', y: names[0] ?? '', xIsVoltage: true };
}

/** 学習器が学習した学習データの、実験の中の名前。 */
function dataName(runs: string, experiment: string, run: string): string {
  const model = experimentStatus(runs, experiment).models.find(m => m.parts.some(p => p.id === run));
  if (!model) throw new Error(`${experiment} に ${run} を使うモデルが無い`);
  return model.data.name;
}

/** 設定の値の並び（{ threshold: 0.1 } → threshold 0.1）。 */
function params(values: Record<string, unknown>): string[] {
  return Object.entries(values).map(
    ([key, value]) => `${key} ${typeof value === 'object' ? JSON.stringify(value) : String(value)}`,
  );
}

/** 組み立ての表。spec の細胞・形・群から。右辺の中身（項ライブラリなど）は型ごとの中身が持つ。 */
export function setupOf(spec: Extract<Spec, { kind: 'part' }>, part: PartJson): PartsReply['setup'] {
  const { group } = spec;
  return [
    { label: '細胞', value: spec.cell },
    { label: '形', value: spec.form },
    { label: '群のゲート', value: group.gates.join('・') },
    { label: '圧縮器', value: [group.compressor, ...params(group.compressor_params)].join('・') },
    { label: '潜在', value: `${part.latents.join('・')}（${group.d} 次元）` },
    { label: '右辺', value: group.rhs },
    { label: '学習した区画', value: part.comps.join('・') },
  ];
}

/** sympy の LaTeX のレート関数 1 つ（α_{n traub}(V)）。 */
const RATE_CALL = String.raw`\\(?:alpha|beta)_\{[^}]*\}\{\\left\(V \\right\)\}`;
/** 項の中のレートの因子。レートの和の括弧（(α_n + β_n)(V)）か、レート関数 1 つ。 */
const RATE = new RegExp(String.raw`\\left\(${RATE_CALL}(?: \+ ${RATE_CALL})+\\right\)|${RATE_CALL}`);

/** 項の LaTeX を、レートの因子（無ければ空）と残りの単項（無ければ 1）に分ける。 */
export function factorOf(feature: string): { rate: string; monomial: string } {
  const rate = feature.match(RATE)?.[0] ?? '';
  const monomial = feature.replace(rate, '').replace(/\s+/g, ' ').trim();
  return { rate, monomial: monomial || '1' };
}

/** SINDy の右辺を、対象ごとに、レートの因子ごとの項の並びにする。因子は項ライブラリに現れる順で、因子の無い項は最後。 */
export function equationsOf(sindy: { targets: string[]; features: string[]; xi: number[][] }): SindyEquation[] {
  const factors = sindy.features.map(factorOf);
  const rates = [...new Set(factors.map(f => f.rate))].sort((a, b) => Number(a === '') - Number(b === ''));
  return sindy.targets.map((target, t) => ({
    target,
    groups: rates.map(rate => ({
      rate,
      terms: factors.flatMap((f, i) => (f.rate === rate ? [{ monomial: f.monomial, xi: sindy.xi[t][i] }] : [])),
    })),
  }));
}

/** 1 列の有限な値の平均と最大。 */
function meanMax(values: Float64Array): { mean: number; max: number } {
  let sum = 0;
  let count = 0;
  let max = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    sum += v;
    count += 1;
    max = Math.max(max, v);
  }
  return { mean: count > 0 ? sum / count : NaN, max };
}

/** 型ごとの中身。右辺の種類で分け、圧縮器の再構成の指標は鍵の頭の圧縮器の種類で読む。 */
export function learnerOf(
  spec: Extract<Spec, { kind: 'part' }>,
  part: PartJson,
  rmse: Reconstruction['rmse'],
): Learner {
  const metric = (key: string) => part.metrics[key] ?? null;
  const reconstruction = (kind: string): Reconstruction => ({
    mse: metric(`${kind}/reconstruction_mse`),
    ratio: metric(`${kind}/reconstruction_mse_ratio`),
    rmse,
  });
  const { closure, preprocessor } = part;
  if (closure.kind === 'kam')
    return { kind: 'kam', criterion: metric('kam/criterion'), reconstruction: reconstruction('kam') };
  if (closure.kind !== 'sindy') throw new Error(`右辺 ${closure.kind} の学習器は答えの形を持たない`);
  const sindy = closure as { targets: string[]; features: string[]; xi: number[][] };
  const own = (): Extract<Learner, { kind: 'sindy' }>['compressor']['own'] => {
    if (preprocessor.kind === 'pca') {
      const pca = preprocessor as { explained_variance_ratio: number[]; n_components: number };
      return { kind: 'pca', ratios: pca.explained_variance_ratio, kept: pca.n_components };
    }
    if (preprocessor.kind === 'ae') return { kind: 'ae', hidden: (preprocessor as { hidden: number }).hidden };
    throw new Error(`圧縮器 ${preprocessor.kind} は答えの形を持たない`);
  };
  const { library_specs: library, ...rest } = spec.group.rhs_params;
  let nnz = 0;
  for (const row of sindy.xi) for (const v of row) if (v !== 0) nnz += 1;
  return {
    kind: 'sindy',
    rhs: {
      library: (library ?? []).map(l => (l.gates ? `${l.type}（${l.gates}）` : l.type)),
      settings: params(rest),
      nnz,
      size: sindy.xi.length * sindy.features.length,
      equations: equationsOf(sindy),
    },
    compressor: { kind: preprocessor.kind, reconstruction: reconstruction(preprocessor.kind), own: own() },
  };
}

/** 学習器 1 本の、区画 comps の線と中身。 */
export function parts(ctx: Context, request: PartsRequest): PartsReply {
  const run = learnedRun(ctx.runs, request.experiment, 'parts', request.part);
  const spec = ctx.files.json<Spec>(run, RUN.spec);
  if (spec.kind !== 'part') throw new Error(`${run} は学習器の run でない`);
  const part = ctx.files.json<PartJson>(run, RUN.part);
  const comps = request.comps ?? part.comps.slice(0, 1);
  const data = `data/${spec.data}`;
  const meta = wavesOf(ctx, data);
  const duration = meta.n[0] * meta.dt;
  const span = spanOf(meta, request.window, duration);
  const idOf = (name: string) => {
    const id = meta.names?.[name];
    if (id === undefined) throw new Error(`${data} に区画 ${name} が無い`);
    return id;
  };
  const dataOf = (name: string, variable: string) => compArray(data, RUN.waves, meta, 0, idOf(name), variable);
  const latentOf = (name: string, latent: string) =>
    fitArray(run, meta, idOf(name), 'latent', part.latents.indexOf(latent));
  const named = (name: string, comp: string) => (comps.length > 1 ? `${name}（${comp}）` : name);
  const draw = (label: string, ref: ArrayRef): Line => ({ label, points: line(ctx.files, ref, span) });
  const axes = pathAxes(part);
  const rmse = part.comps.map(name => ({
    comp: name,
    ...meanMax(ctx.files.column(fitArray(run, meta, idOf(name), 'recon_rmse'))),
  }));
  return {
    run,
    data: dataName(ctx.runs, request.experiment, run),
    label: part.gates.map(gate => gate.toLowerCase()).join('+'),
    setup: setupOf(spec, part),
    comps: part.comps,
    shown: comps,
    latents: part.latents,
    duration,
    window: [span.start, span.end],
    gateOrder: gatesOf(meta, idOf(part.comps[0] ?? comps[0])),
    voltage: comps.map(name => draw(name, dataOf(name, 'V'))),
    gates: comps.flatMap(name => part.gates.map(gate => draw(named(gate, name), dataOf(name, gate)))),
    latent: comps.flatMap(name => part.latents.map(latent => draw(named(latent, name), latentOf(name, latent)))),
    recon: comps.map(name => draw(name, fitArray(run, meta, idOf(name), 'recon_rmse'))),
    manifold: comps.flatMap(name =>
      part.gates.map(gate => ({
        label: named(gate, name),
        points: trajectory(ctx.files, latentOf(name, part.latents[0]), dataOf(name, gate), span),
      })),
    ),
    path: comps.map(name => ({
      label: name,
      points: trajectory(
        ctx.files,
        axes.xIsVoltage ? dataOf(name, 'V') : latentOf(name, axes.x),
        latentOf(name, axes.y),
        span,
      ),
    })),
    axes,
    learner: learnerOf(spec, part, rmse),
  };
}
