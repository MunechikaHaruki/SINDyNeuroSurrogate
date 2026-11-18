// 学習器（runs/parts/<学習器>/）の答え。学習器の run と、その spec が指す学習データの run（data/<spec.data>/。学習しない KAM は指さない）だけを読む。
// 答えの欄は学習器の run の置き場に 1 つずつ対応する：組み立て（spec.json）、decode の曲線（compressor/decode.json）、圧縮器と右辺に固有の中身
// （compressor/own.json・rhs/own.json）、学習データの上の中身（fit/<区画>/：潜在・再構成誤差・潜在の傾きの正解（前進差分）・学習器の右辺の時系列と、全区間のまとめ）。
import type { CompressorOwn, Fit, Line, PartsReply, PartsRequest, RhsOwn, SindyEquation } from '../../shared/api.ts';
import {
  compArray,
  fitArray,
  RUN,
  type ArrayRef,
  type CompressorJson,
  type DecodeJson,
  type FitSummary,
  type RhsJson,
  type WaveMeta,
} from '../../shared/runs.ts';
import type { Spec } from '../../shared/spec.ts';
import { experimentStatus } from '../run/status.ts';
import { learnedRun, type Context } from './experiments.ts';
import { spanOf, wavesOf } from './inputs.ts';
import { line, slopeCompare, trajectory } from './waves.ts';

type PartSpec = Extract<Spec, { kind: 'part' }>;

/** 群の潜在の名前（群の名前で修飾し z1 から。h+n.z1）。 */
export function latentNames(spec: PartSpec): string[] {
  const label = spec.group.gates.map(gate => gate.toLowerCase()).join('+');
  return Array.from({ length: spec.group.d }, (_, i) => `${label}.z${i + 1}`);
}

/** 潜在の軌跡の軸。群の潜在が 2 次元以上なら先頭の 2 つ、1 次元なら V とその潜在。 */
export function pathAxes(latents: string[]): Fit['axes'] {
  if (latents.length >= 2) return { x: latents[0], y: latents[1], xIsVoltage: false };
  return { x: 'V', y: latents[0] ?? '', xIsVoltage: true };
}

/** 学習器が学習した学習データの、実験の中の名前。 */
function dataName(runs: string, experiment: string, run: string): string {
  const model = experimentStatus(runs, experiment).models.find(m => m.parts.some(p => p.id === run));
  if (!model?.data) throw new Error(`${experiment} に ${run} を学習データの上で使うモデルが無い`);
  return model.data.name;
}

/** 設定の値の並び（{ threshold: 0.1 } → threshold 0.1）。 */
function params(values: Record<string, unknown>): string[] {
  return Object.entries(values).map(
    ([key, value]) => `${key} ${typeof value === 'object' ? JSON.stringify(value) : String(value)}`,
  );
}

/** 組み立ての表（spec.json）。細胞・形・群のゲート・圧縮器・潜在・右辺。 */
export function setupOf(spec: PartSpec): PartsReply['setup'] {
  const { group } = spec;
  return [
    { label: '細胞', value: spec.cell },
    { label: '形', value: spec.form },
    { label: '群のゲート', value: group.gates.join('・') },
    { label: '圧縮器', value: [group.compressor, ...params(group.compressor_params)].join('・') },
    { label: '潜在', value: `${latentNames(spec).join('・')}（${group.d} 次元）` },
    { label: '右辺', value: group.rhs },
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

/** decode の曲線（compressor/decode.json）を、潜在の軸ごとのゲートの線にする。 */
export function decodeOf(decode: DecodeJson): PartsReply['decode'] {
  return decode.axes.map(axis => ({
    latent: axis.latent,
    lines: decode.gates.map((gate, i) => ({ label: gate, points: axis.z.map((x, j) => ({ x, y: axis.gates[i][j] })) })),
  }));
}

/** 圧縮器に固有の中身（compressor/own.json）。 */
export function compressorOf(own: CompressorJson): CompressorOwn {
  if (own.kind === 'pca')
    return {
      kind: 'pca',
      ratios: own.explained_variance_ratio,
      kept: own.n_components,
      components: own.components,
      mean: own.mean,
    };
  return own;
}

/** 右辺に固有の中身（rhs/own.json）。SINDy は項を式にまとめ、項ライブラリとほかの設定は spec から。 */
export function rhsOf(spec: PartSpec, own: RhsJson): RhsOwn {
  if (own.kind !== 'sindy') return own;
  const { library_specs: library, ...rest } = spec.group.rhs_params;
  let nnz = 0;
  for (const row of own.xi) for (const v of row) if (v !== 0) nnz += 1;
  return {
    kind: 'sindy',
    library: (library ?? []).map(l => (l.gates ? `${l.type}（${l.gates}）` : l.type)),
    settings: params(rest),
    nnz,
    size: own.xi.length * own.features.length,
    equations: equationsOf(own),
  };
}

/** 学習データの上の中身（fit/）。学習データの run の目録 meta の上で、区画 comps の線と、学習した全区画の全区間のまとめ。 */
function fitOf(
  ctx: Context,
  request: PartsRequest,
  run: string,
  data: string,
  meta: WaveMeta,
  gates: string[],
  latents: string[],
): Fit {
  const nameOf = (id: number) => Object.entries(meta.names ?? {}).find(([, v]) => v === id)?.[0] ?? String(id);
  const learned = ctx.files
    .dirs(run, RUN.fit)
    .map(Number)
    .sort((a, b) => a - b);
  const comps = learned.map(nameOf);
  const shown = request.comps ?? comps.slice(0, 1);
  const duration = meta.n[0] * meta.dt;
  const span = spanOf(meta, request.window, duration);
  const idOf = (name: string) => {
    const id = meta.names?.[name];
    if (id === undefined) throw new Error(`${data} に区画 ${name} が無い`);
    return id;
  };
  const dataOf = (name: string, variable: string) => compArray(data, RUN.waves, meta, 0, idOf(name), variable);
  const arrayOf = (name: string, file: 'latent' | 'slope' | 'rhs', latent: string) =>
    fitArray(run, meta, idOf(name), file, latents.indexOf(latent));
  const named = (name: string, comp: string) => (shown.length > 1 ? `${name}（${comp}）` : name);
  const draw = (label: string, ref: ArrayRef): Line => ({ label, points: line(ctx.files, ref, span) });
  const axes = pathAxes(latents);
  return {
    data: dataName(ctx.runs, request.experiment, run),
    comps,
    shown,
    duration,
    window: [span.start, span.end],
    voltage: shown.map(name => draw(name, dataOf(name, 'V'))),
    gates: shown.flatMap(name => gates.map(gate => draw(named(gate, name), dataOf(name, gate)))),
    latent: shown.flatMap(name => latents.map(latent => draw(named(latent, name), arrayOf(name, 'latent', latent)))),
    recon: shown.map(name => draw(name, fitArray(run, meta, idOf(name), 'recon_rmse'))),
    manifold: shown.flatMap(name =>
      gates.map(gate => ({
        label: named(gate, name),
        points: trajectory(ctx.files, arrayOf(name, 'latent', latents[0]), dataOf(name, gate), span),
      })),
    ),
    path: shown.map(name => ({
      label: name,
      points: trajectory(
        ctx.files,
        axes.xIsVoltage ? dataOf(name, 'V') : arrayOf(name, 'latent', axes.x),
        arrayOf(name, 'latent', axes.y),
        span,
      ),
    })),
    axes,
    slopes: slopeCompare(
      ctx.files,
      shown.flatMap(name =>
        latents.map(latent => ({
          label: named(latent, name),
          slope: arrayOf(name, 'slope', latent),
          rhs: arrayOf(name, 'rhs', latent),
        })),
      ),
      span,
    ),
    reconstruction: learned.map(id => {
      const summary = ctx.files.json<FitSummary>(run, RUN.fitSummary(id));
      return {
        comp: nameOf(id),
        mse: summary.reconstruction_mse,
        ratio: summary.reconstruction_mse_ratio,
        ...meanMax(ctx.files.column(fitArray(run, meta, id, 'recon_rmse'))),
      };
    }),
  };
}

/** 学習器 1 本の答え。置き場ごとに 1 欄。学習データを指さない（KAM の）学習器は学習器の run だけを読む。 */
export function parts(ctx: Context, request: PartsRequest): PartsReply {
  const run = learnedRun(ctx.runs, request.experiment, 'parts', request.part);
  const spec = ctx.files.json<Spec>(run, RUN.spec);
  if (spec.kind !== 'part') throw new Error(`${run} は学習器の run でない`);
  const latents = latentNames(spec);
  const decode = ctx.files.json<DecodeJson>(run, RUN.decode);
  let fit: Fit | null = null;
  if (spec.data !== undefined) {
    const data = `data/${spec.data}`;
    const meta = wavesOf(ctx, data);
    fit = fitOf(ctx, request, run, data, meta, spec.group.gates, latents);
  }
  return {
    run,
    label: spec.group.gates.map(gate => gate.toLowerCase()).join('+'),
    setup: setupOf(spec),
    latents,
    gateOrder: decode.order,
    decode: decodeOf(decode),
    compressor: compressorOf(ctx.files.json<CompressorJson>(run, RUN.compressor)),
    rhs: rhsOf(spec, ctx.files.json<RhsJson>(run, RUN.rhs_own)),
    fit,
  };
}
