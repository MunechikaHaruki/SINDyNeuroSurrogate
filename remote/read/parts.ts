// 学習器（runs/parts/<学習器>/）の答え。学習器の run と、その spec が指す学習データの run（data/<spec.data>/）だけを読む。KAM は学習器を持たず、モデルの答え（models.ts）が持つ。
// 答えの欄は学習器の run の置き場に 1 つずつ対応する：組み立て（spec.json）、ゲートの空間（compressor/decode.json の曲線、学習データの軌道、AE は compressor/plane.json の encoder の格子）、圧縮器と右辺に固有の中身
// （compressor/own.json・rhs/own.json）、学習データの上の中身（fit/<区画>/：潜在の時系列、decoder の曲線に重ねる学習データの点、ゲートごとの全区間の再構成誤差）。
import type {
  CompressorOwn,
  Fit,
  GateSpace,
  Line,
  PartsReply,
  PartsRequest,
  RhsField,
  RhsOwn,
  SindyEquation,
} from '../../shared/api.ts';
import {
  compArray,
  fitArray,
  RUN,
  type ArrayRef,
  type CompressorJson,
  type DecodeJson,
  type PlaneJson,
  type FieldJson,
  type RhsJson,
  type WaveMeta,
} from '../../shared/runs.ts';
import type { Spec } from '../../shared/spec.ts';
import { experimentStatus } from '../run/status.ts';
import { learnedRun, type Context } from './experiments.ts';
import { spanOf, wavesOf } from './inputs.ts';
import { decodedCurves, errorOf, rebuiltOf, reconstructor } from './comparison.ts';
import { line, strided, trajectory } from './waves.ts';

type PartSpec = Extract<Spec, { kind: 'part' }>;

/** 群の潜在の名前（群の名前で修飾し z1 から。h+n.z1）。 */
export function latentNames(spec: PartSpec): string[] {
  const label = spec.group.gates.map(gate => gate.toLowerCase()).join('+');
  return Array.from({ length: spec.group.d }, (_, i) => `${label}.z${i + 1}`);
}

/** 潜在の軌跡の軸。群の潜在が 2 次元以上なら先頭の 2 つ（単位が同じ）、1 次元なら V とその潜在（学習器の潜在は電位でないので単位が違う）。 */
export function pathAxes(latents: string[]): Fit['axes'] {
  if (latents.length >= 2) return { x: latents[0], y: latents[1], xIsVoltage: false, sameScale: true };
  return { x: 'V', y: latents[0] ?? '', xIsVoltage: true, sameScale: false };
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

/** ゲートの空間の軌道に出す点の数の上限（区画ごと）。 */
const SPACE_POINTS = 6000;

/** 圧縮の軸（compressor/decode.json）を、潜在の軸ごとのゲートの空間の点の並びにする。z の増える順。 */
export function axesOf(decode: DecodeJson): GateSpace['axes'] {
  return decode.axes.map(axis => ({
    latent: axis.latent,
    points: axis.z.map((_, j) => decode.gates.map((_, i) => axis.gates[i][j])),
    z: axis.z,
  }));
}

/** 右辺の 1 列（V を固定した z の並びの値 f）の、安定な 0（z の増える向きに正から負へ横切る所）の z と、そこでの傾き ∂f/∂z。
 * 安定な 0 が 2 つ以上あれば、前の列の z（near）に近いもの。無ければ null。 */
export function stableRoot(
  z: number[],
  f: (number | null)[],
  near: number | null,
): { z: number; slope: number } | null {
  let best: { z: number; slope: number } | null = null;
  for (let i = 0; i + 1 < z.length; i++) {
    const a = f[i];
    const b = f[i + 1];
    if (a === null || b === null || !(a > 0 && b <= 0)) continue;
    const root = z[i] + (a / (a - b)) * (z[i + 1] - z[i]);
    const slope = (b - a) / (z[i + 1] - z[i]);
    if (best === null || (near !== null && Math.abs(root - near) < Math.abs(best.z - near))) best = { z: root, slope };
  }
  return best;
}

/** 右辺の格子（z の行 × V の列）から、V ごとの z∞(V) と τ(V) = −1/(∂f/∂z) の線を求める。安定な 0 の無い V は飛ばす。 */
export function steadyOf(
  v: number[],
  z: number[],
  grid: (number | null)[][],
  label: string,
): { steady: Line; tau: Line } {
  const steady: Line = { label, points: [] };
  const tau: Line = { label, points: [] };
  let near: number | null = null;
  for (let col = 0; col < v.length; col++) {
    const root = stableRoot(
      z,
      grid.map(row => row[col]),
      near,
    );
    if (root === null) continue;
    near = root.z;
    steady.points.push({ x: v[col], y: root.z });
    if (root.slope < 0) tau.points.push({ x: v[col], y: -1 / root.slope });
  }
  return { steady, tau };
}

/** 右辺の格子（rhs/field.json）に、学習器と元のモデルの z∞(V) と τ(V) を添える。無ければ（膜電位も同定する形）null。 */
export function fieldOf(field: FieldJson | null): RhsField | null {
  if (!field) return null;
  return {
    V: field.V,
    latents: field.latents.map(entry => {
      const learned = steadyOf(field.V, entry.z, entry.learned, '学習器');
      const original = steadyOf(field.V, entry.z, entry.original, '元のモデル');
      return { ...entry, steady: [original.steady, learned.steady], tau: [original.tau, learned.tau] };
    }),
  };
}

/** run の中の .json の文字を読む。無ければ null。 */
function jsonOrNull<T>(text: string | null): T | null {
  return text === null ? null : (JSON.parse(text) as T);
}

/** encoder の等高線の格子（compressor/plane.json）。ゲートの名前を群のゲートの中の位置にする。無ければ（PCA）空。 */
export function levelsOf(plane: PlaneJson | null, gates: string[]): GateSpace['levels'] {
  if (!plane) return [];
  return plane.planes.map(p => ({
    x: gates.indexOf(p.x),
    y: gates.indexOf(p.y),
    xs: p.xs,
    ys: p.ys,
    latents: p.latents,
  }));
}

/** 圧縮器に固有の中身（compressor/own.json）。PCA は保持した成分の寄与率の和も添える。 */
export function compressorOf(own: CompressorJson): CompressorOwn {
  if (own.kind === 'pca')
    return {
      kind: 'pca',
      ratios: own.explained_variance_ratio,
      kept: own.n_components,
      explained: own.explained_variance_ratio.slice(0, own.n_components).reduce((a, b) => a + b, 0),
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

/** decoder の図に重ねる学習データの点の数の上限（区画ごと）。 */
const DECODED_POINTS = 3000;

/** 学習データの上の中身（fit/）。学習データの run の目録 meta の上で、区画 comps の線と、decoder の曲線に重ねる点と、学習した全区画のゲートごとの再構成誤差。
 * 同じ区画と時間の範囲で、ゲートの空間の学習データの軌道 trail も返す。 */
function fitOf(
  ctx: Context,
  request: PartsRequest,
  run: string,
  data: string,
  meta: WaveMeta,
  decode: DecodeJson,
  own: CompressorJson,
  latents: string[],
): { fit: Fit; trail: GateSpace['data'] } {
  const gates = decode.gates;
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
  const latentColumns = (id: number) => latents.map((_, k) => ctx.files.column(fitArray(run, meta, id, 'latent', k)));
  const rebuild = reconstructor(decode, own);
  const arrayOf = (name: string, file: 'latent', latent: string) =>
    fitArray(run, meta, idOf(name), file, latents.indexOf(latent));
  const named = (name: string, comp: string) => (shown.length > 1 ? `${name}（${comp}）` : name);
  const draw = (label: string, ref: ArrayRef): Line => ({ label, points: line(ctx.files, ref, span) });
  const axes = pathAxes(latents);
  const from = Math.max(0, Math.ceil((span.start - span.t0) / span.dt));
  const to = Math.min(meta.n[0] - 1, Math.floor((span.end - span.t0) / span.dt));
  const trail = shown.map(name => ({
    label: name,
    points: strided(
      gates.map(gate => ctx.files.column(dataOf(name, gate))),
      from,
      to,
      SPACE_POINTS,
    ),
  }));
  const fit: Fit = {
    data: dataName(ctx.runs, request.experiment, run),
    comps,
    shown,
    duration,
    window: [span.start, span.end],
    voltage: shown.map(name => draw(name, dataOf(name, 'V'))),
    gates: shown.flatMap(name => gates.map(gate => draw(named(gate, name), dataOf(name, gate)))),
    latent: shown.flatMap(name => latents.map(latent => draw(named(latent, name), arrayOf(name, 'latent', latent)))),
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
    decoded: decode.axes.map((axis, k) => ({
      latent: axis.latent,
      curves: decodedCurves(decode, k),
      points: shown.flatMap(name => {
        const columns = [
          ctx.files.column(arrayOf(name, 'latent', latents[k])),
          ...gates.map(gate => ctx.files.column(dataOf(name, gate))),
        ];
        const sampled = strided(columns, from, to, DECODED_POINTS);
        return gates.map((gate, g) => ({
          label: named(gate, name),
          points: sampled.map(p => ({ x: p[0], y: p[g + 1] })),
        }));
      }),
      section: decode.axes.length > 1,
      order: decode.order,
    })),
    errors: learned.flatMap(id => {
      const rebuilt = rebuiltOf(rebuild, latentColumns(id), gates.length);
      const comp = nameOf(id);
      return gates.map((gate, g) => ({
        comp,
        gate,
        ...errorOf(ctx.files.column(compArray(data, RUN.waves, meta, 0, id, gate)), rebuilt?.[g] ?? null),
      }));
    }),
  };
  return { fit, trail };
}

/** 学習器 1 本の答え。置き場ごとに 1 欄。 */
export function parts(ctx: Context, request: PartsRequest): PartsReply {
  const run = learnedRun(ctx.runs, request.experiment, 'parts', request.part);
  const spec = ctx.files.json<Spec>(run, RUN.spec);
  if (spec.kind !== 'part') throw new Error(`${run} は学習器の run でない`);
  const latents = latentNames(spec);
  const decode = ctx.files.json<DecodeJson>(run, RUN.decode);
  const own = ctx.files.json<CompressorJson>(run, RUN.compressor);
  const data = `data/${spec.data}`;
  const { fit, trail } = fitOf(ctx, request, run, data, wavesOf(ctx, data), decode, own, latents);
  return {
    run,
    label: spec.group.gates.map(gate => gate.toLowerCase()).join('+'),
    setup: setupOf(spec),
    latents,
    gates: decode.gates,
    gateOrder: decode.order,
    space: {
      axes: axesOf(decode),
      data: trail,
      levels: levelsOf(jsonOrNull<PlaneJson>(ctx.files.text(run, RUN.plane)), decode.gates),
    },
    field: fieldOf(jsonOrNull<FieldJson>(ctx.files.text(run, RUN.field))),
    compressor: compressorOf(own),
    rhs: rhsOf(spec, ctx.files.json<RhsJson>(run, RUN.rhs_own)),
    fit,
  };
}
