// 学習器（runs/parts/<学習器>/）の答え。学習器の run と、その spec が指す学習データの run（data/<spec.data>/）だけを読む：
// 組み立ての表、学習データの上の膜電位・群のゲート・潜在・再構成誤差の時系列、ゲートと潜在の軌跡、学習の指標、PCA の寄与率、SINDy の係数。
import type { Line, PartsReply, PartsRequest } from '../../shared/api.ts';
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

/** 設定の値の並び（{ hidden: 16 } → hidden 16）。 */
function params(values: Record<string, unknown>): string[] {
  return Object.entries(values).map(
    ([key, value]) => `${key} ${typeof value === 'object' ? JSON.stringify(value) : String(value)}`,
  );
}

/** 圧縮器の中身のうち、組み立ての表の圧縮器の行に添えない欄。 */
const ELSEWHERE = new Set(['kind', 'gates', 'explained_variance_ratio', 'n_components']);

/** 組み立ての表。spec の群と、part.json の圧縮器の中身（AE の隠れ層など）から。 */
export function setupOf(spec: Extract<Spec, { kind: 'part' }>, part: PartJson): PartsReply['setup'] {
  const { group } = spec;
  // 種類・ゲート・寄与率は表のほかの行やほかの図が持つので、残りの中身（AE の隠れ層など）だけを圧縮器の行に添える
  const inner = Object.fromEntries(Object.entries(part.preprocessor).filter(([key]) => !ELSEWHERE.has(key)));
  const { library_specs: library, ...rhs } = group.rhs_params;
  const rows = [
    { label: '細胞', value: spec.cell },
    { label: '形', value: spec.form },
    { label: '群のゲート', value: group.gates.join('・') },
    { label: '圧縮器', value: [group.compressor, ...params({ ...group.compressor_params, ...inner })].join('・') },
    { label: '潜在', value: `${part.latents.join('・')}（${group.d} 次元）` },
    { label: '右辺', value: [group.rhs, ...params(rhs)].join('・') },
  ];
  if (library?.length)
    rows.push({
      label: '項ライブラリ',
      value: library.map(l => (l.gates ? `${l.type}（${l.gates}）` : l.type)).join('・'),
    });
  rows.push({ label: '学習した区画', value: part.comps.join('・') });
  return rows;
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
  const pca =
    part.preprocessor.kind === 'pca'
      ? (part.preprocessor as { explained_variance_ratio: number[]; n_components: number })
      : null;
  const sindy =
    part.closure.kind === 'sindy' ? (part.closure as { targets: string[]; features: string[]; xi: number[][] }) : null;
  return {
    run,
    data: dataName(ctx.runs, request.experiment, run),
    label: part.gates.map(gate => gate.toLowerCase()).join('+'),
    setup: setupOf(spec, part),
    comps: part.comps,
    shown: comps,
    latents: part.latents,
    compressor: part.preprocessor.kind,
    closure: part.closure.kind,
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
    metrics: part.metrics,
    fit: part.comps.map(name => ({
      comp: name,
      ...meanMax(ctx.files.column(fitArray(run, meta, idOf(name), 'recon_rmse'))),
    })),
    pca: pca && { ratios: pca.explained_variance_ratio, kept: pca.n_components },
    sindy: sindy && { targets: sindy.targets, features: sindy.features, xi: sindy.xi },
  };
}
