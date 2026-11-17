// 学習器（runs/parts/<学習器>/）の答え。学習器の run だけを読む：学習区画ごとの学習データの群の状態と膜電位の写し、潜在の軌跡、再構成誤差、PCA の寄与率、SINDy の係数。
import type { Line, PartsReply, PartsRequest } from '../../shared/api.ts';
import { fitArray, RUN, type FitMeta, type PartJson } from '../../shared/runs.ts';
import type { Spec } from '../../shared/spec.ts';
import { experimentStatus } from '../run/status.ts';
import { learnedRun, type Context } from './experiments.ts';
import { spanOf } from './inputs.ts';
import { line, trajectory } from './waves.ts';

/** 潜在の軌跡の軸。群の潜在が 2 次元以上なら先頭の 2 つ、1 次元なら V とその潜在。 */
export function pathAxes(part: PartJson): PartsReply['axes'] {
  const names = part.latents;
  if (names.length >= 2) return { x: names[0], y: names[1], xIsVoltage: false };
  return { x: 'V', y: names[0] ?? '', xIsVoltage: true };
}

/** 学習器が学習した学習データの、実験の中の名前（spec が指す run の名前）。 */
function dataName(runs: string, experiment: string, run: string): string {
  const model = experimentStatus(runs, experiment).models.find(m => m.parts.some(p => p.id === run));
  if (!model) throw new Error(`${experiment} に ${run} を使うモデルが無い`);
  return model.data.name;
}

/** 学習器 1 本の、区画 comps の線と中身。 */
export function parts(ctx: Context, request: PartsRequest): PartsReply {
  const run = learnedRun(ctx.runs, request.experiment, 'parts', request.part);
  const spec = ctx.files.json<Spec>(run, RUN.spec);
  if (spec.kind !== 'part') throw new Error(`${run} は学習器の run でない`);
  const part = ctx.files.json<PartJson>(run, RUN.part);
  const meta = ctx.files.json<FitMeta>(run, RUN.fit);
  const comps = request.comps ?? part.comps.slice(0, 1);
  const duration = meta.n * meta.dt;
  const span = spanOf(meta, request.window, duration);
  const idOf = (name: string) => {
    const id = meta.names[name];
    if (id === undefined) throw new Error(`${run} は区画 ${name} で学習していない`);
    return id;
  };
  const latentOf = (name: string, latent: string) => fitArray(run, meta, idOf(name), 'latent', part.latents.indexOf(latent));
  const axes = pathAxes(part);
  const gates: Line[] =
    comps.length > 0
      ? part.gates.map((gate, i) => ({ label: gate.toLowerCase(), points: line(ctx.files, fitArray(run, meta, idOf(comps[0]), 'gates', i), span) }))
      : [];
  const pca = part.preprocessor.kind === 'pca' ? (part.preprocessor as { explained_variance_ratio: number[]; n_components: number }) : null;
  const sindy = part.closure.kind === 'sindy' ? (part.closure as { targets: string[]; features: string[]; xi: number[][] }) : null;
  return {
    run,
    data: dataName(ctx.runs, request.experiment, run),
    label: part.gates.map(gate => gate.toLowerCase()).join('+'),
    comps: part.comps,
    shown: comps,
    latents: part.latents,
    compressor: part.preprocessor.kind,
    closure: part.closure.kind,
    duration,
    gates,
    recon: comps.map(name => ({ label: name, points: line(ctx.files, fitArray(run, meta, idOf(name), 'recon_rmse'), span) })),
    path: comps.map(name => ({
      label: name,
      points: trajectory(ctx.files, axes.xIsVoltage ? fitArray(run, meta, idOf(name), 'V') : latentOf(name, axes.x), latentOf(name, axes.y), span),
    })),
    axes,
    pca: pca && { ratios: pca.explained_variance_ratio, kept: pca.n_components },
    sindy: sindy && { targets: sindy.targets, features: sindy.features, xi: sindy.xi },
  };
}
