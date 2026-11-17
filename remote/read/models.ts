// モデル（runs/models/<モデル>/）の答え。組んだ学習データと学習器の run と、入力の掃引の 1 点での原系との比べ（膜電位、潜在変数と相平面（原系は列の圧縮器で潜在へ写したもの）、全区画の膜電位・内部電流・ゲート、1 組の指標）。
import type { Line, ModelCompare, ModelsReply, ModelsRequest } from '../../shared/api.ts';
import {
  compArray,
  currentArray,
  gatesOf,
  internalArray,
  RUN,
  variablesOf,
  type ArrayRef,
  type WaveMeta,
} from '../../shared/runs.ts';
import { experimentStatus } from '../run/status.ts';
import { type Context, modelOf, planOf, plans } from './experiments.ts';
import { compOf, inputView, originalOf, spanOf, wavesOf } from './inputs.ts';
import { line, trajectory, type Span } from './waves.ts';

/** 原系の run original と置換系の run の、掃引の点 k・区画 comp の線。comps は全区画の図に出す区画、axes は相平面の軸（無い名前なら V と最初の潜在）。replaced でなければ（発散した点）置換系を描かない。 */
function compareLines(
  ctx: Context,
  at: { original: string; run: string; k: number; comp: string; replaced: boolean; projected: boolean },
  span: Span,
  comps: string[],
  axes: { x: string; y: string },
): ModelCompare {
  const { original, run, k, comp, replaced } = at;
  const meta = wavesOf(ctx, original);
  const own = wavesOf(ctx, run);
  const id = meta.names?.[comp] ?? 0;
  const draw = (label: string, ref: ArrayRef): Line => ({ label, points: line(ctx.files, ref, span) });
  const root = RUN.projected(comp);
  const projected = at.projected ? ctx.files.json<WaveMeta>(run, RUN.meta(root)) : undefined;
  const names = projected ? variablesOf(projected, id).filter(name => name !== 'V') : [];
  const ownNames = variablesOf(own, id);
  const voltage = [draw('原系', compArray(original, RUN.waves, meta, k, id, 'V'))];
  if (replaced) voltage.push(draw('置換系', compArray(run, RUN.waves, own, k, id, 'V')));
  const latents: Record<string, Line[]> = {};
  for (const name of names) {
    latents[name] = [draw('原系', compArray(run, root, projected!, k, id, name))];
    if (replaced && ownNames.includes(name))
      latents[name].push(draw('置換系', compArray(run, RUN.waves, own, k, id, name)));
  }
  const choices = ['V', ...names];
  const chosen = {
    x: choices.includes(axes.x) ? axes.x : 'V',
    y: choices.includes(axes.y) ? axes.y : (names[0] ?? 'V'),
  };
  const phase: Line[] = [];
  if (names.length > 0) {
    const x = compArray(run, root, projected!, k, id, chosen.x);
    const y = compArray(run, root, projected!, k, id, chosen.y);
    phase.push({ label: '原系（射影）', points: trajectory(ctx.files, x, y, span) });
    if (replaced && ownNames.includes(chosen.x) && ownNames.includes(chosen.y))
      phase.push({
        label: '置換系',
        points: trajectory(
          ctx.files,
          compArray(run, RUN.waves, own, k, id, chosen.x),
          compArray(run, RUN.waves, own, k, id, chosen.y),
          span,
        ),
      });
  }
  const all: Line[] = [];
  const internal: Line[] = [];
  const gates: Line[] = [];
  for (const name of comps) {
    const compId = meta.names?.[name];
    if (compId === undefined) continue;
    all.push(draw(name, compArray(original, RUN.waves, meta, k, compId, 'V')));
    if (meta.nodes?.includes(compId)) internal.push(draw(name, internalArray(original, RUN.waves, meta, k, compId)));
    for (const gate of gatesOf(meta, compId))
      gates.push(
        draw(comps.length > 1 ? `${gate}（${name}）` : gate, compArray(original, RUN.waves, meta, k, compId, gate)),
      );
  }
  return {
    current: [draw('I_ext', currentArray(original, RUN.waves, meta, k))],
    voltage,
    latents,
    phase,
    axes: chosen,
    comps: all,
    internal,
    gates,
  };
}

/** モデル 1 つと、その入力での比べ。 */
export function models(ctx: Context, request: ModelsRequest): ModelsReply {
  const all = plans();
  const plan = planOf(all, request.experiment);
  const model = modelOf(plan, request.model);
  const found = experimentStatus(ctx.runs, plan.id);
  const reply: ModelsReply = {
    model: model.name,
    input: null,
    comp: '',
    point: 0,
    compare: null,
    waiting: '',
  };
  const name = request.input ?? Object.keys(plan.series)[0];
  if (!name) return { ...reply, waiting: '実験に入力が無い' };
  const original = originalOf(found, name);
  const input = inputView(ctx, plan, name, original);
  if (!input) return { ...reply, waiting: `${name} の元のモデルをまだ回していない` };
  const comp = compOf(input, request.comp);
  const point = Math.min(Math.max(0, request.point ?? input.values.length - 1), Math.max(0, input.points.length - 1));
  const pair = found.pairs.find(p => p.series === name && p.model === model.name);
  if (pair?.state !== 'done')
    return {
      ...reply,
      input,
      comp,
      point,
      waiting: pair ? `${name} をこのモデルでまだ回していない` : `${name} はこのモデルで回す組に無い`,
    };
  const metrics = ctx.metrics({
    original: original.id,
    columns: [pair.run.id],
    comp,
    delta: all.measures.coincidence_window,
  });
  const diverged = metrics.pairs[pair.run.id]?.[point]?.diverged ?? false;
  const span = spanOf(wavesOf(ctx, original.id), request.window, input.duration);
  const lines = compareLines(
    ctx,
    {
      original: original.id,
      run: pair.run.id,
      k: point,
      comp,
      replaced: !diverged,
      projected: input.replaceTargets.includes(comp),
    },
    span,
    request.comps ?? [],
    request.axes ?? { x: 'V', y: '' },
  );
  return { ...reply, input, comp, point, compare: lines };
}
