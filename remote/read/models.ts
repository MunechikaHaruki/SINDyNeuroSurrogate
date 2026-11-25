// モデル（runs/models/<モデル>/）の答え。モデルの run と、KAM のモデルはその spec が指す学習の入力の元のモデルの run だけを読む：組み立て（spec の細胞・形・吸収と、組んだ学習器か KAM の群）と、
// 組んだ 1 刻みの演算の回数（summary.json）と、KAM のモデルの図（kam.json の条件、equivalent/ の区画に入る電流と元のモデルの等価電位、reduced/ の縮約、学習の入力の膜電位。Kepler 1992 の図 2〜4 にあたる）。
import type { CostRow, KamFigures, KamGroup, Line, ModelsReply, ModelsRequest } from '../../shared/api.ts';
import { compArray, currentArray, RUN, type KamJson, type WaveMeta } from '../../shared/runs.ts';
import type { Group, Spec } from '../../shared/spec.ts';
import { specAt } from '../run/places.ts';
import { experimentStatus } from '../run/status.ts';
import { type Context, modelOf, planOf, plans } from './experiments.ts';
import { spanOf } from './inputs.ts';
import { line, trajectory } from './waves.ts';

/** 演算の並び（summary.json の cost/<側>/<演算>）。 */
const OPS = ['exp', 'div', 'pm', 'mul'];

/** summary.json の演算の回数を、演算ごとの行にする。 */
function costRows(summary: Record<string, number>): CostRow[] {
  return OPS.map(op => ({
    op,
    surrogate: summary[`cost/surrogate/${op}`],
    original: summary[`cost/original/${op}`],
    diff: summary[`cost/surr-orig/${op}`],
  }));
}

/** KAM の群の ψ の名前（群のゲートを小文字で + につないで z1 を添える。h+n.z1）。 */
export function kamLatent(group: Group): string {
  return `${group.gates.map(gate => gate.toLowerCase()).join('+')}.z1`;
}

/** 等価電位の線の名前（v_h）。 */
export function potentialLabel(gate: string): string {
  return `v_${gate.toLowerCase()}`;
}

/** KAM のモデルの図。run の equivalent/（元のモデルの等価電位）と reduced/（縮約）から、区画 comp の時間の範囲の線を組む。 */
function kamFigures(ctx: Context, run: string, kam: KamJson, train: string, request: ModelsRequest): KamFigures {
  const full = ctx.files.json<WaveMeta>(run, RUN.meta(RUN.equivalent));
  const reduced = ctx.files.json<WaveMeta>(run, RUN.meta(RUN.reduced));
  const duration = full.n[0] * full.dt;
  const span = spanOf(full, request.window, duration);
  const fullId = full.names?.[kam.comp] ?? 0;
  const reducedId = reduced.names?.[kam.comp] ?? 0;
  const ofFull = (variable: string) => compArray(run, RUN.equivalent, full, 0, fullId, variable);
  const ofReduced = (variable: string) => compArray(run, RUN.reduced, reduced, 0, reducedId, variable);
  const draw = (label: string, ref: ReturnType<typeof ofFull>): Line => ({ label, points: line(ctx.files, ref, span) });
  return {
    train,
    comp: kam.comp,
    duration,
    window: [span.start, span.end],
    current: [draw('I', currentArray(run, RUN.equivalent, full, 0))],
    membrane: [draw('V', ofFull('V')), ...kam.absorbed.map(gate => draw(potentialLabel(gate), ofFull(gate)))],
    potentials: kam.groups.map(group => ({
      latent: group.latent,
      lines: [...group.gates.map(gate => draw(potentialLabel(gate), ofFull(gate))), draw('ψ', ofFull(group.latent))],
    })),
    voltage: [draw('元のモデル', ofFull('V')), draw('縮約', ofReduced('V'))],
    phase: kam.groups.map(group => ({
      latent: group.latent,
      lines: [
        { label: '元のモデル', points: trajectory(ctx.files, ofFull('V'), ofFull(group.latent), span) },
        { label: '縮約', points: trajectory(ctx.files, ofReduced('V'), ofReduced(group.latent), span) },
      ],
    })),
  };
}

/** モデル 1 本の組み立てと演算の回数と、KAM のモデルの図。組み立てはモデルの spec（回す前は送られてきた resolved/ の同じ spec）から読む。 */
export function models(ctx: Context, request: ModelsRequest): ModelsReply {
  const plan = planOf(plans(), request.experiment);
  const model = modelOf(plan, request.model);
  const built = experimentStatus(ctx.runs, plan.id).models.find(m => m.model === model.name);
  if (!built) throw new Error(`${plan.id} のモデル ${model.name} の spec が無い`);
  const { run } = built;
  const done = run.state === 'done';
  const spec = done ? ctx.files.json<Spec>(run.id, RUN.spec) : specAt(run.spec);
  if (spec.kind !== 'model') throw new Error(`${run.id} はモデルの run でない`);
  let kam: ModelsReply['kam'] = null;
  if (spec.kam) {
    const own = done ? ctx.files.json<KamJson>(run.id, RUN.kam) : null;
    const groups: KamGroup[] = spec.kam.groups.map((group, i) => ({
      gates: group.gates,
      latent: kamLatent(group),
      weights: own?.groups[i].weights ?? null,
      criterion: own?.groups[i].criterion ?? null,
    }));
    const figures = own ? kamFigures(ctx, run.id, own, built.train.name, request) : null;
    kam = { groups, figures };
  }
  return {
    model: model.name,
    run: { state: run.state, run: run.id },
    cell: spec.cell,
    form: spec.form,
    absorbed: spec.absorbed ?? [],
    parts: built.parts.map(p => p.name),
    cost: done ? costRows(ctx.files.json<Record<string, number>>(run.id, RUN.summary)) : null,
    kam,
  };
}
