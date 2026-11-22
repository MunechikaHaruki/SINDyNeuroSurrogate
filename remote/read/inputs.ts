// 入力（runs/inputs/<入力>/）の答え。入力の下の run（元のモデル original/ と置き換えた列 <モデル>@<区画>/）だけを読み、実験の系列 1 つを元のモデルと回し終えたモデルの列で並べる：
// 掃引の点ごとの電流と電位の線、原系との指標と発火、頼めば 1 組の潜在空間（原系は列の圧縮器で写した projected/）と、元のモデルの全区画の膜電位・内部電流・ゲート。
import type {
  ColumnView,
  InputsReply,
  InputsRequest,
  InputView,
  Line,
  PairCompare,
  VoltageLine,
  WholeWaves,
  Window,
} from '../../shared/api.ts';
import { compArray, currentArray, gatesOf, internalArray, RUN, variablesOf, type WaveMeta } from '../../shared/runs.ts';
import type { ExperimentStatus, PlanEntry, RunStatus } from '../../shared/spec.ts';
import { experimentStatus } from '../run/status.ts';
import { type Context, modelOf, planOf, plans, reducedOf, stimulusOf } from './experiments.ts';
import { isFraction, line, slopeCompare, trajectory, type Span } from './waves.ts';

/** 元のモデルの run の目録（waves/meta.json）。 */
export function wavesOf(ctx: Context, run: string): WaveMeta {
  return ctx.files.json<WaveMeta>(run, RUN.meta(RUN.waves));
}

/** 範囲 window（省けば全体）の、目録 meta の時刻の並び。 */
export function spanOf(meta: { t0: number; dt: number }, window: Window | undefined, duration: number): Span {
  const [start, end] = window ?? [0, duration];
  return { t0: meta.t0, dt: meta.dt, start, end };
}

/** 実験の系列 1 つの見出し。元のモデルをまだ回していなければ null。 */
export function inputView(ctx: Context, plan: PlanEntry, name: string, original: RunStatus): InputView | null {
  const declared = plan.series[name];
  if (!declared) throw new Error(`${plan.id} に入力 ${name} が無い`);
  if (original.state !== 'done') return null;
  const meta = wavesOf(ctx, original.id);
  const unit = ctx.files.json<{ unit?: string }>(original.id, 'meta.json').unit ?? '';
  const { spec } = declared;
  return {
    name,
    param: declared.param,
    unit,
    values: declared.values,
    points: meta.n.map((_, k) => (declared.param ? `${declared.param} = ${declared.values[k]} ${unit}` : '単発')),
    comps: Object.keys(meta.names ?? { soma: 0 }),
    replaceTargets: declared.replace_targets,
    stimulus: stimulusOf(plan, name),
    duration: meta.n[0] * meta.dt,
    dt: meta.dt,
    spec: {
      target: spec.target,
      current_type: spec.current_type,
      stim: spec.stim ?? 'soma',
      current_params: spec.current_params,
    },
    original: { state: original.state, run: original.id },
  };
}

/** 系列の元のモデルの状態。 */
export function originalOf(found: ExperimentStatus, name: string): RunStatus {
  const series = found.series.find(s => s.series === name);
  if (!series) throw new Error(`${found.name} に入力 ${name} が無い`);
  return series.original;
}

/** 比べる区画。頼まれた名前が無ければ投げ、省けば置き換えた区画のうち刺激する区画（無ければ最初に置き換える区画）。 */
export function compOf(input: InputView, comp: string | undefined): string {
  // 省けば、置き換えた区画のうち刺激する区画（発火を測る所）、無ければ最初に置き換える区画
  const stim = input.replaceTargets.includes(input.spec.stim) ? input.spec.stim : undefined;
  const chosen = comp ?? stim ?? input.replaceTargets[0] ?? input.comps[0];
  if (!input.comps.includes(chosen))
    throw new Error(`${input.name} に区画 ${chosen} が無い（${input.comps.join('、')}）`);
  return chosen;
}

/** 系列を回し終えたモデルの列（宣言の順）。models で絞る。 */
function columnsOf(plan: PlanEntry, found: ExperimentStatus, input: string, models?: string[]): ColumnView[] {
  const list: ColumnView[] = [];
  for (const pair of plan.pairs) {
    if (pair.series !== input || (models && !models.includes(pair.model))) continue;
    const run = found.pairs.find(p => p.series === input && p.model === pair.model)?.run;
    if (run?.state !== 'done') continue;
    const model = modelOf(plan, pair.model);
    list.push({
      model: model.name,
      method: model.method,
      grouping: model.grouping,
      reduced: reducedOf(model),
      run: run.id,
    });
  }
  return list;
}

/** 相平面の軸の単位が同じか。潜在どうしは同じ単位、V と潜在は、潜在が等価電位（KAM の ψ、mV）のときだけ同じ。 */
export function sameScale(axes: { x: string; y: string }, kam: boolean): boolean {
  const unit = (name: string) => (name === 'V' || kam ? 'mV' : 'latent');
  return unit(axes.x) === unit(axes.y);
}

/** 1 組（元のモデル original と列 run）の、掃引の点 k・区画 comp の潜在空間。axes は相平面の軸（無い名前なら V と最初の潜在）。kam はモデルが KAM か（潜在が mV）。
 * projected でなければ（区画を置き換えていない）潜在は無く、replaced でなければ（発散した点）置換系を描かない。 */
function pairOf(
  ctx: Context,
  at: {
    model: string;
    run: string;
    k: number;
    comp: string;
    id: number;
    projected: boolean;
    replaced: boolean;
    kam: boolean;
  },
  span: Span,
  axes: { x: string; y: string },
): PairCompare {
  const { run, k, id, replaced } = at;
  const root = RUN.projected(at.comp);
  const projected = at.projected ? ctx.files.json<WaveMeta>(run, RUN.meta(root)) : undefined;
  const names = projected ? variablesOf(projected, id).filter(name => name !== 'V') : [];
  const own = wavesOf(ctx, run);
  const ownNames = variablesOf(own, id);
  const latents: Record<string, Line[]> = {};
  for (const name of names) {
    latents[name] = [{ label: '原系', points: line(ctx.files, compArray(run, root, projected!, k, id, name), span) }];
    if (replaced && ownNames.includes(name))
      latents[name].push({
        label: '置換系',
        points: line(ctx.files, compArray(run, RUN.waves, own, k, id, name), span),
      });
  }
  const choices = ['V', ...names];
  const chosen = {
    x: choices.includes(axes.x) ? axes.x : 'V',
    y: choices.includes(axes.y) ? axes.y : (names[0] ?? 'V'),
  };
  const phase: Line[] = [];
  if (projected) {
    const x = compArray(run, root, projected, k, id, chosen.x);
    const y = compArray(run, root, projected, k, id, chosen.y);
    phase.push({ label: '原系（射影）', points: trajectory(ctx.files, x, y, span) });
    if (replaced && ownNames.includes(chosen.x) && ownNames.includes(chosen.y)) {
      const ownX = compArray(run, RUN.waves, own, k, id, chosen.x);
      const ownY = compArray(run, RUN.waves, own, k, id, chosen.y);
      phase.push({ label: '置換系', points: trajectory(ctx.files, ownX, ownY, span) });
    }
  }
  const slopes =
    projected && ctx.files.text(run, RUN.meta(RUN.slope(at.comp))) !== null
      ? slopeCompare(
          ctx.files,
          names.map(name => ({
            label: name,
            slope: compArray(run, RUN.slope(at.comp), projected, k, id, name),
            rhs: compArray(run, RUN.rhs(at.comp), projected, k, id, name),
          })),
          span,
        )
      : null;
  return { model: at.model, latents, phase, axes: { ...chosen, sameScale: sameScale(chosen, at.kam) }, slopes };
}

/** 元のモデル original の掃引の点 k の、区画 comps の膜電位・内部電流・ゲート（0〜1 の状態）とゲートでない状態。 */
function wholeOf(ctx: Context, original: string, meta: WaveMeta, k: number, comps: string[], span: Span): WholeWaves {
  const draw = (label: string, ref: Parameters<typeof line>[1]): Line => ({
    label,
    points: line(ctx.files, ref, span),
  });
  const whole: WholeWaves = { voltage: [], internal: [], gates: [], states: [] };
  for (const name of comps) {
    const id = meta.names?.[name];
    if (id === undefined) continue;
    whole.voltage.push(draw(name, compArray(original, RUN.waves, meta, k, id, 'V')));
    if (meta.nodes?.includes(id)) whole.internal.push(draw(name, internalArray(original, RUN.waves, meta, k, id)));
    for (const gate of gatesOf(meta, id)) {
      const ref = compArray(original, RUN.waves, meta, k, id, gate);
      const drawn = draw(comps.length > 1 ? `${gate}（${name}）` : gate, ref);
      (isFraction(ctx.files.column(ref)) ? whole.gates : whole.states).push(drawn);
    }
  }
  return whole;
}

/** 入力 1 つを、回し終えたモデルの列と比べた答え。元のモデルをまだ回していなければ投げる。 */
export function inputs(ctx: Context, request: InputsRequest): InputsReply {
  const all = plans();
  const plan = planOf(all, request.experiment);
  const found = experimentStatus(ctx.runs, plan.id);
  const original = originalOf(found, request.input);
  const input = inputView(ctx, plan, request.input, original);
  if (!input) throw new Error(`${plan.id} の ${request.input} の元のモデルをまだ回していない`);
  const comp = compOf(input, request.comp);
  const columns = columnsOf(plan, found, request.input, request.models);
  const metrics = ctx.metrics({
    original: original.id,
    columns: columns.map(c => c.run),
    comp,
    delta: all.measures.coincidence_window,
  });
  const meta = wavesOf(ctx, original.id);
  const id = meta.names?.[comp] ?? 0;
  const span = spanOf(meta, request.window, input.duration);
  const points = meta.n.map((_, k) => {
    const diverged = columns.filter(c => metrics.pairs[c.run]?.[k]?.diverged).map(c => c.model);
    if (request.waves === false) return { current: [], voltages: [], diverged };
    const voltages: VoltageLine[] = [
      {
        label: '原系',
        model: null,
        points: line(ctx.files, compArray(original.id, RUN.waves, meta, k, id, 'V'), span),
      },
    ];
    for (const c of columns) {
      if (diverged.includes(c.model)) continue;
      const own = wavesOf(ctx, c.run);
      voltages.push({
        label: c.model,
        model: c.model,
        points: line(ctx.files, compArray(c.run, RUN.waves, own, k, id, 'V'), span),
      });
    }
    return { current: line(ctx.files, currentArray(original.id, RUN.waves, meta, k), span), voltages, diverged };
  });
  const byModel = <T>(table: Record<string, T>) => Object.fromEntries(columns.map(c => [c.model, table[c.run]]));
  const pointOf = (k: number) => Math.min(Math.max(0, k), meta.n.length - 1);
  const paired = request.pair && columns.find(c => c.model === request.pair!.model);
  if (request.pair && !paired) throw new Error(`${request.input} を ${request.pair.model} でまだ回していない`);
  const pair =
    paired && request.pair
      ? pairOf(
          ctx,
          {
            model: paired.model,
            run: paired.run,
            k: pointOf(request.pair.point),
            comp,
            id,
            projected: input.replaceTargets.includes(comp),
            replaced: !metrics.pairs[paired.run]?.[pointOf(request.pair.point)]?.diverged,
            kam: modelOf(plan, paired.model).config.groups.every(g => g.compressor === 'kam'),
          },
          span,
          request.pair.axes ?? { x: 'V', y: '' },
        )
      : null;
  const whole = request.whole
    ? wholeOf(ctx, original.id, meta, pointOf(request.whole.point), request.whole.comps, span)
    : null;
  return {
    input,
    comp,
    columns,
    points,
    summaries: byModel(metrics.summaries),
    spikes: { original: metrics.trains.original, columns: byModel(metrics.trains.columns) },
    pair,
    whole,
  };
}
