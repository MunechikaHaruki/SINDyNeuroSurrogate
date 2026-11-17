// 入力（runs/inputs/<入力>/）の答え。実験の系列 1 つを、元のモデルと、回し終えたモデルの列で並べる：掃引の点ごとの電流と電位の線、原系との指標、ラスタ、列の学習側の指標。
import type { ColumnView, InputsReply, InputsRequest, InputView, VoltageLine, Window } from '../../shared/api.ts';
import { compArray, currentArray, RUN, type PartJson, type WaveMeta } from '../../shared/runs.ts';
import type { ExperimentStatus, PlanEntry, RunStatus } from '../../shared/spec.ts';
import { experimentStatus } from '../run/status.ts';
import { type Context, modelOf, planOf, plans, reducedOf, stimulusOf } from './experiments.ts';
import { line, type Span } from './waves.ts';

/** 元のモデルの run の目録（waves/meta.json）。 */
export function wavesOf(ctx: Context, run: string): WaveMeta {
  return ctx.files.json<WaveMeta>(run, RUN.meta(RUN.waves));
}

/** 範囲 window（省けば全体）の、目録 meta の時刻の並び。 */
export function spanOf(meta: WaveMeta, window: Window | undefined, duration: number): Span {
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
    original: { state: original.state, spec: original.spec, run: original.id },
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

/** 列の学習側の指標。学習器ごとの指標（part.json の metrics、群の名前で修飾する）と、モデルの組んだ演算コスト（summary.json）。 */
function trainingOf(ctx: Context, found: ExperimentStatus, model: string): Record<string, number | null> {
  const built = found.models.find(m => m.model === model)!;
  const summary: Record<string, number | null> = {};
  for (const part of built.parts) {
    const json = ctx.files.json<PartJson>(part.id, RUN.part);
    const label = json.gates.map(g => g.toLowerCase()).join('+');
    for (const [key, value] of Object.entries(json.metrics)) summary[`${label}/${key}`] = value;
  }
  return { ...summary, ...ctx.files.json<Record<string, number | null>>(built.run.id, RUN.summary) };
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
  return {
    input,
    comp,
    columns,
    points,
    pairs: byModel(metrics.pairs),
    crossings: byModel(metrics.crossings),
    summaries: byModel(metrics.summaries),
    spikes: { original: metrics.trains.original, columns: byModel(metrics.trains.columns) },
    training: Object.fromEntries(columns.map(c => [c.model, trainingOf(ctx, found, c.model)])),
  };
}
