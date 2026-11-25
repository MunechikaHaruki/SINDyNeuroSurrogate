// 実験の目次、実験 1 つ、run の一覧の答え。宣言の写し（resolved/plans.json）と run の状態（../run/status.ts）から、宣言の順に組む。
// 宣言そのものは読まない（手元の local/ は import しない）。読むのは手元が書いて送った plans.json と resolved/ の spec と runs だけ。
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type {
  Experiments,
  ExperimentView,
  NamedRun,
  Progress,
  RegistryReply,
  RunKind,
  SetupRow,
} from '../../shared/api.ts';
import type { ExperimentStatus, IndexEntry, PlanEntry, PlanModel, Plans, State } from '../../shared/spec.ts';
import { jobs } from '../run-spec.ts';
import { resolvedDir, scan } from '../run/places.ts';
import { experimentStatus } from '../run/status.ts';
import type { SeriesMetrics, SeriesRequest } from './comparison.ts';
import type { RunFiles } from './waves.ts';

/** 答えを組むのに要るもの。runs の置き場、run のファイルの読み方、系列の指標（頼み方ごとに覚える）。 */
export type Context = { runs: string; files: RunFiles; metrics: (request: SeriesRequest) => SeriesMetrics };

/** Γ で「発火の時刻まで合う」とみなす線（見せ方の線で、比べ方の物差しではない）。 */
export const GAMMA_LINE = 0.9;

/** 宣言の写し。手元が resolved/ と一緒に送る。 */
export function plans(): Plans {
  const file = join(resolvedDir(), 'plans.json');
  if (!existsSync(file)) throw new Error('resolved/plans.json が無い（手元で bun local/resolve.ts write して送る）');
  return JSON.parse(readFileSync(file, 'utf8')) as Plans;
}

/** 実験を id で引く。宣言に無ければ投げる。 */
export function planOf(all: Plans, id: string): PlanEntry {
  const plan = all.experiments.find(e => e.id === id);
  if (!plan) throw new Error(`実験 ${id} が宣言に無い`);
  return plan;
}

/** 実験のモデルを名前で引く。宣言に無ければ投げる。 */
export function modelOf(plan: PlanEntry, name: string): PlanModel {
  const model = plan.models.find(m => m.name === name);
  if (!model) throw new Error(`${plan.id} にモデル ${name} が無い`);
  return model;
}

/** まとめて減らした状態の数。群ごとに ゲートの数 − 潜在の本数 の和と、膜電位へ吸収したゲートの数。 */
export function reducedOf(model: PlanModel): number {
  let reduced = (model.config.absorbed ?? []).length;
  for (const g of model.config.groups) reduced += g.gates.length - g.d;
  return reduced;
}

/** 系列の刺激が入る区間 [ms]。前後の silence_duration は電流が 0 の無刺激の区間で、掃引の 0 の点とは別のもの。 */
export function stimulusOf(plan: PlanEntry, series: string): [number, number] {
  const { silence_duration: silence = 0, duration } = plan.series[series].spec.current_params;
  return [silence, duration - silence];
}

/** 実験の進み具合。置換できない組は済んだものとして数える。回す run が無ければ未着手。 */
function progressOf(plan: PlanEntry, found: ExperimentStatus): Progress {
  if (Object.keys(plan.series).length === 0) return '未着手';
  const states: State[] = [...found.models.map(m => m.state), ...found.pairs.map(p => p.state)];
  if (states.includes('running')) return '計算中';
  if (!states.includes('todo')) return '結果あり';
  return states.includes('done') ? '途中' : '未着手';
}

/** 回っている run の数（モデルと組）。 */
function runningOf(found: ExperimentStatus): number {
  return [...found.models, ...found.pairs].filter(r => r.state === 'running').length;
}

/** 実験の目次。実験群と実験は宣言の順。 */
export function experiments(runs: string): Experiments {
  const all = plans();
  let running = 0;
  const groups = all.groups.map(group => ({
    ...group,
    experiments: all.experiments
      .filter(e => e.group === group.name)
      .map(plan => {
        const found = experimentStatus(runs, plan.id);
        running += runningOf(found);
        return {
          id: plan.id,
          description: plan.description,
          progress: progressOf(plan, found),
          runs: {
            parts: learnedOf(plan, found).map(r => r.name),
            models: plan.models.map(m => m.name),
            inputs: Object.keys(plan.series),
            bench: plan.benches.map(b => b.name),
          },
        };
      }),
  }));
  return {
    groups,
    running,
    runs: scan(runs).length,
    measures: { coincidenceWindow: all.measures.coincidence_window, gammaLine: GAMMA_LINE },
  };
}

/** 実験の設定を、宣言の名前と、その名前が解決した値で並べる（元のモデル → 群分け → 仕方 → 系列 → 比べ方）。 */
function setupOf(plan: PlanEntry, window: number): SetupRow[] {
  const series = Object.entries(plan.series);
  const methods = plan.methods.map(method => {
    const config = plan.models.find(m => m.method === method)?.config;
    const g = config?.groups[0];
    const parts = g ? [`圧縮 ${g.compressor}`, `右辺 ${g.rhs}`, `潜在 ${g.d}`] : [];
    if (config?.absorbed?.length) parts.push(`膜電位へ吸収 ${config.absorbed.join('・')}`);
    if (config) {
      const { duration, ...rest } = config.data.current_params;
      parts.push(
        `学習 ${config.data.target}・${duration} ms${Object.entries(rest)
          .map(([k, v]) => `・${k} ${v}`)
          .join('')}`,
      );
    }
    return { name: method, value: parts.join('・') };
  });
  const inputs = series.map(([name, s]) => {
    const params = Object.entries(s.spec.current_params).map(([k, v]) => `${k} ${v}`);
    const parts = [
      s.spec.current_type,
      `刺激 ${s.spec.stim ?? 'soma'}`,
      ...params,
      `置き換え ${s.replace_targets.join('・')}`,
    ];
    if (s.param) parts.push(`掃引 ${s.param} = ${s.values.join(', ')}`);
    return { name, value: parts.join('・') };
  });
  const compare: SetupRow['items'] = [];
  if (plan.pairs.length > 0)
    compare.push({
      name: 'Γ',
      value: `±${window} ms 以内で一致した発火の割合（偶然の一致を引く）・線 Γ ≥ ${GAMMA_LINE}`,
    });
  for (const b of plan.benches)
    compare.push({
      name: b.name,
      value: `細胞 ${b.cells.join('・')}${b.ncu ? '・ncu で 1 刻みの VRAM とレジスタ' : ''}`,
    });
  return [
    {
      label: '元のモデル',
      items: [...new Set(series.map(([, s]) => s.spec.target))].map(t => ({ name: t, value: '' })),
    },
    { label: '群分け', items: plan.groupings.map(g => ({ name: g, value: '' })) },
    { label: '仕方', items: methods },
    { label: '系列', items: inputs },
    ...(compare.length > 0 ? [{ label: '比べ方', items: compare }] : []),
  ];
}

/** 実験が使う run（モデル・学習器・元のモデル（学習の入力も）・組・GPU の実測）の、計算にかかった秒（meta.json の elapsed）の和と、回した数・全部の数。 */
export function rerunCost(found: ExperimentStatus, index: IndexEntry[]): ExperimentView['rerun'] {
  const ids = new Set<string>();
  for (const m of found.models) [m.run.id, ...m.parts.map(p => p.id)].forEach(id => ids.add(id));
  for (const s of found.series) ids.add(s.original.id);
  for (const p of found.pairs) ids.add(p.run.id);
  for (const b of found.benches) ids.add(b.run.id);
  let seconds = 0;
  let done = 0;
  for (const id of ids) {
    const elapsed = index.find(r => r.id === id)?.elapsed;
    if (elapsed === undefined) continue;
    seconds += elapsed;
    done += 1;
  }
  return { seconds, done, total: ids.size };
}

/** 実験のモデルが使う学習器（モデルの宣言の順に、重ねずに）。 */
function learnedOf(plan: PlanEntry, found: ExperimentStatus): NamedRun[] {
  const parts: NamedRun[] = [];
  for (const m of plan.models) {
    const built = found.models.find(f => f.model === m.name);
    if (!built) continue;
    for (const p of built.parts) if (!parts.some(q => q.name === p.name)) parts.push({ name: p.name, state: p.state });
  }
  return parts;
}

/** 実験の学習器を宣言の名前で引き、その run の置き場（parts/<id>）を返す。無いかまだ回していなければ投げる。 */
export function learnedRun(runs: string, id: string, name: string): string {
  const found = experimentStatus(runs, planOf(plans(), id).id);
  const run = found.models.flatMap(m => m.parts).find(r => r.name === name);
  if (!run) throw new Error(`${id} に学習器 ${name} が無い`);
  if (run.state !== 'done') throw new Error(`${id} の学習器 ${name} はまだ回していない`);
  return run.id;
}

/** 実験のモデルが学習した入力（学習の系列）の名前。 */
export function trainOf(runs: string, id: string, model: string): string {
  const built = experimentStatus(runs, planOf(plans(), id).id).models.find(m => m.model === model);
  if (!built) throw new Error(`${id} にモデル ${model} が無い`);
  return built.train.name;
}

/** 実験 1 つの設定と、回す組の状態。並びは宣言の順。 */
export function experiment(runs: string, id: string): ExperimentView {
  const all = plans();
  const plan = planOf(all, id);
  const found = experimentStatus(runs, id);
  const pairs: ExperimentView['pairs'] = [];
  for (const p of plan.pairs) {
    const run = found.pairs.find(q => q.series === p.series && q.model === p.model);
    if (run) pairs.push({ input: p.series, model: p.model, state: run.state });
  }
  return {
    id,
    group: plan.group,
    description: plan.description,
    setup: setupOf(plan, all.measures.coincidence_window),
    rerun: rerunCost(found, scan(runs)),
    inputs: Object.keys(plan.series),
    models: plan.models.map(m => {
      const built = found.models.find(f => f.model === m.name)!;
      return { name: m.name, train: built.train.name, parts: built.parts.map(p => p.name) };
    }),
    parts: learnedOf(plan, found),
    pairs,
    benches: plan.benches.map(b => {
      const run = found.benches.find(f => f.bench === b.name)!;
      return { name: b.name, state: run.state };
    }),
  };
}

/** run の一覧（新しい順）と、外の画面から起こした仕事。名前は宣言から引き、どの実験にも当たらない run は null。 */
export function registry(runs: string): RegistryReply {
  const names: Record<string, string> = {};
  for (const plan of plans().experiments) {
    const found = experimentStatus(runs, plan.id);
    for (const series of found.series) names[series.original.id] = series.series;
    for (const model of found.models) {
      names[model.run.id] = model.model;
      for (const part of model.parts) names[part.id] = part.name;
    }
    for (const pair of found.pairs) names[pair.run.id] = `${pair.series} × ${pair.model}`;
    for (const bench of found.benches) names[bench.run.id] = bench.bench;
  }
  return {
    runs: scan(runs).map(r => ({
      id: r.id,
      place: r.id.split('/')[0] as RunKind,
      name: names[r.id] ?? null,
      given: r.name,
      created: r.created,
    })),
    jobs: jobs(runs),
  };
}
