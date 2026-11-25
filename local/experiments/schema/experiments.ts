// 実験を並べる。実験群ごとに、実験層 config/<群>.yaml（形と組み立ては schema.ts）を読む。
// 決め打ちの実験は config/<実験>.json（形は schema.ts の PlanJson）。ここは並べて、モデルごとの学習の系列（presets.ts の trainSeries）を足し、id・モデルと系列の名前が重ならないことを確かめ、名前から引けるようにするだけ。
// 手元の側の中心。resolve.ts（送る spec を組む）と画面は、ここが返す解決済みの実験と、名前から引く modelOf・seriesOf だけを使う。計算する機械（../../remote/）は読まない。
import { PRESET_SERIES, trainSeries, type Series } from './presets.ts';
import { fromJson, groupPlans, type Model, type Plan } from './schema.ts';
import group1comp from '../config/1comp.yaml';
import group19comp from '../config/19comp.yaml';
import groupGpu from '../config/gpu.yaml';
import neuro2026 from '../config/neuro2026.json' with { type: 'json' };

/** 実験群。読む順に、名前と実験層。 */
const GROUP_SOURCES: { name: string; doc: unknown }[] = [
  { name: '1comp', doc: group1comp },
  { name: '19comp', doc: group19comp },
  { name: 'gpu', doc: groupGpu },
];

/** 実験 1 つ。解決済みの実験に、宣言を置いた実験群の名前を添える。画面は実験群ごとに実験を並べる。 */
export type GroupedPlan = Plan & { group: string };

/** 実験群の名前 → 実験層の frontmatter の description。 */
export const GROUPS: Record<string, string> = {};
/** 実験。実験群の順、実験群の中は書いた順。 */
export const EXPERIMENTS: GroupedPlan[] = [];
/** モデルの名前 → モデル（設定と、仕方と群分けの名前）。 */
export const MODELS: Record<string, Model> = {};
/** 系列の名前 → 掃引（プリセットの系列と、実験が差分で組んだ系列）。 */
export const SERIES: Record<string, Series> = { ...PRESET_SERIES };

/** 鍵を並べ替えた JSON。差分の書き順で中身の比較がずれないようにする。 */
function sorted(value: unknown): string {
  return JSON.stringify(value, (_, v) =>
    v && typeof v === 'object' && !Array.isArray(v)
      ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))
      : v,
  );
}

/** 同じ名前が違う中身を指していれば投げる。 */
function register<T>(table: Record<string, T>, name: string, value: T, where: string): void {
  if (table[name] !== undefined && sorted(table[name]) !== sorted(value))
    throw new Error(`${where}：名前 ${name} がほかの実験と違う中身を指している`);
  table[name] = value;
}

/** 実験に、モデルが学習した入力を系列として先頭に足し、モデルごとにその系列を回す組を足す。学習データはこの系列の元のモデルの run。 */
export function withTraining(plan: Plan): Plan {
  const train: Record<string, Series> = {};
  const pairs: Plan['pairs'] = [];
  for (const m of plan.models) {
    const { name, series } = trainSeries(m.config);
    if (plan.series[name]) throw new Error(`${plan.id}：系列 ${name} は学習の系列の名前と重なる`);
    const input = sorted({ spec: series.spec, param: series.param, values: series.values });
    const same = Object.entries(plan.series).find(([, s]) => sorted({ spec: s.spec, param: s.param, values: s.values }) === input);
    if (same) throw new Error(`${plan.id}：系列 ${same[0]} は ${m.name} の学習の入力と同じ入力（学習の系列 ${name} と見分けられない）`);
    train[name] = series;
    pairs.push({ model: m.name, series: name });
  }
  return { ...plan, series: { ...train, ...plan.series }, pairs: [...pairs, ...plan.pairs] };
}

const resolvedGroups = GROUP_SOURCES.map(g => ({ name: g.name, ...groupPlans(g.name, g.doc) }));
resolvedGroups.push({
  name: 'neuro2026',
  description: 'NEURO2026 で発表した形を今の研究コードで回し直す決め打ちの実験',
  plans: [fromJson('neuro2026-rerun', 'config/neuro2026.json', neuro2026)],
});
for (const group of resolvedGroups) {
  GROUPS[group.name] = group.description;
  for (const declared of group.plans) {
    const plan = withTraining(declared);
    const where = `${group.name} の ${plan.id}`;
    if (EXPERIMENTS.some(e => e.id === plan.id)) throw new Error(`${where}：実験の id がほかの実験群と重なる`);
    for (const m of plan.models) register(MODELS, m.name, m, where);
    for (const [name, s] of Object.entries(plan.series)) register(SERIES, name, s, where);
    EXPERIMENTS.push({ ...plan, group: group.name });
  }
}

/** モデルを名前で引く。知らない名前は投げる。 */
export function modelOf(name: string): Model {
  const m = MODELS[name];
  if (!m) throw new Error(`モデル ${name} がどの実験にも無い`);
  return m;
}

/** 系列を名前で引く。知らない名前は投げる。 */
export function seriesOf(name: string): Series {
  const s = SERIES[name];
  if (!s) throw new Error(`系列 ${name} が無い（${Object.keys(SERIES).join(', ')}）`);
  return s;
}
