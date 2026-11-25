// 実験を並べる。実験群ごとに config/<群>.yaml（形と組み立ては schema.ts、基準のファイルは base.ts）を読む。
// ここは並べて、モデルごとの学習の系列（base.ts の trainSeries）を足し、実験の id が重ならないことを確かめるだけ。モデルと系列の名前は実験の中で一意で、実験をまたいでは同じ名前が違う中身を指してよい。
// 手元の側の中心。resolve.ts（送る spec を組む）と画面は、ここが返す解決済みの実験だけを使う。計算する機械（../../../remote/）は読まない。
import { trainSeries, type Series } from './base.ts';
import { groupPlans, type Plan } from './schema.ts';
import group1comp from '../config/1comp.yaml';
import group19comp from '../config/19comp.yaml';
import groupGpu from '../config/gpu.yaml';
import groupNeuro2026 from '../config/neuro2026.yaml';

/** 実験群。読む順に、名前と YAML。 */
const GROUP_SOURCES: { name: string; doc: unknown }[] = [
  { name: '1comp', doc: group1comp },
  { name: '19comp', doc: group19comp },
  { name: 'gpu', doc: groupGpu },
  { name: 'neuro2026', doc: groupNeuro2026 },
];

/** 実験 1 つ。解決済みの実験に、宣言を置いた実験群の名前を添える。画面は実験群ごとに実験を並べる。 */
export type GroupedPlan = Plan & { group: string };

/** 実験群の名前 → frontmatter の description。 */
export const GROUPS: Record<string, string> = {};
/** 実験。実験群の順、実験群の中は書いた順。 */
export const EXPERIMENTS: GroupedPlan[] = [];

/** 鍵を並べ替えた JSON。差分の書き順で中身の比較がずれないようにする。 */
function sorted(value: unknown): string {
  return JSON.stringify(value, (_, v) =>
    v && typeof v === 'object' && !Array.isArray(v)
      ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))
      : v,
  );
}

/** 実験に、モデルが学習した入力を系列として先頭に足し、モデルごとにその系列を回す組を足す。学習データはこの系列の元のモデルの run。 */
export function withTraining(plan: Plan): Plan {
  const train: Record<string, Series> = {};
  const pairs: Plan['pairs'] = [];
  for (const m of plan.models) {
    const { name, series } = trainSeries(m.config);
    if (plan.series[name]) throw new Error(`${plan.id}：系列 ${name} は学習の系列の名前と重なる`);
    const input = sorted({ spec: series.spec, param: series.param, values: series.values });
    const same = Object.entries(plan.series).find(
      ([, s]) => sorted({ spec: s.spec, param: s.param, values: s.values }) === input,
    );
    if (same)
      throw new Error(
        `${plan.id}：系列 ${same[0]} は ${m.name} の学習の入力と同じ入力（学習の系列 ${name} と見分けられない）`,
      );
    train[name] = series;
    pairs.push({ model: m.name, series: name });
  }
  return { ...plan, series: { ...train, ...plan.series }, pairs: [...pairs, ...plan.pairs] };
}

for (const source of GROUP_SOURCES) {
  const group = groupPlans(source.name, source.doc);
  GROUPS[source.name] = group.description;
  for (const declared of group.plans) {
    const plan = withTraining(declared);
    if (EXPERIMENTS.some(e => e.id === plan.id))
      throw new Error(`${source.name} の ${plan.id}：実験の id がほかの実験と重なる`);
    EXPERIMENTS.push({ ...plan, group: source.name });
  }
}
