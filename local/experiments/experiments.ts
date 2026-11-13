// 実験を並べる。実験群ごとに、実験層 config/<群>.yaml（形と組み立ては schema.ts）と判定層 config/<群>.checks.ts を読む。主張の一文は判定層の docstring で、resolve.ts が抜き出して resolved/claims.json に置いたものを読む。
// 決め打ちの実験は config/<実験>.json（形は plan.ts）。ここは並べて、id・主張の名前・学習 run と系列の名前が重ならないことを確かめ、名前から引けるようにするだけ。
// 手元の側の中心。resolve.ts（送る spec を組む）、judge.ts（主張の判定）、画面は、ここが返す解決済みの実験と、名前から引く trainOf・seriesOf だけを使う。計算する機械（../../remote/）は読まない。
import { PRESET_SERIES, TESTS } from './presets.ts';

import type { Checks, Claim, Model, Plan, Series, Train } from './common.ts';
import { fromJson } from './plan.ts';
import { groupPlans } from './schema.ts';
import { checks as checks1comp } from '../config/1comp.checks.ts';
import group1comp from '../config/1comp.yaml';
import group19comp from '../config/19comp.yaml';
import groupGpu from '../config/gpu.yaml';
import neuro2026 from '../config/neuro2026.json' with { type: 'json' };
import claimTexts from '../../resolved/claims.json' with { type: 'json' };

/** 実験群。読む順に、名前・実験層・判定層（無ければ空）。 */
const GROUP_SOURCES: { name: string; doc: unknown; checks: Checks }[] = [
  { name: '1comp', doc: group1comp, checks: checks1comp },
  { name: '19comp', doc: group19comp, checks: {} },
  { name: 'gpu', doc: groupGpu, checks: {} },
];

/** 実験 1 つ。解決済みの実験に、宣言を置いた実験群の名前を添える。画面は実験群ごとに実験を並べる。 */
export type GroupedPlan = Plan & { group: string };

/** 実験群の名前 → 実験層の frontmatter の description。 */
export const GROUPS: Record<string, string> = {};
/** 実験。実験群の順、実験群の中は書いた順。 */
export const EXPERIMENTS: GroupedPlan[] = [];
/** 主張の名前 → 主張。図（repo の外）とテストが名前で引く。 */
export const CLAIMS: Record<string, Claim> = {};
/** 学習 run の名前 → 学習 run（実験の学習 run とテスト専用の学習）。 */
export const TRAINS: Record<string, Train> = { ...TESTS };
/** 学習 run の名前 → 仕方と群分けの名前。 */
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

const texts = claimTexts as Record<string, string>;
const resolvedGroups = GROUP_SOURCES.map(g => ({ name: g.name, ...groupPlans(g.name, g.doc, g.checks, texts) }));
resolvedGroups.push({
  name: 'neuro2026',
  description: 'NEURO2026 で発表した形を今の研究コードで回し直す決め打ちの実験',
  plans: [fromJson('neuro2026-rerun', 'config/neuro2026.json', neuro2026, {})],
});
for (const group of resolvedGroups) {
  GROUPS[group.name] = group.description;
  for (const plan of group.plans) {
    const where = `${group.name} の ${plan.id}`;
    if (EXPERIMENTS.some(e => e.id === plan.id)) throw new Error(`${where}：実験の id がほかの実験群と重なる`);
    for (const c of plan.claims) {
      if (c.name in CLAIMS) throw new Error(`${where}：主張の名前 ${c.name} が重なる`);
      CLAIMS[c.name] = c;
    }
    for (const m of plan.models) {
      register(TRAINS, m.name, m.train, where);
      register(MODELS, m.name, m, where);
    }
    for (const [name, s] of Object.entries(plan.series)) register(SERIES, name, s, where);
    EXPERIMENTS.push({ ...plan, group: group.name });
  }
}

/** 学習 run を名前で引く。知らない名前は投げる。 */
export function trainOf(name: string): Train {
  const t = TRAINS[name];
  if (!t) throw new Error(`学習 run ${name} がどの実験にも無い`);
  return t;
}

/** 系列を名前で引く。知らない名前は投げる。 */
export function seriesOf(name: string): Series {
  const s = SERIES[name];
  if (!s) throw new Error(`系列 ${name} が無い（${Object.keys(SERIES).join(', ')}）`);
  return s;
}
