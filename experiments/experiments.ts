// 実験を並べる。主題ごとのディレクトリ（sweep/ 改良の探索、property/ 性質の発見、mc/ 拡張）が、宣言 experiments.json とその形 schema.ts と判定 checks.ts を持つ。
// 主題の形は主題が自由に決め、schema.ts が解決済みの実験（common.ts の Plan）を返す。ここは並べて、id・主張の名前・学習 run と系列の名前が重ならないことを確かめ、名前から引けるようにするだけ。
// 段取り（orchestrate.ts）・API・画面は、ここが返す解決済みの実験と、名前から引く trainOf・seriesOf だけを使う。
import { PRESET_SERIES, TESTS } from '../evaluation/orchestrate/conf.ts';
import type { Experiment, Series, Train } from '../evaluation/orchestrate/spec.ts';
import type { Claim, Judge, Model, Plan } from './common.ts';
import { plans as mcPlans } from './mc/schema.ts';
import { JUDGES as PROPERTY } from './property/checks.ts';
import { plans as propertyPlans } from './property/schema.ts';
import { JUDGES as SWEEP } from './sweep/checks.ts';
import { plans as sweepPlans } from './sweep/schema.ts';

/** 主題。読む順（改良の探索、性質の発見、拡張）。judges は主題の判定で、どれも主張に当たる必要がある。 */
const THEMES: { name: string; plans: (judges: Record<string, Judge>) => Plan[]; judges: Record<string, Judge> }[] = [
  { name: 'sweep', plans: sweepPlans, judges: SWEEP },
  { name: 'property', plans: propertyPlans, judges: PROPERTY },
  { name: 'mc', plans: () => mcPlans(), judges: {} },
];

/** 実験。主題の順、主題の中は書いた順。 */
export const EXPERIMENTS: Plan[] = [];
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

for (const theme of THEMES) {
  const own = new Set<string>();
  for (const plan of theme.plans(theme.judges)) {
    const where = `${theme.name}/experiments.json の ${plan.id}`;
    if (EXPERIMENTS.some(e => e.id === plan.id)) throw new Error(`${where}：実験の id がほかの主題と重なる`);
    for (const c of plan.claims) {
      if (c.name in CLAIMS) throw new Error(`${where}：主張の名前 ${c.name} が重なる`);
      own.add(c.name);
      CLAIMS[c.name] = c;
    }
    for (const m of plan.models) {
      register(TRAINS, m.name, m.train, where);
      register(MODELS, m.name, m, where);
    }
    for (const [name, s] of Object.entries(plan.series)) register(SERIES, name, s, where);
    EXPERIMENTS.push(plan);
  }
  for (const name of Object.keys(theme.judges))
    if (!own.has(name)) throw new Error(`${theme.name}/checks.ts：判定 ${name} に当たる主張が experiments.json に無い`);
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

/** 実験の id → 回す組。段取りの状態（status）が読む。 */
export const EXPERIMENT_RUNS: Record<string, Experiment> = {};
for (const e of EXPERIMENTS) EXPERIMENT_RUNS[e.id] = { pairs: e.pairs, benches: e.benches };
