// 実験を並べる。主題ごとのディレクトリ（sweep/ 改良の探索、property/ 性質の発見、mc/ 拡張）の experiments.json（宣言）と checks.ts（判定）を読み、
// 形と、仕方と系列が presets.json にあるかと、主張ごとに判定があるかを確かめて一つの並びにし、段取りの状態（status）に渡す回す組を組む。主張の判定は API の claims が runs から出す。
// 宣言は実験 id → { gates（全部の群分け）か groupings, methods, claims: 主張の名前 → { on: 読む系列, groupings?, comp?, claim } }。回す系列は主張の on を合わせたもの。
import { METHODS, SERIES, modelOf } from '../evaluation/orchestrate/conf.ts';
import type { Experiment } from '../evaluation/orchestrate/spec.ts';
import { groupings, type Claim, type Judge, type Plan } from './common.ts';
import mcDeclaration from './mc/experiments.json' with { type: 'json' };
import { JUDGES as MC } from './mc/checks.ts';
import propertyDeclaration from './property/experiments.json' with { type: 'json' };
import { JUDGES as PROPERTY } from './property/checks.ts';
import sweepDeclaration from './sweep/experiments.json' with { type: 'json' };
import { JUDGES as SWEEP } from './sweep/checks.ts';

type Entry = Record<string, unknown>;

/** 分類。読む順（改良の探索、性質の発見、拡張）。 */
const CATEGORIES: { name: string; declaration: unknown; judges: Record<string, Judge> }[] = [
  { name: 'sweep', declaration: sweepDeclaration, judges: SWEEP },
  { name: 'property', declaration: propertyDeclaration, judges: PROPERTY },
  { name: 'mc', declaration: mcDeclaration, judges: MC },
];

function fail(where: string, message: string): never {
  throw new Error(`${where}：${message}`);
}
function isObject(value: unknown): value is Entry {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function isStrings(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(v => typeof v === 'string');
}

/** 主張の形を確かめ、同じ名前の判定を付ける。 */
function claimOf(where: string, name: string, entry: unknown, judges: Record<string, Judge>): Claim {
  if (!isObject(entry)) fail(where, '主張は物');
  for (const key of Object.keys(entry))
    if (!['on', 'groupings', 'comp', 'claim'].includes(key)) fail(where, `知らない鍵 ${key}`);
  if (typeof entry.claim !== 'string') fail(where, 'claim は文字列');
  if (!isStrings(entry.on) || entry.on.length === 0) fail(where, 'on は判定が読む系列の空でない配列');
  if (entry.groupings !== undefined && !isStrings(entry.groupings)) fail(where, 'groupings は群分けの名前の配列');
  if (entry.comp !== undefined && typeof entry.comp !== 'string') fail(where, 'comp は区画の名前');
  const comp = entry.comp as string | undefined;
  const judge = judges[name];
  if (!judge) fail(where, `主張 ${name} の判定が checks.ts に無い`);
  return {
    name,
    claim: entry.claim,
    series: entry.on.map(s => (comp === undefined ? { name: s } : { name: s, comp })),
    ...(entry.groupings === undefined ? {} : { groupings: entry.groupings }),
    judge,
  };
}

/** 実験の形を確かめ、群分けを名前の並びにする。仕方と系列は presets.json にあるものだけ。 */
function planOf(where: string, id: string, entry: unknown, judges: Record<string, Judge>): Plan {
  if (!isObject(entry)) fail(where, '実験は物');
  for (const key of Object.keys(entry))
    if (!['gates', 'groupings', 'methods', 'claims'].includes(key)) fail(where, `知らない鍵 ${key}`);
  if (!isStrings(entry.methods)) fail(where, 'methods は縮約の仕方の配列');
  for (const method of entry.methods) if (!(method in METHODS)) fail(where, `仕方 ${method} が presets.json に無い`);
  let names: string[];
  if (typeof entry.gates === 'string' && entry.groupings === undefined) names = groupings([...entry.gates]);
  else if (isStrings(entry.groupings) && entry.gates === undefined) names = entry.groupings;
  else fail(where, 'gates（全部の群分けを組むゲートの並び）か groupings（群分けの名前）のどちらか一方を書く');
  if (!isObject(entry.claims) || Object.keys(entry.claims).length === 0)
    fail(where, 'claims は主張の名前 → 主張 の空でない物');
  const claims: Claim[] = [];
  for (const [name, c] of Object.entries(entry.claims))
    claims.push(claimOf(`${where}.claims.${name}`, name, c, judges));
  const series: string[] = [];
  for (const c of claims) for (const { name } of c.series) if (!series.includes(name)) series.push(name);
  for (const name of series) if (!(name in SERIES)) fail(where, `系列 ${name} が presets.json に無い`);
  return { id, groupings: names, methods: entry.methods, series, claims };
}

/** 実験。分類の順、分類の中は書いた順。 */
export const EXPERIMENTS: Plan[] = [];
/** 主張の名前 → 主張。図（repo の外）とテストが名前で引く。 */
export const CLAIMS: Record<string, Claim> = {};
for (const category of CATEGORIES) {
  if (!isObject(category.declaration)) fail(`${category.name}/experiments.json`, '実験の id → 実験 の物');
  const own = new Set<string>();
  for (const id of Object.keys(category.declaration)) {
    if (EXPERIMENTS.some(e => e.id === id))
      fail(`${category.name}/experiments.json`, `実験の id ${id} がほかの分類と重なる`);
    const plan = planOf(`${category.name}/experiments.json の ${id}`, id, category.declaration[id], category.judges);
    for (const c of plan.claims) {
      own.add(c.name);
      if (c.name in CLAIMS) fail(`${category.name}/experiments.json の ${id}`, `主張の名前 ${c.name} が重なる`);
      CLAIMS[c.name] = c;
    }
    EXPERIMENTS.push(plan);
  }
  for (const name of Object.keys(category.judges))
    if (!own.has(name)) fail(`${category.name}/checks.ts`, `判定 ${name} に当たる主張が experiments.json に無い`);
}

/** 実験の回す学習 run と系列の組。仕方の順、群分けの順、系列の順。 */
export function pairsOf(e: Pick<Plan, 'groupings' | 'methods' | 'series'>): Experiment {
  const pairs: Experiment['pairs'] = [];
  for (const method of e.methods)
    for (const grouping of e.groupings)
      for (const series of e.series) pairs.push({ model: modelOf(method, grouping), series });
  return { pairs };
}

/** 実験の id → 回す組。段取りの状態（status）が読む。 */
export const EXPERIMENT_RUNS: Record<string, Experiment> = {};
for (const e of EXPERIMENTS) EXPERIMENT_RUNS[e.id] = pairsOf(e);
