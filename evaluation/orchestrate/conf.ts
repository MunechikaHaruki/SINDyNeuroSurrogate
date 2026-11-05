// 回せるものの宣言を読む。宣言そのものは repo の根の conf.json（縮約の仕方 methods、評価の系列 series、実験 experiments）が持つ。テスト専用の学習はテストの都合なので conf.json に入れず、隣の test-trains.ts が持つ。
// 学習 run は 縮約の仕方 × 群分け で決まり、名前は `<仕方>_<群分け>`（`kam_hn`、`pca_hn_ac`）。群分けは群をゲートの小文字の名前でつなぎ、群どうしを _ で区切る。
// ここは読んで形を確かめ、仕方に群分けを重ねて学習の設定を組むだけ。段取り（orchestrate.ts）がここから spec.json を組む。形は spec.ts。
// conf.json は既定と違う値だけを書き、既定はここが埋める。description はその宣言の理由で、spec には入れない。綴りの確かめはここと、pytest が Python のドメインに当てて見る（tests/test_surrogate.py）。
import declaration from '../../conf.json' with { type: 'json' };
import { TEST_DECLARATIONS } from './test-trains.ts';
import type { Experiment, Group, LibrarySpec, Series, SimSpec, SurrogateConfig, Train } from './spec.ts';

/** 仕方が持つ値。既定値は Python の署名が持つので、書くのは仕方が決める値だけ。 */
type Declared = {
  cell: string;
  form: 'hybrid' | 'sindy';
  /** 教師データを取るモデルと、学習用の電流の長さ（ms）。 */
  target: string;
  duration: number;
  compressor: 'pca' | 'ae' | 'kam';
  /** 群の潜在次元（全群で同じ）。 */
  d: number;
  rhs: 'sindy' | 'ude' | 'kam';
  library_specs?: LibrarySpec[];
  compressor_params?: Record<string, unknown>;
  rhs_params?: Record<string, unknown>;
  /** 膜電位へ吸収するゲート（大文字の状態の名前、`["M"]`）。 */
  absorb?: string[];
};

type Entry = Record<string, unknown>;
/** 仕方に書ける鍵（description のほか）。テストはこれを仕方に重ねて上書きする。 */
const KEYS = [
  'cell',
  'form',
  'target',
  'duration',
  'compressor',
  'd',
  'rhs',
  'library_specs',
  'compressor_params',
  'rhs_params',
  'absorb',
];

function fail(where: string, message: string): never {
  throw new Error(`conf.json の ${where}：${message}`);
}
function isObject(value: unknown): value is Entry {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function isStrings(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(v => typeof v === 'string');
}

/** 細胞のゲートの項（レートとその積）に定数と電位の項を足した、既定の項ライブラリ。 */
function libraryOf(cell: string): LibrarySpec[] {
  const gates = ['gate_forward', 'gate_backward', 'gate_forward_product', 'gate_backward_product'];
  return [...gates.map(g => ({ type: `${cell}_${g}` })), { type: 'basis' }];
}

/** 書かれなかった値を既定で埋める。細胞は Traub、教師は細胞と同じ、圧縮は仕方の名前、右辺は KAM なら KAM でほかは SINDy。 */
function filled(method: string, entry: Entry): Entry {
  const cell = entry.cell ?? 'traub';
  const compressor = entry.compressor ?? method;
  const rhs = entry.rhs ?? (compressor === 'kam' ? 'kam' : 'sindy');
  const library = rhs === 'sindy' ? { library_specs: entry.library_specs ?? libraryOf(String(cell)) } : {};
  return {
    form: 'hybrid',
    duration: 9000,
    d: 1,
    ...entry,
    cell,
    target: entry.target ?? cell,
    compressor,
    rhs,
    ...library,
  };
}

/** 鍵と値の形を確かめる。 */
function declared(where: string, entry: Entry): Declared {
  for (const key of Object.keys(entry)) if (!KEYS.includes(key)) fail(where, `知らない鍵 ${key}`);
  if (typeof entry.cell !== 'string' || typeof entry.target !== 'string') fail(where, 'cell と target は文字列');
  if (entry.form !== 'hybrid' && entry.form !== 'sindy') fail(where, 'form は hybrid か sindy');
  if (typeof entry.duration !== 'number') fail(where, 'duration は数');
  if (entry.compressor !== 'pca' && entry.compressor !== 'ae' && entry.compressor !== 'kam')
    fail(where, 'compressor は pca・ae・kam');
  if (typeof entry.d !== 'number') fail(where, 'd は数');
  if (entry.rhs !== 'sindy' && entry.rhs !== 'ude' && entry.rhs !== 'kam') fail(where, 'rhs は sindy・ude・kam');
  if (
    entry.library_specs !== undefined &&
    !(Array.isArray(entry.library_specs) && entry.library_specs.every(s => isObject(s) && typeof s.type === 'string'))
  )
    fail(where, 'library_specs は type を持つ物の配列');
  if (entry.absorb !== undefined &&
    !(Array.isArray(entry.absorb) && entry.absorb.length > 0 && entry.absorb.every(g => typeof g === 'string')))
    fail(where, 'absorb は状態の名前の空でない配列');
  if (entry.absorb !== undefined && entry.form !== 'hybrid') fail(where, 'absorb は hybrid だけ');
  for (const key of ['compressor_params', 'rhs_params'])
    if (entry[key] !== undefined && !isObject(entry[key])) fail(where, `${key} は物`);
  return entry as Declared;
}

/** description を外した値。 */
function values(entry: Entry): Entry {
  const list: Entry = {};
  for (const key of Object.keys(entry)) if (key !== 'description') list[key] = entry[key];
  return list;
}

/** 群分けの名前（`hn_ac`）を状態の名前の群（[['H','N'],['A','C']]）にする。 */
export function groupsOf(grouping: string): string[][] {
  const groups: string[][] = [];
  for (const block of grouping.split('_')) {
    if (block.length < 2) throw new Error(`群分け ${grouping} の群 ${block} は 2 つ以上のゲートを持たない`);
    groups.push([...block.toUpperCase()]);
  }
  return groups;
}

/** 宣言と群分けを学習 run にする。学習用の電流（train）・刻み・全区画で学習するのは、どの学習でも同じ。 */
function train(name: string, d: Declared, grouping: string): Train {
  const data: SimSpec = {
    target: d.target,
    current_type: 'train',
    dt: 0.01,
    current_params: { duration: d.duration },
  };
  const rhs = d.rhs;
  const groups: Group[] = [];
  const shared = groupsOf(grouping).flat().filter(g => d.absorb?.includes(g));
  if (shared.length > 0) throw new Error(`${name} は ${shared.join('・')} を群と吸収の両方に入れている`);
  for (const gates of groupsOf(grouping)) {
    groups.push({
      gates,
      compressor: d.compressor,
      d: d.d,
      rhs,
      compressor_params: d.compressor_params ?? {},
      rhs_params:
        rhs === 'sindy' && d.library_specs !== undefined ? { library_specs: d.library_specs } : (d.rhs_params ?? {}),
    });
  }
  const config: SurrogateConfig = {
    cell: d.cell,
    form: d.form,
    data,
    groups,
    train_comp: null,
    ...(d.absorb === undefined ? {} : { absorbed: d.absorb }),
  };
  return { name, config };
}

const raw = declaration as unknown as {
  methods: Record<string, Entry>;
  series: Record<string, Entry>;
  experiments: Record<string, Entry>;
};

/** 縮約の仕方。名前 → 宣言。 */
export const METHODS: Record<string, Declared> = {};
for (const name of Object.keys(raw.methods))
  METHODS[name] = declared(`methods.${name}`, filled(name, values(raw.methods[name])));

/** テスト専用の学習 run。電流は 180 ms に縮める。 */
const TESTS: Record<string, Train> = {};
for (const name of Object.keys(TEST_DECLARATIONS)) {
  const { method, grouping, ...rest } = TEST_DECLARATIONS[name];
  if (typeof method !== 'string' || !raw.methods[method])
    fail(`tests.${name}`, `仕方 ${String(method)} が methods に無い`);
  if (typeof grouping !== 'string') fail(`tests.${name}`, 'grouping は群分けの名前');
  const entry = filled(method, {
    ...values(raw.methods[method]),
    duration: 180,
    ...rest,
  });
  TESTS[name] = train(name, declared(`tests.${name}`, entry), grouping);
}
export const TEST_NAMES = Object.keys(TESTS);

/** 仕方と群分けの学習 run の名前（`kam_hn`）。 */
export function modelOf(method: string, grouping: string): string {
  return `${method}_${grouping}`;
}

/** 学習 run の名前を仕方と群分けに分ける。 */
export function partsOf(name: string): { method: string; grouping: string } {
  const cut = name.indexOf('_');
  return { method: name.slice(0, cut), grouping: name.slice(cut + 1) };
}

/** 学習 run の名前（`<仕方>_<群分け>` かテストの名前）を学習 run にする。知らない名前は投げる。 */
export function trainOf(name: string): Train {
  if (TESTS[name]) return TESTS[name];
  const { method, grouping } = partsOf(name);
  if (!METHODS[method]) throw new Error(`学習 ${name} の仕方が methods（${Object.keys(METHODS).join(', ')}）に無い`);
  return train(name, METHODS[method], grouping);
}

/** 系列の鍵（ほかの鍵は電流のパラメータ）。 */
const SERIES_KEYS = ['description', 'target', 'current', 'stim', 'replace_targets', 'sweep'];

/** 系列の形を確かめ、既定を埋めて掃引にする。既定は Traub の単一区画、定常電流、刻み 0.01 ms、刺激前の静穏 10 ms、soma に入れて soma を置き換える。 */
function series(name: string, entry: Entry): Series {
  const where = `series.${name}`;
  const current: Entry = { silence_duration: 10.0 };
  for (const key of Object.keys(entry)) if (!SERIES_KEYS.includes(key)) current[key] = entry[key];
  const replace = entry.replace_targets ?? ['soma'];
  if (!isStrings(replace)) fail(where, 'replace_targets は区画の名前の配列');
  if (!isObject(entry.sweep) || Object.keys(entry.sweep).length !== 1)
    fail(where, 'sweep は { 掃引する電流のパラメータ: 値の配列 } の 1 組');
  const [[param, sweep]] = Object.entries(entry.sweep);
  if (!Array.isArray(sweep) || !sweep.every(v => typeof v === 'number')) fail(where, 'sweep の値は数の配列');
  const spec = {
    target: entry.target ?? 'traub',
    current_type: entry.current ?? 'lin&steady',
    dt: 0.01,
    current_params: current,
    stim: entry.stim ?? 'soma',
  };
  return {
    spec: spec as SimSpec,
    replace_targets: replace,
    param,
    values: sweep,
  };
}

/** 評価の系列。名前 → 掃引。 */
export const SERIES: Record<string, Series> = {};
for (const name of Object.keys(raw.series)) SERIES[name] = series(name, raw.series[name]);

/** ゲートの群分けを全部並べる。2 つ以上のゲートを入れた群だけを名前にし（例 hn_ac）、全部が 1 つずつの分け方は除く。減らす状態の少ない順、同じなら名前の順。 */
export function groupings(gates: string[]): string[] {
  let partitions: string[][][] = [[]];
  for (const gate of gates) {
    const next: string[][][] = [];
    for (const blocks of partitions) {
      for (let i = 0; i < blocks.length; i++) next.push(blocks.map((b, j) => (j === i ? [...b, gate] : b)));
      next.push([...blocks, [gate]]);
    }
    partitions = next;
  }
  const list: { name: string; reduced: number }[] = [];
  for (const blocks of partitions) {
    const groups = blocks.filter(b => b.length > 1);
    if (groups.length === 0) continue;
    let reduced = 0;
    for (const g of groups) reduced += g.length - 1;
    list.push({ name: groups.map(g => g.join('')).join('_'), reduced });
  }
  list.sort((x, y) => x.reduced - y.reduced || x.name.localeCompare(y.name));
  return list.map(g => g.name);
}

/** 主張 1 つ。check は判定の名前（evaluation/judge/checks.ts）、groupings は判定を絞る群分け（省けば実験の全部）。 */
export type Claim = { claim: string; check: string; groupings?: string[] };
/** 実験。description は何を回すかの一文（Markdown）。groupings × methods × series が回す run の直積で、claims は回す前に書いた主張と判定。 */
export type Plan = { id: string; description: string; groupings: string[]; methods: string[]; series: string[]; claims: Claim[] };

/** 実験の鍵。群分けは gates（全部の群分け）か groupings（名前を並べる）のどちらか一方で書く。 */
const EXPERIMENT_KEYS = ['description', 'gates', 'groupings', 'methods', 'series', 'claims'];

/** 主張の形を確かめる。 */
function claimOf(where: string, entry: unknown): Claim {
  if (!isObject(entry)) fail(where, '主張は物');
  for (const key of Object.keys(entry)) if (!['claim', 'check', 'groupings'].includes(key)) fail(where, `知らない鍵 ${key}`);
  if (typeof entry.claim !== 'string') fail(where, 'claim は文字列');
  if (typeof entry.check !== 'string') fail(where, 'check は判定の名前');
  if (entry.groupings !== undefined && !isStrings(entry.groupings)) fail(where, 'groupings は群分けの名前の配列');
  return entry as Claim;
}

/** 実験の形を確かめ、群分けを名前の並びにする。仕方と系列は宣言にあるものだけ。 */
function plan(id: string, entry: Entry): Plan {
  const where = `experiments.${id}`;
  for (const key of Object.keys(entry)) if (!EXPERIMENT_KEYS.includes(key)) fail(where, `知らない鍵 ${key}`);
  if (typeof entry.description !== 'string') fail(where, 'description は文字列');
  if (!isStrings(entry.methods)) fail(where, 'methods は縮約の仕方の配列');
  if (!isStrings(entry.series)) fail(where, 'series は系列の名前の配列');
  for (const method of entry.methods) if (!(method in METHODS)) fail(where, `仕方 ${method} が methods に無い`);
  for (const name of entry.series) if (!(name in SERIES)) fail(where, `系列 ${name} が series に無い`);
  let names: string[];
  if (isStrings(entry.gates) && entry.groupings === undefined) names = groupings(entry.gates);
  else if (isStrings(entry.groupings) && entry.gates === undefined) names = entry.groupings;
  else fail(where, 'gates（全部の群分け）か groupings（群分けの名前）のどちらか一方を書く');
  if (!Array.isArray(entry.claims) || entry.claims.length === 0) fail(where, 'claims は主張の空でない配列');
  const claims: Claim[] = [];
  for (let k = 0; k < entry.claims.length; k++) claims.push(claimOf(`${where}.claims[${k}]`, entry.claims[k]));
  return { id, description: entry.description, groupings: names, methods: entry.methods, series: entry.series, claims };
}

/** 実験。書いた順。 */
export const EXPERIMENTS: Plan[] = [];
for (const id of Object.keys(raw.experiments)) EXPERIMENTS.push(plan(id, raw.experiments[id]));

/** 実験の回す学習 run と系列の組。仕方の順、群分けの順、系列の順。 */
export function pairsOf(e: Pick<Plan, 'groupings' | 'methods' | 'series'>): Experiment {
  const pairs: Experiment['pairs'] = [];
  for (const method of e.methods)
    for (const grouping of e.groupings) for (const series of e.series) pairs.push({ model: modelOf(method, grouping), series });
  return { pairs };
}

/** 実験の id → 回す組。段取りの状態（status）が読む。 */
export const EXPERIMENT_RUNS: Record<string, Experiment> = {};
for (const e of EXPERIMENTS) EXPERIMENT_RUNS[e.id] = pairsOf(e);
