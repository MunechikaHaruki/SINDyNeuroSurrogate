// 回せるものの宣言を読む。宣言そのものは隣の conf.json（縮約の仕方 methods、テスト専用の学習 tests、評価の系列 series）が持つ。
// 学習 run は 縮約の仕方 × 群分け で決まり、名前は `<仕方>_<群分け>`（`kam_hn`、`pca_hn_ac`）。群分けは群をゲートの小文字の名前でつなぎ、群どうしを _ で区切る。
// ここは読んで形を確かめ、仕方に群分けを重ねて学習の設定を組むだけ。段取り（orchestrate.ts）がここから spec.json を組む。形は spec.ts。
// conf.json の description はその宣言の理由で、spec には入れない。綴りの確かめはここと、pytest が Python のドメインに当てて見る（tests/test_surrogate.py）。
import declaration from './conf.json' with { type: 'json' };
import type { Group, LibrarySpec, Series, SimSpec, SurrogateConfig, Train } from './spec.ts';

/** 仕方が書く値。既定値は Python の署名が持つので、書くのは仕方が決める値だけ。 */
type Declared = {
  cell: string;
  form: 'hybrid' | 'sindy';
  /** 教師データを取るモデルと、学習用の電流の長さ（ms）。 */
  target: string;
  duration: number;
  compressor: 'pca' | 'ae' | 'kam';
  /** 群の潜在次元（全群で同じ）。 */
  d: number;
  rhs?: 'sindy' | 'ude' | 'kam';
  library_specs?: LibrarySpec[];
  compressor_params?: Record<string, unknown>;
  rhs_params?: Record<string, unknown>;
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

/** 鍵と値の形を確かめる。 */
function declared(where: string, entry: Entry): Declared {
  for (const key of Object.keys(entry)) if (!KEYS.includes(key)) fail(where, `知らない鍵 ${key}`);
  if (typeof entry.cell !== 'string' || typeof entry.target !== 'string') fail(where, 'cell と target は文字列');
  if (entry.form !== 'hybrid' && entry.form !== 'sindy') fail(where, 'form は hybrid か sindy');
  if (typeof entry.duration !== 'number') fail(where, 'duration は数');
  if (entry.compressor !== 'pca' && entry.compressor !== 'ae' && entry.compressor !== 'kam')
    fail(where, 'compressor は pca・ae・kam');
  if (typeof entry.d !== 'number') fail(where, 'd は数');
  if (entry.rhs !== undefined && entry.rhs !== 'sindy' && entry.rhs !== 'ude' && entry.rhs !== 'kam')
    fail(where, 'rhs は sindy・ude・kam');
  if (
    entry.library_specs !== undefined &&
    !(Array.isArray(entry.library_specs) && entry.library_specs.every(s => isObject(s) && typeof s.type === 'string'))
  )
    fail(where, 'library_specs は type を持つ物の配列');
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
  const data: SimSpec = { target: d.target, current_type: 'train', dt: 0.01, current_params: { duration: d.duration } };
  const rhs = d.rhs ?? 'sindy';
  const groups: Group[] = [];
  for (const gates of groupsOf(grouping)) {
    groups.push({
      gates,
      compressor: d.compressor,
      d: d.d,
      rhs,
      compressor_params: d.compressor_params ?? {},
      rhs_params: rhs === 'sindy' && d.library_specs !== undefined ? { library_specs: d.library_specs } : (d.rhs_params ?? {}),
    });
  }
  const config: SurrogateConfig = { cell: d.cell, form: d.form, data, groups, train_comp: null };
  return { name, config };
}

const raw = declaration as unknown as {
  methods: Record<string, Entry>;
  tests: Record<string, Entry>;
  series: Record<string, Entry>;
};

/** 縮約の仕方。名前 → 宣言。 */
export const METHODS: Record<string, Declared> = {};
for (const name of Object.keys(raw.methods)) METHODS[name] = declared(`methods.${name}`, values(raw.methods[name]));

/** テスト専用の学習。仕方に差分を重ねる。名前は _test_ で始め、どの実験にも入れない。 */
const TESTS: Record<string, Train> = {};
for (const name of Object.keys(raw.tests)) {
  const { method, grouping, ...rest } = values(raw.tests[name]);
  if (typeof method !== 'string' || !METHODS[method]) fail(`tests.${name}`, `仕方 ${String(method)} が methods に無い`);
  if (typeof grouping !== 'string') fail(`tests.${name}`, 'grouping は群分けの名前');
  TESTS[name] = train(name, declared(`tests.${name}`, { ...METHODS[method], ...rest }), grouping);
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

/** 系列の形を確かめ、description を外す。spec はそのまま Python へ渡す。 */
function series(name: string, entry: Entry): Series {
  const where = `series.${name}`;
  if (!isObject(entry.spec)) fail(where, 'spec は物');
  if (!isStrings(entry.replace_targets)) fail(where, 'replace_targets は区画の名前の配列');
  if (entry.param !== null && typeof entry.param !== 'string') fail(where, 'param は文字列か null');
  if (!Array.isArray(entry.values) || !entry.values.every(v => typeof v === 'number')) fail(where, 'values は数の配列');
  return {
    spec: entry.spec as SimSpec,
    replace_targets: entry.replace_targets,
    param: entry.param,
    values: entry.values,
  };
}

/** 評価の系列。名前 → 掃引。 */
export const SERIES: Record<string, Series> = {};
for (const name of Object.keys(raw.series)) SERIES[name] = series(name, raw.series[name]);
