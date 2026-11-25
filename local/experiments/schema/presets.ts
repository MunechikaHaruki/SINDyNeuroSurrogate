// 一段目の設定。値はすべて config/presets.json が持ち、ここは値を持たない：比べる物差し measures、どの実験にも共通の基準 base、
// 名前で指す部品（縮約の仕方 methods、入力 inputs、置き換える区画の範囲 ranges）。二段目（実験群ごとの config/<群>.yaml）は schema.ts。
// 実験は部品を名前か、名前からの差分 `{ name, from, ...差分 }` で指し、ここの関数が学習器の設定・学習の入力・評価の系列にする。
// 混ぜる順は base → 部品 → 差分で、書かなかった欄だけ値から導く（学習の入力の教師は細胞、圧縮は仕方の名前、右辺は圧縮から、項ライブラリは細胞から）。
// テスト専用のモデルは隣の test-models.ts が持つ。形は ../../../shared/spec.ts。綴りの確かめはここと、pytest が Python のドメインに当てて見る（tests/test_surrogate.py）。
import { z } from 'zod';
import declaration from '../config/presets.json' with { type: 'json' };
import { TEST_DECLARATIONS } from './test-models.ts';
import type { Group, Input, LibrarySpec, SimSpec, SurrogateConfig } from '../../../shared/spec.ts';

/** 評価の系列 1 本。入力（電流と掃引）と、置き換えたモデルの側の置換範囲。送る spec ではこの二つに分かれる（入力が run の置き場の inputs/<鍵>/、範囲が列の名前の @ の後ろ）。 */
export type Series = Input & {
  /** 置換するノード名。互換かどうかは Python が名前ごとに検証する。 */
  replace_targets: string[];
};

/** モデル 1 つの宣言。name はモデルの名前（実験の中の `<仕方の名前>.<群分け>` かテストの名前）で、同一性には入らない。config は群ごとの学習器の設定と、それらを組む形。 */
export type ModelDecl = { name: string; config: SurrogateConfig };

/** 宣言を形に当てる。誤りはどこの何かが分かる一文にする。 */
export function parsed<T>(schema: z.ZodType<T>, where: string, value: unknown): T {
  const result = schema.safeParse(value);
  if (!result.success) throw new Error(`${where}：${z.prettifyError(result.error)}`);
  return result.data;
}

type Entry = Record<string, unknown>;

const Presets = z.strictObject({
  measures: z.strictObject({ coincidence_window: z.number().positive() }),
  base: z.strictObject({
    dt: z.number().positive(),
    learner: z.strictObject({ cell: z.string(), form: z.enum(['hybrid', 'sindy']), d: z.number().int().positive() }),
    train: z.strictObject({ duration: z.number().positive() }),
    input: z.record(z.string(), z.unknown()),
    replace: z.array(z.string()).nonempty(),
  }),
  methods: z.record(z.string(), z.record(z.string(), z.unknown())),
  inputs: z.record(z.string(), z.record(z.string(), z.unknown())),
  ranges: z.record(z.string(), z.array(z.string()).nonempty()),
});
const raw = parsed(Presets, 'config/presets.json', declaration);

/** 置き換えたモデルを元のモデルと比べる物差し。coincidence_window は Γ の許容幅 Δ [ms]（2026-10-27 に決めた値）で、API の series に頼み方で渡す。 */
export const MEASURES = raw.measures;
/** 置き換える区画の範囲の既定（範囲の名前の並び）。 */
export const BASE_REPLACE: string[] = raw.base.replace;
/** 部品の名前。手元の CLI が一覧に出す。 */
export const PART_NAMES = {
  methods: Object.keys(raw.methods),
  inputs: Object.keys(raw.inputs),
  ranges: Object.keys(raw.ranges),
};

/** 部品の指し方。名前か、名前を付けた差分 `{ name, from, ...差分 }`（name は実験の中の名前）。 */
export const Ref = z.union([z.string(), z.looseObject({ name: z.string(), from: z.string() })]);
export type Ref = z.infer<typeof Ref>;

/** 指し方の名前と、部品の名前と差分。 */
export function named(ref: Ref): { name: string; from: string; diff: Entry } {
  if (typeof ref === 'string') return { name: ref, from: ref, diff: {} };
  const { name, from, ...diff } = ref;
  return { name, from, diff };
}

/** 部品の名前を引く。無ければ並びを添えて投げる。 */
function preset(table: Record<string, Entry>, from: string, kind: string): Entry {
  const found = table[from];
  if (!found) throw new Error(`${kind} ${from} が presets.json に無い（${Object.keys(table).join(', ')}）`);
  return found;
}

// ---- 学習器（縮約の仕方 × 群分け）と学習の入力

/** 縮約の仕方の値の形。既定値は Python の署名が持つので、書くのは仕方が決める値だけ。 */
const Method = z
  .strictObject({
    cell: z.string(),
    form: z.enum(['hybrid', 'sindy']),
    compressor: z.enum(['pca', 'ae', 'kam']),
    /** 群の潜在次元（全群で同じ）。 */
    d: z.number().int().positive(),
    rhs: z.enum(['sindy', 'ude', 'kam']),
    library_specs: z.array(z.looseObject({ type: z.string(), gates: z.string().optional() })).optional(),
    compressor_params: z.record(z.string(), z.unknown()).optional(),
    rhs_params: z.record(z.string(), z.unknown()).optional(),
    /** 膜電位へ吸収するゲート（大文字の状態の名前、`["M"]`）。 */
    absorb: z.array(z.string()).nonempty().optional(),
  })
  .refine(m => m.absorb === undefined || m.form === 'hybrid', 'absorb は hybrid だけ');
type Method = z.infer<typeof Method>;

/** 細胞のゲートの項（レートとその積）に定数と電位の項を足した、既定の項ライブラリ。 */
function libraryOf(cell: string): LibrarySpec[] {
  const gates = ['gate_forward', 'gate_backward', 'gate_forward_product', 'gate_backward_product'];
  return [...gates.map(g => ({ type: `${cell}_${g}` })), { type: 'basis' }];
}

/** 縮約の仕方を値にする。base の学習器 → 部品 → 差分の順に重ね、書かなかった欄を導く。 */
export function methodOf(from: string, diff: Entry = {}): Method {
  const entry: Entry = { ...raw.base.learner, ...preset(raw.methods, from, '仕方'), ...diff };
  const compressor = entry.compressor ?? from;
  const rhs = entry.rhs ?? (compressor === 'kam' ? 'kam' : 'sindy');
  const library = rhs === 'sindy' ? { library_specs: entry.library_specs ?? libraryOf(String(entry.cell)) } : {};
  return parsed(Method, `仕方 ${from}${Object.keys(diff).length ? ` ${JSON.stringify(diff)}` : ''}`, {
    ...entry,
    compressor,
    rhs,
    ...library,
  });
}

/** 学習の入力の欄。学習用の電流（current_type 'train'）を単発で回すもので、教師のモデル target（省けば細胞）、長さ duration [ms]、パルス列の乱数 seed（省けば Python の既定）。 */
export const Train = z.strictObject({
  target: z.string().optional(),
  duration: z.number().positive().optional(),
  seed: z.number().int().optional(),
});
export type Train = z.infer<typeof Train>;

/** 学習の入力の電流。base の学習の入力に重ねる。 */
function trainData(cell: string, train: Train = {}): SimSpec {
  const t = { ...raw.base.train, ...train };
  return {
    target: t.target ?? cell,
    current_type: 'train',
    dt: raw.base.dt,
    current_params: { duration: t.duration, ...(t.seed === undefined ? {} : { seed: t.seed }) },
  };
}

/** ゲートの群分けを全部並べる。2 つ以上のゲートを入れた群だけを、群の中は `+`、群の間は `_` でつないで名前にし（例 h+n_a+c）、全部が 1 つずつの分け方は除く。減らす状態の少ない順、同じなら名前の順。 */
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
    list.push({ name: groups.map(g => g.join('+')).join('_'), reduced });
  }
  list.sort((x, y) => x.reduced - y.reduced || x.name.localeCompare(y.name));
  return list.map(g => g.name);
}

/** 群分けの名前（`h+n_a+c`）を状態の名前の群（[['H','N'],['A','C']]）にする。 */
export function groupsOf(grouping: string): string[][] {
  const groups: string[][] = [];
  for (const block of grouping.split('_').map(b => b.split('+'))) {
    if (block.length < 2 || block.some(g => g.length !== 1))
      throw new Error(`群分け ${grouping} の群 ${block.join('+')} は 1 文字のゲートを 2 つ以上 + でつないでいない`);
    groups.push(block.map(g => g.toUpperCase()));
  }
  return groups;
}

/** 縮約の仕方と群分けと学習の入力をモデルにする。刻みと全区画で学習するのは、どの学習器でも同じ。name は表示の名前で、同一性には入らない。 */
export function modelFrom(name: string, d: Method, grouping: string, train: Train = {}): ModelDecl {
  const shared = groupsOf(grouping)
    .flat()
    .filter(g => d.absorb?.includes(g));
  if (shared.length > 0) throw new Error(`${name} は ${shared.join('・')} を群と吸収の両方に入れている`);
  const groups: Group[] = groupsOf(grouping).map(gates => ({
    gates,
    compressor: d.compressor,
    d: d.d,
    rhs: d.rhs,
    compressor_params: d.compressor_params ?? {},
    rhs_params:
      d.rhs === 'sindy' && d.library_specs !== undefined ? { library_specs: d.library_specs } : (d.rhs_params ?? {}),
  }));
  const config: SurrogateConfig = {
    cell: d.cell,
    form: d.form,
    data: trainData(d.cell, train),
    groups,
    train_comp: null,
    ...(d.absorb === undefined ? {} : { absorbed: d.absorb }),
  };
  return { name, config };
}

/** モデルの学習の系列。学習用の電流（config.data）を単発で回す入力で、その元のモデルの run が学習器の学習データになる。置き換える区画は base の範囲。
 * 名前は中身から付ける（traub-train-duration-9000）。 */
export function trainSeries(config: SurrogateConfig): { name: string; series: Series } {
  const { target, current_type, current_params } = config.data;
  const name = [target, current_type, ...Object.entries(current_params).map(([k, v]) => `${k}-${v}`)].join('-');
  return { name, series: { spec: config.data, param: null, values: [], replace_targets: rangeOf(BASE_REPLACE[0]) } };
}

// ---- 評価の系列（入力 × 置き換える区画の範囲）

/** 置き換える区画の範囲を名前で引く。 */
export function rangeOf(name: string): string[] {
  const found = raw.ranges[name];
  if (!found) throw new Error(`置き換える範囲 ${name} が presets.json に無い（${Object.keys(raw.ranges).join(', ')}）`);
  return found;
}

/** 入力の差分に書けない欄。電流のパラメータに化けて送る spec に入るのを防ぐ。 */
const NOT_INPUT = ['replace', 'replace_targets', 'name', 'from'];
/** 入力の欄（ほかの欄は電流のパラメータ）。 */
const INPUT_KEYS = ['target', 'current', 'stim', 'sweep', 'dt'];
const InputShape = z.looseObject({
  target: z.string(),
  current: z.string(),
  stim: z.string(),
  dt: z.number(),
  /** { 掃引する電流のパラメータ: 値の配列 } の 1 組。 */
  sweep: z.record(z.string(), z.array(z.number())).refine(s => Object.keys(s).length === 1, 'sweep は 1 組'),
});

/** 入力を値にする。base の入力 → 部品 → 差分の順に重ね、既定にない欄は電流のパラメータにする。 */
export function inputFrom(from: string, diff: Entry = {}): Input {
  const stray = Object.keys(diff).filter(key => NOT_INPUT.includes(key));
  if (stray.length > 0)
    throw new Error(
      `入力 ${from} の差分に入力の欄でない ${stray.join('・')} がある（置き換える範囲は replace か <入力>@<範囲> で書く）`,
    );
  const entry = parsed(InputShape, `入力 ${from}${Object.keys(diff).length ? ` ${JSON.stringify(diff)}` : ''}`, {
    dt: raw.base.dt,
    ...raw.base.input,
    ...preset(raw.inputs, from, '入力'),
    ...diff,
  });
  const current: Entry = {};
  for (const key of Object.keys(entry)) if (!INPUT_KEYS.includes(key)) current[key] = entry[key];
  const [[param, values]] = Object.entries(entry.sweep);
  const spec = {
    target: entry.target,
    current_type: entry.current,
    dt: entry.dt,
    current_params: current,
    stim: entry.stim,
  } as SimSpec;
  return { spec, param, values };
}

/** 全部の入力を読み込むときに一度解き、使わない入力の綴りの誤りもここで落とす。 */
for (const name of PART_NAMES.inputs) inputFrom(name);

/** テスト専用のモデル。仕方に差分を重ね、学習の電流は 180 ms に縮める（教師は target で替えられる）。 */
export const TESTS: Record<string, ModelDecl> = {};
for (const [name, { method, grouping, target, ...rest }] of Object.entries(TEST_DECLARATIONS))
  TESTS[name] = modelFrom(name, methodOf(String(method), rest), String(grouping), {
    duration: 180,
    ...(target === undefined ? {} : { target: String(target) }),
  });
