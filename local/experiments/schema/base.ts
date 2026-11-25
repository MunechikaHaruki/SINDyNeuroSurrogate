// 基準のファイル。送る形そのままの JSON を置き、実験はそこからの差分で書く（実験群は schema.ts）。
// config/parts/<名前>.json は学習器 1 つ（学習の設定 SurrogateConfig から群の並びを除き、群のゲートを持たない群 group を 1 つ持つ形）、
// config/inputs/<名前>.json は評価の系列 1 本（入力と置き換える区画）、config/measures.json は比べる物差し。値の既定はファイルに書き、ここが持つ値は、学習の系列で soma を置き換えることだけ（TRAIN_REPLACE）。
// 差分は JSON Merge Patch（RFC 7386：オブジェクトは欄ごとに重ね、null はその欄を消し、ほかは置き換える。そのため null という値は差分で書けない）で順に当て、当てた結果を送る形の型で確かめる（書ける欄は送る形が決める）。
// テスト専用のモデルは隣の test-models.ts が持つ。形は ../../../shared/spec.ts。綴りの確かめはここと、pytest が Python のドメインに当てて見る（remote/tests/test_surrogate.py）。
import { readFileSync, readdirSync } from 'node:fs';
import { z } from 'zod';
import { TEST_COMMON, TEST_DECLARATIONS } from './test-models.ts';
import type { Group, Input, SurrogateConfig } from '../../../shared/spec.ts';

/** 評価の系列 1 本。入力（電流と掃引）と、置き換えたモデルの側の置換範囲。送る spec ではこの二つに分かれる（入力が run の置き場の inputs/<鍵>/、範囲が列の名前の @ の後ろ）。 */
export type Series = Input & {
  /** 置換するノード名。互換かどうかは Python が名前ごとに検証する。 */
  replace_targets: string[];
};

/** モデル 1 つの宣言。name はモデルの名前（実験の中の `<学習器の名前>.<群分け>` かテストの名前）で、同一性には入らない。config は群ごとの学習器の設定と、それらを組む形。 */
export type ModelDecl = { name: string; config: SurrogateConfig };

/** 宣言を形に当てる。誤りはどこの何かが分かる一文にする。 */
export function parsed<T>(schema: z.ZodType<T>, where: string, value: unknown): T {
  const result = schema.safeParse(value);
  if (!result.success) throw new Error(`${where}：${z.prettifyError(result.error)}`);
  return result.data;
}

type Json = unknown;
const isObject = (v: Json): v is Record<string, Json> => typeof v === 'object' && v !== null && !Array.isArray(v);

/** JSON Merge Patch。オブジェクトは欄ごとに重ね、null はその欄を消し、配列や値は置き換える。 */
export function patched(target: Json, patch: Json): Json {
  if (!isObject(patch)) return structuredClone(patch);
  const out: Record<string, Json> = isObject(target) ? structuredClone(target) : {};
  for (const [key, value] of Object.entries(patch)) {
    if (value === null) delete out[key];
    else out[key] = patched(out[key], value);
  }
  return out;
}

/** 基準のファイルの指し方。ファイルの名前か、名前を付けた差分 `{ name, from, ...差分 }`（name は実験の中の名前）。 */
export const Ref = z.union([z.string(), z.looseObject({ name: z.string(), from: z.string() })]);
export type Ref = z.infer<typeof Ref>;
/** どの基準にも重ねる差分。 */
export const Patch = z.record(z.string(), z.unknown());
export type Patch = z.infer<typeof Patch>;

/** 指し方の名前と、基準のファイルの名前と差分。 */
export function named(ref: Ref): { name: string; from: string; diff: Patch } {
  if (typeof ref === 'string') return { name: ref, from: ref, diff: {} };
  const { name, from, ...diff } = ref;
  return { name, from, diff };
}

const CONFIG = new URL('../config/', import.meta.url);

/** config/<dir>/ の JSON を名前 → 中身で。 */
function files(dir: string): Record<string, Json> {
  const found: Record<string, Json> = {};
  for (const file of readdirSync(new URL(`${dir}/`, CONFIG))
    .filter(f => f.endsWith('.json'))
    .sort())
    found[file.replace(/\.json$/, '')] = JSON.parse(readFileSync(new URL(`${dir}/${file}`, CONFIG), 'utf8'));
  return found;
}

/** 基準のファイルを名前で引く。無ければ並びを添えて投げる。 */
function base(table: Record<string, Json>, from: string, dir: string): Json {
  if (!(from in table)) throw new Error(`config/${dir}/${from}.json が無い（${Object.keys(table).join(', ')}）`);
  return table[from];
}

/** 誤りの文に出す差分。当てた順に並べる。 */
function shown(diff: Patch, commons: Patch[]): string {
  const all = [diff, ...commons].filter(p => Object.keys(p).length > 0);
  return all.length === 0 ? '' : ` に差分 ${all.map(p => JSON.stringify(p)).join(' → ')}`;
}

/** 置き換えたモデルを元のモデルと比べる物差し。coincidence_window は Γ の許容幅 Δ [ms]（2026-10-27 に決めた値）で、API の series に頼み方で渡す。 */
export const MEASURES = parsed(
  z.strictObject({ coincidence_window: z.number().positive() }),
  'config/measures.json',
  JSON.parse(readFileSync(new URL('measures.json', CONFIG), 'utf8')),
);

// ---- 学習器とモデル

const SimSpec = z.strictObject({
  target: z.string(),
  current_type: z.string(),
  dt: z.number().positive(),
  current_params: z.record(z.string(), z.number()),
  stim: z.string().optional(),
});

/** 学習器 1 つの形（config/parts/<名前>.json）。群のゲートは群分けが決める。 */
const Part = z
  .strictObject({
    cell: z.string(),
    form: z.enum(['hybrid', 'sindy']),
    /** 学習の入力の電流。単発で回し、その元のモデルの run が学習データになる。 */
    data: SimSpec,
    group: z.strictObject({
      compressor: z.enum(['pca', 'ae', 'kam']),
      d: z.number().int().positive(),
      rhs: z.enum(['sindy', 'ude', 'kam']),
      compressor_params: z.record(z.string(), z.unknown()),
      rhs_params: z.looseObject({
        library_specs: z.array(z.strictObject({ type: z.string(), gates: z.string().optional() })).optional(),
      }),
    }),
    train_comp: z.string().nullable(),
    /** 膜電位へ吸収するゲート（大文字の状態の名前、`["M"]`）。 */
    absorbed: z.array(z.string()).nonempty().optional(),
  })
  .refine(p => p.absorbed === undefined || p.form === 'hybrid', 'absorbed は hybrid だけ');
export type Part = z.infer<typeof Part>;

const PARTS = files('parts');
/** 学習器の基準のファイルの名前。 */
export const PART_NAMES = Object.keys(PARTS);

/** 学習器を値にする。基準のファイルに、自分の差分、次にどの学習器にも重ねる差分 commons を順に当てて形を確かめる。where は誤りの文に出す場所。 */
export function partOf(ref: Ref, commons: Patch[] = [], where = ''): Part {
  const { from, diff } = named(ref);
  const value = [diff, ...commons].reduce(patched, base(PARTS, from, 'parts'));
  return parsed(Part, `${where}学習器 ${from}${shown(diff, commons)}`, value);
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

/** 学習器を群分けの群の数だけ並べてモデルにする。どの群も同じ学習器の設定で、ゲートだけが違う。name は表示の名前で、同一性には入らない。 */
export function modelFrom(name: string, part: Part, grouping: string): ModelDecl {
  const shared = groupsOf(grouping)
    .flat()
    .filter(g => part.absorbed?.includes(g));
  if (shared.length > 0) throw new Error(`${name} は ${shared.join('・')} を群と吸収の両方に入れている`);
  const groups: Group[] = groupsOf(grouping).map(gates => ({ gates, ...part.group }));
  const { cell, form, data, train_comp, absorbed } = part;
  return { name, config: { cell, form, data, groups, train_comp, ...(absorbed ? { absorbed } : {}) } };
}

// ---- 評価の系列

/** 評価の系列 1 本の形（config/inputs/<名前>.json）。 */
const SeriesShape = z.strictObject({
  spec: SimSpec.required({ stim: true }),
  param: z.string().nullable(),
  values: z.array(z.number()),
  replace_targets: z.array(z.string()).nonempty(),
});

const INPUTS = files('inputs');
/** 系列の基準のファイルの名前。 */
export const INPUT_NAMES = Object.keys(INPUTS);

/** 系列を値にする。基準のファイルに、自分の差分、次にどの系列にも重ねる差分 commons を順に当てて形を確かめる。where は誤りの文に出す場所。 */
export function inputOf(ref: Ref, commons: Patch[] = [], where = ''): Series {
  const { from, diff } = named(ref);
  const value = [diff, ...commons].reduce(patched, base(INPUTS, from, 'inputs'));
  return parsed(SeriesShape, `${where}系列 ${from}${shown(diff, commons)}`, value);
}

/** 学習の系列で置き換える区画。 */
const TRAIN_REPLACE = ['soma'];

/** モデルの学習の系列。学習用の電流（config.data）を単発で回す入力で、その元のモデルの run が学習器の学習データになる。soma を置き換える。
 * 名前は中身から付ける（traub-train-duration-9000）。 */
export function trainSeries(config: SurrogateConfig): { name: string; series: Series } {
  const { target, current_type, current_params } = config.data;
  const name = [target, current_type, ...Object.entries(current_params).map(([k, v]) => `${k}-${v}`)].join('-');
  return { name, series: { spec: config.data, param: null, values: [], replace_targets: TRAIN_REPLACE } };
}

/** 基準のファイルを読み込むときに全部一度確かめ、使わないファイルの誤りもここで落とす。 */
for (const name of PART_NAMES) partOf(name);
for (const name of INPUT_NAMES) inputOf(name);

/** テスト専用のモデル。学習器の基準に差分を重ね、テストの共通の差分（学習の電流を縮める）を最後に当てる。 */
export const TESTS: Record<string, ModelDecl> = {};
for (const [name, { from, grouping, ...diff }] of Object.entries(TEST_DECLARATIONS))
  TESTS[name] = modelFrom(name, partOf({ name, from, ...diff }, [TEST_COMMON], `テスト ${name} の `), grouping);
