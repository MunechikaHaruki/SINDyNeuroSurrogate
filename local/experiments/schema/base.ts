// 基準のファイル。送る形そのままの JSON を置き、実験はそこからの差分で書く（実験群は schema.ts）。
// config/learner.json は学習器 1 つ（学習の設定 SurrogateConfig から群の並びを除き、群のゲートを持たない群 group を 1 つ持つ形）、
// config/input.json は評価の系列 1 本（電流 1 本と置き換える区画）、config/measures.json は比べる物差し。基準はそれぞれ 1 つだけで、どの指し方もそこからの差分。
// 値の既定はファイルに書き、ここが持つ値は、学習の系列で soma を置き換えることだけ（TRAIN_REPLACE）。
// 差分は JSON Merge Patch（RFC 7386：オブジェクトは欄ごとに重ね、null はその欄を消し、ほかは置き換える。そのため null という値は差分で書けない）で順に当て、当てた結果を送る形の型で確かめる（書ける欄は送る形が決める）。
// 学習器は名前（learnerDiff の文法）が宣言そのもので、系列は差分から名前を付ける（seriesName）。掃引（どの電流パラメータをどの値で振るか）は送る形に無く、系列の指し方 `{ sweep, ...差分 }` だけが持つ。
// テスト専用のモデルは隣の test-models.ts が持つ。形は ../../../shared/spec.ts。綴りの確かめはここと、pytest が Python のドメインに当てて見る（remote/tests/test_surrogate.py）。
import { readFileSync } from 'node:fs';
import { z } from 'zod';
import { TEST_COMMON, TEST_DECLARATIONS } from './test-models.ts';
import type { Group, PlanSeries, SurrogateConfig } from '../../../shared/spec.ts';

/** 評価の系列 1 本。電流（掃引する前）と掃引と、置き換えたモデルの側の置換範囲（置換するノード名。互換かどうかは Python が名前ごとに検証する）。
 * 送る spec では、掃引の点ごとの入力（run の置き場の inputs/<鍵>/）と、範囲（列の名前の @ の後ろ）に分かれる。 */
export type Series = PlanSeries;

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

/** 掃引の書き方 `{ <電流パラメータ>: [値...] }`。振るパラメータは 1 つで、値は重ならない。 */
const SweepShape = z
  .record(z.string(), z.array(z.number()).nonempty())
  .refine(s => Object.keys(s).length === 1, '掃引は電流パラメータ 1 つ（{ value: [0, 2.5] } の形）')
  .refine(s => Object.values(s).every(v => new Set(v).size === v.length), '掃引の値が重なる');

/** 基準に重ねる差分。 */
export const Patch = z.record(z.string(), z.unknown());
export type Patch = z.infer<typeof Patch>;
/** 系列の指し方。系列の基準からの差分に、掃引 sweep を添えたもの（名前は中身から付く。seriesName）。 */
export const SeriesRef = z.looseObject({ sweep: SweepShape.optional() });
export type SeriesRef = z.infer<typeof SeriesRef>;

const CONFIG = new URL('../config/', import.meta.url);

/** config/<名前> の JSON。 */
function config(name: string): Json {
  return JSON.parse(readFileSync(new URL(name, CONFIG), 'utf8'));
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
  config('measures.json'),
);

// ---- 学習器とモデル

const SimSpec = z.strictObject({
  target: z.string(),
  current_type: z.string(),
  dt: z.number().positive(),
  current_params: z.record(z.string(), z.number()),
  stim: z.string().optional(),
});

/** 学習器 1 つの形（config/learner.json）。群のゲートは群分けが決める。 */
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

/** 学習器の基準。 */
const LEARNER = config('learner.json');

/** 学習器を値にする。基準に、差分 diff、次に重ねる差分 commons を順に当てて形を確かめる。where は誤りの文に出す場所、label は学習器の名前。 */
export function partOf(diff: Patch, commons: Patch[] = [], where = '', label = '基準'): Part {
  const value = [diff, ...commons].reduce(patched, LEARNER);
  return parsed(Part, `${where}学習器 ${label}${shown(diff, commons)}`, value);
}

const COMPRESSORS = ['pca', 'ae', 'kam'] as const;
const RHS = ['sindy', 'ude', 'kam'] as const;

/** 学習器の名前（`<圧縮器>-<右辺>[-d<潜在>][-absorb-<ゲート>]`、圧縮器と右辺が同じなら 1 語。例 kam、kam-absorb-m、pca-sindy、ae-sindy-d5）を、学習器の基準への差分にする。
 * 名前が宣言そのもので、名前に出ない設定は書けない。文法に合わないか、書き方が一通りでなければ投げる。 */
export function learnerDiff(name: string): Patch {
  const match = /^([a-z]+)(?:-([a-z]+))?(?:-d(\d+))?(?:-absorb-([a-z](?:\+[a-z])*))?$/.exec(name);
  const [, compressor, rhs = compressor, d, absorbed] = match ?? [];
  if (!COMPRESSORS.includes(compressor as never) || !RHS.includes(rhs as never))
    throw new Error(`学習器 ${name} は <圧縮器>-<右辺>[-d<潜在>][-absorb-<ゲート>] でない（圧縮器 ${COMPRESSORS.join('・')}、右辺 ${RHS.join('・')}）`);
  if ((compressor === 'kam') !== (rhs === 'kam')) throw new Error(`学習器 ${name}：KAM は圧縮器と右辺の両方が kam（kam と書く）`);
  if (rhs === 'ude' && compressor !== 'ae') throw new Error(`学習器 ${name}：UDE は AE を一緒に学習するので圧縮器は ae`);
  const group: Patch = { compressor, rhs };
  if (rhs !== 'sindy') group.rhs_params = { library_specs: null };
  if (d !== undefined) group.d = Number(d);
  const diff: Patch = { group };
  if (absorbed) diff.absorbed = absorbed.split('+').map(g => g.toUpperCase());
  const canonical = `${compressor === rhs ? compressor : `${compressor}-${rhs}`}${d !== undefined && d !== '1' ? `-d${d}` : ''}${absorbed ? `-absorb-${absorbed}` : ''}`;
  if (canonical !== name) throw new Error(`学習器 ${name} は ${canonical} と書く`);
  return diff;
}

/** 名前の学習器を値にする。commons は名前の差分の後に重ねる差分（実験の train）。 */
export function learnerOf(name: string, commons: Patch[] = [], where = ''): Part {
  return partOf(learnerDiff(name), commons, where, name);
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

/** 評価の系列 1 本の形（config/input.json）。 */
const SeriesShape = z.strictObject({
  spec: SimSpec.required({ stim: true }),
  replace_targets: z.array(z.string()).nonempty(),
});

/** 系列の基準。 */
const INPUT = config('input.json');

/** 系列の名前。中身から付ける：`<細胞>-<電流の種類（& の後ろ）>@<置き換える区画を + でつないだもの>`（例 traub-steady@soma、traub19-steady@c05+c07+soma+c09+c11）。 */
export function seriesName(series: Pick<Series, 'spec' | 'replace_targets'>): string {
  const kind = series.spec.current_type.split('&').at(-1);
  return `${series.spec.target}-${kind}@${series.replace_targets.join('+')}`;
}

/** 系列を値にする。基準に差分を当てて形を確かめ、指し方の掃引を添えて、中身から名前を付ける。where は誤りの文に出す場所。 */
export function inputOf(ref: SeriesRef, where = ''): { name: string; series: Series } {
  const { sweep: written, ...diff } = parsed(SeriesRef, `${where}系列`, ref);
  const value = patched(INPUT, diff);
  const { spec, replace_targets } = parsed(SeriesShape, `${where}系列${shown(diff, [])}`, value);
  const sweep = written ? (([param, values]) => ({ param, values }))(Object.entries(written)[0]) : null;
  const name = seriesName({ spec, replace_targets });
  if (sweep && sweep.param in spec.current_params)
    throw new Error(`${where}系列 ${name}：掃引する ${sweep.param} を current_params にも書いている（どちらか一方に）`);
  return { name, series: { spec, sweep, replace_targets } };
}

/** 学習の系列で置き換える区画。 */
const TRAIN_REPLACE = ['soma'];

/** モデルの学習の系列。学習用の電流（config.data）を単発で回す入力で、その元のモデルの run が学習器の学習データになる。soma を置き換える。
 * 名前はほかの系列と同じく中身から付ける（traub-train@soma）。 */
export function trainSeries(config: SurrogateConfig): { name: string; series: Series } {
  const series: Series = { spec: config.data, sweep: null, replace_targets: TRAIN_REPLACE };
  return { name: seriesName(series), series };
}

/** 基準のファイルを読み込むときに確かめる。 */
partOf({});
inputOf({});

/** テスト専用のモデル。学習器の基準に差分を重ね、テストの共通の差分（学習の電流を縮める）を最後に当てる。 */
export const TESTS: Record<string, ModelDecl> = {};
for (const [name, { grouping, ...diff }] of Object.entries(TEST_DECLARATIONS))
  TESTS[name] = modelFrom(name, partOf(diff, [TEST_COMMON], `テスト ${name} の `, name), grouping);
