// 二段目の設定。実験群（config/<群>.yaml）は、frontmatter（description）の文書と、群の基準 base・群が動かす欄 vary・実験の並び runs の文書の二つ。
// 書ける欄（設定のパラメタ）はここの PARAMS が持ち、一段目の部品（presets.ts）を名前か差分で指す。base は presets.json の base の上に重なる群の基準で、どの欄も書ける。
// 実験は base から vary に挙げた欄だけを書き換える（群ごとのスキーマ。挙げていない欄を書くと投げる）。
// 返すのはモデルと系列を値で持つ解決済みの実験（Plan）だけで、resolve.ts と画面は宣言の形を知らない。
import { z } from 'zod';
import {
  BASE_REPLACE,
  Ref,
  Train,
  groupings,
  inputFrom,
  methodOf,
  modelFrom,
  named,
  parsed,
  rangeOf,
  type ModelDecl,
  type Series,
} from './presets.ts';

/** 細胞の数。1 刻みのカーネルのブロック（128 細胞）の倍数。 */
const Cells = z.array(z.number().int().positive().multipleOf(128)).nonempty();

/** 設定のパラメタ。実験の run は、ここの欄の値の組。
 * methods は縮約の仕方、groupings は群分けの名前の並びか `{ all: ゲート }`（その全部の群分け）、train は学習の入力、
 * series は評価の入力（`<入力>@<範囲>` と書けば置き換える範囲をその系列だけ決める）、replace は置き換える区画の範囲の名前の並び（入力ごとに範囲の数だけ系列ができる）、
 * benches は GPU で測る系列の名前で、その系列で元のモデルと全部のモデルを細胞の数 cells で測る（ncu なら Nsight Compute で 1 刻みのカーネルも）。 */
export const PARAMS = {
  methods: z.array(Ref).nonempty(),
  groupings: z.union([
    z.array(z.string()).nonempty(),
    z.strictObject({ all: z.array(z.string().length(1)).nonempty() }),
  ]),
  train: Train,
  series: z.array(Ref).nonempty(),
  replace: z.array(z.string()).nonempty(),
  benches: z.array(z.string()).nonempty(),
  cells: Cells,
  ncu: z.boolean(),
};
export type Param = keyof typeof PARAMS;
const PARAM_NAMES = Object.keys(PARAMS) as [Param, ...Param[]];
/** 設定。どの欄も省ける（省いた欄は base か presets.json の base から）。 */
export type Setting = { [K in Param]?: z.infer<(typeof PARAMS)[K]> };

/** GPU の実測 1 つ。model はモデルの名前（null なら元のモデル）、cells は並べる細胞の数、ncu は Nsight Compute で 1 刻みのカーネルを測るか。 */
export type Bench = { name: string; model: string | null; series: string; cells: number[]; ncu: boolean };

/** モデル 1 つ。name は実験の中の名前（`<仕方の名前>.<群分け>`）、method と grouping は図が列を引く名前。 */
export type Model = ModelDecl & { method: string; grouping: string };

/** 解決済みの実験。モデルと系列を値で持ち、pairs が回す組（モデルの名前 × 系列の名前）。
 * groupings と methods は画面が絞り込みに使う名前の並び。名前は実験の中で一意。 */
export type Plan = {
  id: string;
  groupings: string[];
  methods: string[];
  models: Model[];
  series: Record<string, Series>;
  pairs: { model: string; series: string }[];
  /** GPU の実測。無ければ空。 */
  benches: Bench[];
};

/** 実験群の二つの文書の外形。runs の中身は vary が決める群のスキーマで後から当てる。 */
const GroupDocument = z.tuple([
  z.strictObject({ description: z.string().min(1) }),
  z.strictObject({
    base: z.strictObject(PARAMS).partial(),
    vary: z.array(z.enum(PARAM_NAMES)).refine(v => new Set(v).size === v.length, 'vary に同じ欄が二度ある'),
    runs: z.record(z.string(), z.unknown()).refine(runs => Object.keys(runs).length > 0, 'runs が空'),
  }),
]);

/** 実験群のスキーマ。実験は vary に挙げた欄だけを書ける（何を見る実験かの説明は冊の側 outputs/experiments/groups/<群>.yaml が持つ）。 */
function runSchema(vary: Param[]) {
  const fields: Partial<Record<Param, z.ZodType>> = {};
  for (const key of vary) fields[key] = PARAMS[key].optional();
  return z.strictObject(fields);
}

/** 系列の指し方を、置き換える範囲ごとの系列にする。`<入力>@<範囲>` はその範囲だけ。範囲が一つなら名前は書いたまま、二つ以上なら `<名前>@<範囲>`。 */
function seriesOf(ref: z.infer<typeof Ref>, replace: string[]): [string, Series][] {
  const { name, from, diff } = named(ref);
  const parts = from.split('@');
  if (parts.length > 2 || name.split('@').length > 2) throw new Error(`系列 ${name} の @ は入力と範囲の間の一つだけ`);
  const [input, pinned] = parts;
  const value = inputFrom(input, diff);
  if (pinned !== undefined) return [[name, { ...value, replace_targets: rangeOf(pinned) }]];
  if (replace.length === 1) return [[name, { ...value, replace_targets: rangeOf(replace[0]) }]];
  return replace.map(range => [`${name}@${range}`, { ...value, replace_targets: rangeOf(range) }]);
}

/** 解決済みの実験を組む。回す組は 仕方 × 群分け × 系列 の直積で、学習の入力は全部のモデルで同じ。 */
export function build(where: string, id: string, setting: Setting): Plan {
  const { methods, groupings: grouping, series: refs } = setting;
  if (!methods) throw new Error(`${where}：仕方 methods が無い（base か run に書く）`);
  if (!grouping) throw new Error(`${where}：群分け groupings が無い（base か run に書く）`);
  if (!refs) throw new Error(`${where}：系列 series が無い（base か run に書く）`);
  const groupingList = Array.isArray(grouping) ? grouping : groupings(grouping.all);
  const replace = setting.replace ?? BASE_REPLACE;
  replace.forEach(rangeOf);
  const series: Record<string, Series> = {};
  for (const ref of refs)
    for (const [name, value] of seriesOf(ref, replace)) {
      if (series[name]) throw new Error(`${where}：系列の名前 ${name} が二つある`);
      series[name] = value;
    }
  const models: Model[] = [];
  for (const ref of methods) {
    const { name, from, diff } = named(ref);
    const method = methodOf(from, diff);
    for (const g of groupingList)
      models.push({ ...modelFrom(`${name}.${g}`, method, g, setting.train), method: name, grouping: g });
  }
  if (new Set(models.map(m => m.name)).size !== models.length) throw new Error(`${where}：モデルの名前が重なる`);
  const pairs = models.flatMap(m => Object.keys(series).map(s => ({ model: m.name, series: s })));
  const benches: Bench[] = [];
  for (const s of setting.benches ?? []) {
    if (!series[s]) throw new Error(`${where}：GPU の実測の系列 ${s} を回さない（${Object.keys(series).join(', ')}）`);
    if (!setting.cells) throw new Error(`${where}：GPU の実測の細胞の数 cells が無い`);
    for (const model of [null, ...models.map(m => m.name)])
      benches.push({
        name: `${model ?? 'original'}@${s}`,
        model,
        series: s,
        cells: setting.cells,
        ncu: setting.ncu ?? false,
      });
  }
  return {
    id,
    groupings: groupingList,
    methods: methods.map(m => named(m).name),
    models,
    series,
    pairs,
    benches,
  };
}

/** 実験群 1 つを解決済みの実験にする。doc は実験群の YAML を読んだ二つの文書。実験は群の基準に、書いた欄だけを丸ごと重ねる（学習の入力 train だけは欄ごとに混ぜる）。 */
export function groupPlans(name: string, doc: unknown): { description: string; plans: Plan[] } {
  const where = `config/${name}.yaml`;
  const [front, body] = parsed(GroupDocument, where, doc);
  const schema = runSchema(body.vary);
  const plans = Object.entries(body.runs).map(([id, value]) => {
    const at = `${where} の ${id}`;
    const run = parsed(schema, `${at}（書けるのは vary の ${body.vary.join('・') || 'なし'}）`, value ?? {}) as Setting;
    const train = body.base.train || run.train ? { ...body.base.train, ...run.train } : undefined;
    return build(at, id, { ...body.base, ...run, ...(train ? { train } : {}) });
  });
  return { description: front.description, plans };
}
