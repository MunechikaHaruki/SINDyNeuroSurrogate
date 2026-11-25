// 実験群（config/<群>.yaml）。frontmatter（description）の文書と、群の基準 base と実験の並び runs の文書の二つ。
// 実験は基準のファイル（base.ts）を名前か差分で指し、群の base に書いた欄だけを重ねる。書ける欄はここの FIELDS、差分に書ける欄は送る形（base.ts の型）が決める。
// 何を回して何を見る実験かの説明は冊の側（outputs/experiments/groups/<群>.yaml）が持つ。
// 返すのはモデルと系列を値で持つ解決済みの実験（Plan）だけで、resolve.ts と画面は宣言の形を知らない。
import { z } from 'zod';
import {
  Patch,
  Ref,
  groupings,
  inputOf,
  modelFrom,
  named,
  parsed,
  partOf,
  type ModelDecl,
  type Series,
} from './base.ts';

/** 細胞の数。1 刻みのカーネルのブロック（128 細胞）の倍数。 */
const Cells = z.array(z.number().int().positive().multipleOf(128)).nonempty();

/** 実験の欄。parts は学習器の基準（config/parts/）、part はどの学習器にも重ねる差分、groupings は群分けの名前の並びか `{ all: ゲート }`（その全部の群分け）、
 * inputs は評価の系列の基準（config/inputs/）、input はどの系列にも重ねる差分、replace は置き換える範囲（名前 → 区画。系列ごとに範囲の数だけ系列ができ、名前は `<系列>@<範囲>`）、
 * benches は GPU で測る系列の名前で、その系列で元のモデルと全部のモデルを細胞の数 cells で測る（ncu なら Nsight Compute で 1 刻みのカーネルも）。 */
const FIELDS = {
  parts: z.array(Ref).nonempty(),
  part: Patch,
  groupings: z.union([
    z.array(z.string()).nonempty(),
    z.strictObject({ all: z.array(z.string().length(1)).nonempty() }),
  ]),
  inputs: z.array(Ref).nonempty(),
  input: Patch,
  replace: z.record(z.string(), z.array(z.string()).nonempty()),
  benches: z.array(z.string()).nonempty(),
  cells: Cells,
  ncu: z.boolean(),
};
const Setting = z.strictObject(FIELDS).partial();
export type Setting = z.infer<typeof Setting>;

/** GPU の実測 1 つ。model はモデルの名前（null なら元のモデル）、cells は並べる細胞の数、ncu は Nsight Compute で 1 刻みのカーネルを測るか。 */
export type Bench = { name: string; model: string | null; series: string; cells: number[]; ncu: boolean };

/** モデル 1 つ。name は実験の中の名前（`<学習器の名前>.<群分け>`）、method と grouping は図が列を引く名前。 */
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

/** 実験群の二つの文書。実験は欄を書かなければ空（群の base のまま）。 */
const GroupDocument = z.tuple([
  z.strictObject({ description: z.string().min(1) }),
  z.strictObject({
    base: Setting,
    runs: z.record(z.string(), Setting.nullable()).refine(runs => Object.keys(runs).length > 0, 'runs が空'),
  }),
]);

/** 解決済みの実験を組む。回す組は 学習器 × 群分け × 系列 の直積。 */
export function build(where: string, id: string, setting: Setting, patches?: { part: Patch[]; input: Patch[] }): Plan {
  const commons = patches ?? { part: setting.part ? [setting.part] : [], input: setting.input ? [setting.input] : [] };
  const { parts, groupings: grouping, inputs } = setting;
  if (!parts) throw new Error(`${where}：学習器 parts が無い（base か run に書く）`);
  if (!grouping) throw new Error(`${where}：群分け groupings が無い（base か run に書く）`);
  if (!inputs) throw new Error(`${where}：系列 inputs が無い（base か run に書く）`);
  const groupingList = Array.isArray(grouping) ? grouping : groupings(grouping.all);
  const series: Record<string, Series> = {};
  const add = (name: string, value: Series) => {
    if (series[name]) throw new Error(`${where}：系列の名前 ${name} が二つある`);
    if (name.split('@').length > 2) throw new Error(`${where}：系列の名前 ${name} の @ は系列と範囲の間の一つだけ`);
    series[name] = value;
  };
  for (const ref of inputs) {
    const { name } = named(ref);
    const value = inputOf(ref, commons.input, `${where}：`);
    if (!setting.replace) add(name, value);
    else
      for (const [range, targets] of Object.entries(setting.replace))
        add(`${name}@${range}`, { ...value, replace_targets: targets });
  }
  const models: Model[] = [];
  for (const ref of parts) {
    const { name } = named(ref);
    const part = partOf(ref, commons.part, `${where}：`);
    for (const g of groupingList) models.push({ ...modelFrom(`${name}.${g}`, part, g), method: name, grouping: g });
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
  return { id, groupings: groupingList, methods: parts.map(p => named(p).name), models, series, pairs, benches };
}

/** 実験群 1 つを解決済みの実験にする。doc は実験群の YAML を読んだ二つの文書。
 * 実験は群の base に書いた欄を丸ごと重ねる。差分の欄（part・input）だけは、群の差分、実験の差分の順に当てる（合成すると null が消えるので、順に当てる）。 */
export function groupPlans(name: string, doc: unknown): { description: string; plans: Plan[] } {
  const where = `config/${name}.yaml`;
  const [front, body] = parsed(GroupDocument, where, doc);
  const plans = Object.entries(body.runs).map(([id, value]) => {
    const run = value ?? {};
    const both = (key: 'part' | 'input') => [body.base[key], run[key]].filter((p): p is Patch => p !== undefined);
    return build(`${where} の ${id}`, id, { ...body.base, ...run }, { part: both('part'), input: both('input') });
  });
  return { description: front.description, plans };
}
