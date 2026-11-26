// 実験の宣言（config/experiments.yaml）。frontmatter（description）の文書と、群 → 実験の文書の二つ。群は冊のページの単位で、道案内の一文 description と実験の並び experiments を持つ。
// 実験は欄を全部自分で書く。学習器は名前（base.ts の learnerDiff の文法）で、系列は基準からの差分で指し、系列の名前は中身から付く（seriesName）。実験どうしで同じものは YAML のアンカー（&名前・*名前・<<:）で使い回し、ここは上書きの規則を持たない。
// 書ける欄はここの Setting、差分に書ける欄は送る形（base.ts の型）が決める。何を回して何を見る実験かの説明は冊の側（outputs/experiments/groups/<群>.yaml）が持つ。
// 返すのはモデルと系列を値で持つ解決済みの実験（Plan）だけで、resolve.ts と画面は宣言の形を知らない。
import { z } from 'zod';
import { Patch, SeriesRef, groupings, inputOf, learnerOf, modelFrom, parsed, type ModelDecl, type Series } from './base.ts';

/** 細胞の数。1 刻みのカーネルのブロック（128 細胞）の倍数。 */
const Cells = z.array(z.number().int().positive().multipleOf(128)).nonempty();

/** 実験の欄。learners は学習器の名前（kam、pca-sindy など）、train は学習の電流（学習器の data）への差分で、実験のどのモデルも同じ学習の入力を使う。
 * groupings は群分けの名前の並びか `{ all: ゲート }`（その全部の群分け）、inputs は評価の系列の基準からの差分（掃引 sweep と置き換える区画 replace_targets もここに書く）。
 * bench の series は系列の名前（中身から付いたもの）。
 * bench は GPU の実測で、系列 series ごとに元のモデルと全部のモデルを細胞の数 cells で測る（ncu なら Nsight Compute で 1 刻みのカーネルも）。 */
const Setting = z.strictObject({
  learners: z.array(z.string()).nonempty(),
  train: Patch.optional(),
  groupings: z.union([
    z.array(z.string()).nonempty(),
    z.strictObject({ all: z.array(z.string().length(1)).nonempty() }),
  ]),
  inputs: z.array(SeriesRef).nonempty(),
  bench: z.strictObject({ series: z.array(z.string()).nonempty(), cells: Cells, ncu: z.boolean() }).optional(),
});
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

/** 群 1 つ。 */
const Group = z.strictObject({
  description: z.string().min(1),
  experiments: z.record(z.string(), Setting).refine(e => Object.keys(e).length > 0, 'experiments が空'),
});
/** 宣言のファイルの二つの文書。 */
const Document = z.tuple([
  z.strictObject({ description: z.string().min(1) }),
  z.strictObject({ groups: z.record(z.string(), Group).refine(g => Object.keys(g).length > 0, 'groups が空') }),
]);

/** 解決済みの実験を組む。回す組は 学習器 × 群分け × 系列 の直積。 */
export function build(where: string, id: string, setting: Setting): Plan {
  const { learners, train, groupings: grouping, inputs, bench } = setting;
  const groupingList = Array.isArray(grouping) ? grouping : groupings(grouping.all);
  const series: Record<string, Series> = {};
  for (const ref of inputs) {
    const { name, series: value } = inputOf(ref, `${where}：`);
    if (series[name]) throw new Error(`${where}：系列 ${name} が二つある（名前は細胞・電流の種類・置き換える区画から付く）`);
    series[name] = value;
  }
  const models: Model[] = [];
  for (const name of learners) {
    const part = learnerOf(name, train ? [{ data: train }] : [], `${where}：`);
    for (const g of groupingList) models.push({ ...modelFrom(`${name}.${g}`, part, g), method: name, grouping: g });
  }
  if (new Set(models.map(m => m.name)).size !== models.length) throw new Error(`${where}：モデルの名前が重なる`);
  const pairs = models.flatMap(m => Object.keys(series).map(s => ({ model: m.name, series: s })));
  const benches: Bench[] = [];
  for (const s of bench?.series ?? []) {
    if (!series[s]) throw new Error(`${where}：GPU の実測の系列 ${s} を回さない（${Object.keys(series).join(', ')}）`);
    for (const model of [null, ...models.map(m => m.name)])
      benches.push({ name: `${model ?? 'original'}@${s}`, model, series: s, cells: bench!.cells, ncu: bench!.ncu });
  }
  return { id, groupings: groupingList, methods: learners, models, series, pairs, benches };
}

/** 宣言を群ごとの解決済みの実験にする（群も実験も書いた順）。doc は宣言の YAML を読んだ二つの文書。 */
export function groupPlans(doc: unknown): { name: string; description: string; plans: Plan[] }[] {
  const where = 'config/experiments.yaml';
  const [, body] = parsed(Document, where, doc);
  return Object.entries(body.groups).map(([name, group]) => ({
    name,
    description: group.description,
    plans: Object.entries(group.experiments).map(([id, setting]) => build(`${where} の ${name} の ${id}`, id, setting)),
  }));
}
