// 実験群の実験層（config/<群>.yaml）の形と、解決済みの実験（common.ts の Plan）への組み立て。
// 実験層は frontmatter（description）の文書と、共通の設定 setup と回す run の並び runs の文書の二つ。run に書いた欄だけが setup を置き換える。
// 返すのは学習 run と系列を値で持つ解決済みの実験だけで、resolve.ts と画面は実験層の形を知らない。
import { z } from 'zod';
import { Ref, methodOf, seriesFrom, trainFrom } from './presets.ts';

import { groupings, type Model, type Plan, type Bench, type Series } from './common.ts';

/** 仕方の指し方。プリセットの名前か、名前を付けた差分 `{ name, from, ...差分 }`（name は表示と判定の列の名前）。 */
export const MethodRef = z.union([z.string(), z.looseObject({ name: z.string(), from: z.string() })]);
/** 系列の指し方。仕方と同じ。 */
export const SeriesRef = MethodRef;

/** setup と run が持てる欄。群分けは、ゲートの並び gates（全部の群分けを組む）か、群分けの名前 groupings のどちらか。
 * benches は GPU で測る系列で、その系列で元のモデルと run の学習 run を全部、細胞の数 cells で測る（ncu なら Nsight Compute で 1 刻みのカーネルも）。 */
const Fields = {
  methods: z.array(MethodRef).nonempty().optional(),
  gates: z.array(z.string().length(1)).nonempty().optional(),
  groupings: z.array(z.string()).nonempty().optional(),
  series: z.array(SeriesRef).nonempty().optional(),
  benches: z.array(z.string()).nonempty().optional(),
  /** 細胞の数。1 刻みのカーネルのブロック（128 細胞）の倍数。 */
  cells: z.array(z.number().int().positive().multipleOf(128)).nonempty().optional(),
  ncu: z.boolean().optional(),
};

/** 実験層の二つの文書。frontmatter は実験群の道案内の一文だけ。 */
const GroupDocument = z.tuple([
  z.strictObject({ description: z.string().min(1) }),
  z.strictObject({
    setup: z.strictObject(Fields),
    runs: z
      .record(z.string(), z.strictObject({ description: z.string().min(1), ...Fields }))
      .refine(runs => Object.keys(runs).length > 0, 'runs が空'),
  }),
]);

/** 宣言を形に当てる。誤りはどこの何かが分かる一文にする。 */
export function parsed<T>(schema: z.ZodType<T>, where: string, value: unknown): T {
  const result = schema.safeParse(value);
  if (!result.success) throw new Error(`${where}：${z.prettifyError(result.error)}`);
  return result.data;
}

/** 指し方の名前と、presets.ts に渡す指し方。 */
function named(ref: z.infer<typeof MethodRef>): { name: string; ref: Ref } {
  if (typeof ref === 'string') return { name: ref, ref };
  const { name, ...rest } = ref;
  return { name, ref: rest as Ref };
}

type Run = { description: string } & { [K in keyof typeof Fields]?: z.infer<(typeof Fields)[K]> };

/** run に setup を重ねる。run が群分け（gates か groupings）を書いたら、setup の群分けは使わない。 */
function merged(setup: Omit<Run, 'description'>, run: Run): Run {
  const base = run.gates || run.groupings ? { ...setup, gates: undefined, groupings: undefined } : setup;
  return { ...base, ...Object.fromEntries(Object.entries(run).filter(([, v]) => v !== undefined)) } as Run;
}

/** 解決済みの実験を組む。回す組は 仕方 × 群分け × 系列 の直積。 */
export function build(where: string, id: string, run: Run): Plan {
  if (!run.methods) throw new Error(`${where}：仕方 methods が無い（setup か run に書く）`);
  if (!run.series) throw new Error(`${where}：系列 series が無い（setup か run に書く）`);
  if (!!run.gates === !!run.groupings) throw new Error(`${where}：群分けは gates か groupings のどちらか一つを書く`);
  const groupingList = run.gates ? groupings(run.gates) : run.groupings!;
  const series: Record<string, Series> = {};
  for (const s of run.series) {
    const { name, ref } = named(s);
    series[name] = seriesFrom(ref);
  }
  const models: Model[] = [];
  for (const m of run.methods) {
    const { name, ref } = named(m);
    const declared = methodOf(ref);
    for (const grouping of groupingList) {
      const model = `${name}.${grouping}`;
      models.push({ name: model, method: name, grouping, train: trainFrom(model, declared, grouping) });
    }
  }
  const pairs = models.flatMap(m => Object.keys(series).map(s => ({ model: m.name, series: s })));
  const benches: Bench[] = [];
  for (const s of run.benches ?? []) {
    if (!series[s]) throw new Error(`${where}：GPU の実測の系列 ${s} を run が回さない`);
    if (!run.cells) throw new Error(`${where}：GPU の実測の細胞の数 cells が無い`);
    for (const model of [null, ...models.map(m => m.name)])
      benches.push({ name: `${model ?? 'original'}@${s}`, model, series: s, cells: run.cells, ncu: run.ncu ?? false });
  }
  return {
    id,
    description: run.description,
    groupings: groupingList,
    methods: run.methods.map(m => named(m).name),
    models,
    series,
    pairs,
    benches,
    ...(run.gates ? { gates: run.gates } : {}),
  };
}

/** 実験群 1 つを解決済みの実験にする。doc は実験層の YAML を読んだ二つの文書。 */
export function groupPlans(name: string, doc: unknown): { description: string; plans: Plan[] } {
  const where = `config/${name}.yaml`;
  const [front, body] = parsed(GroupDocument, where, doc);
  const plans = Object.entries(body.runs).map(([id, run]) => build(`${where} の ${id}`, id, merged(body.setup, run)));
  return { description: front.description, plans };
}
