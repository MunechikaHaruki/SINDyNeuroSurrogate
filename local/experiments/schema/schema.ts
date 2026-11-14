// 宣言の二つの書き方と、どちらからも組む解決済みの実験（Plan）。
// 実験群の実験層（config/<群>.yaml）は frontmatter（description）の文書と、共通の設定 setup と回す run の並び runs の文書の二つ。run に書いた欄だけが setup を置き換え、プリセット（presets.ts）を名前か差分で指す。
// 決め打ちの構成（組み合わせを展開しない実験）は、プリセットを通さず、学習の設定と系列を値のまま config/<実験>.json に書く（PlanJson）。
// 返すのは学習 run と系列を値で持つ解決済みの実験だけで、resolve.ts と画面は宣言の形を知らない。
import { z } from 'zod';
import type { SurrogateConfig } from '../../../shared/spec.ts';
import { Ref, groupings, methodOf, parsed, seriesFrom, trainFrom, type Series, type Train } from './presets.ts';

/** 細胞の数。1 刻みのカーネルのブロック（128 細胞）の倍数。 */
const Cells = z.array(z.number().int().positive().multipleOf(128)).nonempty();

/** GPU の実測 1 つ。model は学習 run の名前（null なら元のモデル）、cells は並べる細胞の数、ncu は Nsight Compute で 1 刻みのカーネルを測るか。 */
const BenchValue = z.strictObject({
  name: z.string(),
  model: z.string().nullable(),
  series: z.string(),
  cells: Cells,
  ncu: z.boolean(),
});
export type Bench = z.infer<typeof BenchValue>;

/** 学習 run 1 本。name は実験の中の名前（`<仕方の名前>.<群分け>`）、method と grouping は図が列を引く名前。 */
export type Model = { name: string; method: string; grouping: string; train: Train };

/** 解決済みの実験。description は何を回して何を見るかの一文（Markdown）。学習 run と系列を値で持ち、pairs が回す組（学習 run の名前 × 系列の名前）。
 * groupings と methods は画面が絞り込みに使う名前の並び。 */
export type Plan = {
  id: string;
  description: string;
  groupings: string[];
  methods: string[];
  models: Model[];
  series: Record<string, Series>;
  pairs: { model: string; series: string }[];
  /** GPU の実測。無ければ空。 */
  benches: Bench[];
  /** 群分けを全部回すときの、群分けを組むゲート（画面が群分けの掃引の節を出す）。 */
  gates?: string[];
};

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
  cells: Cells.optional(),
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

// ---- 決め打ちの実験（config/<実験>.json）

const SimSpec = z.strictObject({
  target: z.string(),
  current_type: z.string(),
  dt: z.number(),
  current_params: z.record(z.string(), z.number()),
  stim: z.string().optional(),
});

const Group = z.strictObject({
  gates: z.array(z.string()).nonempty(),
  compressor: z.enum(['pca', 'ae', 'kam']),
  d: z.number().int().positive(),
  rhs: z.enum(['sindy', 'ude', 'kam']),
  compressor_params: z.record(z.string(), z.unknown()),
  rhs_params: z.looseObject({
    library_specs: z.array(z.strictObject({ type: z.string(), gates: z.string().optional() })).optional(),
  }),
});

/** 学習の設定（../../shared/spec.ts の SurrogateConfig）。 */
export const Config = z.strictObject({
  cell: z.string(),
  form: z.enum(['hybrid', 'sindy']),
  data: SimSpec,
  groups: z.array(Group).nonempty(),
  train_comp: z.string().nullable(),
  absorbed: z.array(z.string()).optional(),
});

/** 評価の系列（presets.ts の Series）。 */
export const SeriesValue = z.strictObject({
  spec: SimSpec.required({ stim: true }),
  replace_targets: z.array(z.string()).nonempty(),
  param: z.string().nullable(),
  values: z.array(z.number()),
});

/** 解決済みの実験。学習 run は名前・仕方と群分けの名前（図と判定が列を引く）・設定、系列は名前 → 値。回す組 pairs は学習 run × 系列の名前。 */
export const PlanJson = z.strictObject({
  description: z.string().min(1),
  trains: z
    .array(z.strictObject({ name: z.string(), method: z.string(), grouping: z.string(), config: Config }))
    .nonempty(),
  series: z.record(z.string(), SeriesValue),
  pairs: z.array(z.strictObject({ model: z.string(), series: z.string() })),
  benches: z.array(BenchValue),
});
export type PlanJson = z.infer<typeof PlanJson>;

/** 決め打ちの実験の JSON を形に当て、名前の参照を確かめて実験にする。誤りはどこの何かが分かる一文にする。 */
export function fromJson(id: string, where: string, value: unknown): Plan {
  const p = parsed(PlanJson, where, value);
  const trains = new Set(p.trains.map(t => t.name));
  for (const pair of p.pairs) {
    if (!trains.has(pair.model)) throw new Error(`${where}：組の学習 run ${pair.model} が trains に無い`);
    if (!p.series[pair.series]) throw new Error(`${where}：組の系列 ${pair.series} が series に無い`);
  }
  for (const b of p.benches) {
    if (b.model !== null && !trains.has(b.model)) throw new Error(`${where}：GPU の実測 ${b.name} の学習 run が無い`);
    if (!p.series[b.series]) throw new Error(`${where}：GPU の実測 ${b.name} の系列が無い`);
  }
  return {
    id,
    description: p.description,
    groupings: [...new Set(p.trains.map(t => t.grouping))],
    methods: [...new Set(p.trains.map(t => t.method))],
    models: p.trains.map(t => ({
      name: t.name,
      method: t.method,
      grouping: t.grouping,
      train: { name: t.name, config: t.config as SurrogateConfig },
    })),
    series: p.series as Record<string, Series>,
    pairs: p.pairs,
    benches: p.benches,
  };
}
