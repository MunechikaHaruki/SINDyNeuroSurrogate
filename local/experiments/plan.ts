// 解決済みの実験の JSON の形。プリセットを通さず、学習の設定と系列を値のまま書く。
// 決め打ちの構成（組み合わせを展開しない実験）は、この形の config/<実験>.json を手で書く。送る spec.json は、どの実験の分も隣の resolve.ts が resolved/ にそのまま書き出す。
import { z } from 'zod';
import type { SurrogateConfig } from '../../shared/spec.ts';
import type { Plan, Bench, Series } from './common.ts';

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

/** 評価の系列（common.ts の Series）。 */
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
  benches: z.array(
    z.strictObject({
      name: z.string(),
      model: z.string().nullable(),
      series: z.string(),
      cells: z.array(z.number().int().positive().multipleOf(128)).nonempty(),
      ncu: z.boolean(),
    }),
  ),
});
export type PlanJson = z.infer<typeof PlanJson>;

/** 解決済みの実験の JSON を形に当て、名前の参照を確かめて実験にする。誤りはどこの何かが分かる一文にする。 */
export function fromJson(id: string, where: string, value: unknown): Plan {
  const result = PlanJson.safeParse(value);
  if (!result.success) throw new Error(`${where}：${z.prettifyError(result.error)}`);
  const p = result.data;
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
    benches: p.benches as Bench[],
  };
}
