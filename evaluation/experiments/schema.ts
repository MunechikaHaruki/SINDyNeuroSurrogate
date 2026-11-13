// 実験の宣言（config/<主題>.yaml）の形と、解決済みの実験（common.ts の Plan）への組み立て。主題は改良の探索 sweep、性質の発見 property、拡張 mc、決め打ち fixed（config/<実験>.json、形は plan.ts）。
// 形は主題ごとに zod で決め、問いに合わせて自由に変えてよい。返すのは学習 run と系列を値で持つ解決済みの実験だけで、段取り（orchestrate.ts）・API・画面は主題の形を知らない。
import { z } from 'zod';
import { Ref, methodOf, seriesFrom, trainFrom } from './presets.ts';
import type { Bench, Series } from '../orchestrate/spec.ts';
import { groupings, type Claim, type Judge, type Model, type Plan } from './common.ts';
import mc from '../../config/mc.yaml';
import property from '../../config/property.yaml';
import sweep from '../../config/sweep.yaml';
import neuro2026 from '../../config/neuro2026.json' with { type: 'json' };
import { fromJson } from './plan.ts';

/** 仕方の指し方。プリセットの名前か、名前を付けた差分 `{ name, from, ...差分 }`（name は表示と判定の列の名前）。 */
export const MethodRef = z.union([z.string(), z.looseObject({ name: z.string(), from: z.string() })]);
/** 系列の指し方。仕方と同じ。 */
export const SeriesRef = MethodRef;
/** 主張。読む系列 on、判定を絞る群分け、回す前に書いた一文。 */
export const ClaimShape = z.strictObject({
  on: z.array(z.string()).nonempty(),
  groupings: z.array(z.string()).optional(),
  claim: z.string(),
});

/** GPU の実測。model は実験の学習 run の名前か original（元のモデル）、series は実験が回す系列。名前は `<model>@<series>`。 */
export const BenchShape = z.strictObject({
  models: z.array(z.string()).nonempty(),
  series: z.string(),
  /** 細胞の数。1 刻みのカーネルのブロック（128 細胞）の倍数。 */
  cells: z.array(z.number().int().positive().multipleOf(128)).nonempty(),
  /** Nsight Compute で 1 刻みのカーネルを測るか。 */
  ncu: z.boolean().optional(),
});

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

/** 解決済みの実験を組む。回す組は 仕方 × 群分け × 系列 の直積で、系列は series と主張の on を合わせたもの。 */
export function build(
  where: string,
  e: {
    id: string;
    description: string;
    groupings: string[];
    methods: z.infer<typeof MethodRef>[];
    series?: z.infer<typeof SeriesRef>[];
    claims?: Record<string, z.infer<typeof ClaimShape>>;
    benches?: z.infer<typeof BenchShape>[];
  },
  judges: Record<string, Judge>,
): Plan {
  const series: Record<string, Series> = {};
  for (const s of e.series ?? []) {
    const { name, ref } = named(s);
    series[name] = seriesFrom(ref);
  }
  const claims: Claim[] = [];
  for (const [name, c] of Object.entries(e.claims ?? {})) {
    const judge = judges[name];
    if (!judge) throw new Error(`${where}：主張 ${name} の判定が checks.ts に無い`);
    for (const s of c.on) if (!series[s]) series[s] = seriesFrom(s);
    claims.push({ name, claim: c.claim, series: c.on, ...(c.groupings ? { groupings: c.groupings } : {}), judge });
  }
  if (Object.keys(series).length === 0) throw new Error(`${where}：回す系列が無い（series か claims の on を書く）`);
  const models: Model[] = [];
  for (const m of e.methods) {
    const { name, ref } = named(m);
    const declared = methodOf(ref);
    for (const grouping of e.groupings) {
      const model = `${name}.${grouping}`;
      models.push({ name: model, method: name, grouping, train: trainFrom(model, declared, grouping) });
    }
  }
  const pairs = models.flatMap(m => Object.keys(series).map(s => ({ model: m.name, series: s })));
  const benches: Bench[] = [];
  for (const b of e.benches ?? []) {
    if (!series[b.series]) throw new Error(`${where}：GPU の実測の系列 ${b.series} を実験が回さない`);
    for (const model of b.models) {
      if (model !== 'original' && !models.some(m => m.name === model))
        throw new Error(`${where}：GPU の実測の ${model} は実験の学習 run でも original でもない`);
      const name = `${model}@${b.series}`;
      if (benches.some(x => x.name === name)) throw new Error(`${where}：GPU の実測 ${name} が重なる`);
      benches.push({
        name,
        model: model === 'original' ? null : model,
        series: b.series,
        cells: b.cells,
        ncu: b.ncu ?? false,
      });
    }
  }
  return {
    id: e.id,
    description: e.description,
    groupings: e.groupings,
    methods: e.methods.map(m => named(m).name),
    models,
    series,
    pairs,
    benches,
    claims,
  };
}

/** 改良の探索。ゲートの並び gates から全部の群分けを組み、仕方ごとに回す。主張は判定を絞る群分けを持てる。 */
const Sweep = z.record(
  z.string(),
  z.strictObject({
    description: z.string().min(1),
    gates: z.array(z.string().length(1)).nonempty(),
    methods: z.array(MethodRef).nonempty(),
    series: z.array(SeriesRef).optional(),
    claims: z.record(z.string(), ClaimShape),
  }),
);

/** 性質の発見。群分けを名前で書き、主張が読む系列を回す。 */
const Property = z.record(
  z.string(),
  z.strictObject({
    description: z.string().min(1),
    groupings: z.array(z.string()).nonempty(),
    methods: z.array(MethodRef).nonempty(),
    series: z.array(SeriesRef).optional(),
    claims: z.record(z.string(), ClaimShape),
  }),
);

/** 拡張。判定を持たず、群分けと仕方と回す系列と、あれば GPU の実測を書く。 */
const Mc = z.record(
  z.string(),
  z.strictObject({
    description: z.string().min(1),
    groupings: z.array(z.string()).nonempty(),
    methods: z.array(MethodRef).nonempty(),
    series: z.array(SeriesRef).nonempty(),
    benches: z.array(BenchShape).optional(),
  }),
);

/** 主題。読む順（改良の探索、性質の発見、拡張、決め打ち）。plans は宣言 `<name>.yaml` を形に当て、判定を引いて解決済みの実験にする。決め打ち fixed は、組み合わせを展開しない実験を解決済みの形で手で書いた `config/<実験>.json`（形は plan.ts）。 */
export const THEMES: { name: string; plans: (judges: Record<string, Judge>) => Plan[] }[] = [
  {
    name: 'sweep',
    plans: judges =>
      Object.entries(parsed(Sweep, 'sweep.yaml', sweep)).map(([id, e]) =>
        build(`sweep.yaml の ${id}`, { id, ...e, groupings: groupings(e.gates) }, judges),
      ),
  },
  {
    name: 'property',
    plans: judges =>
      Object.entries(parsed(Property, 'property.yaml', property)).map(([id, e]) =>
        build(`property.yaml の ${id}`, { id, ...e }, judges),
      ),
  },
  {
    name: 'mc',
    plans: judges =>
      Object.entries(parsed(Mc, 'mc.yaml', mc)).map(([id, e]) => build(`mc.yaml の ${id}`, { id, ...e }, judges)),
  },
  {
    name: 'fixed',
    plans: judges => [fromJson('neuro2026-rerun', 'neuro2026.json', neuro2026, judges)],
  },
];
