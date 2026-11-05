// 主題のスキーマ（<主題>/schema.ts）が使う部品。主題は自分の experiments.json の形を zod で宣言し、ここの build で解決済みの実験（Plan）にする。
// 解決済みの実験は、学習 run と系列を値で持つ。段取り（orchestrate.ts）・API・画面はこれだけを読み、主題のスキーマを知らない。
import { z } from 'zod';
import { Ref, methodOf, seriesFrom, trainFrom } from '../evaluation/orchestrate/conf.ts';
import type { Series } from '../evaluation/orchestrate/spec.ts';
import type { Claim, Judge, Model, Plan } from './common.ts';

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

/** 宣言を形に当てる。誤りはどこの何かが分かる一文にする。 */
export function parsed<T>(schema: z.ZodType<T>, where: string, value: unknown): T {
  const result = schema.safeParse(value);
  if (!result.success) throw new Error(`${where}：${z.prettifyError(result.error)}`);
  return result.data;
}

/** 指し方の名前と、conf.ts に渡す指し方。 */
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
      const model = `${name}_${grouping}`;
      models.push({ name: model, method: name, grouping, train: trainFrom(model, declared, grouping) });
    }
  }
  const pairs = models.flatMap(m => Object.keys(series).map(s => ({ model: m.name, series: s })));
  return {
    id: e.id,
    description: e.description,
    groupings: e.groupings,
    methods: e.methods.map(m => named(m).name),
    models,
    series,
    pairs,
    claims,
  };
}
