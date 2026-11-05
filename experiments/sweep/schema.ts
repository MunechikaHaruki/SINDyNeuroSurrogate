// 改良の探索の宣言の形。ゲートの並び gates から全部の群分けを組み、仕方ごとに回す。主張は判定を絞る群分けを持てる。
import { z } from 'zod';
import { groupings, type Judge, type Plan } from '../common.ts';
import { ClaimShape, MethodRef, SeriesRef, build, parsed } from '../parts.ts';
import declaration from './experiments.json' with { type: 'json' };

const Shape = z.record(
  z.string(),
  z.strictObject({
    description: z.string().min(1),
    gates: z.string(),
    methods: z.array(MethodRef).nonempty(),
    series: z.array(SeriesRef).optional(),
    claims: z.record(z.string(), ClaimShape),
  }),
);

export function plans(judges: Record<string, Judge>): Plan[] {
  return Object.entries(parsed(Shape, 'sweep/experiments.json', declaration)).map(([id, e]) =>
    build(`sweep/experiments.json の ${id}`, { id, ...e, groupings: groupings([...e.gates]) }, judges),
  );
}
