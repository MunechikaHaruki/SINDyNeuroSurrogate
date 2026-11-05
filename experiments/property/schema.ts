// 性質の発見の宣言の形。群分けを名前で書き、主張が読む系列を回す。
import { z } from 'zod';
import type { Judge, Plan } from '../common.ts';
import { ClaimShape, MethodRef, SeriesRef, build, parsed } from '../parts.ts';
import declaration from './experiments.json' with { type: 'json' };

const Shape = z.record(
  z.string(),
  z.strictObject({
    description: z.string().min(1),
    groupings: z.array(z.string()).nonempty(),
    methods: z.array(MethodRef).nonempty(),
    series: z.array(SeriesRef).optional(),
    claims: z.record(z.string(), ClaimShape),
  }),
);

export function plans(judges: Record<string, Judge>): Plan[] {
  return Object.entries(parsed(Shape, 'property/experiments.json', declaration)).map(([id, e]) =>
    build(`property/experiments.json の ${id}`, { id, ...e }, judges),
  );
}
