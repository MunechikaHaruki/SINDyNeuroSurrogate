// 拡張の宣言の形。判定を持たず、群分けと仕方と回す系列だけを書く。
import { z } from 'zod';
import type { Plan } from '../common.ts';
import { MethodRef, SeriesRef, build, parsed } from '../parts.ts';
import declaration from './experiments.json' with { type: 'json' };

const Shape = z.record(
  z.string(),
  z.strictObject({
    description: z.string().min(1),
    groupings: z.array(z.string()).nonempty(),
    methods: z.array(MethodRef).nonempty(),
    series: z.array(SeriesRef).nonempty(),
  }),
);

export function plans(): Plan[] {
  return Object.entries(parsed(Shape, 'mc/experiments.json', declaration)).map(([id, e]) =>
    build(`mc/experiments.json の ${id}`, { id, ...e }, {}),
  );
}
