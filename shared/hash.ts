// spec の同一性の鍵。手元の resolve.ts（学習 run を評価の spec の中でハッシュで指す）と計算する機械の段取り（run の置き場）が同じ関数で組む。
// ハッシュを組むのはここだけ（Python は組まない。JS と Python で数値の JSON 表記が違うので、鍵を 2 か所で組まない）。
import { createHash } from 'node:crypto';
import type { Spec } from './spec.ts';

/** 鍵を整列した JSON。同じ中身なら同じ文字列になる。 */
export function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (typeof value === 'object' && value !== null) {
    const keys = Object.keys(value).sort();
    return `{${keys.map(k => `${JSON.stringify(k)}:${canonical((value as Record<string, unknown>)[k])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

/** 中身の 12 桁のハッシュ。 */
export function digest(value: unknown): string {
  return createHash('sha1').update(canonical(value)).digest('hex').slice(0, 12);
}

/** spec の同一性の鍵。 */
export function hashOf(spec: Spec): string {
  return digest(spec);
}
