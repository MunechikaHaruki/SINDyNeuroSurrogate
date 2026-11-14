// 波形。run のファイルを読んで覚え（runs/ の外を指すパスは弾き、.f64 は丸ごと読んで古く使ったものから忘れる）、回線に出す形（表示の範囲で間引いた線・相平面の軌跡・分布、表示に要る桁）にする。
import { readFileSync } from 'node:fs';
import { join, normalize } from 'node:path';
import type { ArrayRef, Bin, Point } from '../../shared/api.ts';

// ---- run のファイルを読む

/** 覚えておく .f64 の合計の上限 [バイト]。超えたら古く使ったものから忘れる。 */
const BUDGET = 2 ** 31;

/** runs の置き場 1 つの読み方。 */
export type RunFiles = {
  /** run の中の .f64 の 1 列（変数ごとに n 個ずつ続いて並んだうちの column 番目）。 */
  column: (ref: ArrayRef) => Float64Array;
  /** run の中の .json。 */
  json: <T>(run: string, file: string) => T;
};

/** runs の置き場 runs を読む。読んだ .f64 はこの中で覚える。 */
export function runFiles(runs: string): RunFiles {
  const arrays = new Map<string, Float64Array>();
  let held = 0;

  /** run の下のファイルの絶対パス。runs/ の外を指すものは弾く。 */
  function inRuns(run: string, file: string): string {
    const relative = normalize(join(run, file));
    if (relative.startsWith('..') || relative.startsWith('/')) throw new Error(`${run}/${file} は runs の外`);
    return join(runs, relative);
  }

  /** run の中の .f64 を丸ごと。 */
  function f64(run: string, file: string): Float64Array {
    const key = `${run}/${file}`;
    const hit = arrays.get(key);
    if (hit) {
      arrays.delete(key);
      arrays.set(key, hit);
      return hit;
    }
    if (!file.endsWith('.f64')) throw new Error(`${key} は .f64 でない`);
    const bytes = readFileSync(inRuns(run, file));
    const values = new Float64Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    arrays.set(key, values);
    held += values.byteLength;
    for (const [old, array] of arrays) {
      if (held <= BUDGET || old === key) break;
      arrays.delete(old);
      held -= array.byteLength;
    }
    return values;
  }

  function column(ref: ArrayRef): Float64Array {
    const all = f64(ref.run, ref.file);
    const { column: at, n } = ref;
    if (
      !Number.isInteger(at) ||
      at < 0 ||
      !Number.isInteger(n) ||
      n <= 0 ||
      all.length % n !== 0 ||
      all.length < (at + 1) * n
    )
      throw new Error(`${ref.run}/${ref.file}（${all.length} 個）に ${at} 列目（${n} 点ずつ）が無い`);
    return all.subarray(ref.column * ref.n, (ref.column + 1) * ref.n);
  }

  function json<T>(run: string, file: string): T {
    if (!file.endsWith('.json')) throw new Error(`${run}/${file} は .json でない`);
    return JSON.parse(readFileSync(inRuns(run, file), 'utf8')) as T;
  }

  return { column, json };
}

// ---- 回線に出す形

/** 回線に出す線の点。表示に要る桁だけに丸める（時刻は 0.01 ms の刻みが 1 万 ms まで残る 7 桁、値は 6 桁）。 */
export function compact(points: Point[]): Point[] {
  return points.map(p => ({ x: Number(p.x.toPrecision(7)), y: Number(p.y.toPrecision(6)) }));
}

/** 時刻 [ms] の範囲 [start, end] に入る添字の範囲。 */
function span(length: number, t0: number, dt: number, start: number, end: number): [number, number] {
  return [Math.max(0, Math.ceil((start - t0) / dt)), Math.min(length - 1, Math.floor((end - t0) / dt))];
}

/**
 * 表示用に間引く。[start, end] ms を buckets 個に区切り、区間ごとに最小と最大を時刻順に残す。
 * スパイクの山と谷は潰れずに残り、点の数は 2 × buckets 以下になる。有限でない値は捨てる。
 */
export function decimate(
  values: Float64Array,
  t0: number,
  dt: number,
  start: number,
  end: number,
  buckets: number,
): Point[] {
  const points: Point[] = [];
  const [first, last] = span(values.length, t0, dt, start, end);
  const width = (last - first + 1) / buckets;
  for (let b = 0; b < buckets; b++) {
    const from = first + Math.floor(b * width);
    // 最後の区間は last まで（width の丸めで最後の点が落ちないように）
    const to = b === buckets - 1 ? last : Math.min(last, first + Math.floor((b + 1) * width) - 1);
    let low = -1;
    let high = -1;
    for (let i = from; i <= to; i++) {
      if (!Number.isFinite(values[i])) continue;
      if (low < 0 || values[i] < values[low]) low = i;
      if (high < 0 || values[i] > values[high]) high = i;
    }
    if (low < 0) continue;
    const earlier = Math.min(low, high);
    const later = Math.max(low, high);
    points.push({ x: t0 + earlier * dt, y: values[earlier] });
    if (later !== earlier) points.push({ x: t0 + later * dt, y: values[later] });
  }
  return points;
}

/** 相平面の軌跡。時刻順に結ぶので、[start, end] ms から等間隔に max 点まで取る。どちらかが有限でない点は捨てる。 */
export function path(
  xs: Float64Array,
  ys: Float64Array,
  t0: number,
  dt: number,
  start: number,
  end: number,
  max: number,
): Point[] {
  const points: Point[] = [];
  const [first, last] = span(Math.min(xs.length, ys.length), t0, dt, start, end);
  const stride = Math.max(1, Math.floor((last - first) / max));
  for (let i = first; i <= last; i += stride)
    if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) points.push({ x: xs[i], y: ys[i] });
  return points;
}

/** [low, high] を bins 個の等しい幅に分けて数える。範囲の外と有限でない値は数えない。 */
export function histogram(values: Float64Array, low: number, high: number, bins: number): Bin[] {
  const width = (high - low) / bins;
  const counts = new Array<number>(bins).fill(0);
  for (const value of values) {
    if (!Number.isFinite(value) || value < low || value > high) continue;
    const index = Math.min(bins - 1, Math.floor((value - low) / width));
    counts[index] += 1;
  }
  const list: Bin[] = [];
  for (let i = 0; i < bins; i++) list.push({ center: low + (i + 0.5) * width, count: counts[i] });
  return list;
}

/** 列ごとの分布。区間は全部の列の有限な値の最小から最大でそろえる。 */
export function histograms(values: Float64Array[], bins: number): Bin[][] {
  let low = Infinity;
  let high = -Infinity;
  for (const array of values)
    for (const v of array) {
      if (!Number.isFinite(v)) continue;
      low = Math.min(low, v);
      high = Math.max(high, v);
    }
  if (!(low < high)) return values.map(() => []);
  return values.map(array => histogram(array, low, high, bins));
}
