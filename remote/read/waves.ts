// 波形。run のファイルを読んで覚え（runs/ の外を指すパスは弾き、.f64 は丸ごと読んで古く使ったものから忘れる）、回線に出す形（表示の範囲で間引いた線・相平面の軌跡、表示に要る桁）にする。
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, normalize } from 'node:path';
import type { Line, Point, SlopeCompare } from '../../shared/api.ts';
import type { ArrayRef } from '../../shared/runs.ts';

// ---- run のファイルを読む

/** 覚えておく .f64 の合計の上限 [バイト]。超えたら古く使ったものから忘れる。 */
const BUDGET = 2 ** 31;

/** runs の置き場 1 つの読み方。 */
export type RunFiles = {
  /** run の中の .f64 の 1 列（変数ごとに n 個ずつ続いて並んだうちの column 番目）。 */
  column: (ref: ArrayRef) => Float64Array;
  /** run の中の .json。 */
  json: <T>(run: string, file: string) => T;
  /** run の中の .json か .csv を、ファイルのままの文字で。無ければ null。 */
  text: (run: string, file: string) => string | null;
  /** run の中の dir の下のディレクトリの名前。 */
  dirs: (run: string, dir?: string) => string[];
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

  function text(run: string, file: string): string | null {
    if (!file.endsWith('.json') && !file.endsWith('.csv')) throw new Error(`${run}/${file} は .json か .csv でない`);
    const path = inRuns(run, file);
    return existsSync(path) ? readFileSync(path, 'utf8') : null;
  }

  function dirs(run: string, dir = '.'): string[] {
    const path = inRuns(run, dir);
    if (!existsSync(path)) return [];
    const names: string[] = [];
    for (const entry of readdirSync(path, { withFileTypes: true })) if (entry.isDirectory()) names.push(entry.name);
    return names;
  }

  return { column, json, text, dirs };
}

// ---- 回線に出す形

/** 時間の線を間引くときの区間の数。1 本あたり最大でこの 2 倍の点になる。 */
export const BUCKETS = 400;
/** 相平面と潜在の軌跡の 1 本あたりの点の上限。軌跡の形が見える数に留め、回線に出す量を抑える。 */
export const PATH_POINTS = 2000;

/** 時刻の並び（t0 と dt）と、時間の範囲 [start, end] ms。 */
export type Span = { t0: number; dt: number; start: number; end: number };

/** .f64 の 1 列を、範囲 span で BUCKETS 区間に間引いた線。 */
export function line(files: RunFiles, ref: ArrayRef, span: Span): Point[] {
  return arrayLine(files.column(ref), span);
}

/** 計算した 1 列を、範囲 span で BUCKETS 区間に間引いた線。 */
export function arrayLine(values: Float64Array, span: Span): Point[] {
  return compact(decimate(values, span.t0, span.dt, span.start, span.end, BUCKETS));
}

/** 再構成の図の既定の幅 [ms] と、最初の発火の前に空ける時間 [ms]。 */
const ZOOM = { width: 50, before: 5 };

/** 再構成の図の範囲。頼まれた zoom があればそれを span に収め、無ければ span の中で膜電位 v が最初に 0 mV を上向きに横切る時刻の ZOOM.before 前から ZOOM.width（発火が無ければ span の頭から）。 */
export function zoomOf(v: Float64Array | null, span: Span, zoom: [number, number] | undefined): [number, number] {
  const clip = (start: number, width: number): [number, number] => {
    const from = Math.max(span.start, Math.min(start, span.end - width));
    return [from, Math.min(span.end, from + width)];
  };
  if (zoom) return clip(zoom[0], zoom[1] - zoom[0]);
  if (v) {
    const first = Math.max(1, Math.ceil((span.start - span.t0) / span.dt));
    const last = Math.min(v.length - 1, Math.floor((span.end - span.t0) / span.dt));
    for (let i = first; i <= last; i++)
      if (v[i - 1] < 0 && v[i] >= 0) return clip(span.t0 + i * span.dt - ZOOM.before, ZOOM.width);
  }
  return clip(span.start, ZOOM.width);
}

/** .f64 の 2 列を、範囲 span で PATH_POINTS 点までの軌跡にしたもの。 */
export function trajectory(files: RunFiles, x: ArrayRef, y: ArrayRef, span: Span): Point[] {
  return compact(path(files.column(x), files.column(y), span.t0, span.dt, span.start, span.end, PATH_POINTS));
}

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

/** 1 列の有限な値がどれも 0〜1 に収まるか（ゲートの開いている割合か）。目録の gate の印は Traub の Ca 濃度 XI のような状態にも付くので、ゲートの図に載せるかは値で分ける。 */
export function isFraction(values: Float64Array): boolean {
  const slack = 1e-6;
  for (const v of values) if (Number.isFinite(v) && (v < -slack || v > 1 + slack)) return false;
  return true;
}

/** 2 列の、どちらも有限な時刻での差の RMSE と、1 列目（正解）の標準偏差。有限な時刻が無ければどちらも NaN。 */
export function gapOf(truth: Float64Array, guess: Float64Array): { rmse: number; scale: number } {
  let count = 0;
  let sum = 0;
  let squares = 0;
  let gap = 0;
  for (let i = 0; i < truth.length; i++) {
    const t = truth[i];
    const g = guess[i];
    if (!Number.isFinite(t) || !Number.isFinite(g)) continue;
    count += 1;
    sum += t;
    squares += t * t;
    gap += (g - t) ** 2;
  }
  if (count === 0) return { rmse: NaN, scale: NaN };
  const mean = sum / count;
  return { rmse: Math.sqrt(gap / count), scale: Math.sqrt(Math.max(0, squares / count - mean * mean)) };
}

/** 潜在ごとの傾きの比べ。slope は正解（前進差分）、rhs は学習器の右辺の列。線と散布は範囲 span、ずれは全区間。 */
export function slopeCompare(
  files: RunFiles,
  latents: { label: string; slope: ArrayRef; rhs: ArrayRef }[],
  span: Span,
): SlopeCompare {
  const lines: Record<string, Line[]> = {};
  for (const l of latents)
    lines[l.label] = [
      { label: '正解', points: line(files, l.slope, span) },
      { label: '右辺', points: line(files, l.rhs, span) },
    ];
  return {
    lines,
    match: latents.map(l => ({ label: l.label, points: trajectory(files, l.slope, l.rhs, span) })),
    error: latents.map(l => ({ latent: l.label, ...gapOf(files.column(l.slope), files.column(l.rhs)) })),
  };
}

/** 列 columns（ゲートごと）の、添字 [from, to] を上限 limit 点まで等間隔に間引いたゲートの空間の点。有限でない点は捨てる。 */
export function strided(columns: Float64Array[], from: number, to: number, limit: number): number[][] {
  const step = Math.max(1, Math.ceil((to - from + 1) / limit));
  const points: number[][] = [];
  for (let i = from; i <= to; i += step) {
    const point = columns.map(c => Number(c[i].toPrecision(6)));
    if (point.every(Number.isFinite)) points.push(point);
  }
  return points;
}
