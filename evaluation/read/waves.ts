// run の波形（neurosurrogate/run.py の waves/ の形式）の目録と、読んだバイト列の切り分け方、表示用の間引き。
// ファイルは読まない。読むのは計算する機械の API（evaluation/serve/）で、回線には間引いたものだけを出す。

/** 1 列の波形の目録（waves/meta.json）。features は [comp_id, 変数名, ゲートか] の並び。 */
export type WaveMeta = {
  t0: number;
  dt: number;
  n: number[];
  features: [number, string, boolean][];
  nodes: number[] | null;
  names: Record<string, number> | null;
};

export type Point = { x: number; y: number };

/** comp `compId` の変数名（ファイルの中の並び順）。 */
export function variablesOf(meta: WaveMeta, compId: number): string[] {
  const names: string[] = [];
  for (const feature of meta.features) if (feature[0] === compId) names.push(feature[1]);
  return names;
}

/** comp `compId` のゲートの変数名。 */
export function gatesOf(meta: WaveMeta, compId: number): string[] {
  const names: string[] = [];
  for (const feature of meta.features) if (feature[0] === compId && feature[2]) names.push(feature[1]);
  return names;
}

/** 変数ごとに n 個ずつ続いて並んだ float64 を変数ごとの列に分ける（comp<id>.f64、I_internal.f64、fit の x・latent）。 */
export function decodeComp(all: Float64Array, variables: string[], n: number): Record<string, Float64Array> {
  if (all.length !== variables.length * n)
    throw new Error(`長さ ${all.length} が ${variables.length} 変数 × ${n} 点と合わない`);
  const table: Record<string, Float64Array> = {};
  for (let i = 0; i < variables.length; i++) table[variables[i]] = all.subarray(i * n, (i + 1) * n);
  return table;
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
