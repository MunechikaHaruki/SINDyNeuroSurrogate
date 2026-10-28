// 元のモデルと置き換えたモデルの 1 区画・1 点の指標。波形（waves/）とスパイクの特徴量（spikes.json）から計算する。
// スパイクの特徴量は eFEL が持つので Python（neurosurrogate/spikes.py）が書き、比べる計算はここが持つ。

/** 特徴量の名前 → 値の並び（spikes.json の 1 区画・1 点）。null は eFEL の nan。 */
export type Spikes = Record<string, (number | null)[]>;
/** comp 名 → 点ごとの特徴量（spikes.json）。 */
export type SpikesFile = Record<string, Spikes[]>;
/** 1 ペア（原系, 置換系）の 1 comp の指標。null は値が無い（nan）。 */
export type PairMetrics = {
  rows: Record<string, [number | null, number | null]>;
  scalars: Record<string, number | null>;
  spikes: Record<string, [(number | null)[], (number | null)[]]>;
  diverged: boolean;
};

/** AP ごとの特徴量（neurosurrogate/spikes.py の AP_FEATURES）。比べる AP は読む側が選ぶ。 */
export const AP_FEATURES = [
  'peak_voltage',
  'AP_amplitude',
  'AP_begin_voltage',
  'AP_rise_rate',
  'AP_fall_rate',
  'AP_duration_half_width',
  'AP_rise_time',
  'AP_fall_time',
  'AHP_depth',
  'AHP_time_from_peak',
];
/** |V| [mV] の発散の閾値（生理的な範囲は ±200 程度）。 */
const DIVERGE_V = 1e3;
/** スパイクの形を比べる窓の半幅 [ms]。 */
const HALF_WINDOW = 2.0;

/** 有限でない値は null にする（JSON の nan と同じ扱い）。 */
function finite(value: number): number | null {
  return Number.isFinite(value) ? value : null;
}

/** 電位が NaN・無限大を含むか、生理的にあり得ない大きさへ振り切れたか。 */
export function diverged(v: Float64Array): boolean {
  for (const x of v) if (!Number.isFinite(x) || Math.abs(x) > DIVERGE_V) return true;
  return false;
}

function mean(values: number[]): number {
  let sum = 0;
  for (const x of values) sum += x;
  return sum / values.length;
}

/** 母標準偏差（numpy の np.std）。 */
function std(values: number[]): number {
  const m = mean(values);
  let sum = 0;
  for (const x of values) sum += (x - m) * (x - m);
  return Math.sqrt(sum / values.length);
}

/** null を除いた値。1 つでも null（nan）があれば numpy と同じく結果は nan なので、そのときは undefined。 */
function present(values: (number | null)[] | undefined): number[] | undefined {
  if (!values || values.length === 0) return undefined;
  const list: number[] = [];
  for (const x of values) {
    if (x === null) return undefined;
    list.push(x);
  }
  return list;
}

/** 値の並びの統計。空か nan を含めば null。 */
function stat(values: (number | null)[] | undefined, fn: (values: number[]) => number): number | null {
  const list = present(values);
  return list ? finite(fn(list)) : null;
}

/** 山ごとに ±HALF_WINDOW ms を切り出して平均した形。切り出せる山が無ければ undefined。 */
function template(v: Float64Array, peaks: (number | null)[], dt: number): Float64Array | undefined {
  const half = Math.trunc(HALF_WINDOW / dt);
  const sum = new Float64Array(2 * half + 1);
  let count = 0;
  for (const peak of peaks) {
    if (peak === null || peak - half < 0 || peak + half + 1 > v.length) continue;
    for (let i = 0; i < sum.length; i++) sum[i] += v[peak - half + i];
    count++;
  }
  if (count === 0) return undefined;
  for (let i = 0; i < sum.length; i++) sum[i] /= count;
  return sum;
}

/** Pearson の相関（np.corrcoef）。 */
function correlation(a: Float64Array, b: Float64Array): number {
  const ma = mean(Array.from(a));
  const mb = mean(Array.from(b));
  let ab = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    ab += (a[i] - ma) * (b[i] - mb);
    aa += (a[i] - ma) * (a[i] - ma);
    bb += (b[i] - mb) * (b[i] - mb);
  }
  return Math.max(-1, Math.min(1, ab / Math.sqrt(aa * bb)));
}

/** 1 ペアの 1 comp の全指標。rows は両方にある指標の [原系, 置換系]、scalars は比べて決まる指標、spikes は AP ごとの特徴量。 */
export function compare(
  original: Float64Array,
  surrogate: Float64Array,
  o: Spikes,
  s: Spikes,
  dt: number,
): PairMetrics {
  let squared = 0;
  let absolute = 0;
  for (let i = 0; i < original.length; i++) {
    const error = original[i] - surrogate[i];
    squared += error * error;
    absolute += Math.abs(error);
  }
  const oIsi = stat(o.ISI_values, mean);
  const sIsi = stat(s.ISI_values, mean);
  const oShape = template(original, o.peak_indices ?? [], dt);
  const sShape = template(surrogate, s.peak_indices ?? [], dt);
  const spikes: PairMetrics['spikes'] = {};
  for (const name of AP_FEATURES) spikes[name] = [o[name] ?? [], s[name] ?? []];
  return {
    rows: {
      spike_count: [(o.peak_indices ?? []).length, (s.peak_indices ?? []).length],
      latency: [o.time_to_first_spike?.[0] ?? null, s.time_to_first_spike?.[0] ?? null],
      mean_isi: [oIsi, sIsi],
      std_isi: [stat(o.ISI_values, std), stat(s.ISI_values, std)],
    },
    scalars: {
      rmse: finite(Math.sqrt(squared / original.length)),
      mae: finite(absolute / original.length),
      periodicity_gap: oIsi === null || sIsi === null ? null : Math.abs(oIsi - sIsi),
      spike_shape_corr: oShape && sShape ? finite(correlation(oShape, sShape)) : null,
    },
    spikes,
    diverged: diverged(surrogate),
  };
}

/** 0 mV を上向きにまたいだ時刻 [ms] と、そのスパイクの山 [mV]。eFEL を通さないので、TS の縮約モデルにも同じ物差しを当てられる。 */
export type Crossings = { times: number[]; peaks: number[] };

export function crossings(v: ArrayLike<number>, dt: number): Crossings {
  const times: number[] = [];
  const peaks: number[] = [];
  for (let i = 1; i < v.length; i++) {
    if (!(v[i - 1] < 0 && v[i] >= 0)) continue;
    times.push(i * dt);
    let top = v[i];
    for (let j = i; j < v.length && v[j] >= 0; j++) top = Math.max(top, v[j]);
    peaks.push(top);
  }
  return { times, peaks };
}

/** 元のモデルとの差。peak は山の平均の差（モデル − 元）、lastShift は両方にある最後の山の時刻の差の大きさ。 */
export type CrossingDiff = { count: [number, number]; peak: number | null; lastShift: number | null };

export function crossingDiff(original: Crossings, model: Crossings): CrossingDiff {
  const n = Math.min(original.times.length, model.times.length);
  const peak = original.peaks.length > 0 && model.peaks.length > 0 ? mean(model.peaks) - mean(original.peaks) : null;
  return {
    count: [original.times.length, model.times.length],
    peak,
    lastShift: n > 0 ? Math.abs(model.times[n - 1] - original.times[n - 1]) : null,
  };
}

/** Γ の許容幅 Δ [ms]（2026-10-27 に決めた物差し）。 */
export const COINCIDENCE_WINDOW = 2;

/** coincidence factor Γ（Kistler et al. 1997）。±Δ 以内で一致した発火の数を偶然の一致で補正する。1 が完全一致、0 が偶然と同じ。
 * どちらにも発火が無ければ決まらないので null。ν は置き換えたモデルの発火率、duration は波形の長さ [ms]。 */
export function coincidence(original: number[], model: number[], duration: number, delta = COINCIDENCE_WINDOW): number | null {
  if (original.length + model.length === 0) return null;
  let matched = 0;
  let j = 0;
  for (const t of original) {
    while (j < model.length && model[j] < t - delta) j++;
    if (j < model.length && model[j] <= t + delta) {
      matched++;
      j++;
    }
  }
  const nu = model.length / duration;
  const expected = 2 * nu * delta * original.length;
  return (matched - expected) / (0.5 * (original.length + model.length)) / (1 - 2 * nu * delta);
}

/** 発火率 [Hz]。duration は数える区間の長さ [ms]。f–I 曲線の 1 点。 */
export function rate(times: number[], duration: number): number {
  return (times.length / duration) * 1000;
}

/** 発火の始まり（rheobase）。掃引の値を小さい順に見て、初めて発火した値。どこでも発火しなければ null。 */
export function rheobase(values: number[], counts: number[]): number | null {
  const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const first = order.find((i) => counts[i] > 0);
  return first === undefined ? null : values[first];
}
