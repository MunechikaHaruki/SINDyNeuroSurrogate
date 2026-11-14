// 比べ。run は生データ（波形）だけを持つので、波形から先の量は全部ここが頼まれるたびに純粋関数で計算する：
// 波形 1 本の特徴量（eFEL・0 mV の上向きの横切り・山の平均の形・発散）、原系と置き換えた列の点ごとの指標（rmse・mae、ISI の統計、周期の差、形の相関、横切りで数えた差、
// coincidence factor Γ）と、列ごとのまとめ（決まる点での Γ の平均、発火の始まり）。eFEL は Python（neurosurrogate/spikes.py）を子プロセスで起こし、波形を標準入力で渡す。
// 物差しの値（Γ の許容幅 Δ）は手元が頼み方で渡す。run は書き換わらないので、特徴量は run・区画ごとに、答えは頼み方ごとに覚えておく。
import type { ColumnSummary, CrossingDiff, CrossingRow, PairMetrics, SeriesMetrics, SeriesRequest } from '../../shared/api.ts';
import { compArray, RUN, type WaveMeta } from '../../shared/runs.ts';
import type { Spec } from '../../shared/spec.ts';
import { PROJECT, PYTHON } from '../run/places.ts';
import type { RunFiles } from './waves.ts';

/** eFEL の特徴量の名前 → 値の並び（波形 1 本）。null は eFEL の nan。 */
export type Spikes = Record<string, (number | null)[]>;
/** 0 mV を上向きにまたいだ時刻 [ms] と、そのスパイクの山 [mV]。 */
export type Crossings = { times: number[]; peaks: number[] };
/** 波形 1 本の特徴量。shape は山ごとに ±2 ms を切り出して平均した形（切り出せる山が無ければ null）。 */
export type TraceFeatures = { spikes: Spikes; crossings: Crossings; shape: number[] | null; diverged: boolean };
/** 原系と置き換えた列の、1 点の膜電位の差。値が決まらなければ（nan）null。 */
export type Distance = { rmse: number | null; mae: number | null };

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
/** スパイクの形を切り出す窓の半幅 [ms]。 */
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

/** 0 mV を上向きにまたいだ時刻 [ms] と、そのスパイクの山 [mV]。eFEL を通さないので、TS の縮約モデルにも同じ物差しを当てられる。 */
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

/** 山ごとに ±HALF_WINDOW ms を切り出して平均した形。切り出せる山が無ければ null。 */
export function shape(v: Float64Array, peaks: (number | null)[], dt: number): number[] | null {
  const half = Math.trunc(HALF_WINDOW / dt);
  const sum = new Float64Array(2 * half + 1);
  let count = 0;
  for (const peak of peaks) {
    if (peak === null || peak - half < 0 || peak + half + 1 > v.length) continue;
    for (let i = 0; i < sum.length; i++) sum[i] += v[peak - half + i];
    count++;
  }
  if (count === 0) return null;
  return Array.from(sum, x => x / count);
}

/** 原系と置き換えた列の膜電位の差。 */
export function distance(original: Float64Array, surrogate: Float64Array): Distance {
  let squared = 0;
  let absolute = 0;
  for (let i = 0; i < original.length; i++) {
    const error = original[i] - surrogate[i];
    squared += error * error;
    absolute += Math.abs(error);
  }
  return { rmse: finite(Math.sqrt(squared / original.length)), mae: finite(absolute / original.length) };
}

/** 波形の並びの eFEL の特徴量。Python を一度だけ起こし、波形を続けて標準入力へ流す。 */
export function efel(traces: Float64Array[], dt: number): Spikes[] {
  if (traces.length === 0) return [];
  const header = JSON.stringify(traces.map(v => ({ dt, n: v.length })));
  const input = new Uint8Array(traces.reduce((sum, v) => sum + v.byteLength, 0));
  let at = 0;
  for (const v of traces) {
    input.set(new Uint8Array(v.buffer, v.byteOffset, v.byteLength), at);
    at += v.byteLength;
  }
  const done = Bun.spawnSync([...PYTHON, '-m', 'neurosurrogate.spikes', header], {
    cwd: PROJECT,
    stdin: input,
    env: { ...process.env, JAX_PLATFORMS: 'cpu' },
  });
  if (done.exitCode !== 0) throw new Error(`eFEL が落ちた：${done.stderr.toString()}`);
  return JSON.parse(done.stdout.toString()) as Spikes[];
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

/** Pearson の相関（np.corrcoef）。 */
function correlation(a: number[], b: number[]): number {
  const ma = mean(a);
  const mb = mean(b);
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

/** 1 ペアの 1 comp の全指標を、両方の特徴量と波形の差から組む。 */
export function compare(o: TraceFeatures, s: TraceFeatures, d: Distance): PairMetrics {
  const oIsi = stat(o.spikes.ISI_values, mean);
  const sIsi = stat(s.spikes.ISI_values, mean);
  const spikes: PairMetrics['spikes'] = {};
  for (const name of AP_FEATURES) spikes[name] = [o.spikes[name] ?? [], s.spikes[name] ?? []];
  return {
    rows: {
      spike_count: [(o.spikes.peak_indices ?? []).length, (s.spikes.peak_indices ?? []).length],
      latency: [o.spikes.time_to_first_spike?.[0] ?? null, s.spikes.time_to_first_spike?.[0] ?? null],
      mean_isi: [oIsi, sIsi],
      std_isi: [stat(o.spikes.ISI_values, std), stat(s.spikes.ISI_values, std)],
    },
    scalars: {
      rmse: d.rmse,
      mae: d.mae,
      periodicity_gap: oIsi === null || sIsi === null ? null : Math.abs(oIsi - sIsi),
      spike_shape_corr: o.shape && s.shape ? finite(correlation(o.shape, s.shape)) : null,
    },
    spikes,
    diverged: s.diverged,
  };
}

/** 0 mV の横切りで数えた、元のモデルとの差。 */
export function crossingDiff(original: Crossings, model: Crossings): CrossingDiff {
  const n = Math.min(original.times.length, model.times.length);
  const peak = original.peaks.length > 0 && model.peaks.length > 0 ? mean(model.peaks) - mean(original.peaks) : null;
  return {
    count: [original.times.length, model.times.length],
    peak,
    lastShift: n > 0 ? Math.abs(model.times[n - 1] - original.times[n - 1]) : null,
  };
}

/** coincidence factor Γ（Kistler et al. 1997 の式(4.1)、https://doi.org/10.1162/neco.1997.9.5.1015）。±Δ 以内で一致した発火の数を偶然の一致で補正する。
 * 1 が完全一致、0 が偶然と同じ。全体を幅 2Δ のビン K 個に分け、偶然の一致を N_model N_original / K、正規化を 1 − N_original / K とする。
 * どちらにも発火が無ければ決まらないので null。duration は波形の長さ [ms]、delta は許容幅 Δ [ms]（物差しは頼む側が決める）。 */
export function coincidence(original: number[], model: number[], duration: number, delta: number): number | null {
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
  const bins = duration / (2 * delta);
  const expected = (model.length * original.length) / bins;
  const normalization = 1 - original.length / bins;
  return (matched - expected) / (0.5 * (original.length + model.length)) / normalization;
}

/** 決まる点（どちらかが発火した点）での Γ の平均。決まる点が無ければ null。 */
export function meanGamma(gammas: (number | null)[]): number | null {
  let sum = 0;
  let n = 0;
  for (const g of gammas) {
    if (g === null) continue;
    sum += g;
    n++;
  }
  return n > 0 ? sum / n : null;
}

/** 発火の始まり（rheobase）。掃引の値を小さい順に見て、初めて発火した値。どこでも発火しなければ null。 */
export function rheobase(values: number[], counts: number[]): number | null {
  const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const first = order.find(i => counts[i] > 0);
  return first === undefined ? null : values[first];
}

/** 系列の、列ごと・点ごとの原系との指標と、列ごとのまとめ。duration は点ごとの波形の長さ [ms]、values は掃引の値（掃引しなければ空）。 */
export function seriesMetrics(
  original: TraceFeatures[],
  columns: Record<string, TraceFeatures[]>,
  distances: Record<string, Distance[]>,
  duration: number[],
  values: number[],
  delta: number,
): SeriesMetrics {
  const result: SeriesMetrics = {
    pairs: {},
    crossings: {},
    summaries: {},
    times: { original: original.map(f => f.crossings.times), columns: {} },
  };
  for (const [run, features] of Object.entries(columns)) {
    result.pairs[run] = features.map((f, k) => compare(original[k], f, distances[run][k]));
    const rows: CrossingRow[] = features.map((f, k) => ({
      ...crossingDiff(original[k].crossings, f.crossings),
      gamma: coincidence(original[k].crossings.times, f.crossings.times, duration[k], delta),
    }));
    result.crossings[run] = rows;
    const summary: ColumnSummary = {
      gamma: meanGamma(rows.map(r => r.gamma)),
      rheobase:
        values.length === 0
          ? [null, null]
          : [
              rheobase(
                values,
                rows.map(r => r.count[0]),
              ),
              rheobase(
                values,
                rows.map(r => r.count[1]),
              ),
            ],
    };
    result.summaries[run] = summary;
    result.times.columns[run] = features.map(f => f.crossings.times);
  }
  return result;
}

/** 系列の比べを、読み方 files で組む。特徴量は run・区画ごとに、答えは頼み方ごとに覚えておく。 */
export function seriesComparison(files: RunFiles): (request: SeriesRequest) => SeriesMetrics {
  const known = new Map<string, TraceFeatures[]>();
  const answers = new Map<string, SeriesMetrics>();

  /** run の点ごとの、区画 comp の膜電位。 */
  function voltages(run: string, comp: string): { v: Float64Array[]; dt: number } {
    const meta = files.json<WaveMeta>(run, RUN.meta(RUN.waves));
    const compId = meta.names ? meta.names[comp] : 0;
    if (compId === undefined) throw new Error(`${run} に区画 ${comp} が無い（${Object.keys(meta.names!).join('、')}）`);
    return { v: meta.n.map((_, k) => files.column(compArray(run, RUN.waves, meta, k, compId, 'V'))), dt: meta.dt };
  }

  /** run ごと・点ごとの、区画 comp の膜電位の特徴量。まだ無い run だけ、刻みごとに eFEL を一度起こす。 */
  function features(runs: string[], comp: string): Record<string, TraceFeatures[]> {
    const missing = runs.filter(run => !known.has(`${run} ${comp}`));
    const read = new Map(missing.map(run => [run, voltages(run, comp)]));
    for (const dt of new Set([...read.values()].map(r => r.dt))) {
      const group = missing.filter(run => read.get(run)!.dt === dt);
      const found = efel(
        group.flatMap(run => read.get(run)!.v),
        dt,
      );
      let at = 0;
      for (const run of group) {
        const list = read.get(run)!.v.map((v, k) => ({
          spikes: found[at + k],
          crossings: crossings(v, dt),
          shape: shape(v, found[at + k].peak_indices ?? [], dt),
          diverged: diverged(v),
        }));
        at += list.length;
        known.set(`${run} ${comp}`, list);
      }
    }
    return Object.fromEntries(runs.map(run => [run, known.get(`${run} ${comp}`)!]));
  }

  return request => {
    const key = JSON.stringify(request);
    const hit = answers.get(key);
    if (hit) return hit;
    const { original, columns, comp, delta } = request;
    if (!(Number.isFinite(delta) && delta > 0)) throw new Error(`Γ の許容幅 delta ${delta} が正の数でない`);
    const found = features([original, ...columns], comp);
    const base = voltages(original, comp).v;
    const distances: Record<string, Distance[]> = {};
    const own: Record<string, TraceFeatures[]> = {};
    for (const run of columns) {
      distances[run] = voltages(run, comp).v.map((v, k) => distance(base[k], v));
      own[run] = found[run];
    }
    const meta = files.json<WaveMeta>(original, RUN.meta(RUN.waves));
    const spec = files.json<Spec>(original, 'spec.json');
    const values = spec.kind === 'sim' ? spec.input.values : [];
    const result = seriesMetrics(
      found[original],
      own,
      distances,
      meta.n.map(n => n * meta.dt),
      values,
      delta,
    );
    answers.set(key, result);
    return result;
  };
}
