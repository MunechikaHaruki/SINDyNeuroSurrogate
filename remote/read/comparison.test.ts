import { expect, test } from 'bun:test';
import { coincidence, compare, crossingDiff, crossings, distance, diverged, efel, meanGamma, rheobase, seriesMetrics, shape, type TraceFeatures } from './comparison.ts';

/** 山を 2 つ持つ波形（dt = 0.5 ms で 2 ms の窓は ±4 点）。 */
const v = new Float64Array([
  -65, -65, -65, -65, -65, 0, 30, 0, -65, -65, -65, -65, -65, -65, 0, 30, 0, -65, -65, -65, -65, -65, -65,
]);

test('crossings count upward passes through 0 mV with their peaks', () => {
  expect(crossings(v, 0.5)).toEqual({ times: [2.5, 7], peaks: [30, 30] });
  expect(crossings(new Float64Array([-65, -65, -65, -65, -65, -65, 0, 20, 0, -65]), 0.5)).toEqual({
    times: [3],
    peaks: [20],
  });
});

test('the shape averages ±2 ms around the peaks that fit inside the trace', () => {
  const found = shape(v, [6, 15], 0.5)!;
  expect(found).toHaveLength(9);
  expect(found[4]).toBe(30);
  expect(shape(v, [], 0.5)).toBeNull();
  expect(shape(v, [1], 0.5)).toBeNull();
});

test('the distance is the root mean square and the mean of the absolute differences', () => {
  const flat = new Float64Array(v.length).fill(-65);
  expect(distance(v, v)).toEqual({ rmse: 0, mae: 0 });
  // 差は 2 つの山の 65, 95, 65 だけ
  const d = distance(v, flat);
  expect(d.mae).toBeCloseTo((2 * (65 + 95 + 65)) / v.length, 12);
  expect(d.rmse).toBeCloseTo(Math.sqrt((2 * (65 * 65 + 95 * 95 + 65 * 65)) / v.length), 12);
});

test('a NaN or a runaway potential counts as diverged and gives no distance', () => {
  const broken = Float64Array.from(v);
  broken[3] = NaN;
  expect(diverged(broken)).toBe(true);
  expect(diverged(new Float64Array([0, 2000]))).toBe(true);
  expect(diverged(v)).toBe(false);
  expect(distance(v, broken).rmse).toBeNull();
});

test('eFEL runs once over several traces and finds no spike in a flat one', () => {
  const dt = 0.1;
  const spiking = new Float64Array(2000).map((_, i) => (i % 200 < 10 ? 30 : -65));
  const [some, none] = efel([spiking, new Float64Array(500).fill(-65)], dt);
  expect(some.peak_indices!.length).toBeGreaterThan(0);
  expect(none.peak_indices).toEqual([]);
});

test('the mean Γ skips undecided points', () => {
  expect(meanGamma([1, null, 0.5])).toBe(0.75);
  expect(meanGamma([null, null])).toBeNull();
});

test('rheobase is the smallest swept value that fires', () => {
  expect(rheobase([0, 2.5, 5], [0, 0, 4])).toBe(5);
  expect(rheobase([5, 0], [1, 0])).toBe(5);
  expect(rheobase([0, 2.5], [0, 0])).toBeNull();
});

/** 2 つの山を持つ波形 1 本の特徴量（山の時刻は 2.5 ms と 7 ms）。 */
const spiking: TraceFeatures = {
  spikes: { peak_indices: [6, 15], ISI_values: [4.5], time_to_first_spike: [3], peak_voltage: [30, 30] },
  crossings: { times: [2.5, 7], peaks: [30, 30] },
  shape: [-65, -65, -65, 0, 30, 0, -65, -65, -65],
  diverged: false,
};
const silent: TraceFeatures = {
  spikes: { peak_indices: [], ISI_values: [], time_to_first_spike: [] },
  crossings: { times: [], peaks: [] },
  shape: null,
  diverged: false,
};

test('a surrogate identical to the original has no error and the same spike shape', () => {
  const m = compare(spiking, spiking, { rmse: 0, mae: 0 });
  expect(m.rows.spike_count).toEqual([2, 2]);
  expect(m.rows.latency).toEqual([3, 3]);
  expect(m.rows.mean_isi).toEqual([4.5, 4.5]);
  expect(m.rows.std_isi).toEqual([0, 0]);
  expect(m.scalars).toEqual({ rmse: 0, mae: 0, periodicity_gap: 0, spike_shape_corr: 1 });
  expect(m.spikes.peak_voltage).toEqual([
    [30, 30],
    [30, 30],
  ]);
  expect(m.spikes.AHP_depth).toEqual([[], []]);
  expect(m.diverged).toBe(false);
});

test('a silent surrogate has no spike metrics', () => {
  const m = compare(spiking, silent, { rmse: 1, mae: 1 });
  expect(m.rows.spike_count).toEqual([2, 0]);
  expect(m.rows.latency).toEqual([3, null]);
  expect(m.rows.mean_isi).toEqual([4.5, null]);
  expect(m.scalars.periodicity_gap).toBeNull();
  expect(m.scalars.spike_shape_corr).toBeNull();
  // eFEL の nan（JSON の null）を含む ISI の統計は nan
  const nan = { ...spiking, spikes: { ...spiking.spikes, ISI_values: [null] } };
  expect(compare(spiking, nan, { rmse: 0, mae: 0 }).rows.mean_isi).toEqual([4.5, null]);
});

test('the crossing diff compares the last shared spike', () => {
  const found = spiking.crossings;
  expect(crossingDiff(found, { times: [3], peaks: [20] })).toEqual({ count: [2, 1], peak: -10, lastShift: 0.5 });
  expect(crossingDiff(found, { times: [], peaks: [] })).toEqual({ count: [2, 0], peak: null, lastShift: null });
});

test('the series metrics pair each column with the original point by point', () => {
  const m = seriesMetrics([spiking], { r: [spiking] }, { r: [{ rmse: 0, mae: 0 }] }, [11.5], [5], 2);
  expect(m.times).toEqual({ original: [[2.5, 7]], columns: { r: [[2.5, 7]] } });
  expect(m.crossings.r[0].count).toEqual([2, 2]);
  expect(m.crossings.r[0].gamma).toBeCloseTo(1);
  expect(m.pairs.r).toHaveLength(1);
  expect(m.summaries.r).toEqual({ gamma: m.crossings.r[0].gamma, rheobase: [5, 5] });
});

test('coincidence is 1 for the same spikes, near 0 for unrelated ones, and null with no spikes', () => {
  expect(coincidence([10, 50, 90], [10, 50, 90], 100, 2)).toBeCloseTo(1);
  expect(coincidence([10, 50, 90], [11.5, 51, 88.5], 100, 2)).toBeCloseTo(1);
  expect(coincidence([10, 50, 90], [30, 70], 100, 2)).toBeLessThan(0);
  expect(coincidence([], [], 100, 2)).toBeNull();
  expect(coincidence([10], [], 100, 2)).toBe(0);
});

test('coincidence follows Kistler et al. 1997 eq. (4.1) when the rates differ', () => {
  // 10 s で元 300 回、置き換え 450 回、一致 280 回。K = 2500、Γ = (280 − 450·300/2500) / 375 / (1 − 300/2500)
  const original: number[] = [];
  const model: number[] = [];
  for (let i = 0; i < 300; i++) original.push(10 + i * 33);
  for (let i = 0; i < 280; i++) model.push(10 + i * 33);
  for (let i = 0; i < 170; i++) model.push(26.5 + i * 33);
  model.sort((a, b) => a - b);
  const expected = (280 - (450 * 300) / 2500) / (0.5 * (300 + 450)) / (1 - 300 / 2500);
  expect(coincidence(original, model, 10000, 2)).toBeCloseTo(expected, 6);
});
