import { expect, test } from 'bun:test';
import { coincidence, compare, crossingDiff, crossings, diverged, rate, rheobase, type Spikes } from './compare.ts';

/** 山を 2 つ持つ波形（dt = 0.5 ms で 2 ms の窓は ±4 点）。 */
const v = new Float64Array([
  -65, -65, -65, -65, -65, 0, 30, 0, -65, -65, -65, -65, -65, -65, 0, 30, 0, -65, -65, -65, -65, -65, -65,
]);
const spikes: Spikes = { peak_indices: [6, 15], ISI_values: [4.5], time_to_first_spike: [3], peak_voltage: [30, 30] };

test('a surrogate identical to the original has no error and the same spike shape', () => {
  const m = compare(v, v, spikes, spikes, 0.5);
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

test('a silent surrogate has no spike metrics, and its errors are the mean of the differences', () => {
  const flat = new Float64Array(v.length).fill(-65);
  const m = compare(v, flat, spikes, { peak_indices: [], ISI_values: [], time_to_first_spike: [] }, 0.5);
  expect(m.rows.spike_count).toEqual([2, 0]);
  expect(m.rows.latency).toEqual([3, null]);
  expect(m.rows.mean_isi).toEqual([4.5, null]);
  expect(m.scalars.periodicity_gap).toBeNull();
  expect(m.scalars.spike_shape_corr).toBeNull();
  // 差は 2 つの山の 65, 95, 65 だけ
  expect(m.scalars.mae).toBeCloseTo((2 * (65 + 95 + 65)) / v.length, 12);
  expect(m.scalars.rmse).toBeCloseTo(Math.sqrt((2 * (65 * 65 + 95 * 95 + 65 * 65)) / v.length), 12);
});

test('a NaN or a runaway potential counts as diverged and gives no waveform error', () => {
  const broken = Float64Array.from(v);
  broken[3] = NaN;
  expect(diverged(broken)).toBe(true);
  expect(diverged(new Float64Array([0, 2000]))).toBe(true);
  expect(compare(v, broken, spikes, spikes, 0.5).scalars.rmse).toBeNull();
  // eFEL の nan（JSON の null）を含む ISI の統計は nan
  expect(compare(v, v, spikes, { ...spikes, ISI_values: [null] }, 0.5).rows.mean_isi).toEqual([4.5, null]);
});

test('crossings count upward passes through 0 mV with their peaks, and the diff compares the last shared spike', () => {
  const found = crossings(v, 0.5);
  expect(found).toEqual({ times: [2.5, 7], peaks: [30, 30] });
  const shifted = crossings(new Float64Array([-65, -65, -65, -65, -65, -65, 0, 20, 0, -65]), 0.5);
  expect(shifted).toEqual({ times: [3], peaks: [20] });
  expect(crossingDiff(found, shifted)).toEqual({ count: [2, 1], peak: -10, lastShift: 0.5 });
  expect(crossingDiff(found, { times: [], peaks: [] })).toEqual({ count: [2, 0], peak: null, lastShift: null });
});

test('coincidence is 1 for the same spikes, near 0 for unrelated ones, and null with no spikes', () => {
  expect(coincidence([10, 50, 90], [10, 50, 90], 100)).toBeCloseTo(1);
  expect(coincidence([10, 50, 90], [11.5, 51, 88.5], 100)).toBeCloseTo(1);
  expect(coincidence([10, 50, 90], [30, 70], 100)).toBeLessThan(0);
  expect(coincidence([], [], 100)).toBeNull();
  expect(coincidence([10], [], 100)).toBe(0);
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
  expect(coincidence(original, model, 10000)).toBeCloseTo(expected, 6);
});

test('rate and rheobase read the f–I curve of a sweep', () => {
  expect(rate([1, 2, 3], 300)).toBeCloseTo(10);
  expect(rheobase([0, 2.5, 5], [0, 0, 4])).toBe(5);
  expect(rheobase([5, 0], [1, 0])).toBe(5);
  expect(rheobase([0, 2.5], [0, 0])).toBeNull();
});
