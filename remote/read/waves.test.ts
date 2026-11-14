import { expect, test } from 'bun:test';
import { decimate, histogram, histograms, path } from './waves.ts';

test('decimation keeps each bucket’s extremes in time order, so a spike survives', () => {
  const values = new Float64Array(1000).fill(-65);
  values[503] = 40; // スパイクの頂上
  values[507] = -80; // 後過分極の谷
  const points = decimate(values, 0, 0.01, 0, 9.99, 50);
  expect(points.length).toBeLessThanOrEqual(100);
  let top = -Infinity;
  let bottom = Infinity;
  for (const point of points) {
    top = Math.max(top, point.y);
    bottom = Math.min(bottom, point.y);
  }
  expect([top, bottom]).toEqual([40, -80]);
  for (let i = 1; i < points.length; i++) expect(points[i].x).toBeGreaterThanOrEqual(points[i - 1].x);
});

test('decimation reads only the shown window and skips values that are not finite', () => {
  const values = new Float64Array([0, 1, Number.NaN, 3, 4, 5, 6, 7, 8, 9]);
  const points = decimate(values, 0, 1, 2, 5, 4);
  const xs: number[] = [];
  for (const point of points) xs.push(point.x);
  expect(xs).toEqual([3, 4, 5]);
});

test('a path keeps time order and fewer than twice max points of the window', () => {
  const xs = new Float64Array(1000).map((_, i) => i);
  const ys = new Float64Array(1000).map((_, i) => -i);
  const points = path(xs, ys, 0, 1, 100, 899, 50);
  expect(points.length).toBeLessThan(100);
  expect(points[0]).toEqual({ x: 100, y: -100 });
  for (let i = 1; i < points.length; i++) expect(points[i].x).toBeGreaterThan(points[i - 1].x);
});

test('the histogram counts only finite values in range, and puts the top edge in the last bin', () => {
  expect(histogram(new Float64Array([0, 0.5, 1, 2, NaN, -1]), 0, 2, 2)).toEqual([
    { center: 0.5, count: 2 },
    { center: 1.5, count: 2 },
  ]);
});

test('histograms share one range across the columns', () => {
  const [a, b] = histograms([new Float64Array([0, 1]), new Float64Array([2, 3])], 3);
  expect(a.reduce((n, bin) => n + bin.count, 0)).toBe(2);
  expect(b[2].count).toBe(2);
  expect(histograms([new Float64Array([1, 1])], 3)).toEqual([[]]);
});
