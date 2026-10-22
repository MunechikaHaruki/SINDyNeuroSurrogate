import { expect, test } from 'bun:test';
import { decimate, decodeComp, declaredNames, experimentColumns, experimentTrains, variablesOf, type WaveMeta } from './runs.ts';
import type { ExperimentStatus } from '../orchestrate/spec.ts';

const meta: WaveMeta = {
  t0: 0,
  dt: 0.5,
  n: [4],
  features: [
    [0, 'V', false],
    [1, 'V', false],
    [0, 'M', true],
    [0, 'H', true],
  ],
  nodes: null,
  names: null,
};

test('a comp file is split into one column per variable, in the order the features list them', () => {
  expect(variablesOf(meta, 0)).toEqual(['V', 'M', 'H']);
  // この repo は comp の変数を変数ごとに続けて書く（V の 4 点、M の 4 点、H の 4 点）。
  const raw = new Float64Array([-60, -59, -58, -57, 0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6]);
  const table = decodeComp(raw.buffer, variablesOf(meta, 0), meta.n[0]);
  expect(Array.from(table.V)).toEqual([-60, -59, -58, -57]);
  expect(Array.from(table.M)).toEqual(Array.from(new Float64Array([0.1, 0.2, 0.3, 0.4])));
  expect(Array.from(table.H)).toEqual(Array.from(new Float64Array([0.9, 0.8, 0.7, 0.6])));
  expect(() => decodeComp(raw.buffer, ['V', 'M'], 4)).toThrow();
});

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

test('columns are the finished replaced runs of an experiment, in preset and sweep order', () => {
  const experiment: ExperimentStatus = {
    name: 'e',
    presets: [
      { preset: 'p', state: 'done', runs: [{ name: 'p n=1', id: 't1', state: 'done' }, { name: 'p n=2', id: 't2', state: 'done' }] },
      { preset: 'q', state: 'todo', runs: [{ name: 'q', id: 't3', state: 'todo' }] },
    ],
    series: [{ series: 's', original: { name: 's', id: 'o', state: 'done' } }],
    pairs: [
      { series: 's', preset: 'p', state: 'todo', runs: [{ name: 'p n=1', id: 'r1', state: 'done' }, { name: 'p n=2', id: 'r2', state: 'todo' }] },
      { series: 's', preset: 'q', state: 'todo', runs: [{ name: 'q', id: 'r3', state: 'todo' }] },
    ],
  };
  expect(experimentColumns(experiment, 's')).toEqual([{ run: 'r1', train: 't1', trainName: 'p n=1' }]);
  expect(experimentTrains(experiment).map(r => r.id)).toEqual(['t1', 't2']);
  expect(declaredNames({ experiments: [experiment], jobs: [] }).r1).toBe('s × p n=1');
});
