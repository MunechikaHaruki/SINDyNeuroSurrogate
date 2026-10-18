import { expect, test } from 'bun:test';
import { decimate, decodeComp, listColumns, listSeries, listTrainRuns, variablesOf, type WaveMeta } from './runs.ts';
import { PRESETS, type IndexEntry, type SimSpec } from '../orchestrate/conf.ts';

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
  const raw = new Float32Array([-60, -59, -58, -57, 0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6]);
  const table = decodeComp(raw.buffer, variablesOf(meta, 0), meta.n[0]);
  expect(Array.from(table.V)).toEqual([-60, -59, -58, -57]);
  expect(Array.from(table.M)).toEqual(Array.from(new Float32Array([0.1, 0.2, 0.3, 0.4])));
  expect(Array.from(table.H)).toEqual(Array.from(new Float32Array([0.9, 0.8, 0.7, 0.6])));
  expect(() => decodeComp(raw.buffer, ['V', 'M'], 4)).toThrow();
});

test('decimation keeps each bucket’s extremes in time order, so a spike survives', () => {
  const values = new Float32Array(1000).fill(-65);
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
  const values = new Float32Array([0, 1, Number.NaN, 3, 4, 5, 6, 7, 8, 9]);
  const points = decimate(values, 0, 1, 2, 5, 4);
  const xs: number[] = [];
  for (const point of points) xs.push(point.x);
  expect(xs).toEqual([3, 4, 5]);
});

test('runs are grouped into series by their original run and what they replace', () => {
  const sim: SimSpec = { target: 'traub19', current_type: 'lin&steady', dt: 0.01, current_params: {} };
  const series = { spec: sim, param: 'value', values: [0, 5] };
  const runs: IndexEntry[] = [
    { hash: 'r2', kind: 'replaced', name: 'all', created: '3', spec: { format: 1, kind: 'replaced', series: { ...series, replace_targets: ['soma', 'c00'] }, original: 'o', train: 't' } },
    { hash: 'r1', kind: 'replaced', name: 'soma', created: '2', unit: 'μA/cm²', spec: { format: 1, kind: 'replaced', series: { ...series, replace_targets: ['soma'] }, original: 'o', train: 't' } },
    { hash: 'o', kind: 'original', name: 'soma', created: '1', spec: { format: 1, kind: 'original', series } },
    { hash: 'p', kind: 'original', name: 'pulse', created: '1', spec: { format: 1, kind: 'original', series } },
    { hash: 't', kind: 'train', name: 'preset', created: '0', spec: { format: 1, kind: 'train', surrogate: PRESETS._test_hh_sindy[0].config } },
  ];
  const list = listSeries(runs);
  const names: string[] = [];
  for (const entry of list) names.push(`${entry.name}:${entry.run}`);
  expect(names).toEqual(['all:o', 'soma:o', 'pulse:p']);
  expect(listColumns(runs, list[1])).toEqual([{ run: 'r1', train: 't', trainName: 'preset' }]);
  expect(listColumns(runs, list[2])).toEqual([]);
  expect(listTrainRuns(runs)).toEqual([{ hash: 't', name: 'preset' }]);
});
