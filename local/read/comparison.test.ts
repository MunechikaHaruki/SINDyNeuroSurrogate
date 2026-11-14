import { expect, test } from 'bun:test';
import type { ExperimentStatus } from '../../shared/spec.ts';
import { experimentColumns, meanGamma, rheobase } from './comparison.ts';

test('columns are the finished replaced runs of an experiment, in declared order', () => {
  const experiment: ExperimentStatus = {
    name: 'e',
    models: [
      { model: 'kam.h+n', state: 'done', run: { name: 'kam.h+n', id: 't1', spec: '', state: 'done' } },
      { model: 'pca-sindy.h+n', state: 'todo', run: { name: 'pca-sindy.h+n', id: 't2', spec: '', state: 'todo' } },
    ],
    series: [{ series: 's', original: { name: 's', id: 'o', spec: '', state: 'done' } }],
    pairs: [
      { series: 's', model: 'kam.h+n', state: 'done', run: { name: 'kam.h+n', id: 'r1', spec: '', state: 'done' } },
      {
        series: 's',
        model: 'pca-sindy.h+n',
        state: 'todo',
        run: { name: 'pca-sindy.h+n', id: 'r2', spec: '', state: 'todo' },
      },
    ],
    benches: [],
  };
  expect(experimentColumns(experiment, 's')).toEqual([
    { run: 'r1', train: 't1', trainName: 'kam.h+n', model: 'kam.h+n', method: 'kam', grouping: 'h+n' },
  ]);
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
