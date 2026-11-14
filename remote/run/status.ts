// 実験の状態。resolved/ の実験ごとに、学習 run・系列と組・GPU の実測の置き場と状態を組み、画面から起こした仕事と合わせて返す。並びは名前の順で、宣言の順は手元が持つ。
import { join } from 'node:path';
import type { ExperimentStatus, RunStatus, State, Status } from '../../shared/spec.ts';
import { jobRunning, jobs } from './run-spec.ts';
import { existsSync, readdirSync } from 'node:fs';
import { dirOf, RESOLVED, runState, specAt } from './places.ts';

/** resolved/ の spec 1 つの名前・置き場・状態。 */
function runStatus(runs: string, path: string, name: string): RunStatus {
  const id = dirOf(specAt(path));
  return { name, id, spec: path, state: runState(runs, id) };
}

/** ディレクトリの下の .json の名前（.json を除いたもの）を名前の順に。無ければ空。 */
function specNames(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter(f => f.endsWith('.json'))
    .map(f => f.slice(0, -'.json'.length))
    .sort();
}

/** ディレクトリの下のディレクトリの名前を名前の順に。無ければ空。 */
function subdirs(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir, { withFileTypes: true })
    .filter(e => e.isDirectory())
    .map(e => e.name)
    .sort();
}

/** 仕事が回っていれば計算中、でなければ run の状態。 */
function withJob(runs: string, run: RunStatus): State {
  return jobRunning(runs, { spec: run.spec }) ? 'running' : run.state;
}

/** resolved/<実験>/ の spec の、置き場と状態。学習 run は train/、系列は sim/<系列>/ の original と学習 run の名前、GPU の実測は bench/。並びは名前の順で、宣言の順は使う側が持つ。 */
function experimentStatus(runs: string, name: string): ExperimentStatus {
  const root = join(RESOLVED, name);
  const result: ExperimentStatus = { name, models: [], series: [], pairs: [], benches: [] };
  for (const model of specNames(join(root, 'train'))) {
    const run = runStatus(runs, `${name}/train/${model}.json`, model);
    result.models.push({ model, state: withJob(runs, run), run });
  }
  for (const series of subdirs(join(root, 'sim'))) {
    const original = runStatus(runs, `${name}/sim/${series}/original.json`, series);
    result.series.push({ series, original });
    for (const model of specNames(join(root, 'sim', series))) {
      if (model === 'original') continue;
      const run = runStatus(runs, `${name}/sim/${series}/${model}.json`, model);
      const train = result.models.find(m => m.model === model)?.run.state;
      const running = jobRunning(runs, { spec: run.spec }) || [original.state, train, run.state].includes('running');
      result.pairs.push({ series, model, state: running ? 'running' : run.state, run });
    }
  }
  for (const bench of specNames(join(root, 'bench'))) {
    const run = runStatus(runs, `${name}/bench/${bench}.json`, `${name} ${bench}`);
    result.benches.push({ bench, state: withJob(runs, run), run });
  }
  return result;
}

/** resolved/ の実験ごとの状態と、外の画面から起こした仕事。実験は resolved/ の下のディレクトリで、宣言は読まない。 */
export function status(runs: string): Status {
  return { experiments: subdirs(RESOLVED).map(name => experimentStatus(runs, name)), jobs: jobs(runs) };
}
