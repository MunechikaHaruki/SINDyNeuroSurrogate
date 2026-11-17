// 実験の状態。resolved/ の実験ごとに、モデル（とその学習器・学習データ）・系列と組・GPU の実測の置き場と状態を組み、画面から起こした仕事と合わせて返す。並びは名前の順で、宣言の順は答えを組む側（../read/）が宣言の写しで並べる。
import { join } from 'node:path';
import type { ExperimentStatus, RunStatus, State, Status } from '../../shared/spec.ts';
import { jobRunning, jobs } from '../run-spec.ts';
import { existsSync, readdirSync } from 'node:fs';
import { hashOf } from '../../shared/hash.ts';
import { dirOf, resolvedDir, runState, specAt } from './places.ts';

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

/** resolved/<実験>/<dir>/ の spec の、ハッシュ → 状態。モデルが指す学習器と、学習器が指す学習データを引く。 */
function byHash(runs: string, name: string, dir: string): Map<string, RunStatus> {
  const found = new Map<string, RunStatus>();
  for (const spec of specNames(join(resolvedDir(), name, dir))) {
    const path = `${name}/${dir}/${spec}.json`;
    found.set(hashOf(specAt(path)), runStatus(runs, path, spec));
  }
  return found;
}

/** モデルの状態。組んであれば済み、どれかが回っていれば計算中。 */
function modelState(runs: string, run: RunStatus, pointed: RunStatus[]): State {
  if (run.state === 'done') return 'done';
  return [run, ...pointed].some(r => withJob(runs, r) === 'running') ? 'running' : run.state;
}

/** resolved/<実験>/ の spec の、置き場と状態。モデルは models/（指す学習器は parts/、学習データは data/）、系列は sim/<系列>/ の original とモデルの名前、GPU の実測は bench/。並びは名前の順で、宣言の順は使う側が持つ。 */
export function experimentStatus(runs: string, name: string): ExperimentStatus {
  const root = join(resolvedDir(), name);
  const result: ExperimentStatus = { name, models: [], series: [], pairs: [], benches: [] };
  const parts = byHash(runs, name, 'parts');
  const data = byHash(runs, name, 'data');
  for (const model of specNames(join(root, 'models'))) {
    const path = `${name}/models/${model}.json`;
    const spec = specAt(path);
    if (spec.kind !== 'model') throw new Error(`resolved/${path} はモデルの spec でない`);
    const run = runStatus(runs, path, model);
    const own = spec.parts.map(hash => parts.get(hash)!);
    const first = specAt(own[0].spec);
    if (first.kind !== 'part') throw new Error(`resolved/${own[0].spec} は学習器の spec でない`);
    const trained = data.get(first.data)!;
    result.models.push({ model, state: modelState(runs, run, [...own, trained]), run, parts: own, data: trained });
  }
  for (const series of subdirs(join(root, 'sim'))) {
    const original = runStatus(runs, `${name}/sim/${series}/original.json`, series);
    result.series.push({ series, original });
    for (const model of specNames(join(root, 'sim', series))) {
      if (model === 'original') continue;
      const run = runStatus(runs, `${name}/sim/${series}/${model}.json`, model);
      const train = result.models.find(m => m.model === model)?.state;
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
  return { experiments: subdirs(resolvedDir()).map(name => experimentStatus(runs, name)), jobs: jobs(runs) };
}
