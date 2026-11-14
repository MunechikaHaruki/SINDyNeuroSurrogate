// エージェント用の CLI。`SINDY_API=http://<host>:8787 bun local/read/cli.ts experiments`（実験の状態）と `… table <実験>...`（元のモデルとの差の表をタブ区切りで）。
// 画面と同じ読み方（machine.ts・comparison.ts）で、計算する機械の API へ直に届ける。
import { EXPERIMENTS, SERIES } from '../resolve/experiments.ts';
import type { Plan } from '../resolve/schema.ts';
import type { ExperimentStatus, IndexEntry, State, Status } from '../../shared/spec.ts';
import type { CrossingDiff, SeriesMetrics, SeriesRequest } from '../../shared/api.ts';
import { connect, direct, listRuns, readMetrics, readStatus } from './machine.ts';
import { COINCIDENCE_WINDOW, experimentColumns, meanGamma, rheobase, seriesEntry, stimulusWindow } from './comparison.ts';

/** 系列の指標の読み方（計算する機械の API の series）。 */
type Measure = (request: SeriesRequest) => Promise<SeriesMetrics>;

/** 符号付きの小数 1 桁（−1.2、+0.5）。 */
function signed(value: number): string {
  return `${value > 0 ? '+' : '−'}${Math.abs(value).toFixed(1)}`;
}

/** 1 列（学習 run 1 つ）の、掃引の点ごとの元のモデルとの差と Γ（どちらも発火しない点は null）。 */
type Row = { name: string; diffs: CrossingDiff[]; gammas: (number | null)[] };

/** 発火回数の差を「全点で一致」か「2.5 で +2、10 で −1」の形に書く。掃引しない系列は値を書かない。 */
function countText(row: Row, values: number[]): string {
  const parts: string[] = [];
  for (let k = 0; k < row.diffs.length; k++) {
    const delta = row.diffs[k].count[1] - row.diffs[k].count[0];
    if (delta === 0) continue;
    const sign = delta > 0 ? '+' : '−';
    parts.push(values.length > 0 ? `${values[k]} で ${sign}${Math.abs(delta)}` : `${sign}${Math.abs(delta)}`);
  }
  return parts.length === 0 ? '全点で一致' : parts.join('、');
}

/** 発火のある点での、山の平均の差（モデル − 元）の最小と最大 [mV]。発火が無ければ —。 */
function peakText(row: Row): string {
  let low: number | null = null;
  let high: number | null = null;
  for (const diff of row.diffs) {
    if (diff.peak === null) continue;
    low = low === null ? diff.peak : Math.min(low, diff.peak);
    high = high === null ? diff.peak : Math.max(high, diff.peak);
  }
  if (low === null || high === null) return '—';
  return `${signed(low)} 〜 ${signed(high)}`;
}

/** Γ の、決まる点での平均。決まる点が無ければ —。 */
function gammaText(row: Row): string {
  const mean = meanGamma(row.gammas);
  return mean === null ? '—' : mean.toFixed(2);
}

/** 発火の始まり（rheobase）を「元 → モデル」で書く。掃引しない系列は —。 */
function rheobaseText(row: Row, values: number[]): string {
  if (values.length === 0) return '—';
  const show = (x: number | null) => (x === null ? '無し' : `${x}`);
  const original = rheobase(
    values,
    row.diffs.map(d => d.count[0]),
  );
  const model = rheobase(
    values,
    row.diffs.map(d => d.count[1]),
  );
  return original === model ? `${show(model)}（一致）` : `${show(original)} → ${show(model)}`;
}

/** 実験の 1 系列の表。comp は比べた区画（系列が置き換える最初の区画）。 */
type Table = { experiment: string; series: string; comp: string; values: number[]; unit: string; rows: Row[] };

/** 実験の系列を、回し終えた列ごとに元のモデルと比べる。系列を省けば実験の最初の系列。元のモデルを回していなければ投げる。 */
async function compareExperiment(
  measure: Measure,
  index: IndexEntry[],
  experiment: ExperimentStatus,
  series?: string,
): Promise<Table> {
  const name = series ?? experiment.series[0].series;
  const entry = seriesEntry(index, experiment, name);
  if (!entry) throw new Error(`${experiment.name} の ${name} の元のモデルをまだ回していない`);
  const comp = SERIES[name].replace_targets[0];
  const columns = experimentColumns(experiment, name);
  const metrics = await measure({
    original: entry.run,
    columns: columns.map(c => c.run),
    comp,
    delta: COINCIDENCE_WINDOW,
  });
  const rows: Row[] = [];
  for (const column of columns) {
    const found = metrics.crossings[column.run] ?? [];
    rows.push({
      name: column.trainName,
      diffs: found.map(({ count, peak, lastShift }) => ({ count, peak, lastShift })),
      gammas: found.map(r => r.gamma),
    });
  }
  return { experiment: experiment.name, series: name, comp, values: entry.values, unit: entry.unit, rows };
}

/** 実験の状態。 */
type Progress = '未着手' | '途中' | '計算中' | '結果あり';

/** 実験を並べる順。手を動かす所を先に。 */
const PROGRESS_ORDER: Progress[] = ['結果あり', '計算中', '途中', '未着手'];

/** 実験の学習と評価の状態を並べる。 */
function statesOf(found: ExperimentStatus): State[] {
  const states: State[] = [];
  for (const model of found.models) states.push(model.state);
  for (const pair of found.pairs) states.push(pair.state);
  return states;
}

/** 実験の状態を run から導く。置換できない組は済んだものとして数える。回す run が無ければ未着手。 */
function progressOf(experiment: Plan, status: Status): Progress {
  if (Object.keys(experiment.series).length === 0) return '未着手';
  const found = status.experiments.find(e => e.name === experiment.id);
  if (!found) throw new Error(`実験 ${experiment.id} の状態が無い`);
  const states = statesOf(found);
  if (states.includes('running')) return '計算中';
  if (!states.includes('todo')) return '結果あり';
  return states.some(s => s === 'done') ? '途中' : '未着手';
}

/** エージェント用：`experiments` は実験を状態ごとに説明の最初の段落を添えて、`table <実験>...` は実験の系列ごとの差の表をタブ区切りで書き出す。 */
async function main([command, ...names]: string[]): Promise<void> {
  const base = process.env.SINDY_API;
  if (!base) throw new Error('SINDY_API に計算する機械の API（http://<host>:8787）を渡す');
  connect(direct(base));
  const current = await readStatus();
  if (command === 'experiments') {
    for (const progress of PROGRESS_ORDER)
      for (const e of EXPERIMENTS)
        if (progressOf(e, current) === progress) console.log(`${progress}\t${e.id}\t${e.description.split('\n\n')[0]}`);
    return;
  }
  if (command !== 'table' || names.length === 0)
    throw new Error('experiments か、table と実験の id（研究コードの experiments/）を渡す');
  const index = await listRuns();
  const measure: Measure = readMetrics;
  for (const name of names) {
    const experiment = current.experiments.find(e => e.name === name);
    if (!experiment) throw new Error(`実験 ${name} が研究コードの experiments/ に無い`);
    for (const { series } of experiment.series) {
      const table = await compareExperiment(measure, index, experiment, series);
      console.log(
        `# ${table.experiment} / ${table.series} / ${table.comp}（掃引 ${table.values.join(', ')} ${table.unit}、刺激 ${stimulusWindow(table.series).join('〜')} ms。その外は無刺激）`,
      );
      console.log('列\tΓ（Δ = 2 ms）\t発火の始まり\t発火回数の差\tピークの差 (mV)');
      for (const row of table.rows) {
        const cells = [gammaText(row), rheobaseText(row, table.values), countText(row, table.values), peakText(row)];
        console.log(`${row.name}\t${cells.join('\t')}`);
      }
    }
  }
}

if (import.meta.main) await main(process.argv.slice(2));
