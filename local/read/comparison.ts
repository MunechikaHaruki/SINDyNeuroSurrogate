// 比べ。実験の系列 1 つについて、原系と、学習 run で置き換えて回した列を、宣言（../resolve/experiments.ts）に照らして組み、区画 1 つの原系との指標まで読む。
// 画面の図と表、エージェントの CLI の材料。計算する機械には隣の machine.ts を通して問い合わせる。
// 比べる物差し（Γ の許容幅と線、Γ の平均、発火の始まり）もここが持つ。
import { MODELS, SERIES } from '../resolve/experiments.ts';
import type { ExperimentStatus, IndexEntry, Status } from '../../shared/spec.ts';
import type { SeriesMetrics } from '../../shared/api.ts';
import { RUN, type SpikesFile, type WaveMeta } from '../../shared/runs.ts';
import { readJson, readMetrics } from './machine.ts';

// ---- 物差し

/** Γ で「発火の時刻まで合う」とみなす線。 */
export const GAMMA_LINE = 0.9;

/** Γ の許容幅 Δ [ms]（2026-10-27 に決めた物差し）。計算する機械には series の頼み方で渡す。 */
export const COINCIDENCE_WINDOW = 2;

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

// ---- 比べ

/** 評価の系列 1 つ。原系の run と、どこを置換したか。 */
export type SeriesEntry = {
  name: string;
  /** 元のモデルの run の id */
  run: string;
  param: string | null;
  values: number[];
  unit: string;
  replaceTargets: string[];
};

/** 系列を 1 つの学習 run で置換して回した列。run と train は id、model は学習 run の名前（`<仕方>.<群分け>`）、method と grouping はその仕方と群分けの名前。 */
export type Column = { run: string; train: string; trainName: string; model: string; method: string; grouping: string };

/** 実験の系列を、原系 run の一覧の行から読む。原系をまだ回していなければ undefined。 */
export function seriesEntry(runs: IndexEntry[], experiment: ExperimentStatus, name: string): SeriesEntry | undefined {
  const declared = experiment.series.find(s => s.series === name);
  const row = runs.find(r => r.id === declared?.original.id);
  if (!row || row.spec.kind !== 'sim') return undefined;
  const input = row.spec.input;
  return {
    name,
    run: row.id,
    param: input.param,
    values: input.values,
    unit: row.unit ?? '',
    replaceTargets: SERIES[name].replace_targets,
  };
}

/** 系列の刺激が入る区間 [ms]。前後の silence_duration は電流が 0 の無刺激の区間で、掃引の 0 の点とは別のもの。 */
export function stimulusWindow(name: string): [number, number] {
  const { silence_duration: silence = 0, duration } = SERIES[name].spec.current_params;
  return [silence, duration - silence];
}

/** 実験の系列を、その実験の学習 run で置換して回し終えた列。宣言の順。 */
export function experimentColumns(experiment: ExperimentStatus, series: string): Column[] {
  const list: Column[] = [];
  for (const pair of experiment.pairs) {
    if (pair.series !== series || pair.run.state !== 'done') continue;
    const train = experiment.models.find(m => m.model === pair.model)!.run;
    const declared = MODELS[pair.model];
    if (!declared) throw new Error(`学習 run ${pair.model} がどの実験の宣言にも無い`);
    const { method, grouping } = declared;
    list.push({ run: pair.run.id, train: train.id, trainName: pair.model, model: pair.model, method, grouping });
  }
  return list;
}

/** 系列 1 つの、原系と回し終えた全列の目録・スパイクの特徴量・学習の要約。キーは run の id。 */
export type SeriesData = {
  entry: SeriesEntry;
  columns: Column[];
  originalMeta: WaveMeta;
  metas: Record<string, WaveMeta>;
  spikes: Record<string, SpikesFile>;
  summaries: Record<string, Record<string, number | null>>;
};

/** 系列を読む。原系をまだ回していなければ投げる。 */
export async function loadSeries(index: IndexEntry[], experiment: ExperimentStatus, name: string): Promise<SeriesData> {
  const entry = seriesEntry(index, experiment, name);
  if (!entry) throw new Error(`${experiment.name} の ${name} の元のモデルをまだ回していない`);
  const columns = experimentColumns(experiment, name);
  const data: SeriesData = {
    entry,
    columns,
    originalMeta: await readJson<WaveMeta>(entry.run, RUN.meta(RUN.waves)),
    metas: {},
    spikes: {},
    summaries: {},
  };
  const reads: Promise<void>[] = [
    readJson<SpikesFile>(entry.run, RUN.spikes).then(s => void (data.spikes[entry.run] = s)),
  ];
  for (const column of columns) {
    reads.push(readJson<WaveMeta>(column.run, RUN.meta(RUN.waves)).then(m => void (data.metas[column.run] = m)));
    reads.push(readJson<SpikesFile>(column.run, RUN.spikes).then(s => void (data.spikes[column.run] = s)));
    reads.push(
      readJson<Record<string, number | null>>(column.train, RUN.summary).then(
        s => void (data.summaries[column.run] = s),
      ),
    );
  }
  await Promise.all(reads);
  return data;
}

/** 系列の列（省けば全列）の、1 区画での原系との指標（計算する機械が波形から出す）。 */
function loadMetrics(data: SeriesData, comp: string, columns: Column[] = data.columns): Promise<SeriesMetrics> {
  return readMetrics({ original: data.entry.run, columns: columns.map(c => c.run), comp, delta: COINCIDENCE_WINDOW });
}

/** 比べ 1 つ。実験の系列 1 つを、絞り込みで残った列と 1 区画の指標まで読んだもの。画面の図と表の材料。
 * names は列の短い名前（キーは学習 run の id）、comp は比べた区画（省けば系列が置き換える最初の区画）、domain は系列の全体の時間 [ms]。 */
export type Comparison = {
  experiment: string;
  data: SeriesData;
  metrics: SeriesMetrics;
  columns: Column[];
  names: Record<string, string>;
  comp: string;
  domain: [number, number];
};

/** 実験の系列を、群分けと仕方で絞った列と、1 区画の指標まで読む。実験が無いか原系をまだ回していなければ投げる。 */
export async function loadComparison(
  index: IndexEntry[],
  status: Status,
  experiment: string,
  series: string,
  only: { groupings?: string[]; methods?: string[]; comp?: string } = {},
): Promise<Comparison> {
  const found = status.experiments.find(e => e.name === experiment);
  if (!found) throw new Error(`実験 ${experiment} が宣言に無い`);
  const data = await loadSeries(index, found, series);
  const columns = columnsOf(data, only);
  const comp = only.comp ?? data.entry.replaceTargets[0] ?? 'soma';
  const metrics = await loadMetrics(data, comp, columns);
  const domain: [number, number] = [0, data.originalMeta.n[0] * data.originalMeta.dt];
  return { experiment, data, metrics, columns, names: shortNames(columns), comp, domain };
}

/** 列が掃引の点 k で発散したか（置換系の電位が NaN・無限大か ±1000 mV を超えた）。 */
export function divergedAt(c: Comparison, column: Column, k: number): boolean {
  return c.metrics.pairs[column.run]?.[k]?.diverged ?? false;
}

/** 列のうち、学習 run の群分けと仕方が絞り込みに入るもの。省いた絞り込みは全部を通す。 */
function columnsOf(data: SeriesData, only: { groupings?: string[]; methods?: string[] } = {}): Column[] {
  const list: Column[] = [];
  for (const column of data.columns) {
    if (only.groupings && !only.groupings.includes(column.grouping)) continue;
    if (only.methods && !only.methods.includes(column.method)) continue;
    list.push(column);
  }
  return list;
}

/** 凡例と表の見出しに使う短い名前（キーは学習 run の id）。全列の名前に共通する語を省く。 */
function shortNames(columns: Column[]): Record<string, string> {
  const names: Record<string, string> = {};
  for (const column of columns) {
    const kept: string[] = [];
    for (const word of column.trainName.split(' ')) {
      let shared = columns.length > 1;
      for (const other of columns) if (!other.trainName.split(' ').includes(word)) shared = false;
      if (!shared) kept.push(word);
    }
    names[column.train] = kept.length > 0 ? kept.join(' ') : column.trainName;
  }
  return names;
}
