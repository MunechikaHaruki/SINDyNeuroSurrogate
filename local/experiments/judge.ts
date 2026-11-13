// 主張の判定。計算する機械が返した量（API の series の口の Γ と発火の回数）を、判定層（config/<群>.checks.ts）の線で判定する。
// 判定は宣言を持つ手元が出す。計算する機械は宣言を知らず、量だけを返す。問い合わせ方（measure）は使う側が渡す。
import type { SeriesMetrics, SeriesRequest } from '../../shared/api.ts';
import type { ExperimentStatus, IndexEntry, Status } from '../../shared/spec.ts';
import { COINCIDENCE_WINDOW, rowsOf, type Claim, type Column, type Judgement, type Row } from './common.ts';
import { MODELS, seriesOf, type GroupedPlan } from './experiments.ts';

/** 主張 1 つの判定。name は主張の名前、rows は判定の行、判定できなければ null と理由。 */
export type ClaimJudgement = {
  experiment: string;
  claim: string;
  name: string;
  rows: Judgement[] | null;
  error?: string;
};

/** 系列 1 つの量を計算する機械に頼む関数。 */
export type Measure = (request: SeriesRequest) => Promise<SeriesMetrics>;

/** 実験の系列 1 つを、置き換えて回し終えた列ごとの判定の値にする（列は宣言の順）。元のモデルをまだ回していなければ投げる。 */
async function table(
  measure: Measure,
  index: IndexEntry[],
  plan: GroupedPlan,
  found: ExperimentStatus,
  name: string,
): Promise<Row[]> {
  const original = found.series.find(s => s.series === name)?.original;
  const entry = index.find(r => r.id === original?.id);
  if (!original || !entry || entry.spec.kind !== 'sim')
    throw new Error(`${plan.id} の ${name} の元のモデルをまだ回していない`);
  const columns: Column[] = [];
  for (const p of plan.pairs) {
    if (p.series !== name) continue;
    const pair = found.pairs.find(x => x.series === name && x.model === p.model);
    if (pair?.run.state !== 'done') continue;
    columns.push({ run: pair.run.id, method: MODELS[p.model].method, grouping: MODELS[p.model].grouping });
  }
  const request = {
    original: original.id,
    columns: columns.map(c => c.run),
    comp: seriesOf(name).replace_targets[0],
    delta: COINCIDENCE_WINDOW,
  };
  return rowsOf(columns, entry.spec.input.values, (await measure(request)).crossings);
}

/** 主張 1 つを判定する。 */
async function judge(
  measure: Measure,
  index: IndexEntry[],
  plan: GroupedPlan,
  found: ExperimentStatus,
  claim: Claim,
): Promise<Judgement[]> {
  const tables: Row[][] = [];
  for (const name of claim.series) tables.push(await table(measure, index, plan, found, name));
  return claim.judge(tables, claim.groupings);
}

/** 実験の主張ごとの判定（実験の順、主張の順）。判定できない主張は理由を添える。 */
export async function judgeClaims(
  experiments: GroupedPlan[],
  status: Status,
  index: IndexEntry[],
  measure: Measure,
): Promise<ClaimJudgement[]> {
  const list: ClaimJudgement[] = [];
  for (const plan of experiments) {
    const found = status.experiments.find(x => x.name === plan.id);
    for (const c of plan.claims) {
      const base = { experiment: plan.id, claim: c.claim, name: c.name };
      try {
        if (!found) throw new Error(`計算する機械の resolved/ に ${plan.id} が無い（送り直す）`);
        list.push({ ...base, rows: await judge(measure, index, plan, found, c) });
      } catch (err) {
        list.push({ ...base, rows: null, error: err instanceof Error ? err.message : String(err) });
      }
    }
  }
  return list;
}
