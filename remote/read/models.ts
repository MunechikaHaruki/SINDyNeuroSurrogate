// モデル（runs/models/<モデル>/）の答え。モデルの run だけを読む：組み立て（spec の細胞・形・吸収と、組んだ学習器）と、組んだ 1 刻みの演算の回数（summary.json）。
import type { CostRow, ModelsReply, ModelsRequest } from '../../shared/api.ts';
import { RUN } from '../../shared/runs.ts';
import type { Spec } from '../../shared/spec.ts';
import { specAt } from '../run/places.ts';
import { experimentStatus } from '../run/status.ts';
import { type Context, modelOf, planOf, plans } from './experiments.ts';

/** 演算の並び（summary.json の cost/<側>/<演算>）。 */
const OPS = ['exp', 'div', 'pm', 'mul'];

/** summary.json の演算の回数を、演算ごとの行にする。 */
function costRows(summary: Record<string, number>): CostRow[] {
  return OPS.map(op => ({
    op,
    surrogate: summary[`cost/surrogate/${op}`],
    original: summary[`cost/original/${op}`],
    diff: summary[`cost/surr-orig/${op}`],
  }));
}

/** モデル 1 本の組み立てと演算の回数。組み立てはモデルの spec（回す前は送られてきた resolved/ の同じ spec）から読む。 */
export function models(ctx: Context, request: ModelsRequest): ModelsReply {
  const plan = planOf(plans(), request.experiment);
  const model = modelOf(plan, request.model);
  const built = experimentStatus(ctx.runs, plan.id).models.find(m => m.model === model.name);
  if (!built) throw new Error(`${plan.id} のモデル ${model.name} の spec が無い`);
  const { run } = built;
  const spec = run.state === 'done' ? ctx.files.json<Spec>(run.id, RUN.spec) : specAt(run.spec);
  if (spec.kind !== 'model') throw new Error(`${run.id} はモデルの run でない`);
  return {
    model: model.name,
    run: { state: run.state, run: run.id },
    cell: spec.cell,
    form: spec.form,
    absorbed: spec.absorbed ?? [],
    parts: built.parts.map(p => p.name),
    cost: run.state === 'done' ? costRows(ctx.files.json<Record<string, number>>(run.id, RUN.summary)) : null,
  };
}
