// 学習データ（runs/data/<学習データ>/）の答え。run が持つ波形そのものを、選んだ区画ごとに返す：注入した電流、膜電位、ゲートとゲートでない状態（Ca 濃度など）、内部電流（区画がつながる型だけ）。
import type { DataReply, DataRequest, Line } from '../../shared/api.ts';
import { compArray, currentArray, gatesOf, internalArray, RUN, type ArrayRef, type WaveMeta } from '../../shared/runs.ts';
import type { Spec } from '../../shared/spec.ts';
import { learnedRun, type Context } from './experiments.ts';
import { spanOf, wavesOf } from './inputs.ts';
import { isFraction, line } from './waves.ts';

/** 学習データ 1 本の、区画 comps の線。 */
export function data(ctx: Context, request: DataRequest): DataReply {
  const run = learnedRun(ctx.runs, request.experiment, 'data', request.data);
  const spec = ctx.files.json<Spec>(run, RUN.spec);
  if (spec.kind !== 'data') throw new Error(`${run} は学習データの run でない`);
  const meta: WaveMeta = wavesOf(ctx, run);
  const stim = spec.data.stim ?? 'soma';
  const all = meta.names ? Object.keys(meta.names) : [stim];
  const comps = request.comps ?? [all.includes(stim) ? stim : all[0]];
  const duration = meta.n[0] * meta.dt;
  const span = spanOf(meta, request.window, duration);
  const draw = (label: string, ref: ArrayRef): Line => ({ label, points: line(ctx.files, ref, span) });
  const reply: DataReply = {
    run,
    stim,
    comps: all,
    shown: comps,
    duration,
    current: [draw('I_ext', currentArray(run, RUN.waves, meta, 0))],
    voltage: [],
    gates: [],
    states: [],
    internal: [],
  };
  for (const name of comps) {
    const id = meta.names?.[name];
    if (id === undefined) throw new Error(`${run} に区画 ${name} が無い`);
    reply.voltage.push(draw(name, compArray(run, RUN.waves, meta, 0, id, 'V')));
    for (const gate of gatesOf(meta, id)) {
      const ref = compArray(run, RUN.waves, meta, 0, id, gate);
      const drawn = draw(comps.length > 1 ? `${gate}（${name}）` : gate, ref);
      (isFraction(ctx.files.column(ref)) ? reply.gates : reply.states).push(drawn);
    }
    if (all.length > 1 && meta.nodes?.includes(id)) reply.internal.push(draw(name, internalArray(run, RUN.waves, meta, 0, id)));
  }
  return reply;
}
