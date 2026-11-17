// 学習データ（runs/data/<学習データ>/）の答え。注入した電流と区画ごとの膜電位の線、学習が踏んだ電位の分布。
import type { DataReply, DataRequest, Line } from '../../shared/api.ts';
import { compArray, currentArray, RUN, type WaveMeta } from '../../shared/runs.ts';
import type { Spec } from '../../shared/spec.ts';
import type { Context } from './experiments.ts';
import { spanOf, wavesOf } from './inputs.ts';
import { compact, histograms, line } from './waves.ts';

/** 電位の分布の区間の数。 */
const BINS = 60;

/** 学習データの run の区画 name の変数 1 つ。 */
export function dataArray(run: string, meta: WaveMeta, name: string, variable: string) {
  const id = meta.names?.[name];
  if (id === undefined) throw new Error(`${run} に区画 ${name} が無い`);
  return compArray(run, RUN.waves, meta, 0, id, variable);
}

/** 学習データ 1 本の、区画 comps の線と分布。 */
export function data(ctx: Context, request: DataRequest): DataReply {
  const { run } = request;
  const spec = ctx.files.json<Spec>(run, RUN.spec);
  if (spec.kind !== 'data') throw new Error(`${run} は学習データの run でない`);
  const meta = wavesOf(ctx, run);
  const stim = spec.data.stim ?? 'soma';
  const all = meta.names ? Object.keys(meta.names) : [stim];
  const comps = request.comps ?? [all.includes(stim) ? stim : all[0]];
  const duration = meta.n[0] * meta.dt;
  const span = spanOf(meta, request.window, duration);
  const voltages = comps.map(name => ctx.files.column(dataArray(run, meta, name, 'V')));
  const coverage: Line[] = histograms(voltages, BINS).map((bins, i) => ({
    label: comps[i],
    points: compact(bins.map(bin => ({ x: bin.center, y: bin.count }))),
  }));
  return {
    run,
    stim,
    comps: all,
    shown: comps,
    duration,
    current: [{ label: 'I_ext', points: line(ctx.files, currentArray(run, RUN.waves, meta, 0), span) }],
    voltage: comps.map(name => ({ label: name, points: line(ctx.files, dataArray(run, meta, name, 'V'), span) })),
    coverage,
  };
}
