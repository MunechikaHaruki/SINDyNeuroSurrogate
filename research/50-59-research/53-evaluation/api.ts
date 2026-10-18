// 図のボタンから学習と評価を起こす口。開発サーバーが /api/face/research/50-59-research/53-evaluation に載せる（context-router の .web/routes/api/face）。
// 受けるのは conf.ts に宣言した preset 名と系列名だけで、段取り（orchestrate.ts）の関数を呼ぶ。任意のコマンドは通さない。
import { PRESETS, SERIES, type Status } from './conf.ts';
import { RUNS, launch, status } from './orchestrate.ts';

/** 宣言の学習と評価の状態。図はこれを見て「計算中」と未実行の組を出す。 */
export function GET(): Status {
  return status(RUNS);
}

/** { action: 'train', preset } か { action: 'eval', series, preset } を切り離して回す。同じ組が回っていれば起こさない。 */
export function POST(body: unknown): { started: boolean } {
  const { action, preset, series } = (body ?? {}) as Record<string, unknown>;
  if (typeof preset !== 'string' || !Object.hasOwn(PRESETS, preset))
    throw new Error(`宣言に無い preset ${String(preset)}`);
  if (action === 'train') return { started: launch(RUNS, { action, preset }) };
  if (action !== 'eval') throw new Error(`知らない action ${String(action)}`);
  if (typeof series !== 'string' || !Object.hasOwn(SERIES, series)) throw new Error(`宣言に無い系列 ${String(series)}`);
  return { started: launch(RUNS, { action, series, preset }) };
}
