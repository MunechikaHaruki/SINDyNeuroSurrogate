// 操作盤の冊（53.03）のボタンから学習と評価を起こす口。開発サーバーが /api/resources/50-59-research/53-evaluation/orchestrate に載せる（context-router の resources/.web/routes/api/resources）。
// 受けるのは conf.ts に宣言した preset 名と系列名だけで、段取り（orchestrate.ts）の関数を呼ぶ。任意のコマンドは通さない。
import { PRESETS, SERIES } from './conf.ts';
import type { Status } from './spec.ts';
import { RUNS, launch, status } from './orchestrate.ts';

/** 宣言の学習と評価の状態と、起こした仕事のログの末尾。操作盤はこれを見て状態とボタンを出す。 */
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
