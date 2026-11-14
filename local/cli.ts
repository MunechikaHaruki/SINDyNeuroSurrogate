// エージェント用の CLI。`SINDY_API=http://<host>:8787 bun local/cli.ts` で、実験を状態（結果あり・計算中・途中・未着手）ごとに、説明の最初の段落を添えてタブ区切りで書く。
// 冊と同じく使う側として計算する機械の API（口の形は shared/api.ts）を直に叩き、宣言（local/experiments/schema/）と API の実験の状態から組む。
import { EXPERIMENTS } from './experiments/schema/experiments.ts';
import type { Plan } from './experiments/schema/schema.ts';
import type { State, Status } from '../shared/spec.ts';

/** 実験の状態。 */
type Progress = '未着手' | '途中' | '計算中' | '結果あり';

/** 実験を並べる順。手を動かす所を先に。 */
const PROGRESS_ORDER: Progress[] = ['結果あり', '計算中', '途中', '未着手'];

/** 計算する機械の API の実験ごとの状態。届かないか口が断れば、その文で投げる。 */
async function readStatus(): Promise<Status> {
  const base = process.env.SINDY_API;
  if (!base) throw new Error('SINDY_API に計算する機械の API（http://<host>:8787）を渡す');
  const response = await fetch(`${base}/status`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
    signal: AbortSignal.timeout(60_000),
  });
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()) as Status;
}

/** 実験の状態を run から導く。置換できない組は済んだものとして数える。回す run が無ければ未着手。 */
function progressOf(experiment: Plan, status: Status): Progress {
  if (Object.keys(experiment.series).length === 0) return '未着手';
  const found = status.experiments.find(e => e.name === experiment.id);
  if (!found) throw new Error(`実験 ${experiment.id} の状態が無い`);
  const states: State[] = [...found.models.map(m => m.state), ...found.pairs.map(p => p.state)];
  if (states.includes('running')) return '計算中';
  if (!states.includes('todo')) return '結果あり';
  return states.includes('done') ? '途中' : '未着手';
}

if (import.meta.main) {
  const status = await readStatus();
  for (const progress of PROGRESS_ORDER)
    for (const e of EXPERIMENTS)
      if (progressOf(e, status) === progress) console.log(`${progress}\t${e.id}\t${e.description.split('\n\n')[0]}`);
}
