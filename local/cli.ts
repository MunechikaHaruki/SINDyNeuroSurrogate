// エージェント用の CLI。`SINDY_API=http://<host>:8787 bun local/cli.ts` で、実験を状態（結果あり・計算中・途中・未着手）ごとに、説明の最初の段落を添えてタブ区切りで書く。
// 冊と同じく使う側として計算する機械の API（口の形は shared/api.ts）の実験の目次を読むだけで、組み立ては計算する機械が済ませている。
import type { Experiments, Progress } from '../shared/api.ts';

/** 実験を並べる順。手を動かす所を先に。 */
const PROGRESS_ORDER: Progress[] = ['結果あり', '計算中', '途中', '未着手'];

/** 計算する機械の API の実験の目次。届かないか口が断れば、その文で投げる。 */
async function readExperiments(): Promise<Experiments> {
  const base = process.env.SINDY_API;
  if (!base) throw new Error('SINDY_API に計算する機械の API（http://<host>:8787）を渡す');
  const response = await fetch(`${base}/experiments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
    signal: AbortSignal.timeout(60_000),
  });
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()) as Experiments;
}

if (import.meta.main) {
  const all = (await readExperiments()).groups.flatMap(g => g.experiments);
  for (const progress of PROGRESS_ORDER)
    for (const e of all) if (e.progress === progress) console.log(`${progress}\t${e.id}`);
}
