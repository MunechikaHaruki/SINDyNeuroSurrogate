// 計算する機械に問い合わせる口。計算する機械の API（remote/serve/serve.ts。口の形は shared/api.ts）を手元から呼び、run の小さい JSON、間引いた線・軌跡・分布、指標だけを受け取る（波形は手元に運ばない）。仕事を起こすのもここ。
// 実験の状態は宣言の順（../resolve/experiments.ts）に並べ直して返す。口への届け方は使う側が connect で差す（CLI は direct で API へ直に、画面は自分の中継へ）。run は書き換わらないので、答えは覚えておく。
import { EXPERIMENTS } from '../resolve/experiments.ts';
import type { IndexEntry, Job, Status } from '../../shared/spec.ts';
import type {
  Api,
  ArrayRef,
  Bin,
  PathRequest,
  Point,
  SeriesMetrics,
  SeriesRequest,
  TraceRequest,
} from '../../shared/api.ts';
import type { WaveMeta } from '../../shared/runs.ts';

/** API の口 op を 1 つ呼ぶ。届かないか口が断れば、その文で投げる。 */
export type Call = <Op extends keyof Api>(op: Op, body: Api[Op]['body']) => Promise<Api[Op]['reply']>;

/** 指標は大きい run を読むので待つ [ms]。 */
const TIMEOUT = 120_000;

/** API（base は `http://<host>:8787` の形）へ直に届ける。 */
export function direct(base: string): Call {
  return async (op, body) => {
    let response: Response;
    try {
      response = await fetch(`${base}/${op}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(TIMEOUT),
      });
    } catch (e) {
      throw new Error(`計算する機械（${base}）に届かない：${e instanceof Error ? e.message : String(e)}`);
    }
    if (!response.ok) throw new Error(await response.text());
    return await response.json();
  };
}

let transport: Call | undefined;

/** 口への届け方を差す。読む前に一度だけ呼ぶ。 */
export function connect(to: Call): void {
  transport = to;
}

/** 差した届け方で口 op を呼ぶ。 */
export function call<Op extends keyof Api>(op: Op, body: Api[Op]['body']): Promise<Api[Op]['reply']> {
  if (!transport) throw new Error('研究コードの API への届け方を connect で差していない');
  return transport(op, body);
}

/** run は書き換わらないので、同じ問い合わせの答え（run の JSON、分布、指標）を覚えておく。失敗は覚えない。 */
const answered = new Map<string, Promise<unknown>>();

function remembered<Op extends keyof Api>(op: Op, body: Api[Op]['body']): Promise<Api[Op]['reply']> {
  const key = JSON.stringify([op, body]);
  if (!answered.has(key)) {
    const reply = call(op, body);
    reply.catch(() => answered.delete(key));
    answered.set(key, reply);
  }
  return answered.get(key) as Promise<Api[Op]['reply']>;
}

/** run の一覧（新しい順）。今の形式の run だけ。まだ何も回していなければ空。 */
export function listRuns(): Promise<IndexEntry[]> {
  return call('index', {});
}

/** 学習か評価を切り離して回す。同じ組が回っていれば何もしない。 */
export async function start(job: Job): Promise<void> {
  await call('launch', { job });
}

/** run の小さい JSON 1 つ。 */
export function readJson<T>(run: string, path: string): Promise<T> {
  return remembered('json', { run, path }) as Promise<T>;
}

/** 時間の図の線（間引いたもの）。まとめて 1 往復で頼む。表示の範囲ごとに変わるので覚えない。 */
export function readTraces(requests: TraceRequest[]): Promise<Point[][]> {
  return call('traces', { requests });
}

/** 相平面の軌跡。表示の範囲ごとに変わるので覚えない。 */
export function readPaths(requests: PathRequest[]): Promise<Point[][]> {
  return call('paths', { requests });
}

/** 列ごとの分布（区間は全部の列で揃える）。 */
export function readHistograms(arrays: ArrayRef[], bins: number): Promise<Bin[][]> {
  return remembered('histograms', { arrays, bins });
}

/** 系列の、列ごと・点ごとの原系との指標。 */
export function readMetrics(request: SeriesRequest): Promise<SeriesMetrics> {
  return remembered('series', request);
}

/** 時間の図の 1 本を、表示の範囲と区間の数で頼む形にする。 */
export function trace(array: ArrayRef, meta: WaveMeta, domain: [number, number], buckets: number): TraceRequest {
  return { array, t0: meta.t0, dt: meta.dt, start: domain[0], end: domain[1], buckets };
}

/** 計算する機械が名前の順で返す状態を、宣言の順（../resolve/experiments.ts）に並べ直す。宣言に無いものは後ろに名前の順で残す。 */
export function ordered(status: Status): Status {
  const rank = <T>(list: T[], key: (x: T) => string, order: string[]) =>
    [...list].sort((a, b) => {
      const [i, j] = [order.indexOf(key(a)), order.indexOf(key(b))];
      return (i < 0 ? Infinity : i) - (j < 0 ? Infinity : j);
    });
  const ids = EXPERIMENTS.map(e => e.id);
  const experiments = rank(status.experiments, e => e.name, ids).map(found => {
    const plan = EXPERIMENTS.find(e => e.id === found.name);
    if (!plan) return found;
    const pairs = plan.pairs.map(p => `${p.series} ${p.model}`);
    return {
      ...found,
      models: rank(
        found.models,
        m => m.model,
        plan.models.map(m => m.name),
      ),
      series: rank(found.series, s => s.series, Object.keys(plan.series)),
      pairs: rank(found.pairs, p => `${p.series} ${p.model}`, pairs),
      benches: rank(
        found.benches,
        b => b.bench,
        plan.benches.map(b => b.name),
      ),
    };
  });
  return { ...status, experiments };
}

/** 研究コードの実験ごとの状態と、起こした仕事。並びは宣言の順。 */
export async function readStatus(): Promise<Status> {
  return ordered(await call('status', {}));
}

/** 回っているものの数。減ったら何かが回し終わったので、run の一覧を読み直す。 */
export function runningCount(status: Status): number {
  let count = 0;
  for (const experiment of status.experiments) {
    for (const model of experiment.models) if (model.state === 'running') count++;
    for (const pair of experiment.pairs) if (pair.state === 'running') count++;
  }
  return count;
}
