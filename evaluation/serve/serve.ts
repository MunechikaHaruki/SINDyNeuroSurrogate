// 計算する機械の上で runs の隣に立ち、結果を小さい JSON にして返す API（Bun.serve、POST /<口>、既定のポート 8787）。口の形は api.ts。
// 波形は回線に出さず、間引いた線・軌跡・分布・指標にしてから返す。run は書き換わらないので、読んだファイルと指標は覚えておく。
// 回すのは段取り（orchestrate.ts）の関数だけで、任意のコマンドは通さない。runs の置き場は SINDY_RUNS（イメージでは /runs）。
//   bun evaluation/serve/serve.ts
import { readFileSync } from 'node:fs';
import { join, normalize } from 'node:path';
import { RUNS, launch, scan, status } from '../orchestrate/orchestrate.ts';
import { SERIES } from '../orchestrate/conf.ts';
import { EXPERIMENTS, EXPERIMENT_RUNS } from '../../experiments/experiments.ts';
import { FORMAT, type ExperimentStatus, type IndexEntry, type Job } from '../orchestrate/spec.ts';
import { rowsOf, type Claim, type Column, type Judgement, type Row } from '../../experiments/common.ts';
import { coincidence, compare, crossingDiff, crossings, type PairMetrics, type SpikesFile } from '../read/compare.ts';
import { histogram, type Bin } from '../read/model-json.ts';
import { decimate, path, variablesOf, type Point, type WaveMeta } from '../read/waves.ts';
import type { Api, ArrayRef, ClaimJudgement, CrossingRow, SeriesMetrics, SeriesRequest } from './api.ts';

const PORT = Number(process.env.SINDY_PORT ?? 8787);
/** 覚えておく .f64 の合計の上限 [バイト]。超えたら古く使ったものから忘れる。 */
const BUDGET = 2 ** 31;

/** 回線に出す線の点。表示に要る桁だけに丸める（時刻は 0.01 ms の刻みが 1 万 ms まで残る 7 桁、値は 6 桁）。 */
function compact(points: Point[]): Point[] {
  return points.map(p => ({ x: Number(p.x.toPrecision(7)), y: Number(p.y.toPrecision(6)) }));
}

type Handlers = { [Op in keyof Api]: (body: Api[Op]['body']) => Api[Op]['reply'] };

/** 仕事の欄を、段取りが知る形だけに組み直す（余分な欄で仕事の鍵が変わり、重複の防止をすり抜けないように）。名前は launch が確かめる。 */
function checkJob(job: Job): Job {
  if (job?.action === 'train') return { action: 'train', model: job.model };
  if (job?.action === 'eval') return { action: 'eval', series: job.series, model: job.model };
  throw new Error(`知らない仕事 ${JSON.stringify(job)}`);
}

/** runs の置き場 1 つの口。読んだ .f64 と指標はこの中で覚える。 */
export function handlers(runs: string): Handlers {
  const arrays = new Map<string, Float64Array>();
  const metrics = new Map<string, SeriesMetrics>();
  let held = 0;

  /** run の下のファイルの絶対パス。runs/ の外を指すものは弾く。 */
  function inRuns(run: string, file: string): string {
    const relative = normalize(join(run, file));
    if (relative.startsWith('..') || relative.startsWith('/')) throw new Error(`${run}/${file} は runs の外`);
    return join(runs, relative);
  }

  /** run の中の .f64 を丸ごと。 */
  function f64(run: string, file: string): Float64Array {
    const key = `${run}/${file}`;
    const hit = arrays.get(key);
    if (hit) {
      arrays.delete(key);
      arrays.set(key, hit);
      return hit;
    }
    if (!file.endsWith('.f64')) throw new Error(`${key} は .f64 でない`);
    const bytes = readFileSync(inRuns(run, file));
    const values = new Float64Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    arrays.set(key, values);
    held += values.byteLength;
    for (const [old, array] of arrays) {
      if (held <= BUDGET || old === key) break;
      arrays.delete(old);
      held -= array.byteLength;
    }
    return values;
  }

  function column(ref: ArrayRef): Float64Array {
    const all = f64(ref.run, ref.file);
    const { column: at, n } = ref;
    if (
      !Number.isInteger(at) ||
      at < 0 ||
      !Number.isInteger(n) ||
      n <= 0 ||
      all.length % n !== 0 ||
      all.length < (at + 1) * n
    )
      throw new Error(`${ref.run}/${ref.file}（${all.length} 個）に ${at} 列目（${n} 点ずつ）が無い`);
    return all.subarray(ref.column * ref.n, (ref.column + 1) * ref.n);
  }

  function json<T>(run: string, file: string): T {
    if (!file.endsWith('.json')) throw new Error(`${run}/${file} は .json でない`);
    return JSON.parse(readFileSync(inRuns(run, file), 'utf8')) as T;
  }

  /** 1 点の 1 区画の膜電位。 */
  function voltage(run: string, meta: WaveMeta, point: number, compId: number): Float64Array {
    return column({
      run,
      file: `waves/${point}/comp${compId}.f64`,
      column: variablesOf(meta, compId).indexOf('V'),
      n: meta.n[point],
    });
  }

  /** 系列の、列ごと・点ごとの原系との指標と、0 mV の横切りで数えた差と Γ、発火の時刻。 */
  function seriesMetrics(request: SeriesRequest): SeriesMetrics {
    const key = JSON.stringify(request);
    const hit = metrics.get(key);
    if (hit) return hit;
    const meta = json<WaveMeta>(request.original, 'waves/meta.json');
    const compId = meta.names ? meta.names[request.comp] : 0;
    if (compId === undefined)
      throw new Error(`${request.original} に区画 ${request.comp} が無い（${Object.keys(meta.names!).join('、')}）`);
    const original = meta.n.map((_, k) => voltage(request.original, meta, k, compId));
    const originalSpikes = json<SpikesFile>(request.original, 'spikes.json')[request.comp];
    const result: SeriesMetrics = {
      pairs: {},
      crossings: {},
      times: { original: original.map(v => crossings(v, meta.dt).times), columns: {} },
    };
    for (const run of request.columns) {
      const columnMeta = json<WaveMeta>(run, 'waves/meta.json');
      const spikes = json<SpikesFile>(run, 'spikes.json')[request.comp];
      const pairs: PairMetrics[] = [];
      const rows: CrossingRow[] = [];
      const times: number[][] = [];
      for (let k = 0; k < meta.n.length; k++) {
        const v = voltage(run, columnMeta, k, compId);
        if (originalSpikes && spikes) pairs.push(compare(original[k], v, originalSpikes[k], spikes[k], meta.dt));
        const before = crossings(original[k], meta.dt);
        const after = crossings(v, meta.dt);
        rows.push({
          ...crossingDiff(before, after),
          gamma: coincidence(before.times, after.times, original[k].length * meta.dt),
        });
        times.push(after.times);
      }
      if (originalSpikes && spikes) result.pairs[run] = pairs;
      result.crossings[run] = rows;
      result.times.columns[run] = times;
    }
    metrics.set(key, result);
    return result;
  }

  /** 列ごとの分布。区間は全部の列の有限な値の最小から最大。 */
  function histograms(refs: ArrayRef[], bins: number): Bin[][] {
    const values = refs.map(column);
    let low = Infinity;
    let high = -Infinity;
    for (const array of values)
      for (const v of array) {
        if (!Number.isFinite(v)) continue;
        low = Math.min(low, v);
        high = Math.max(high, v);
      }
    if (!(low < high)) return values.map(() => []);
    return values.map(array => histogram(array, low, high, bins));
  }

  /** 実験の系列 1 つを、置き換えて回し終えた列ごとの判定の値にする。元のモデルをまだ回していなければ投げる。 */
  function table(index: IndexEntry[], found: ExperimentStatus, name: string): Row[] {
    const original = found.series.find(s => s.series === name)?.original;
    const entry = index.find(r => r.id === original?.id);
    if (!original || !entry || entry.spec.kind !== 'sim')
      throw new Error(`${found.name} の ${name} の元のモデルをまだ回していない`);
    const columns: Column[] = [];
    for (const pair of found.pairs) {
      if (pair.series !== name || pair.run.state !== 'done') continue;
      columns.push({ run: pair.run.id, model: pair.model });
    }
    const request = { original: original.id, columns: columns.map(c => c.run), comp: SERIES[name].replace_targets[0] };
    return rowsOf(columns, entry.spec.input.values, seriesMetrics(request).crossings);
  }

  /** 主張 1 つを判定する。 */
  function judge(index: IndexEntry[], found: ExperimentStatus, claim: Claim): Judgement[] {
    const tables: Row[][] = [];
    for (const name of claim.series) tables.push(table(index, found, name));
    return claim.judge(tables, claim.groupings);
  }

  /** 実験の主張ごとの判定。判定できない主張は理由を添える。 */
  function claims(): ClaimJudgement[] {
    const index = scan(runs);
    const current = status(runs, EXPERIMENT_RUNS);
    const list: ClaimJudgement[] = [];
    for (const e of EXPERIMENTS) {
      const found = current.experiments.find(x => x.name === e.id)!;
      for (const c of e.claims) {
        const base = { experiment: e.id, claim: c.claim, name: c.name };
        try {
          list.push({ ...base, rows: judge(index, found, c) });
        } catch (err) {
          list.push({ ...base, rows: null, error: err instanceof Error ? err.message : String(err) });
        }
      }
    }
    return list;
  }

  return {
    index: () => scan(runs).filter(e => e.spec.format === FORMAT),
    status: () => status(runs, EXPERIMENT_RUNS),
    claims,
    launch: body => ({ started: launch(runs, checkJob(body.job)) }),
    json: body => json(body.run, body.path),
    traces: body => body.requests.map(r => compact(decimate(column(r.array), r.t0, r.dt, r.start, r.end, r.buckets))),
    paths: body => body.requests.map(r => compact(path(column(r.x), column(r.y), r.t0, r.dt, r.start, r.end, r.max))),
    histograms: body => histograms(body.arrays, body.bins),
    series: seriesMetrics,
  };
}

/** POST /<口> の JSON を口に渡す応答。口が投げた誤りは 400 で文を返す。 */
export function server(runs: string): (request: Request) => Promise<Response> {
  const ops = handlers(runs);
  return async request => {
    const op = new URL(request.url).pathname.slice(1);
    if (request.method !== 'POST') return new Response('POST だけ受ける', { status: 405 });
    if (!Object.hasOwn(ops, op)) return new Response(`知らない口 ${op}`, { status: 404 });
    // JSON だけ受ける。ブラウザで開いた他のサイトが text/plain の POST で仕事を起こせないよう、preflight の要る形に限る
    if (!request.headers.get('content-type')?.startsWith('application/json'))
      return new Response('JSON だけ受ける', { status: 415 });
    try {
      const body = await request.json();
      return Response.json((ops[op as keyof Api] as (body: unknown) => unknown)(body));
    } catch (e) {
      return new Response(e instanceof Error ? e.message : String(e), { status: 400 });
    }
  };
}

if (import.meta.main) {
  Bun.serve({ port: PORT, hostname: '0.0.0.0', fetch: server(RUNS) });
  console.log(`runs ${RUNS} を :${PORT} で返す`);
}
