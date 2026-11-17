// 計算する機械の側の通信。runs の隣に立つ API（Bun.serve、POST /<口>、既定のポート 8787）。口の形は shared/api.ts で、ここは口を回す側（run/）と答えを作る側（read/）の関数に渡すだけ。
// 口は実験の目次と実験 1 つ、run の置き場ごとの答え（inputs・models・data・parts・bench）、run の一覧、仕事を起こす口。run の保存の形を知るのはこちらだけで、
// 波形は回線に出さず、間引いた線・軌跡・分布・指標にしてから返す。宣言は読まない。知っているのは resolved/（spec と宣言の写し plans.json）と runs だけ。回すのは段取りの関数だけで、任意のコマンドは通さない。
// runs の置き場は SINDY_RUNS（イメージでは /runs）。
//   bun remote/serve.ts
import type { Job } from '../shared/spec.ts';
import type { Api } from '../shared/api.ts';
import { bench } from './read/bench.ts';
import { seriesComparison } from './read/comparison.ts';
import { data } from './read/data.ts';
import { type Context, experiment, experiments, registry } from './read/experiments.ts';
import { inputs } from './read/inputs.ts';
import { models } from './read/models.ts';
import { parts } from './read/parts.ts';
import { runFiles } from './read/waves.ts';
import { launch } from './run-spec.ts';
import { RUNS } from './run/places.ts';

const PORT = Number(process.env.SINDY_PORT ?? 8787);

type Handlers = { [Op in keyof Api]: (body: Api[Op]['body']) => Api[Op]['reply'] };

/** 仕事の欄を、段取りが知る形だけに組み直す（余分な欄で仕事の鍵が変わり、重複の防止をすり抜けないように）。パスは launch が確かめる。 */
function checkJob(job: Job): Job {
  if (typeof job?.spec === 'string') return { spec: job.spec };
  throw new Error(`知らない仕事 ${JSON.stringify(job)}`);
}

/** runs の置き場 1 つの口。読んだ .f64・特徴量・指標はこの中で覚える。 */
export function handlers(runs: string): Handlers {
  const files = runFiles(runs);
  const ctx: Context = { runs, files, metrics: seriesComparison(files) };
  return {
    experiments: () => experiments(runs),
    experiment: body => experiment(runs, body.experiment),
    inputs: body => inputs(ctx, body),
    models: body => models(ctx, body),
    data: body => data(ctx, body),
    parts: body => parts(ctx, body),
    bench: body => bench(ctx, body),
    registry: () => registry(runs),
    launch: body => ({ started: launch(runs, checkJob(body.job)) }),
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
