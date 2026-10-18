<script lang="ts">
  import { onMount } from 'svelte';
  import Dashboard from '$lib/Dashboard.svelte';
  import Card from '$lib/Card.svelte';
  import Choice from '$lib/Choice.svelte';
  import { listRuns, readStatus, runningCount, start } from './read/runs.ts';
  import type { IndexEntry, Job, JobState, State, Status } from './orchestrate/conf.ts';

  /** 状態を見に行く間隔（ms） */
  const POLL = 5000;
  const STATE_LABEL: Record<State, string> = { todo: '未実行', running: '計算中', done: '済み', unreplaceable: '置換できない' };
  const JOB_LABEL: Record<JobState['state'], string> = { running: '計算中', done: '済み', failed: '失敗' };

  let status = $state<Status>({ trains: [], pairs: [], jobs: [] });
  let index = $state<IndexEntry[]>([]);
  let error = $state('');
  let kind = $state<'all' | IndexEntry['kind']>('all');

  /** 宣言にある preset と系列の名前（表の行と列）。 */
  /** テスト専用の preset（_test_*）は出さない。 */
  let trains = $derived(status.trains.filter(t => !t.preset.startsWith('_test_')));
  let presets = $derived(trains.map(t => t.preset));
  let seriesNames = $derived([...new Set(status.pairs.map(p => p.series))]);
  /** 計算中と失敗の仕事。済んだ仕事は出さない。 */
  let shownJobs = $derived(status.jobs.filter(j => j.state !== 'done'));
  let names = $derived.by(() => {
    const table: Record<string, string> = {};
    for (const run of index) table[run.hash] = run.name;
    return table;
  });
  let shownRuns = $derived(kind === 'all' ? index : index.filter(r => r.kind === kind));
  /** 同じ種類・同じ名前の run のうち一番新しいもの。一覧は新しい順なので最初に出たもの。 */
  let latest = $derived.by(() => {
    const newest: Record<string, string> = {};
    for (const run of index) if (!(`${run.kind}/${run.name}` in newest)) newest[`${run.kind}/${run.name}`] = run.hash;
    return Object.values(newest);
  });

  /** preset × 系列の組の状態。 */
  function pairState(series: string, preset: string): State {
    for (const pair of status.pairs) if (pair.series === series && pair.preset === preset) return pair.state;
    return 'todo';
  }

  /** 長い系列名を _ の後ろで折り返せるようにする（宣言の名前は英数字と _ だけ）。 */
  function breakable(name: string): string {
    return name.replaceAll('_', '_<wbr>');
  }

  /** 仕事を一行で言う。 */
  function jobLabel(job: Job): string {
    if (job.action === 'train') return `学習 ${job.preset}`;
    return `評価 ${job.series} × ${job.preset}`;
  }

  /** 置換系 run が名指す学習 run と原系 run。 */
  function references(run: IndexEntry): string {
    if (run.spec.kind !== 'replaced') return '';
    const train = names[run.spec.train] ?? '';
    return `${train}（${run.spec.train.slice(0, 6)}）／原系 ${run.spec.original.slice(0, 6)}`;
  }

  async function refresh(): Promise<void> {
    status = await readStatus();
  }

  /** 仕事を起こして状態を読み直す。 */
  async function launch(job: Job): Promise<void> {
    try {
      await start(job);
      await refresh();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  onMount(() => {
    void (async () => {
      index = await listRuns();
      await refresh();
    })().catch(e => (error = e instanceof Error ? e.message : String(e)));
    // 回っているものが減ったら、回し終えた run が index.json に載ったので読み直す。
    const timer = setInterval(() => {
      void (async () => {
        const before = runningCount(status);
        await refresh();
        if (runningCount(status) < before) index = await listRuns();
      })().catch(e => (error = e instanceof Error ? e.message : String(e)));
    }, POLL);
    return () => clearInterval(timer);
  });
</script>

<Dashboard title="SINDy サロゲート：run の操作盤" summary="evaluation/orchestrate/conf.ts に宣言した学習と評価を画面から回す。preset ごとの学習、preset × 系列の評価の状態とボタン、計算中・失敗した仕事のログの末尾、runs/ の全 run の一覧を並べる。">
  {#snippet sidebar()}
    {#if error}<p class="note" aria-live="polite">停止：{error}</p>{/if}
    <section>
      <h3>状態</h3>
      <p class="note">計算中 {runningCount(status)} 件、失敗 {status.jobs.filter(j => j.state === 'failed').length} 件、run {index.length} 本。{POLL / 1000} 秒ごとに読み直す。</p>
      <p class="note">回す条件は <code>evaluation/orchestrate/conf.ts</code> の宣言だけで、ここでは選んで押すだけ。同じ条件の run があれば回さない。</p>
    </section>
    <section>
      <h3>run の一覧</h3>
      <Choice
        label="種類"
        bind:value={kind}
        options={[
          { value: 'all', label: 'すべて' },
          { value: 'train', label: '学習' },
          { value: 'original', label: '原系' },
          { value: 'replaced', label: '置換系' },
        ]}
      />
    </section>
  {/snippet}

  {#snippet left()}
    <Card kicker="仕事" title="計算中と失敗" subtitle={shownJobs.length > 0 ? `${shownJobs.length} 件` : 'いま回っているものも、失敗したものも無い'}>
      {#each shownJobs as job (job.started)}
        <figure>
          <figcaption>{jobLabel(job.job)} <small>{JOB_LABEL[job.state]}・{job.started.slice(0, 19).replace('T', ' ')}</small></figcaption>
          <pre>{job.log.join('\n')}</pre>
          {#if job.state === 'failed'}
            <button type="button" onclick={() => void launch(job.job)}>もう一度回す</button>
          {/if}
        </figure>
      {/each}
      {#snippet info()}
        <p>操作盤から起こした仕事は <code>runs/.jobs/</code> に、仕事の中身・pid・ログ・終了コードを置く。プロセスが生きていれば計算中、終了コードが 0 なら済み、それ以外か終わりを書かずに消えたものは失敗。ログは末尾の 30 行。もう一度回すと、同じ仕事の失敗の記録は消える。</p>
      {/snippet}
    </Card>

    <Card kicker="学習" title="preset ごとの学習" subtitle="学習 run を用意するだけで、評価はしない">
      <table>
        <thead><tr><th>preset</th><th>状態</th><th></th></tr></thead>
        <tbody>
          {#each trains as train (train.preset)}
            <tr>
              <td><code>{train.preset}</code></td>
              <td>{STATE_LABEL[train.state]}</td>
              <td><button type="button" disabled={train.state !== 'todo'} onclick={() => void launch({ action: 'train', preset: train.preset })}>学習する</button></td>
            </tr>
          {/each}
        </tbody>
      </table>
      {#snippet info()}
        <p>掃引の preset は点ごとに学習 run を作り、全部そろうと「済み」になる。</p>
      {/snippet}
    </Card>

    <Card kicker="評価" title="preset × 系列の評価" subtitle="原系と置換系を回し、原系と比べた指標を置換系 run に書く">
      <table>
        <thead>
          <tr><th>preset ＼ 系列</th>{#each seriesNames as series (series)}<th>{@html breakable(series)}</th>{/each}</tr>
        </thead>
        <tbody>
          {#each presets as preset (preset)}
            <tr>
              <td><code>{preset}</code></td>
              {#each seriesNames as series (series)}
                {@const state = pairState(series, preset)}
                <td>
                  {#if state === 'done'}
                    <a href={`53.04-evaluation?series=${encodeURIComponent(series)}`}>図を見る</a>
                  {:else if state === 'todo'}
                    <button type="button" onclick={() => void launch({ action: 'eval', series, preset })}>回す</button>
                  {:else}
                    {STATE_LABEL[state]}
                  {/if}
                </td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
      {#snippet info()}
        <p>まだ学習していない preset でも押せる。そのときは学習から始める。置換できない組（学習した区画の形と、置換する区画が合わない）は一度回すと分かり、以後は「置換できない」と出る。済んだ組は別冊「SINDy サロゲート：評価の比較」の図で見る。</p>
      {/snippet}
    </Card>

    <Card kicker="一覧" title="runs/ の run" subtitle={`${shownRuns.length} 本（新しい順）。● は同じ種類・同じ名前の中で一番新しいもの`}>
      <table>
        <thead><tr><th>種類</th><th>名前</th><th>作った日時</th><th>ハッシュ</th><th>学習／原系</th></tr></thead>
        <tbody>
          {#each shownRuns as run (run.hash)}
            <tr>
              <td>{run.kind}</td>
              <td>{latest.includes(run.hash) ? '● ' : ''}{run.name}</td>
              <td>{run.created.slice(5, 16).replace('T', ' ')}</td>
              <td><code>{run.hash}</code></td>
              <td>{references(run)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </Card>
  {/snippet}
</Dashboard>
