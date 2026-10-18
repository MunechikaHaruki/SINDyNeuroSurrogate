<script lang="ts">
  import { onMount } from 'svelte';
  import Dashboard from '$lib/Dashboard.svelte';
  import Card from '$lib/Card.svelte';
  import Choice from '$lib/Choice.svelte';
  import { declaredNames, listRuns, readStatus, runningCount, start } from './read/runs.ts';
  import { PRESETS } from './orchestrate/conf.ts';
  import type { IndexEntry, Job, JobState, State, Status } from './orchestrate/spec.ts';

  /** 状態を見に行く間隔（ms） */
  const POLL = 5000;
  const STATE_LABEL: Record<State, string> = { todo: '未実行', running: '計算中', done: '済み', unreplaceable: '置換できない' };
  const JOB_LABEL: Record<JobState['state'], string> = { running: '計算中', done: '済み', failed: '失敗' };

  let status = $state<Status>({ experiments: [], jobs: [] });
  let index = $state<IndexEntry[]>([]);
  let error = $state('');
  let experimentName = $state('');
  let kind = $state<'all' | IndexEntry['kind']>('all');

  let experiment = $derived(status.experiments.find(e => e.name === experimentName) ?? status.experiments[0]);
  // 名前を選んでいなければ最初の実験になるので、選択欄にもその名前を出す。
  $effect(() => {
    if (experiment && experimentName !== experiment.name) experimentName = experiment.name;
  });

  let presets = $derived(experiment ? experiment.presets.map(p => p.preset) : []);
  let seriesNames = $derived(experiment ? experiment.series.map(s => s.series) : []);
  /** 計算中と失敗の仕事。済んだ仕事は出さない。 */
  let shownJobs = $derived(status.jobs.filter(j => j.state !== 'done'));
  let names = $derived(declaredNames(status));
  let shownRuns = $derived(kind === 'all' ? index : index.filter(r => r.kind === kind));

  /** 実験の中の preset の学習の状態。 */
  function trainState(preset: string): State {
    return experiment?.presets.find(p => p.preset === preset)?.state ?? 'todo';
  }

  /** 実験の中の preset × 系列の組の状態。 */
  function pairState(series: string, preset: string): State {
    return experiment?.pairs.find(p => p.series === series && p.preset === preset)?.state ?? 'todo';
  }

  /** 系列の列に、評価を終えた組が一つでもあるか。あれば見出しから評価の図へ行ける。 */
  function anyDone(series: string): boolean {
    return experiment?.pairs.some(p => p.series === series && p.state === 'done') ?? false;
  }

  /** 設定の木を「spec.datasets.target」のような鍵と値の組に開く。配列は一つの値として扱う。 */
  function flatten(value: unknown, prefix: string, out: Record<string, unknown>): void {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      out[prefix] = value;
      return;
    }
    for (const key of Object.keys(value)) {
      const path = prefix === '' ? key : `${prefix}.${key}`;
      flatten((value as Record<string, unknown>)[key], path, out);
    }
  }

  /** 値を表に書く形にする。文字列はそのまま、項ライブラリは項の種類を並べる。 */
  function show(value: unknown): string {
    if (typeof value === 'string') return value;
    if (!Array.isArray(value)) return JSON.stringify(value);
    const items: string[] = [];
    for (const item of value) items.push(typeof item === 'object' && item !== null && 'type' in item ? String(item.type) : JSON.stringify(item));
    return items.join('\n');
  }

  /** 実験の preset の設定を鍵ごとに引く表。掃引で run ごとに違う値は「, 」で並べ、全 preset で同じ値の鍵は共通に回す。 */
  let settings = $derived.by(() => {
    const keys: string[] = [];
    const table: Record<string, Record<string, string>> = {};
    for (const preset of presets) {
      const values: Record<string, string[]> = {};
      for (const run of PRESETS[preset] ?? []) {
        const leaves: Record<string, unknown> = {};
        flatten(run.config, '', leaves);
        for (const key of Object.keys(leaves)) {
          if (!keys.includes(key)) keys.push(key);
          values[key] ??= [];
          if (!values[key].includes(show(leaves[key]))) values[key].push(show(leaves[key]));
        }
      }
      table[preset] = {};
      for (const key of Object.keys(values)) table[preset][key] = values[key].join(', ');
    }
    const varying: string[] = [];
    const common: { key: string; value: string }[] = [];
    for (const key of keys) {
      const seen: string[] = [];
      for (const preset of presets) if (!seen.includes(table[preset][key] ?? '')) seen.push(table[preset][key] ?? '');
      if (seen.length === 1 && presets.length > 1) common.push({ key, value: seen[0] });
      else varying.push(key);
    }
    return { varying, common, table };
  });

  /** 長い名前を _ の後ろで折り返せるようにする（宣言の名前は英数字と _ だけ）。 */
  function breakable(name: string): string {
    return name.replaceAll('_', '_<wbr>');
  }

  /** 別の冊へ、いま選んでいる実験と組を渡すリンク。 */
  function link(book: string, params: Record<string, string>): string {
    return `${book}?${new URLSearchParams({ experiment: experiment?.name ?? '', ...params }).toString()}`;
  }

  /** 仕事を一行で言う。 */
  function jobLabel(job: Job): string {
    if (job.action === 'train') return `学習 ${job.preset}`;
    return `評価 ${job.series} × ${job.preset}`;
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
    experimentName = new URLSearchParams(window.location.search).get('experiment') ?? '';
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

<Dashboard title="SINDy サロゲート：run の操作盤" summary="evaluation/orchestrate/conf.ts に宣言した実験（見比べる preset と系列の組）を選び、学習と評価を画面から回す。実験の組の状態とボタン、preset の設定、計算中・失敗した仕事のログの末尾、runs/ の全 run の一覧を並べる。">
  {#snippet sidebar()}
    {#if error}<p class="note" aria-live="polite">停止：{error}</p>{/if}
    <section>
      <h3>実験</h3>
      <Choice label="見比べる組" bind:value={experimentName} options={status.experiments.map(e => ({ value: e.name, label: e.name }))} />
    </section>
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

    <Card kicker="実験" title="学習と評価" subtitle="行は preset、列は学習とこの実験の系列">
      <table>
        <thead>
          <tr
            ><th>preset</th><th>学習</th>{#each seriesNames as series (series)}<th
                >{@html breakable(series)}{#if anyDone(series)}<br /><a href={link('53.05-evaluation', { series })}>図を見る</a>{/if}</th
              >{/each}</tr
          >
        </thead>
        <tbody>
          {#each presets as preset (preset)}
            {@const train = trainState(preset)}
            <tr>
              <td><code>{@html breakable(preset)}</code></td>
              <td>
                {#if train === 'done'}
                  <a href={link('53.04-trained-model', { preset })}>中身を見る</a>
                {:else if train === 'todo'}
                  <button type="button" onclick={() => void launch({ action: 'train', preset })}>学習する</button>
                {:else}
                  {STATE_LABEL[train]}
                {/if}
              </td>
              {#each seriesNames as series (series)}
                {@const state = pairState(series, preset)}
                <td>
                  {#if state === 'todo'}
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
        <p>掃引の preset は点ごとに学習 run を作り、全部そろうと「済み」になる。評価はまだ学習していない preset でも押せて、そのときは学習から始める。原系と置換系を回し、原系と比べた指標を置換系 run に書く。置換できない組（学習した区画の形と、置換する区画が合わない）は一度回すと分かり、以後は「置換できない」と出る。評価の図は系列ごとに一枚で、その実験の preset を並べて比べるので、系列の見出しから開く。</p>
      {/snippet}
    </Card>

    <Card kicker="実験" title="preset の設定" subtitle="evaluation/orchestrate/conf.ts の PRESETS が組んだ設定をそのまま並べる">
      <div class="wide">
        <table>
          <thead>
            <tr
              ><th>preset ＼ 鍵</th>{#each settings.varying as key (key)}<th>{@html key.replaceAll('.', '.<wbr>')}</th>{/each}</tr
            >
          </thead>
          <tbody>
            {#each presets as preset (preset)}
              <tr>
                <td><code>{@html breakable(preset)}</code></td>
                {#each settings.varying as key (key)}
                  <td
                    >{#if settings.table[preset]?.[key]}{#each settings.table[preset][key].split('\n') as line, i (i)}{#if i > 0}<br />{/if}<code>{line}</code>{/each}{/if}</td
                  >
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      {#if settings.common.length > 0}
        <p class="note">
          共通：{#each settings.common as item, i (item.key)}{#if i > 0}、{/if}<code>{item.key}</code> = <code>{item.value.replaceAll('\n', ', ')}</code>{/each}
        </p>
      {/if}
      {#snippet info()}
        <p>学習の設定の木（<code>SurrogateConfig</code>）を鍵ごとに開いたもの。preset が書かず、Python の署名の既定値に任せた鍵は出ない。この実験のどの preset でも同じ値の鍵は、表の下の「共通」に一度だけ書く。掃引の preset は点ごとに違う値を「, 」で並べる。</p>
      {/snippet}
    </Card>

    <Card kicker="一覧" title="runs/ の run" subtitle={`${shownRuns.length} 本（新しい順）。名前はどの実験にも当たらない run だけ、回したときの名前で出す`}>
      <table>
        <thead><tr><th>種類</th><th>名前</th><th>作った日時</th><th>場所</th></tr></thead>
        <tbody>
          {#each shownRuns as run (run.id)}
            <tr>
              <td>{run.kind}</td>
              <td
                >{#if run.id in names}{names[run.id]}{:else}<small>実験に無い：{run.name}</small>{/if}</td
              >
              <td>{run.created.slice(5, 16).replace('T', ' ')}</td>
              <td><code>{run.id}</code></td>
            </tr>
          {/each}
        </tbody>
      </table>
    </Card>
  {/snippet}
</Dashboard>
