<script lang="ts">
  import { eq, tex } from '$lib/tex.ts';
  import Dashboard from '$lib/Dashboard.svelte';
  import Card from '$lib/Card.svelte';
  import Range from '$lib/Range.svelte';
  import { equilibria, psiPartitions, type GateName, type Partition } from './gate-reduction-model.ts';

  /** 電位だけで動く 6 ゲート。s,r は Ca 系と一緒に元のまま残す。 */
  const panelGates: GateName[] = ['m', 'h', 'n', 'a', 'b', 'c'];
  /** φ 群の選び方。φ に入れられるのは m だけ（比較図「φ 群」）。 */
  const phiChoices: GateName[][] = [['m'], []];
  /** I₀ = 0 の安定な平衡点（静止電位）。式(10)はここで読む。 */
  const rest = equilibria(0)[0];

  /** 評価するモデル一つ：φ 群と、残りのゲートの ψ 群の組み分け。 */
  type Model = { phi: GateName[]; partition: Partition };

  /** φ 群の選び方ごとに、残りのゲートの組み分けをすべて並べる。 */
  function allModels(): Model[] {
    const list: Model[] = [];
    for (const phi of phiChoices) {
      const remaining: GateName[] = [];
      for (const name of panelGates) if (!phi.includes(name)) remaining.push(name);
      for (const partition of psiPartitions(remaining, rest)) list.push({ phi, partition });
    }
    return list;
  }
  /** 全モデルを、式(10)の小さい順、同じなら変数の少ない順に並べておく。ε で絞っても順は変わらない。 */
  const models = allModels().sort(byCriterion);

  /** 式(10)の左辺の目安。Kepler は値を定めていない。 */
  let epsilon = $state(0.6);
  /** ε 以下のモデル。 */
  let shown = $derived(models.filter(model => model.partition.worst <= epsilon));

  /** 並べ順：式(10)の小さい順、同じなら変数の少ない順。 */
  function byCriterion(x: Model, y: Model): number {
    return x.partition.worst - y.partition.worst || x.partition.blocks.length - y.partition.blocks.length;
  }
  /** モデルを「Φ{m}、ψ{h,n}、ψ{a,c}、b」の形に書く。 */
  function modelLabel(model: Model): string {
    const parts: string[] = [];
    if (model.phi.length > 0) parts.push(`Φ{${model.phi.join(',')}}`);
    for (const block of model.partition.blocks) parts.push(block.length === 1 ? block[0] : `ψ{${block.join(',')}}`);
    return parts.join('、');
  }
</script>

<Dashboard title="Traub 単一区画：縮約モデル群の評価" summary="比較図で作れる縮約モデルを並べ、元の 11 状態系をどの程度再現できているかを評価する。まず基本的な力学的特徴で足切りし、残ったモデルの忠実さを比べる。" complete={false}>
  {#snippet sidebar()}
    <p class="note">縮約モデルの作り方と群分けは <a href="gate-reduction-comparison">別冊「Traub 単一区画：ゲートの組を選び、縮約と比べる」</a> にある。</p>
    <section>
      <h3>評価するモデル</h3>
      <Range label="式(10) の目安 ε" bind:value={epsilon} min={0.01} max={1} step={0.01} display={epsilon.toFixed(2)} />
      <p class="note">φ 群に m を入れるか入れないかと、残りのゲートの ψ 群の組み分けを総当たりで作る（全 {models.length} 通り）。そのうち、静止電位での式(10)の左辺がすべて ε 以下のもの {shown.length} 通りを並べる。</p>
    </section>
  {/snippet}

  {#snippet left()}
    <Card kicker="一段目" title="足切り" subtitle={`ε = ${epsilon.toFixed(2)} 以下の ${shown.length} 通り`}>
      <table>
        <thead><tr><th>群分け</th><th>変数</th><th>式(10)</th></tr></thead>
        <tbody>
          {#each shown as model (modelLabel(model))}
            <tr>
              <td>{modelLabel(model)}</td>
              <td>{model.partition.blocks.length}</td>
              <td>{model.partition.worst > 0 ? model.partition.worst.toPrecision(2) : '—'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
      {#snippet info()}
        {@html eq`\Big|\sum_{j\in\Psi_k}\alpha_jk_j-k_i\Big|\,\Big[\sum_{j\in\Psi_k}k_j\Big]^{-1}\ll1 \quad (i\in\Psi_k)`}
        <p>式(10)の列は、2 つ以上のゲートの ψ 群すべてでの左辺の最大。静止電位（{@html tex`I_0=0`} の安定な平衡点、{rest.toFixed(1)} mV）で読む。ψ 群がなければ —。変数は 6 ゲートが縮約後に何本になるか。</p>
      {/snippet}
    </Card>

    <Card kicker="二段目" title="忠実さ" subtitle="足切りを通ったモデルが、元のモデルをどの程度再現するか">
      <p class="note">指標はまだない。</p>
    </Card>
  {/snippet}
</Dashboard>
