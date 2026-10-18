<script lang="ts">
  import Mermaid from '$lib/Mermaid.svelte';
  import Page from '$lib/Page.svelte';
  import Section from '$lib/Section.svelte';

  /** 層の依存。矢印は「import する」向き。tests/test_conventions.py の _LAYERS から推移的な辺を除いたもの。 */
  const layers = `flowchart RL
    base["base<br/>core/ と __init__"]
    neurons["neurons<br/>neurons/"]
    sim_desc["sim_desc<br/>sim/ の spec・result・waveform"]
    surrogate["surrogate<br/>surrogate/"]
    sim_exec["sim_exec<br/>sim/run"]
    neurons --> base
    sim_desc --> neurons
    surrogate --> sim_desc
    sim_exec --> surrogate
  `;

  /** repo の中の入口と、結果の置き場。 */
  const flow = `flowchart LR
    conf["53-evaluation/conf.ts<br/>何を回すか"] --> orchestrate["53-evaluation/orchestrate.ts<br/>just train と just eval"]
    orchestrate -- spec.json --> run["neurosurrogate/run.py<br/>計算"]
    run --> domain["neurosurrogate/<br/>ドメイン層"]
    run --> runs[("runs/&lt;ハッシュ&gt;/")]
    orchestrate -- index.json --> runs
    runs -- 開発サーバー --> research["research/<br/>図"]
  `;
</script>

<Page title="SINDyNeuroSurrogate の構成" summary="研究コードの層と依存の向き、学習と評価の段取りから runs/ を経て図に届くまでの流れ。">
  <Section id="sec-1" title="層の依存">
    <p>ドメイン層（<code>neurosurrogate/</code>）は 5 つの層に分かれ、矢印の向きにだけ import できる。左の層ほど基盤で、<code>base</code> は他の層を一切 import しない。許可は推移的にも明示してあり、図では直接の辺だけを描く。この向きは <code>tests/test_conventions.py</code> の <code>_LAYERS</code> が検査する。</p>
    <Mermaid label="ドメイン層の依存" code={layers} />
  </Section>
  <Section id="sec-2" title="入口と結果の流れ">
    <p>TS は <code>research/</code>、Python は <code>neurosurrogate/</code> に置く。何を回すかは <code>53-evaluation/conf.ts</code> が型付きで宣言し、段取り（<code>53-evaluation/orchestrate.ts</code>、bun で実行）がそこから spec.json を組んでハッシュを取る。同じ spec は同じハッシュなので、<code>runs/&lt;ハッシュ&gt;/</code> があれば回さない。重い計算は Python の <code>neurosurrogate/run.py</code> だけが持ち、spec.json を 1 つ受け取って run のディレクトリを埋める。<code>run.py</code> は上の 5 層のどれからも import されない最上位の層（<code>entry</code>）で、run の保存形式を知るのはここだけ。図は Python を import せず、開発サーバーが配る <code>runs/</code> のデータだけを読む。</p>
    <Mermaid label="入口から図までの流れ" code={flow} />
  </Section>
</Page>
