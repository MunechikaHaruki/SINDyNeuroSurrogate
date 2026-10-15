<script lang="ts">
  import Mermaid from '$lib/Mermaid.svelte';
  import Page from '$lib/Page.svelte';

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
    conf["scripts/conf<br/>Hydra の設定"] --> main["scripts/main.py<br/>学習"]
    catalog["scripts/catalog.py<br/>何を回すか"] --> evaluate["scripts/evaluate.py<br/>just eval"]
    main --> domain["neurosurrogate/<br/>ドメイン層"]
    evaluate --> domain
    main --> io["scripts/mlflow_io/"]
    evaluate --> io
    io --> mlflow[("MLflow")]
    mlflow -- REST --> research["research/<br/>図"]
  `;
</script>

<Page title="SINDyNeuroSurrogate の構成" summary="研究コードの層と依存の向き、学習と評価の入口から MLflow を経て図に届くまでの流れ。" complete={false}>
  <h2>層の依存</h2>
  <p>ドメイン層（<code>neurosurrogate/</code>）は 5 つの層に分かれ、矢印の向きにだけ import できる。左の層ほど基盤で、<code>base</code> は他の層を一切 import しない。許可は推移的にも明示してあり、図では直接の辺だけを描く。この向きは <code>tests/test_conventions.py</code> の <code>_LAYERS</code> が検査する。</p>
  <Mermaid label="ドメイン層の依存" code={layers} />

  <h2>入口と結果の流れ</h2>
  <p>重い計算は CLI の入口（学習と <code>just eval</code>）だけが持ち、結果は <code>scripts/mlflow_io/</code> を通して MLflow に置く。ドメイン層は MLflow も Hydra も知らない。図は Python を import せず、MLflow の REST からデータだけを読む。</p>
  <Mermaid label="入口から図までの流れ" code={flow} />
</Page>
