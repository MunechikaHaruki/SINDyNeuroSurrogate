<script lang="ts">
  import Mermaid from '$lib/Mermaid.svelte';
  import Page from '$lib/Page.svelte';
  import Section from '$lib/Section.svelte';

  /** 三つの置き場と、その間を行き来するもの。線の上の文字が受け渡すデータ。 */
  const map = `flowchart LR
    subgraph ts["research/（TS と Svelte）"]
      conf["宣言<br/>conf.ts"]
      orch["段取り<br/>orchestrate.ts"]
      api["ボタンの口<br/>api.ts"]
      figs["図<br/>53.03・53.04 ほか"]
    end
    subgraph py["neurosurrogate/（Python）"]
      entry["計算の入口<br/>run.py"]
      domain["ドメイン層<br/>シミュレータと学習"]
    end
    runs[("runs/<br/>結果")]
    conf --> orch
    orch -- "spec.json" --> entry
    entry --> domain
    entry -- "波形・係数・指標" --> runs
    orch -- "index.json" --> runs
    runs -- "開発サーバー" --> figs
    figs -- "ボタン" --> api
    api -- "宣言の名前" --> orch
  `;

  /** just eval を一回打ったときの順序。 */
  const sequence = `sequenceDiagram
    participant U as 本人
    participant O as 段取り（orchestrate.ts）
    participant R as runs/
    participant P as 計算（run.py）
    participant F as 図（53.04）
    U->>O: just eval traub19_somastim poster_traub_ae
    O->>O: conf.ts から spec を組み、ハッシュを取る
    O->>R: runs/ハッシュ/ はあるか
    alt もうある
      R-->>O: ある（回さない）
    else まだない
      O->>P: spec.json と置き場を渡す
      P->>P: シミュレーション・置換・指標
      P->>R: ディレクトリを書く
    end
    O->>R: index.json を書き直す
    F->>R: index.json と run を読む
  `;

  /** 図のボタンを押したときの順序。 */
  const button = `sequenceDiagram
    participant U as 本人
    participant F as 図（53.04）
    participant A as 口（api.ts）
    participant O as 段取り（切り離した子プロセス）
    participant R as runs/
    U->>F: 「poster_traub_ae を回す」を押す
    F->>A: 系列名と preset 名を POST
    A->>A: 宣言にある名前か、同じ組が回っていないか
    A->>O: 起こして、すぐ返す
    loop 5 秒ごと
      F->>A: 状態を聞く
      A->>R: ハッシュ.partial を書くプロセスは生きているか
      A-->>F: 計算中
    end
    O->>R: just eval と同じ手順で書き、index.json を書き直す
    F->>A: 状態を聞く（計算中が減った）
    F->>R: index.json を読み直して描く
  `;

  /** 3 種類の run の参照関係。矢印は「spec の中でハッシュを名指す」向き。 */
  const kinds = `flowchart LR
    train["学習 run<br/>poster_traub_ae"]
    original["原系 run<br/>traub19_somastim"]
    replaced["置換系 run<br/>traub19_somastim × poster_traub_ae"]
    replaced -- "train" --> train
    replaced -- "original" --> original
  `;

  /** ドメイン層の依存。矢印は「import する」向き。tests/test_conventions.py の _LAYERS から推移的な辺を除いたもの。 */
  const layers = `flowchart RL
    base["base<br/>core/"]
    neurons["neurons<br/>neurons/"]
    sim_desc["sim_desc<br/>sim/ の spec・result・waveform"]
    surrogate["surrogate<br/>surrogate/"]
    sim_exec["sim_exec<br/>sim/run.py"]
    entry["entry<br/>run.py"]
    neurons --> base
    sim_desc --> neurons
    surrogate --> sim_desc
    sim_exec --> surrogate
    entry --> sim_exec
  `;

  /** サロゲートの 3 部品。学習は前処理 → 閉包項の順で、定式化が両方を組み合わせる。 */
  const parts = `flowchart LR
    gates["ゲートの軌道"] -- "前処理<br/>PCA か AE" --> latent["潜在 z"]
    latent -- "閉包項<br/>SINDy か UDE" --> rhs["dz/dt の式"]
    ansatz["定式化<br/>sindy・hybrid・ude"] -.組み合わせる.-> latent
    ansatz -.組み合わせる.-> rhs
  `;
</script>

<Page title="SINDyNeuroSurrogate の構成" summary="研究コードの全体像。三つの置き場（Python の neurosurrogate/、TS の research/、結果の runs/）、just train・just eval から図に届くまでの順序、何を回すかの宣言、3 種類の run、Python のドメイン層の依存とサロゲートの部品、どの図がどこを読むか、守っている約束と検査。">
  <p>研究コードは、計算する Python、何を回すかを決めて結果を見る TS、その間に置く結果の三つに分かれる。この冊は、それぞれがどこにあり、何を受け渡し、どの約束で分けているかを、全体から細部へ順に並べる。</p>

  <Section id="sec-map" title="全体の地図">
    <p>置き場は三つで、言語の境目はデータだけが越える。TS は Python を import せず、Python は TS を知らない。</p>
    <Mermaid label="三つの置き場" code={map} />
    <table>
      <thead><tr><th>置き場</th><th>言語</th><th>持つもの</th></tr></thead>
      <tbody>
        <tr><td><code>neurosurrogate/</code></td><td>Python</td><td>シミュレータ（JAX）、サロゲートの学習（pysindy・AE・PCA）、指標（eFEL）、計算の入口 <code>run.py</code></td></tr>
        <tr><td><code>research/</code></td><td>TS・Svelte</td><td>何を回すかの宣言 <code>conf.ts</code>、段取り <code>orchestrate.ts</code>、図のボタンの口 <code>api.ts</code>、図（冊）</td></tr>
        <tr><td><code>runs/</code></td><td>データ</td><td>回した結果。1 run が 1 ディレクトリで、Git には入れない</td></tr>
      </tbody>
    </table>
    <div class="panel">
      <p>重い計算だけが Python にあり、それ以外（何を回すか、もう回したか、どう並べて見るか）は TS が持つ。</p>
    </div>
  </Section>

  <Section id="sec-2" title="一回の実験の流れ">
    <p>入口は <code>just train &lt;preset&gt;</code>（学習）と <code>just eval &lt;系列&gt; &lt;preset&gt;</code>（評価）の二つで、どちらも段取りを呼ぶ。</p>
    <Mermaid label="just eval を打ってから図に出るまで" code={sequence} />
    <ol>
      <li><b>spec を組む。</b>段取りが <code>conf.ts</code> の宣言から、回す条件を JSON（spec）にする。</li>
      <li><b>ハッシュを取る。</b>spec の鍵を並べ替えてから SHA-1 を取り、先頭 12 文字を run の名前にする。同じ条件は書き方の順が違っても同じ名前になる。</li>
      <li><b>回したかを見る。</b><code>runs/&lt;ハッシュ&gt;/</code> があれば回さない。評価では、原系 run と学習 run が先にあるかも同じ方法で見る。</li>
      <li><b>計算する。</b>無ければ Python の <code>run.py</code> に spec.json と置き場を渡す。<code>run.py</code> はシミュレーション・置換・指標を計算し、ディレクトリを書く。途中で落ちても半端な run を残さないよう、一時ディレクトリに書いてから名前を変える。</li>
      <li><b>一覧を書き直す。</b>段取りが <code>runs/index.json</code>（全 run の一覧）を作り直す。</li>
      <li><b>図が読む。</b>図は開発サーバーから <code>index.json</code> と run のファイルを読んで描く。</li>
    </ol>
    <p>一覧は <code>just runs list</code>、宣言にある preset と系列の名前は <code>just runs -h</code> で出る。</p>
    <p>図のボタンからも同じ段取りを起こせる。53.04 の「比べる学習 run」には宣言にあってまだ回していない preset が、53.03 にはまだ学習していない preset がボタンで並ぶ。受け付けるのは <code>conf.ts</code> に宣言した名前だけで、宣言は画面から変えられない。</p>
    <Mermaid label="図のボタンを押してから図に出るまで" code={button} />
    <ul>
      <li><b>切り離して回す。</b>口（<code>api.ts</code>）は段取りを子プロセスで起こしてすぐ返す。開発サーバーを再起動しても計算は続く。</li>
      <li><b>計算中が分かる。</b><code>run.py</code> は書いている間 <code>runs/&lt;ハッシュ&gt;.partial/</code> に自分の pid を置く。そのプロセスが生きていれば計算中で、落ちたあとの残骸なら次に回すとき片付ける。</li>
      <li><b>二度回さない。</b>同じ組を続けて押しても、組のハッシュで回っている段取りを見つけて起こさない。原系を共有する二つの評価が重なったときは、後の方が先の計算の終わりを待つ。</li>
      <li><b>置換できない組は覚える。</b>置換できるかは Python だけが知るので、一度回して置換できないと分かった組は <code>runs/&lt;ハッシュ&gt;.unreplaceable</code> を置き、ボタンから消す。</li>
    </ul>
  </Section>

  <Section id="sec-conf" title="何を回すかの宣言">
    <p>回す条件は、すべて <code>research/50-59-research/53-evaluation/conf.ts</code> に型付きで書く。宣言は三種類ある。</p>
    <table>
      <thead><tr><th>宣言</th><th>中身</th><th>例</th></tr></thead>
      <tbody>
        <tr><td><code>PRESETS</code></td><td>学習の設定。名前から学習 run の列を引く</td><td><code>poster_traub_ae</code>：19 区画の軌道で学習し、6 ゲートを AE で潜在 5 本へ縮める。s, r は物理のまま</td></tr>
        <tr><td>掃引</td><td>一つの preset で値を振り、点ごとに学習 run を作る</td><td><code>traub_useMC19CompForTeachingData</code>：潜在次元 6, 7, 8 の 3 本</td></tr>
        <tr><td><code>SERIES</code></td><td>評価の系列。入力電流・振る値・どこを置換するか</td><td><code>traub19_somastim</code>：soma に一定電流を 0〜10 µA/cm² の 5 点で入れ、soma を置換する</td></tr>
      </tbody>
    </table>
    <p>preset は共通の土台を深いマージ（<code>merge</code>）で上書きして作る。値を選んだ理由は、宣言の隣のコメントに書く。図のつまみ（どの区画を見るか、表示の範囲）は宣言に持たず、図が持つ。</p>
  </Section>

  <Section id="sec-runs" title="run の置き場">
    <p>run は 3 種類ある。置換系 run は、どの学習 run とどの原系 run を比べたかを、spec の中でハッシュで名指す。</p>
    <Mermaid label="3 種類の run の参照" code={kinds} />
    <table>
      <thead><tr><th>種類</th><th>何か</th><th>ディレクトリの中身</th></tr></thead>
      <tbody>
        <tr><td>学習 run</td><td>学習したサロゲート一つ</td><td><code>surrogate/</code>（学習済みのもの）、<code>description/</code>（係数・前処理・学習データ・潜在の軌跡）</td></tr>
        <tr><td>原系 run</td><td>置換しない元のモデルで、系列を回した波形</td><td><code>waves/</code></td></tr>
        <tr><td>置換系 run</td><td>系列を、指定した区画をサロゲートに置き換えて回した波形と、原系との比較</td><td><code>waves/</code>、<code>metrics.json</code>（指標）、<code>summary.json</code>（学習側の指標）、<code>projected/</code>（原系のゲートを潜在へ写したもの）</td></tr>
      </tbody>
    </table>
    <p>どの run にも <code>spec.json</code>（回した条件）と <code>meta.json</code>（種類・表示名・作った日時）がある。波形は little-endian の float32 をそのまま並べたファイルで、形の説明は <code>waves/meta.json</code> にある。</p>
    <div class="panel">
      <p>原系 run は置換範囲を spec に含めない。soma だけ置換する系列と全区画を置換する系列は、同じ原系 run を共有する。</p>
    </div>
  </Section>

  <Section id="sec-1" title="Python のドメイン層">
    <p><code>neurosurrogate/</code> は 6 つの層に分かれ、矢印の向きにだけ import できる。左の <code>base</code> ほど基盤で、ほかの層を一切 import しない。</p>
    <Mermaid label="ドメイン層の依存" code={layers} />
    <table>
      <thead><tr><th>層</th><th>場所</th><th>すること</th></tr></thead>
      <tbody>
        <tr><td><code>base</code></td><td><code>core/</code></td><td>区画のネットワークの型、JAX のシミュレータ（Euler 法を <code>lax.scan</code> で回す）、波形の座標、演算コストの集計、発散の判定</td></tr>
        <tr><td><code>neurons</code></td><td><code>neurons/</code></td><td>HH・Traub の区画の式、Traub の 19 区画モデルの組み方、サロゲートが学習する状態と物理のまま解く状態の分け方</td></tr>
        <tr><td><code>sim_desc</code></td><td><code>sim/</code> の spec・result・waveform</td><td>実験の記述（何の入力をどこに入れるか）、結果の器、原系と置換系を比べる指標</td></tr>
        <tr><td><code>surrogate</code></td><td><code>surrogate/</code></td><td>サロゲートの学習、区画への置換（どこに当てられるかの判定も）、学習したものの中身の書き出し</td></tr>
        <tr><td><code>sim_exec</code></td><td><code>sim/run.py</code></td><td>記述とサロゲートを掛け合わせて、系列を 1 列回す</td></tr>
        <tr><td><code>entry</code></td><td><code>run.py</code></td><td>spec.json を受け取り、run のディレクトリを書く。run の保存形式を知るのはここだけ</td></tr>
      </tbody>
    </table>

    <Section id="sec-parts" title="サロゲートの 3 部品">
      <p>サロゲートは三つの部品を差し替えて作る。前処理がゲートを潜在に縮め、閉包項が潜在の時間発展を式として学び、定式化がその二つをどう組むかを決める。</p>
      <Mermaid label="サロゲートの 3 部品" code={parts} />
      <table>
        <thead><tr><th>部品</th><th>場所</th><th>選べるもの</th></tr></thead>
        <tbody>
          <tr><td>前処理</td><td><code>surrogate/parts/preprocessor/</code></td><td>PCA、AE</td></tr>
          <tr><td>閉包項</td><td><code>surrogate/parts/closure/</code></td><td>SINDy（項の候補から疎な係数を選ぶ）、UDE（ニューラルネット）</td></tr>
          <tr><td>定式化</td><td><code>surrogate/parts/ansatz/</code></td><td>sindy（全部を学習）、hybrid（一部の状態を物理のまま解く）、ude</td></tr>
        </tbody>
      </table>
      <p>学習の順は、前処理を先に当ててから閉包項を同定する。式の形は <a href="../52-theory/52.02-surrogate-model">別冊「Traub 区画のサロゲートの数理構成」</a>にある。</p>
    </Section>
  </Section>

  <Section id="sec-figures" title="図とデータ">
    <p>図は Python を import せず、データだけを読む。読むものは冊によって二通りある。</p>
    <table>
      <thead><tr><th>冊</th><th>読むもの</th></tr></thead>
      <tbody>
        <tr><td>52.01 KAM の適用、53.01 組の比較、53.02 縮約モデル群の評価</td><td><code>52-theory/traub.ts</code>（ブラウザで動く Traub の単一区画のシミュレータ）</td></tr>
        <tr><td>53.03 学習 run の中身</td><td>学習 run の <code>description/</code></td></tr>
        <tr><td>53.04 評価の比較</td><td>原系 run と置換系 run の <code>waves/</code>、<code>metrics.json</code></td></tr>
        <tr><td>51.01、52.02、54.01</td><td>なし（式と構成だけ）</td></tr>
      </tbody>
    </table>
    <div class="panel caution">
      <p>Traub のシミュレータは、Python（<code>neurons/traub.py</code>）と TS（<code>traub.ts</code>）の二か所にある。ずれは <code>traub.reference.json</code> の基準値をテストで突き合わせて見張っている。</p>
    </div>
  </Section>

  <Section id="sec-rules" title="守っている約束">
    <table>
      <thead><tr><th>約束</th><th>理由</th><th>確かめ方</th></tr></thead>
      <tbody>
        <tr><td>図と Python は import し合わない</td><td>言語の境目をデータだけにして、片側を読めばもう片側を読まなくて済むようにする</td><td>図の import は <code>python3 .agents/check.py</code> が実在を見る</td></tr>
        <tr><td>ドメイン層は層の向きにだけ import する</td><td>下の層を、上の層を知らずに読めるようにする</td><td><code>tests/test_conventions.py</code></td></tr>
        <tr><td>ドメイン層は記録・設定・描画の道具を import しない</td><td>計算だけを持ち、置き場が変わっても計算を直さずに済むようにする</td><td><code>tests/test_conventions.py</code></td></tr>
        <tr><td>ハッシュを組むのは TS だけ</td><td>JS と Python では数値の JSON の書き方が違い、二か所で組むと名前がずれる</td><td>決まりだけで、検査はない</td></tr>
        <tr><td>形式の版を二か所で揃える</td><td>形式を変えたら版を上げ、古い run を回し直させる</td><td><code>run.py</code> の <code>_FORMAT</code> と <code>conf.ts</code> の <code>FORMAT</code> を手で揃える。検査はない</td></tr>
        <tr><td>記録した run を書き換えない</td><td>ハッシュが同じなら中身も同じ、を保つ</td><td>図は読むだけ。ボタンの口は宣言の組を回すだけで、run を書き換える口は無い</td></tr>
      </tbody>
    </table>
  </Section>
</Page>
