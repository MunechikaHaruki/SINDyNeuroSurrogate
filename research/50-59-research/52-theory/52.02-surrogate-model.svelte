<script lang="ts">
  import { eq, tex } from '$lib/tex.ts';
  import Mermaid from '$lib/Mermaid.svelte';
  import Page from '$lib/Page.svelte';
  import Section from '$lib/Section.svelte';

  // ---- 全ゲートをまとめて圧縮するモデル（NEURO2026） ----

  /** SINDy の項の数。レート項が圧縮前のゲートごとに α, β, α·z_j, β·z_j、基底が V, z_j, 1。 */
  function libraryTerms(rateGates: number, latents: number): number {
    const rateTerms = 2 * rateGates;
    const productTerms = 2 * rateGates * latents;
    const basisTerms = latents + 2;
    return rateTerms + productTerms + basisTerms;
  }
  /** NEURO2026 の構成：6 ゲートを潜在 5 本へ、レート項は 8 ゲートぶん。 */
  const posterTerms = libraryTerms(8, 5);

  const posterStep = `flowchart LR
    z["潜在 z（5 本）"] -- decoder --> x["6 ゲート<br/>m n c a h b"]
    ca["s r q χ<br/>物理のまま"] --> I["イオン電流 I_ion"]
    x --> I
    I --> dV["dV/dt"]
    z --> theta["Θ(z, V)"]
    dV -. V .-> theta
    theta -- Ξ --> dz["dz/dt"]
  `;
</script>

<Page title="Traub 区画のサロゲートの数理構成" summary="Traub の区画のサロゲートのうち、Python の研究コードで学習・積分・評価まで動いている構成の数理。物理として残す部分、全ゲートをまとめて圧縮する AE + SINDy、その Multi-Compartment での学習と置換。">
  <p>Traub の 19 区画モデルを、状態を減らしたサロゲートに置き換える。この冊は、Python の研究コードで学習・積分・評価まで動いている構成を、何を物理のまま残し何をどう学習するかの式の形で並べる。</p>

  <Section id="sec-physics" title="物理として残す部分">
    <p>サロゲートが学習するのは、電位だけで動く 6 ゲート {@html tex`m,n,c,a,h,b`} の時間発展だけである。膜電位の式と Ca 系は物理の式のまま解く。</p>
    {@html eq`C\frac{dV}{dt}=I_{\mathrm{ext}}-\underbrace{I_{\mathrm{ion}}(V;\ m,h,n,a,b,c;\ s,r,q,\chi)}_{\text{物理の式}}`}
    {@html eq`\frac{d\chi}{dt}=-\kappa\,I_{\mathrm{Ca}}(V,s,r)-0.075\,\chi,\qquad \frac{dq}{dt}=\min(2\times10^{-5}\chi,\,0.01)(1-q)-0.001\,q`}
    <p>膜電位を残すので、区画間の結合（軸方向の電流）はそのまま足せる。s, r を学習側に入れると、{@html tex`I_{\mathrm{Ca}}=g_{\mathrm{Ca}}s^2r(V-E_{\mathrm{Ca}})`} の誤差が {@html tex`s`} の 2 乗で増え、積分器である {@html tex`\chi`} に溜まる。そのため s, r も物理側に回す。</p>
    <p>コンダクタンス密度 {@html tex`g`} は {@html tex`I_{\mathrm{ion}}`} にしか入らない。学習するゲートの式は {@html tex`g`} を含まないので、密度の違う区画にも同じサロゲートを当てられる（<a href="#sec-poster-mc">全ゲートのモデルの Multi-Compartment での学習と置換</a>）。</p>
  </Section>
  <Section id="sec-poster" title="全ゲートをまとめて圧縮するモデル">
    <p>NEURO2026 で発表した形。6 ゲートを AutoEncoder で潜在変数 {@html tex`z`} に縮め、{@html tex`z`} の時間発展を SINDy で式として同定する（<a href="https://doi.org/10.1073/pnas.1906995116">Champion et al. 2019</a>）。</p>
    {@html eq`z=E(x)\in\mathbb{R}^{d},\qquad \hat{x}=D(z)\in\mathbb{R}^{6},\qquad x=(m,n,c,a,h,b)`}
    <p>{@html tex`E`} と {@html tex`D`} は隠れ層 16 の tanh の層を一つずつ持つ。{@html tex`E`} は学習と初期値にだけ使い、積分では {@html tex`D`} だけを毎刻み呼ぶ。潜在変数の式は、候補の項 {@html tex`\Theta`} の線形和で書く。</p>
    {@html eq`\frac{dz}{dt}=\Xi\,\Theta(z,V),\qquad \Theta=\big[\underbrace{\alpha_x(V),\ \beta_x(V)}_{\text{レート}},\ \underbrace{\alpha_x(V)z_j,\ \beta_x(V)z_j}_{\text{レート}\times\text{潜在}},\ V,\ z_j,\ 1\big]`}
    <p>項はゲートの式 {@html tex`dx/dt=\alpha_x(1-x)-\beta_x x`} に現れる形を潜在変数へ移したもので、{@html tex`x`} は s, r を含む 8 ゲートを走る。項の数はレートのゲート数 {@html tex`G`} と潜在次元 {@html tex`d`} で決まる。</p>
    {@html eq`|\Theta|=2G+2Gd+d+2,\qquad G=8,\ d=5\ \Rightarrow\ |\Theta|=${posterTerms},\quad \Xi\ \text{は}\ 5\times${posterTerms}`}
    <Mermaid label="NEURO2026 のモデルの 1 刻み" code={posterStep} />
    <div class="panel caution">
      <p><b>この形の限界</b></p>
      <p>6 → 5 より先で再現が崩れ、圧縮率が頭打ちになった。積分の毎刻みに decoder を通すので、減らした状態の代わりに計算が増える。同定した {@html tex`\Xi`} の係数の 79.6% が非零で、式として読めない。原因は、時定数の違うゲートを一つの潜在空間に混ぜていることだと見ている（確かめてはいない）。</p>
    </div>
  </Section>
  <Section id="sec-poster-mc" title="全ゲートのモデルの Multi-Compartment での学習と置換">
    <p>Traub 原著はチャネルの動き方を全区画で同じとし、区画で違うのはコンダクタンス密度だけである（<a href="https://doi.org/10.1152/jn.1991.66.2.635">Traub et al. 1991</a>）。密度は物理側の {@html tex`I_{\mathrm{ion}}`} にしか入らないので、全区画で共通のサロゲートを 1 つ作る。</p>
    {@html eq`C\frac{dV_c}{dt}=I_{\mathrm{ext},c}-I_{\mathrm{ion}}(V_c;\ \hat{x}_c;\ g_c)-\sum_{c'\sim c}\frac{V_c-V_{c'}}{R_{cc'}},\qquad \frac{dz_c}{dt}=\Xi\,\Theta(z_c,V_c)`}
    <p>{@html tex`c`} は区画、{@html tex`g_c`} はその区画の密度、和は隣の区画 {@html tex`c'`} を走る。{@html tex`\Xi`} と AE は区画によらず一つで、区画ごとに持つのは状態だけである。</p>
    <p>学習データは 19 区画すべての軌道から取る。置換は soma だけと全区画の二通りで、元の系と電位の誤差、発火数、スパイクの特徴量で比べている。</p>
  </Section>
</Page>
