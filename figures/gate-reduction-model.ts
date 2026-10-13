// Traub 単一区画の 11 状態系と、Kepler–Abbott–Marder 法で任意の群分けにまとめた縮約系。DOM に触らない。
// 節は原著の順：Traub の元の系 → §2 等価電位 → §3 重みと整合性条件 → §6 縮約後の運動方程式 → 積分と比較。
// Traub のレート・密度パラメータ: SINDyNeuroSurrogate 290ee405b8d0。
//
// 読み手は本人なので、式と 1 対 1 に読めることを速さや短さより優先する。
// 状態は名前で読み書きし（Gates、Full、Reduced）、番号の並びは状態ベクトルとの出入口にだけ置く。
// 数式の項は 1 行に 1 つずつ書き、分割代入・map/reduce の連なり・タプル型は使わない。
import { Solver } from 'odex';

/** 一つのゲートの、開くレート α と閉じるレート β（ms⁻¹）。 */
export type Rate = { alpha: number; beta: number };
/** ゲート 8 種。並びは状態ベクトルと同じ。 */
export const gateNames = ['m', 'h', 'n', 'a', 'b', 's', 'r', 'c'] as const;
export type GateName = (typeof gateNames)[number];
/** 8 つのゲートの開き具合（0〜1）。 */
export type Gates = Record<GateName, number>;
/** 元の 11 状態。χ は細胞内 Ca 濃度に比例する量。 */
export type Full = { v: number; gates: Gates; q: number; chi: number };
/** 群分け（traub-apply-kam「群分け」）。φ 群は膜電位と一緒にまとめるゲート、ψ 群は 2 つ以上をまとめる組。書かないゲートは 1 つだけの群。 */
/** φ 群の重み α₀ の扱い。'one' は φ 群を速いゲートだけとみなして α₀ = 1、'solve' は Kepler 式(9)の根を各時刻に求める。 */
export type Alpha0Mode = 'one' | 'solve';
export type Grouping = { phi: GateName[]; psi: GateName[][]; alpha0?: Alpha0Mode };
/** 入力電流の平均 I₀ と振幅 A（µA/cm²）。 */
export type Config = { mean: number; amplitude: number };
/** 時刻（ms）と、その時刻の状態ベクトル。 */
export type Sample = { time: number; state: number[] };

export const duration = 500;
const warmup = 200;
const sampleStep = 0.1;
const capacitance = 3;
const kappa = 17402 * 3.32e-5;
/** 最大コンダクタンス（mS/cm²）。 */
const g = { L: 0.1, Na: 30, Ca: 4, DR: 15, A: 5, AHP: 0.8, C: 10 };
/** 逆転電位（mV）。 */
const E = { L: -60, Na: 55, Ca: 80, K: -75 };

// ---- 状態ベクトルとの出入口 ----
// odex は number[] しか扱えないので、番号と名前の対応はこの四つの関数だけが持つ。

/** 11 状態のベクトルを名前で読む。 */
export function readFull(y: number[]): Full {
  return {
    v: y[0],
    gates: { m: y[1], h: y[2], n: y[3], a: y[4], b: y[5], s: y[6], r: y[7], c: y[8] },
    q: y[9],
    chi: y[10],
  };
}
/** 11 状態のベクトルに戻す。 */
function writeFull(state: Full): number[] {
  const x = state.gates;
  return [state.v, x.m, x.h, x.n, x.a, x.b, x.s, x.r, x.c, state.q, state.chi];
}
/** 縮約系の状態ベクトルの並び。φ 群を除き、ゲートの並びで最初に出た順に群を置く。1 つだけの群はゲート値、2 つ以上の群は ψ（mV）を持つ。 */
export function groupLayout(grouping: Grouping): GateName[][] {
  const groups: GateName[][] = [];
  for (const name of gateNames) {
    if (grouping.phi.includes(name)) continue;
    let group: GateName[] = [name];
    for (const candidate of grouping.psi) if (candidate.includes(name)) group = candidate;
    if (group[0] === name) groups.push(group);
  }
  return groups;
}
/** 縮約系の状態ベクトルから、φ = V と 8 つのゲートを作り直して 11 状態の形で読む（traub-apply-kam「縮約モデルの一般形」）。 */
export function readReduced(grouping: Grouping, y: number[]): Full {
  const layout = groupLayout(grouping);
  const v = y[0];
  const gates = steadyGates(rates(v));
  for (let k = 0; k < layout.length; k++) {
    const group = layout[k];
    if (group.length === 1) {
      gates[group[0]] = y[1 + k];
      continue;
    }
    const recovered = steadyGates(rates(y[1 + k]));
    for (const name of group) gates[name] = recovered[name];
  }
  return { v, gates, q: y[1 + layout.length], chi: y[2 + layout.length] };
}

// ---- Traub の元の系：C dV/dt + I(V,{x}) = I_ext、dx/dt = k(V)[x̄(V) − x] ----

/** z / (e^z − 1)。z ≈ 0 では 0/0 を避けてテイラー展開を使う。 */
function bernoulli(z: number): number {
  if (Math.abs(z) < 1e-4) return 1 - z / 2 + (z * z) / 12 - z ** 4 / 720;
  return z / Math.expm1(z);
}
/** bernoulli の z による微分。n̄' の解析式に使う。 */
function bernoulliSlope(z: number): number {
  if (Math.abs(z) < 1e-4) return -0.5 + z / 6 - z ** 3 / 180;
  return (Math.expm1(z) - z * Math.exp(z)) / Math.expm1(z) ** 2;
}

/** 8 つのゲートの [開くレート α, 閉じるレート β]（ms⁻¹）。u = V + 60 は原著の基準電位。 */
export function rates(v: number): Record<GateName, Rate> {
  const u = v + 60;
  // c と r は電位で場合分けする。α を先に決め、β を α との差で置く。
  const alphaC = v <= -10 ? Math.exp((u - 10) / 11 - (u - 6.5) / 27) / 18.975 : 2 * Math.exp(-(u - 6.5) / 27);
  const alphaR = v <= -60 ? 0.005 : Math.exp(-u / 20) / 200;
  return {
    m: { alpha: 1.28 * bernoulli((13.1 - u) / 4), beta: 1.4 * bernoulli((u - 40.1) / 5) },
    h: { alpha: 0.128 * Math.exp((17 - u) / 18), beta: 4 / (1 + Math.exp((40 - u) / 5)) },
    n: { alpha: 0.08 * bernoulli((35.1 - u) / 5), beta: 0.25 * Math.exp((20 - u) / 40) },
    a: { alpha: 0.2 * bernoulli((13.1 - u) / 10), beta: 0.175 * bernoulli((u - 40.1) / 10) },
    b: { alpha: 0.0016 * Math.exp((-13 - u) / 18), beta: 0.05 / (1 + Math.exp((10.1 - u) / 5)) },
    s: { alpha: 1.6 / (1 + Math.exp(-0.072 * (u - 65))), beta: 0.1 * bernoulli((u - 51.1) / 5) },
    r: { alpha: alphaR, beta: v <= -60 ? 0 : 0.005 - alphaR },
    c: { alpha: alphaC, beta: v <= -10 ? 2 * Math.exp(-(u - 6.5) / 27) - alphaC : 0 },
  };
}
/** 定常値 x̄ = α/(α+β)。 */
export function steady(rate: Rate): number {
  return rate.alpha / (rate.alpha + rate.beta);
}
/** 緩和の速さ k = α+β。 */
export function speed(rate: Rate): number {
  return rate.alpha + rate.beta;
}
/** すべてのゲートを定常値にそろえる。 */
function steadyGates(rate: Record<GateName, Rate>): Gates {
  return {
    m: steady(rate.m),
    h: steady(rate.h),
    n: steady(rate.n),
    a: steady(rate.a),
    b: steady(rate.b),
    s: steady(rate.s),
    r: steady(rate.r),
    c: steady(rate.c),
  };
}

/** Ca 電流 I_Ca = g_Ca s² r (V − E_Ca)。 */
function calciumCurrent(v: number, s: number, r: number): number {
  return g.Ca * s * s * r * (v - E.Ca);
}
/** 全イオン電流 I(V,{x})（外向き正）。漏れ → Na → Ca → K の順に足す。 */
function ionicCurrent(v: number, x: Gates, q: number, chi: number): number {
  const potassium = g.DR * x.n + g.A * x.a * x.b + g.AHP * q + g.C * x.c * Math.min(1, chi / 250);
  return g.L * (v - E.L) + g.Na * x.m * x.m * x.h * (v - E.Na) + calciumCurrent(v, x.s, x.r) + potassium * (v - E.K);
}
/** 一つのゲートの dx/dt = α(1 − x) − βx（= k[x̄(V) − x]）。 */
function gateDerivative(rate: Rate, x: number): number {
  return rate.alpha * (1 - x) - rate.beta * x;
}
/** 8 つのゲートの dx/dt。 */
function gateDerivatives(rate: Record<GateName, Rate>, x: Gates): Gates {
  return {
    m: gateDerivative(rate.m, x.m),
    h: gateDerivative(rate.h, x.h),
    n: gateDerivative(rate.n, x.n),
    a: gateDerivative(rate.a, x.a),
    b: gateDerivative(rate.b, x.b),
    s: gateDerivative(rate.s, x.s),
    r: gateDerivative(rate.r, x.r),
    c: gateDerivative(rate.c, x.c),
  };
}
/** AHP ゲート q の dq/dt。開く速さは χ に比例し、上限で頭打ちになる。 */
function ahpDerivative(q: number, chi: number): number {
  return Math.min(0.00002 * chi, 0.01) * (1 - q) - 0.001 * q;
}
/** Ca 濃度 χ の dχ/dt。流入で増え、一定の割合で抜ける。 */
function calciumDerivative(v: number, s: number, r: number, chi: number): number {
  return -kappa * calciumCurrent(v, s, r) - 0.075 * chi;
}

/** 元の 11 状態系の右辺 dy/dt。 */
export function fullDerivative(y: number[], current: number): number[] {
  const state = readFull(y);
  const x = state.gates;
  return writeFull({
    v: (current - ionicCurrent(state.v, x, state.q, state.chi)) / capacitance,
    gates: gateDerivatives(rates(state.v), x),
    q: ahpDerivative(state.q, state.chi),
    chi: calciumDerivative(state.v, x.s, x.r, state.chi),
  });
}
/** 電位 v で全状態を定常にした 11 状態。ゲートは x̄(v)、χ と q は dχ/dt = dq/dt = 0 の解。 */
export function steadyState(v: number): number[] {
  const x = steadyGates(rates(v));
  const chi = (-kappa * calciumCurrent(v, x.s, x.r)) / 0.075;
  const alphaQ = Math.min(0.00002 * chi, 0.01);
  const q = alphaQ / (alphaQ + 0.001);
  return writeFull({ v, gates: x, q, chi });
}
/** 定常 I-V 曲線 I_ss(V)：全状態を定常にしたときの全イオン電流。 */
export function steadyCurrent(v: number): number {
  const state = readFull(steadyState(v));
  return ionicCurrent(v, state.gates, state.q, state.chi);
}
/** 定電流 current での平衡点 V*（mV）。I_ss(V) = current の根を −74.5〜54.5 mV で探す。安定性は見ない。 */
export function equilibria(current: number): number[] {
  const residual = (v: number): number => steadyCurrent(v) - current;
  const roots: number[] = [];
  for (let i = 0; i < 258; i++) {
    let lo = -74.5 + i * 0.5;
    let hi = lo + 0.5;
    // 符号が変わる格子区間だけを二分法で詰める。右端でちょうど 0 の根は次の区間で拾う。
    if (Math.sign(residual(lo)) === Math.sign(residual(hi))) continue;
    if (residual(hi) === 0) continue;
    for (let step = 0; step < 60; step++) {
      const mid = (lo + hi) / 2;
      if (Math.sign(residual(mid)) === Math.sign(residual(lo))) lo = mid;
      else hi = mid;
    }
    roots.push((lo + hi) / 2);
  }
  return roots;
}

// ---- §2 等価電位：ゲート値 x を x = x̄(v) となる電位 v で表す ----

/** h,n の定常値 x̄(v) と傾き x̄'(v)。傾きは商の微分による解析式。 */
export function recoverySteady(v: number): { h: number; n: number; slopeH: number; slopeN: number } {
  const u = v + 60;
  const alphaH = 0.128 * Math.exp((17 - u) / 18);
  const betaH = 4 / (1 + Math.exp((40 - u) / 5));
  const alphaN = 0.08 * bernoulli((35.1 - u) / 5);
  const betaN = 0.25 * Math.exp((20 - u) / 40);
  const slopeH = ((-alphaH / 18) * betaH - (alphaH * betaH * (1 - betaH / 4)) / 5) / (alphaH + betaH) ** 2;
  const slopeN = (-0.016 * bernoulliSlope((35.1 - u) / 5) * betaN + (alphaN * betaN) / 40) / (alphaN + betaN) ** 2;
  return { h: alphaH / (alphaH + betaH), n: alphaN / (alphaN + betaN), slopeH, slopeN };
}
/** 等価電位を探す範囲（mV）。x̄ はこの中で単調。r̄ は −60 mV 以下、c̄ は −10 mV 以上で 1 に張り付くので、そこを外す。 */
const equivalentRange: Record<GateName, { low: number; high: number }> = {
  m: { low: -120, high: 80 },
  h: { low: -120, high: 80 },
  n: { low: -120, high: 80 },
  a: { low: -120, high: 80 },
  b: { low: -120, high: 80 },
  s: { low: -120, high: 80 },
  r: { low: -60, high: 80 },
  c: { low: -120, high: -10 },
};
/** 定常値 x̄ が平らな電位区間（条件 1 が破れる範囲）。幅 step の格子で、隣の点と x̄ が等しい区間をつなぐ。 */
export function flatRanges(name: GateName, low: number, high: number, step: number): { low: number; high: number }[] {
  const ranges: { low: number; high: number }[] = [];
  let start: number | undefined;
  const count = Math.round((high - low) / step);
  for (let i = 0; i < count; i++) {
    // 格子点は小数第 6 位で丸め、step の足し算の誤差で境界がずれないようにする。
    const left = Number((low + i * step).toFixed(6));
    const right = Number((low + (i + 1) * step).toFixed(6));
    const flat = steady(rates(left)[name]) === steady(rates(right)[name]);
    if (flat && start === undefined) start = left;
    if (!flat && start !== undefined) {
      ranges.push({ low: start, high: left });
      start = undefined;
    }
  }
  if (start !== undefined) ranges.push({ low: start, high });
  return ranges;
}
/** 原著 §2：等価電位 v = x̄⁻¹(x)。x が探す範囲での x̄ の値域の外なら NaN。単調なので二分法で解く。 */
export function equivalentPotential(name: GateName, x: number): number {
  let low = equivalentRange[name].low;
  let high = equivalentRange[name].high;
  const signAtLow = Math.sign(steady(rates(low)[name]) - x);
  const signAtHigh = Math.sign(steady(rates(high)[name]) - x);
  if (signAtLow === signAtHigh) return NaN;
  for (let step = 0; step < 40; step++) {
    const mid = (low + high) / 2;
    if (Math.sign(steady(rates(mid)[name]) - x) === signAtLow) low = mid;
    else high = mid;
  }
  return (low + high) / 2;
}

// ---- §3 重みと整合性条件 ----

/** 感度 D_i = ∂F/∂v_i = (∂I/∂x_i) x̄_i'(v_i) を、回復群 h,n について返す。 */
function recoverySensitivities(v: number, m: number, slopeH: number, slopeN: number): number[] {
  return [g.Na * m * m * (v - E.Na) * slopeH, g.DR * (v - E.K) * slopeN];
}
/** 原著式(6)：群の重み α_i = D_i / Σ_j D_j。群の大きさは問わない。 */
export function recoveryWeights(sensitivity: number[]): number[] {
  let total = 0;
  for (const d of sensitivity) total += d;
  const weights: number[] = [];
  for (const d of sensitivity) weights.push(d / total);
  return weights;
}
/** 原著式(10)の左辺 |Σ_j α_j k_j − k_i| / Σ_j k_j を、群の各ゲート i について返す。 */
export function speedConsistency(weights: number[], speeds: number[]): number[] {
  let mean = 0;
  for (let j = 0; j < speeds.length; j++) mean += weights[j] * speeds[j];
  let total = 0;
  for (const k of speeds) total += k;
  const values: number[] = [];
  for (const k of speeds) values.push(Math.abs(mean - k) / total);
  return values;
}
/** 等価電位がすべて V に一致する点（m=m̄(V), v_h=v_n=V）での判定量。原著の評価点で、−75 < V < 55 mV で有効。 */
export function criteria(v: number): {
  dh: number;
  dn: number;
  kh: number;
  kn: number;
  alphaH: number;
  speedH: number;
  speedN: number;
} {
  const rate = rates(v);
  const m = steady(rate.m);
  const eq = recoverySteady(v);
  const sensitivity = recoverySensitivities(v, m, eq.slopeH, eq.slopeN);
  const weights = recoveryWeights(sensitivity);
  const kh = speed(rate.h);
  const kn = speed(rate.n);
  const consistency = speedConsistency(weights, [kh, kn]);
  return {
    dh: sensitivity[0],
    dn: sensitivity[1],
    kh,
    kn,
    alphaH: weights[0],
    speedH: consistency[0],
    speedN: consistency[1],
  };
}

// ---- §3 を任意の組へ：選んだゲートの群について、全状態が V で定常な点で判定する ----

/** ∂I/∂x：全イオン電流の、ゲート x による偏微分（mS/cm²）。「正の係数 × 駆動力 V − E_x」の形になる。 */
function currentSlope(name: GateName, v: number, x: Gates, chi: number): number {
  if (name === 'm') return g.Na * 2 * x.m * x.h * (v - E.Na);
  if (name === 'h') return g.Na * x.m * x.m * (v - E.Na);
  if (name === 's') return g.Ca * 2 * x.s * x.r * (v - E.Ca);
  if (name === 'r') return g.Ca * x.s * x.s * (v - E.Ca);
  if (name === 'n') return g.DR * (v - E.K);
  if (name === 'a') return g.A * x.b * (v - E.K);
  if (name === 'b') return g.A * x.a * (v - E.K);
  return g.C * Math.min(1, chi / 250) * (v - E.K);
}
/** ゲート x が入る電流の逆転電位 E_x（mV）。 */
export function reversalOf(name: GateName): number {
  if (name === 'm' || name === 'h') return E.Na;
  if (name === 's' || name === 'r') return E.Ca;
  return E.K;
}
/**
 * x̄'(v)：定常値の傾き。c だけは解析的に、c̄ = e^{(u−10)/11}/37.95（v ≤ −10）、c̄ = 1（v > −10）を微分する。
 * c̄ は −10 mV で 1.000094 から 1 へ跳ぶので、差分がそこをまたぐと偽の負の傾きになる。残りは幅 ±10⁻⁴ mV の中心差分で求める。
 */
export function steadySlope(name: GateName, v: number): number {
  if (name === 'c') return v <= -10 ? steady(rates(v).c) / 11 : 0;
  const width = 1e-4;
  const above = steady(rates(v + width)[name]);
  const below = steady(rates(v - width)[name]);
  return (above - below) / (2 * width);
}
/** 原著 §3 の感度 D_x = ∂F/∂v_x = (∂I/∂x) x̄'(v)。全状態が電位 v で定常な点（v_x = v）で評価する。 */
export function sensitivity(name: GateName, v: number): number {
  const state = readFull(steadyState(v));
  return currentSlope(name, v, state.gates, state.chi) * steadySlope(name, v);
}
/** 回復側の群（ψ）の判定。原著式(6)の重みと、式(10)の左辺を群の各ゲートについて返す。 */
export function groupCriteria(names: GateName[], v: number): { weights: number[]; consistency: number[] } {
  const rate = rates(v);
  const sensitivities: number[] = [];
  const speeds: number[] = [];
  for (const name of names) {
    sensitivities.push(sensitivity(name, v));
    speeds.push(speed(rate[name]));
  }
  const weights = recoveryWeights(sensitivities);
  return { weights, consistency: speedConsistency(weights, speeds) };
}
/** 膜電位側の群（φ）の判定。原著式(11)の左辺 |∂F/∂v_μ| / (C k_μ)。符号は sensitivitySignBands で見る。 */
export function voltageCriteria(name: GateName, v: number): { ratio: number } {
  const d = sensitivity(name, v);
  const k = speed(rates(v)[name]);
  return { ratio: Math.abs(d) / (capacitance * k) };
}

// ---- ψ 群の組み分け：符号のそろう群だけで作り、式(10)で読む ----

/** 感度が正のゲート（h, n, a, c）。ψ 群は符号のそろうゲートどうしで作る。 */
export const positiveGates: GateName[] = ['h', 'n', 'a', 'c'];
/** ψ 群の組み分けの一つ。key は「h,n|a,c|b」の形。worst は 2 つ以上の群での式(10)の左辺の最大で、群がなければ 0。 */
export type Partition = { key: string; blocks: GateName[][]; worst: number };

/** 群ごとの式(10)の左辺の最大を、組み分け全体で取る。電位 v で評価する。 */
function worstOfBlocks(blocks: GateName[][], v: number): number {
  let worst = 0;
  for (const block of blocks) {
    if (block.length < 2) continue;
    for (const value of groupCriteria(block, v).consistency) worst = Math.max(worst, value);
  }
  return worst;
}
/** 組み分けの key「h,n|a,c|b」。 */
function partitionKey(blocks: GateName[][]): string {
  const parts: string[] = [];
  for (const block of blocks) parts.push(block.join(','));
  return parts.join('|');
}
/** names を、感度の符号がそろう群だけで組み分ける方法をすべて並べ、電位 v での式(10)を添える。 */
export function psiPartitions(names: GateName[], v: number): Partition[] {
  const list: Partition[] = [];
  const blocks: GateName[][] = [];
  /** index 番目のゲートを、既にある群のどれかか、新しい群に入れる。 */
  function place(index: number): void {
    if (index === names.length) {
      const copy: GateName[][] = [];
      for (const block of blocks) copy.push(block.slice());
      list.push({ key: partitionKey(copy), blocks: copy, worst: worstOfBlocks(copy, v) });
      return;
    }
    const name = names[index];
    for (const block of blocks) {
      if (positiveGates.includes(block[0]) !== positiveGates.includes(name)) continue;
      block.push(name);
      place(index + 1);
      block.pop();
    }
    blocks.push([name]);
    place(index + 1);
    blocks.pop();
  }
  place(0);
  return list;
}

// ---- 感度の符号：sign D_x = sign x̄'(v_x) · sign(V − E_x) ----
// E_K < V < E_Na なら駆動力の符号はゲートごとに一定なので、D_x の符号は等価電位 v_x だけで決まる。平衡点は要らない。

/** 符号。正・負・ちょうど 0 の三つ。 */
export type Sign = 1 | -1 | 0;

/** 実数の符号。 */
function signOf(value: number): Sign {
  if (value > 0) return 1;
  if (value < 0) return -1;
  return 0;
}
/** E_K < V < E_Na での駆動力 V − E_x の符号。Na と Ca の電流で負、K の電流で正。 */
export function driveSign(name: GateName): Sign {
  return reversalOf(name) > E.K ? -1 : 1;
}
/** 感度 D_x の符号を、等価電位 v_x の区間ごとに返す。幅 step の格子の各区間の中点で x̄' の符号を見て、同じ符号の区間をつなぐ。 */
export function sensitivitySignBands(
  name: GateName,
  low: number,
  high: number,
  step: number,
): { low: number; high: number; sign: Sign }[] {
  const bands: { low: number; high: number; sign: Sign }[] = [];
  const count = Math.round((high - low) / step);
  for (let i = 0; i < count; i++) {
    const left = Number((low + i * step).toFixed(6));
    const right = Number((low + (i + 1) * step).toFixed(6));
    const sign = signOf(steadySlope(name, (left + right) / 2) * driveSign(name));
    const last = bands[bands.length - 1];
    if (last !== undefined && last.sign === sign) last.high = right;
    else bands.push({ low: left, high: right, sign });
  }
  return bands;
}

// ---- 群を作れる条件を、元モデルの軌道の上で確かめる ----
// 縮約後の重みは各時刻の状態で決まるので、平衡点だけでなく、元モデルが実際にたどる状態で条件 1,2 を見る。

/** 群を作れるかの判定。等価電位がない → 感度の符号が合わない → 感度がすべて 0 で重みが決まらない、の順に見る。 */
export type Verdict = 'ok' | 'noEquivalent' | 'wrongSign' | 'noWeight';
/** 一つの状態での判定と、そのときの感度 D と重み α。φ 群は重みを使わないので NaN を入れる。 */
export type GroupCheck = { verdict: Verdict; sensitivities: number[]; weights: number[] };

/** 状態 state での感度 D_x = (∂I/∂x) x̄'(v_x)。v_x は今のゲート値の等価電位で、なければ NaN（条件 1）。 */
export function stateSensitivity(name: GateName, state: Full): number {
  const equivalent = equivalentPotential(name, state.gates[name]);
  if (Number.isNaN(equivalent)) return NaN;
  return currentSlope(name, state.v, state.gates, state.chi) * steadySlope(name, equivalent);
}
/** 条件 1,2 を状態 state で確かめる。φ 群は D_μ がすべて負、ψ 群は D_i が同符号で、すべて 0 ではないこと。 */
export function checkGroup(names: GateName[], state: Full, voltage: boolean): GroupCheck {
  const sensitivities: number[] = [];
  for (const name of names) sensitivities.push(stateSensitivity(name, state));
  let missing = 0;
  let positive = 0;
  let negative = 0;
  for (const d of sensitivities) {
    if (Number.isNaN(d)) missing++;
    else if (d > 0) positive++;
    else if (d < 0) negative++;
  }
  let verdict: Verdict = 'ok';
  if (missing > 0) verdict = 'noEquivalent';
  else if (voltage ? negative < names.length : positive > 0 && negative > 0) verdict = 'wrongSign';
  else if (!voltage && positive + negative === 0) verdict = 'noWeight';
  const weights = voltage ? [] : recoveryWeights(sensitivities);
  if (voltage) for (let i = 0; i < names.length; i++) weights.push(NaN);
  return { verdict, sensitivities, weights };
}
/** 元モデルの軌道の各標本で checkGroup を行い、判定ごとの時間（標本数 × 0.1 ms）を足し上げる。 */
export function checkTrajectory(
  names: GateName[],
  samples: Sample[],
  voltage: boolean,
): { checks: GroupCheck[]; durations: Record<Verdict, number> } {
  const checks: GroupCheck[] = [];
  const durations: Record<Verdict, number> = { ok: 0, noEquivalent: 0, wrongSign: 0, noWeight: 0 };
  for (const sample of samples) {
    const check = checkGroup(names, readFull(sample.state), voltage);
    checks.push(check);
    durations[check.verdict] += sampleStep;
  }
  return { checks, durations };
}

// ---- §6 縮約後の運動方程式：C dφ/dt + α₀F(φ,ψ) = α₀I_ext、dψ_k/dt = Σ_i α_i f_i(φ,ψ_k) ----

/** ψ 群の dψ/dt = Σ_i α_i f_i。α_i f_i = (∂I/∂x_i) k_i[x̄_i(V) − x̄_i(ψ)] / Σ_j D_j で、x̄_i'(ψ) は式(1)の分母と重みの分子で打ち消す。 */
function groupDerivative(group: GateName[], psi: number, v: number, x: Gates, chi: number): number {
  const rate = rates(v);
  const recovered = rates(psi);
  let total = 0;
  let positive = 0;
  let negative = 0;
  for (const name of group) {
    const d = currentSlope(name, v, x, chi) * steadySlope(name, psi);
    total += d;
    if (d > 0) positive++;
    if (d < 0) negative++;
  }
  // 正の重みが成立しない領域を、クリップや分母への微小数の加算で隠さない。
  if ((positive > 0 && negative > 0) || total === 0) {
    throw new Error(`正の重みの適用範囲外: V=${v.toFixed(3)}, ψ=${psi.toFixed(3)} mV, 群 {${group.join(',')}}`);
  }
  let sum = 0;
  for (const name of group) {
    const drive = steady(rate[name]) - steady(recovered[name]);
    sum += currentSlope(name, v, x, chi) * speed(rate[name]) * drive;
  }
  return sum / total;
}
/** D₀ = ∂F/∂V：ゲートを止めて電位だけを動かしたときの膜の全コンダクタンス（mS/cm²）。 */
function membraneConductance(x: Gates, q: number, chi: number): number {
  const potassium = g.DR * x.n + g.A * x.a * x.b + g.AHP * q + g.C * x.c * Math.min(1, chi / 250);
  return g.L + g.Na * x.m * x.m * x.h + g.Ca * x.s * x.s * x.r + potassium;
}
/**
 * Kepler 式(9)の根 α₀。y = α₀ΣD とおくと、D_μ < 0 なら y < min Ck_μ で残差が単調に増え、根がただ一つある。
 * これが k_μ → ∞ で α₀ → 1 となる根なので、二分法で求める。φ 群が空なら α₀ = 1。
 */
export function phiWeight(d0: number, sensitivities: number[], speeds: number[]): number {
  if (sensitivities.length === 0) return 1;
  let total = d0;
  let high = Infinity;
  for (let i = 0; i < sensitivities.length; i++) {
    if (!(sensitivities[i] < 0)) throw new Error(`φ 群の感度が負でない: D=${sensitivities[i]}`);
    total += sensitivities[i];
    high = Math.min(high, capacitance * speeds[i]);
  }
  // 閾値付近では D_m が D₀ を上回り ΣD < 0 になるが、根の選び方は変わらない。ΣD = 0 だけは α₀ が決まらない。
  if (total === 0) throw new Error('式(9)の ΣD が 0');
  /** 式(9)の左辺を y = α₀ΣD の関数として書いたもの。 */
  function residual(y: number): number {
    let value = y - d0;
    for (let i = 0; i < sensitivities.length; i++) {
      const ck = capacitance * speeds[i];
      value -= (ck * sensitivities[i]) / (ck - y);
    }
    return value;
  }
  let low = high - 1;
  while (residual(low) > 0) low = high - 2 * (high - low);
  let upper = high;
  for (let i = 0; i < 200; i++) {
    const middle = (low + upper) / 2;
    if (residual(middle) > 0) upper = middle;
    else low = middle;
  }
  return (low + upper) / 2 / total;
}
/** 縮約系の状態で α₀ を決める。感度は V = v_μ = φ と作り直したゲートで評価する。 */
function membraneWeight(grouping: Grouping, state: Full): number {
  if (grouping.alpha0 !== 'solve') return 1;
  const rate = rates(state.v);
  const sensitivities: number[] = [];
  const speeds: number[] = [];
  for (const name of grouping.phi) {
    sensitivities.push(currentSlope(name, state.v, state.gates, state.chi) * steadySlope(name, state.v));
    speeds.push(speed(rate[name]));
  }
  return phiWeight(membraneConductance(state.gates, state.q, state.chi), sensitivities, speeds);
}
/** 縮約系の右辺 dy/dt。C dφ/dt = α₀[I_ext − F]。α₀ は grouping.alpha0 で 1 と置くか式(9)を解くかを選ぶ。 */
export function reducedDerivative(grouping: Grouping, y: number[], current: number): number[] {
  const layout = groupLayout(grouping);
  const state = readReduced(grouping, y);
  const x = state.gates;
  const dx = gateDerivatives(rates(state.v), x);
  const alpha0 = membraneWeight(grouping, state);
  const dy: number[] = [(alpha0 * (current - ionicCurrent(state.v, x, state.q, state.chi))) / capacitance];
  for (let k = 0; k < layout.length; k++) {
    const group = layout[k];
    if (group.length === 1) dy.push(dx[group[0]]);
    else dy.push(groupDerivative(group, y[1 + k], state.v, x, state.chi));
  }
  dy.push(ahpDerivative(state.q, state.chi));
  dy.push(calciumDerivative(state.v, x.s, x.r, state.chi));
  return dy;
}

// ---- 積分と比較 ----

/** 時刻 t の注入電流。周期 80 ms と 80/√2 ms の正弦波を足した準周期波形。 */
export function input(t: number, config: Config): number {
  const wave = Math.sin((2 * Math.PI * t) / 80) + 0.5 * Math.sin((2 * Math.PI * Math.SQRT2 * t) / 80);
  return config.mean + (config.amplitude * wave) / 1.5;
}
/** V = −60 mV の定常状態。群分けを渡せば縮約系の並びで返し、等価電位が V に一致するので ψ = V。 */
export function initial(grouping?: Grouping): number[] {
  const full = readFull(steadyState(-60));
  if (grouping === undefined) return writeFull(full);
  const y = [full.v];
  for (const group of groupLayout(grouping)) y.push(group.length === 1 ? full.gates[group[0]] : full.v);
  y.push(full.q, full.chi);
  return y;
}
/** 群分けを渡せば縮約系、渡さなければ元の 11 状態系の右辺。 */
function derivativeOf(grouping?: Grouping): (y: number[], current: number) => number[] {
  if (grouping === undefined) return fullDerivative;
  return function (y: number[], current: number): number[] {
    return reducedDerivative(grouping, y, current);
  };
}
/** 無入力で 200 ms 落ち着かせてから、入力を 500 ms 与えて積分し、0.1 ms ごとの標本を返す。群分けがなければ元モデル。 */
export function simulate(config: Config, grouping?: Grouping, tolerance = 1e-7, maxStep = 0.1): Sample[] {
  const derivative = derivativeOf(grouping);
  const size = initial(grouping).length;
  const options = {
    relativeTolerance: tolerance,
    absoluteTolerance: tolerance * 0.1,
    maxStepSize: maxStep,
    maxSteps: 100000,
  };
  // 待機と刺激の境界をまたいで補間しないよう、積分器を二つに分ける。
  const settle = new Solver((_t, y) => derivative(y, 0), size, options);
  const settled = settle.integrate(-warmup, initial(grouping))(0);
  const solver = new Solver((t, y) => derivative(y, input(t, config)), size, options);
  // integrate は時刻を渡すと状態を返す補間関数を返す。時刻は増える向きにしか渡せない。
  const evaluate = solver.integrate(0, settled);
  const result: Sample[] = [];
  for (let i = 0; i <= Math.round(duration / sampleStep); i++) {
    const time = i * sampleStep;
    const state = [...evaluate(time)];
    if (!state.every(Number.isFinite)) throw new Error(`非有限の状態: t=${time} ms`);
    result.push({ time, state });
  }
  return result;
}
/** 膜電位が 0 mV を上向きにまたいだ時刻（ms）。標本間は線形補間する。 */
export function spikes(samples: Sample[]): number[] {
  const times: number[] = [];
  for (let i = 1; i < samples.length; i++) {
    const left = samples[i - 1];
    const right = samples[i];
    if (left.state[0] < 0 && right.state[0] >= 0) {
      const ratio = left.state[0] / (right.state[0] - left.state[0]);
      times.push(left.time - ratio * (right.time - left.time));
    }
  }
  return times;
}
/** 標本を 11 状態の形で読む。群分けがなければ元モデルの標本。 */
export function readSample(sample: Sample, grouping?: Grouping): Full {
  if (grouping === undefined) return readFull(sample.state);
  return readReduced(grouping, sample.state);
}
/** 同じ時刻の値どうしの RMSE。 */
export function rmse(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum / a.length);
}
