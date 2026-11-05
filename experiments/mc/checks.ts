// 拡張。縮約を Traub の 19 区画に当てる実験の判定。宣言は隣の experiments.json で、主張の名前ごとに判定の関数をここに置く。
import { GAMMA_LINE, below, mean, one, shown, type Judgement, type Row, type Judge } from '../common.ts';

/** 多区画への当てはめ：hn の KAM の soma の平均 Γ は、Na と DR を持つ 5 区画を置き換えた方が soma だけより高い。 */
function naDrBetter([soma, nadr]: Row[][]): Judgement[] {
  const from = mean(one(soma, 'hn', 'kam'));
  const to = mean(one(nadr, 'hn', 'kam'));
  return [
    {
      label: 'soma の平均 Γ（soma だけ → 5 区画）',
      value: `${shown(from)} → ${shown(to)}`,
      line: '上がる',
      ok: from === null || to === null ? null : to > from,
    },
  ];
}

/** 全ゲートを一つにまとめる形：係数の半分超が非零で、1 刻みの演算がどの種類も元より減らずどれかが増え、刺激 0 で元は発火しサロゲートの時刻は合わない。 */
function posterShape([rows]: Row[][]): Judgement[] {
  const row = one(rows, 'mhnabc', 'poster');
  const nnz = row?.summary['nnz_ratio'] ?? null;
  const costs = Object.entries(row?.summary ?? {}).filter(([k]) => k.startsWith('cost/surr-orig/'));
  const zero = row ? row.values.indexOf(0) : -1;
  const [original] = zero >= 0 ? (row?.counts[zero] ?? [0, 0]) : [0, 0];
  const gamma = zero >= 0 ? (row?.gammas[zero] ?? null) : null;
  return [
    { label: '非零の係数の割合', value: shown(nnz), line: '> 0.5', ok: nnz === null ? null : nnz > 0.5 },
    {
      label: '1 刻みの演算の、元との差（種類ごと）',
      value:
        costs
          .map(([k, v]) => `${k.slice('cost/surr-orig/'.length)} ${v === null ? '—' : (v > 0 ? '+' : '') + v}`)
          .join('・') || '—',
      line: 'どれも ≥ 0、どれかが > 0',
      ok:
        costs.length === 0 || costs.some(([, v]) => v === null)
          ? null
          : costs.every(([, v]) => v! >= 0) && costs.some(([, v]) => v! > 0),
    },
    {
      label: '刺激 0 の、元の発火の回数と Γ',
      value: zero < 0 ? '刺激 0 の点が無い' : `${original}、Γ ${shown(gamma)}`,
      line: `元 > 0、Γ < ${GAMMA_LINE}`,
      ok: zero < 0 ? null : original > 0 ? below(gamma) : false,
    },
  ];
}

/** 主張の名前 → 判定。 */
export const JUDGES: Record<string, Judge> = {
  'nadr-better': naDrBetter,
  'poster-shape': posterShape,
};
