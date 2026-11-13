// 実験群 1comp（1comp.yaml）の判定層。主張は checks の各項目の docstring に、回す前に書く（結果を見てから書かない）。
// 項目は、読む run（1comp.yaml の runs）と系列、絞る群分け、判定の関数を持つ。判定の関数の docstring は、主張を何で確かめるか（線）を書く。
import {
  GAMMA_LINE,
  NAMES,
  above,
  all,
  blocksOf,
  below,
  groupingsOf,
  mean,
  none,
  one,
  shown,
  type Checks,
  type Judgement,
  type Row,
} from '../evaluation/experiments/common.ts';

/** 比べる縮約の仕方。 */
const METHODS = ['kam', 'pca-sindy', 'ae-sindy'];


/** ゲートの組と再現：a+c は KAM・PCA・AE のどれでも線以上で、ほかにそういう組は無い。KAM で n を h 以外のゲートとまとめた組は線を下回る。 */
function acOnly([rows]: Row[][]): Judgement[] {
  const allAbove = (g: string) => all(METHODS.map(m => above(mean(one(rows, g, m)))));
  const others = groupingsOf(rows).filter(g => g !== 'a+c');
  const nMixed = groupingsOf(rows).filter(g =>
    blocksOf(g).some(block => block.includes('n') && block.some(x => x !== 'n' && x !== 'h')),
  );
  const nGammas = nMixed.map(g => mean(one(rows, g, 'kam')));
  return [
    {
      label: 'a+c の平均 Γ（KAM・PCA・AE）',
      value: METHODS.map(m => `${NAMES[m]} ${shown(mean(one(rows, 'a+c', m)))}`).join('・'),
      line: `3 つとも ≥ ${GAMMA_LINE}`,
      ok: allAbove('a+c'),
    },
    {
      label: 'a+c 以外で 3 つとも線以上の組',
      line: '0 個',
      ...none(
        others.filter(g => allAbove(g) === true),
        others.filter(g => allAbove(g) === null),
      ),
    },
    {
      label: 'KAM で n を h 以外とまとめた組の平均 Γ の最大',
      value: `${shown(Math.max(...nGammas.map(v => v ?? -Infinity)))}（${nMixed.length} 組）`,
      line: `どれも < ${GAMMA_LINE}`,
      ok: nMixed.length === 0 ? null : all(nGammas.map(below)),
    },
  ];
}

/** m の吸収：図の組は吸収しない KAM で線以上で、吸収すると発火の回数の差はどの点でも一つ以内のまま、平均 Γ が線を下回る。 */
function absorbM([rows]: Row[][], groupings: string[] = groupingsOf(rows)): Judgement[] {
  const pairs = groupings.map(g => ({ g, kam: one(rows, g, 'kam'), absorbed: one(rows, g, 'kam-absorb-m') }));
  return [
    {
      label: '吸収しない KAM の平均 Γ',
      value: pairs.map(p => `${p.g} ${shown(mean(p.kam))}`).join('・'),
      line: `どれも ≥ ${GAMMA_LINE}`,
      ok: all(pairs.map(p => above(mean(p.kam)))),
    },
    {
      label: '吸収した KAM の発火の回数の差の最大',
      value: pairs
        .map(p => `${p.g} ${p.absorbed ? Math.max(...p.absorbed.counts.map(([o, m]) => Math.abs(o - m))) : '—'}`)
        .join('・'),
      line: 'どれも ≤ 1',
      ok: all(pairs.map(p => (p.absorbed ? p.absorbed.counts.every(([o, m]) => Math.abs(o - m) <= 1) : null))),
    },
    {
      label: '吸収した KAM の平均 Γ',
      value: pairs.map(p => `${p.g} ${shown(mean(p.absorbed))}`).join('・'),
      line: `どれも < ${GAMMA_LINE}`,
      ok: all(pairs.map(p => below(mean(p.absorbed)))),
    },
  ];
}


/** 入力と再現：h+n は 3 つの仕方のどれでも Kistler の揺らぐ電流の方が平均 Γ が高く、最も強い揺らぎで線以上なのは KAM だけ。 */
function fluctuatingInput([steady, kistler]: Row[][]): Judgement[] {
  const rises = METHODS.map(m => ({ m, from: mean(one(steady, 'h+n', m)), to: mean(one(kistler, 'h+n', m)) }));
  const values = kistler[0]?.values ?? [];
  const strongest = values.indexOf(Math.max(...values));
  const atStrongest = METHODS.map(m => ({ m, ok: above(one(kistler, 'h+n', m)?.gammas[strongest]) }));
  const passing = atStrongest.filter(a => a.ok === true).map(a => a.m);
  return [
    {
      label: 'h+n の平均 Γ（定常電流 → 揺らぐ電流）',
      value: rises.map(r => `${NAMES[r.m]} ${shown(r.from)} → ${shown(r.to)}`).join('・'),
      line: '3 つとも上がる',
      ok: all(rises.map(r => (r.from === null || r.to === null ? null : r.to > r.from))),
    },
    {
      label: `最も強い揺らぎ（${values[strongest] ?? '—'}）で線以上の仕方`,
      value: passing.length > 0 ? passing.map(m => NAMES[m]).join('・') : 'なし',
      line: 'KAM だけ',
      ok: strongest < 0 || atStrongest.some(a => a.ok === null) ? null : passing.length === 1 && passing[0] === 'kam',
    },
  ];
}

export const checks = {
  /** 単一区画で、KAM・PCA・AE のどれでも発火の時刻まで合うのは a と c をまとめる組だけで、KAM は n を h 以外のゲートとまとめると時刻が合わなくなる。 */
  'only-a+c-keeps-spike-timing': { run: 'all-groupings-1comp', series: ['traub-1comp-steady'], judge: acOnly },
  /** m を膜電位へ吸収すると、吸収しなければ時刻まで合っていた組でも、発火の回数の差は一つ以内のまま、時刻が合わなくなる。 */
  'absorbing-m-loses-spike-timing': {
    run: 'all-groupings-1comp',
    series: ['traub-1comp-steady'],
    groupings: ['a+c', 'h+a', 'h+c', 'h+a+c'],
    judge: absorbM,
  },
  /** h+n をまとめた縮約は、どの仕方でも Kistler 1997 の揺らぐ電流のほうが定常電流より元の発火の時刻に合う。最も強い揺らぎで時刻まで合うのは KAM だけである。 */
  'fluctuating-input-keeps-timing-better': {
    run: 'fluctuating-input-1comp',
    series: ['traub-1comp-steady', 'traub-1comp-kistler'],
    judge: fluctuatingInput,
  },
} satisfies Checks;
