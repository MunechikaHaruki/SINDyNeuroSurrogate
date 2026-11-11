// 主張の判定。宣言（config/<主題>.yaml）の主張の名前ごとに判定の関数を置き、API の claims が runs から判定して返す。判定は探索の問い（改良の探索 sweep、性質の発見 property）にだけ置く。
import {
  GAMMA_LINE,
  NAMES,
  above,
  all,
  below,
  groupingsOf,
  mean,
  none,
  one,
  shown,
  type Judgement,
  type Row,
  type Judge,
} from './common.ts';

/** 比べる縮約の仕方。 */
const METHODS = ['kam', 'pca', 'ae'];

// 改良の探索（sweep.yaml）

/** ゲートの組と再現：ac は KAM・PCA・AE のどれでも線以上で、ほかにそういう組は無い。KAM で n を h 以外のゲートとまとめた組は線を下回る。 */
function acOnly([rows]: Row[][]): Judgement[] {
  const allAbove = (g: string) => all(METHODS.map(m => above(mean(one(rows, g, m)))));
  const others = groupingsOf(rows).filter(g => g !== 'ac');
  const nMixed = groupingsOf(rows).filter(g =>
    g.split('_').some(block => block.includes('n') && [...block].some(x => x !== 'n' && x !== 'h')),
  );
  const nGammas = nMixed.map(g => mean(one(rows, g, 'kam')));
  return [
    {
      label: 'ac の平均 Γ（KAM・PCA・AE）',
      value: METHODS.map(m => `${NAMES[m]} ${shown(mean(one(rows, 'ac', m)))}`).join('・'),
      line: `3 つとも ≥ ${GAMMA_LINE}`,
      ok: allAbove('ac'),
    },
    {
      label: 'ac 以外で 3 つとも線以上の組',
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
  const pairs = groupings.map(g => ({ g, kam: one(rows, g, 'kam'), kamm: one(rows, g, 'kamm') }));
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
        .map(p => `${p.g} ${p.kamm ? Math.max(...p.kamm.counts.map(([o, m]) => Math.abs(o - m))) : '—'}`)
        .join('・'),
      line: 'どれも ≤ 1',
      ok: all(pairs.map(p => (p.kamm ? p.kamm.counts.every(([o, m]) => Math.abs(o - m) <= 1) : null))),
    },
    {
      label: '吸収した KAM の平均 Γ',
      value: pairs.map(p => `${p.g} ${shown(mean(p.kamm))}`).join('・'),
      line: `どれも < ${GAMMA_LINE}`,
      ok: all(pairs.map(p => below(mean(p.kamm)))),
    },
  ];
}

// 性質の発見（property.yaml）

/** 入力と再現：hn は 3 つの仕方のどれでも Kistler の揺らぐ電流の方が平均 Γ が高く、最も強い揺らぎで線以上なのは KAM だけ。 */
function fluctuatingInput([steady, kistler]: Row[][]): Judgement[] {
  const rises = METHODS.map(m => ({ m, from: mean(one(steady, 'hn', m)), to: mean(one(kistler, 'hn', m)) }));
  const values = kistler[0]?.values ?? [];
  const strongest = values.indexOf(Math.max(...values));
  const atStrongest = METHODS.map(m => ({ m, ok: above(one(kistler, 'hn', m)?.gammas[strongest]) }));
  const passing = atStrongest.filter(a => a.ok === true).map(a => a.m);
  return [
    {
      label: 'hn の平均 Γ（定常電流 → 揺らぐ電流）',
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

/** 主張の名前 → 判定。 */
export const JUDGES: Record<string, Judge> = {
  'ac-only': acOnly,
  'absorb-m': absorbM,
  'fluctuating-input': fluctuatingInput,
};
