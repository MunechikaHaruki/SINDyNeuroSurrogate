# todo

研究コードの手入れ。やっても数値も主張も変わらないもの（バグ、重複、形式）だけを置く。結果が変わるものは context-router の jdex の future-works（54）の冊に書く。

## やること

次の一手は子の `- [ ]` に書く。済んだ行は消してよい。

- [ ] Traub の密度などのパラメータを `neurosurrogate/physics/traub.json` に移し、Python（`TraubParams`）と TS（`traub.ts` の `g`・`E`・`capacitance`・`kappa`）の二重定義をなくす
  - [ ] traub19 が区画ごとに上書きする値と、単一区画の既定値の境目を決める
