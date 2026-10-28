# todo

研究コードの手入れ。やっても数値も主張も変わらないもの（バグ、重複、形式）と、コードが原著とずれている箇所を置く。ずれを直して結果が変わるかは context-router の jdex の研究の実験（55.11 から）で確かめる。

## やること

次の一手は子の `- [ ]` に書く。済んだ行は消してよい。

- [ ] Traub の密度などのパラメータを `neurosurrogate/physics/traub.json` に移し、Python（`TraubParams`）と TS（`traub.ts` の `g`・`E`・`capacitance`・`kappa`）の二重定義をなくす
  - [ ] traub19 が区画ごとに上書きする値と、単一区画の既定値の境目を決める
- [ ] Traub19 の basal と apical の寸法が原著と逆（2026-09-28 原著 p.637 で確認）。原著は apical（10 区画）が 2.89 µm・120 µm・2188 µm²、basilar（8 区画）が 2.42 µm・110 µm・1673 µm²。
  コードは index 0..7 に apical の寸法、9..18 に basilar の寸法を入れている。φ（7769 と 5941）の比 1.308 は面積比 2188/1673 と一致し、φ は原著どおり 1..8 が basilar。
  参考にした snsbook の code/part1/multi/traub.c も同じく逆で、そこから引き継いだ。逆のままだと入力区画（原著 3・15）と basal・soma・apical の添字がずれる。
  - [ ] 原著どおりの版 `traub19_paper` を既定にするかを決める（研究の実験 extend）
