# Neuro Surrogate

HH 型マルチコンパートメントニューロンの一部を、学習したサロゲート方程式へ置換して
再現性と演算コストを評価する研究領域。

## Language

**Surrogate scope**:
同じ種類・物理パラメータ両立規則に基づく、サロゲートの学習対象および適用可能対象。
_Avoid_: Train source, replaceables

**Surrogate specification**:
何を、どのデータと定式化で学習するかを宣言し、学習後も由来として残る仕様。
_Avoid_: Surrogate metadata

**Surrogate**:
学習仕様と学習済みの座標変換・閉包項を一体として扱うサロゲートモデル。
_Avoid_: Surrogate bundle

**Hybrid split**:
hybrid 定式化で、どの状態までを学習し、どこからを物理式で解き続けるかの分割。
分割位置も残す物理もニューロンモデルの性質なので `neurons` が持つ。
_Avoid_: Hybrid physics, physics config
