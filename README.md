---
description: "研究コードの repo の入口。要約と、全体の地図・実験の流れ・結果の置き場の図"
---

# SINDyNeuroSurrogate

神経細胞モデル（Traub、Hodgkin–Huxley）のゲート変数をまとめて減らし、状態の少ない代理モデル（サロゲート）に置き換える研究コード。まとめ方は二通りで、式から導く KAM 縮約と、データから学ぶ「圧縮 + [SINDy](https://arxiv.org/abs/1904.02107)」を同じ段取りで比べる。

脳の大規模シミュレーションでは、細胞を数千並べると計算より先にメモリの帯域で詰まる。区画あたりの状態を減らせば、そのぶんメモリが減る。

詳しい設計と規則は [AGENTS.md](AGENTS.md)、ファイルごとの役割は [remote/neurosurrogate/AGENTS.md](remote/neurosurrogate/AGENTS.md) にある。

## 全体の地図

手元と計算する機械は JSON だけを受け渡す。行きは `resolved/` の spec、帰りは API の小さい JSON。両方が使うコードは `shared/` だけ。最上位も `local/`（手元）・`remote/`（計算する機械）・`shared/`・`resolved/` に分かれていて、計算する機械へ送るのは `remote/`・`shared/`・`resolved/` だけ。

```mermaid
flowchart LR
  subgraph local["手元 local/"]
    config["人が書く宣言<br/>config/（プリセット、実験層、判定層）"]
    exp["宣言を読む・spec を組む・主張の判定<br/>experiments/"]
  end
  resolved[("送る spec<br/>resolved/")]
  subgraph machine["計算する機械 remote/（Dockerfile のイメージ）"]
    orch["段取り<br/>orchestrate/"]
    subgraph py["neurosurrogate/（Python）"]
      entry["計算の入口<br/>run.py"]
      surr["surrogate/<br/>学習して置き換える"]
      phys["physics/<br/>元のモデルを解く"]
    end
    runs[("runs/<br/>結果")]
    serve["結果を返す API<br/>serve/"]
  end
  format["受け渡す JSON の形<br/>shared/"]
  view["画面<br/>repo の外"]
  config --> exp
  exp -- "spec.json" --> resolved
  resolved --> orch
  orch -- "spec.json" --> entry
  entry --> surr
  surr --> phys
  entry -- "波形・係数・スパイクの特徴量" --> runs
  runs --> serve
  serve -- "状態、間引いた線と指標（JSON）" --> exp
  serve -- "間引いた線と指標（JSON）" --> view
  exp -- "主張の判定" --> view
  view -- "回す仕事（resolved/ のパス）" --> serve
  serve --> orch
  format -.- exp
  format -.- orch
  format -.- serve
```

## 一回の実験の流れ

```mermaid
sequenceDiagram
  participant L as 手元（local/）
  participant U as 画面
  participant A as API（remote/serve/serve.ts）
  participant O as 段取り（remote/orchestrate/orchestrate.ts）
  participant R as runs/
  participant P as 計算（Python）
  L->>L: 宣言から spec を組み、resolved/ に書く
  L->>A: resolved/ を送る（remote.sh）
  U->>A: resolved/ のパスで仕事を起こす
  A->>O: 切り離して回させる
  O->>R: spec の置き場はあるか
  alt もうある
    R-->>O: ある（回さない）
  else まだない
    O->>P: spec.json と置き場を渡す
    P->>P: シミュレーション・置換・特徴量
    P->>R: ディレクトリを書く
  end
  O->>R: 一覧を書き直す
  U->>A: 線と指標を頼む
  A->>R: 波形を読む
  A-->>U: 間引いた線と指標
  L->>A: 系列の量を頼む
  A-->>L: Γ と発火の回数
  L-->>U: 主張の判定
```

## 結果の置き場

```mermaid
flowchart LR
  subgraph models["runs/models/"]
    train["学習 run"]
  end
  subgraph inputs["runs/inputs/（入力ごと）"]
    original["original/<br/>元のモデル"]
    replaced["置き換えたモデル"]
  end
  replaced -- "学習 run を名指す" --> train
  replaced -. "比べる相手" .-> original
```
