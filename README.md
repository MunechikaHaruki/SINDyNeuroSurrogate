# SINDyNeuroSurrogate

神経細胞モデル（Traub、Hodgkin–Huxley）のゲート変数をまとめて減らし、状態の少ない代理モデル（サロゲート）に置き換える研究コード。まとめ方は二通りで、式から導く KAM 縮約と、データから学ぶ「圧縮 + [SINDy](https://arxiv.org/abs/1904.02107)」を同じ段取りで比べる。

脳の大規模シミュレーションでは、細胞を数千並べると計算より先にメモリの帯域で詰まる。区画あたりの状態を減らせば、そのぶんメモリが減る。

## 全体の地図

置き場は三つで、間を渡るのはデータだけ。TS は Python を import せず、Python は TS を知らない。

```mermaid
flowchart LR
  subgraph ts["evaluation/（TS）"]
    conf["宣言<br/>conf.json"]
    orch["段取り<br/>orchestrate.ts"]
    read["指標と読み方<br/>read/"]
  end
  subgraph py["neurosurrogate/（Python）"]
    entry["計算の入口<br/>run.py"]
    surr["surrogate/<br/>学習して置き換える"]
    phys["physics/<br/>元のモデルを解く"]
  end
  runs[("runs/<br/>結果")]
  conf --> orch
  orch -- "spec.json" --> entry
  entry --> surr
  surr --> phys
  entry -- "波形・係数・スパイクの特徴量" --> runs
  orch -- "一覧 index.json" --> runs
  runs --> read
```

- **宣言**：何を回すか（縮約の仕方、群分け、入力の系列）を書く。
- **段取り**：宣言から条件（spec.json）を組み、まだ無い run だけを Python に回させる。
- **計算**：元のモデルを解き、サロゲートを学習し、置き換えて回す。
- **結果**：図はこの repo では描かない。結果はデータとして runs/ に残し、読んで描く側は repo の外に置く。

## 一回の実験の流れ

条件が同じ run はもうあるので、回し直さない。

```mermaid
sequenceDiagram
  participant U as 人
  participant O as 段取り（TS）
  participant R as runs/
  participant P as 計算（Python）
  U->>O: 系列と学習 run の名前を渡す
  O->>O: 宣言から spec を組み、置き場を決める
  O->>R: 置き場はあるか
  alt もうある
    R-->>O: ある（回さない）
  else まだない
    O->>P: spec.json と置き場を渡す
    P->>P: シミュレーション・置換・特徴量
    P->>R: ディレクトリを書く
  end
  O->>R: 一覧を書き直す
```

## 結果の置き場

保存の単位は「学習したモデル」と「入力」の二つ。置き換えたモデルの結果は、使った学習 run を名指し、同じ入力で回した元のモデルと比べる。

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

## Python と TS の境界

| 渡るもの | 書く側 | 読む側 |
| --- | --- | --- |
| 実験の条件（spec.json） | TS（段取り） | Python |
| 実験の結果（runs/） | Python | TS |
| ゲートのレートの式（physics/ の JSON） | 人 | Python と TS |

形式の細部と設計の判断基準は [DESIGN.md](DESIGN.md)、ファイルごとの役割は [neurosurrogate/FILES.md](neurosurrogate/FILES.md) にある。

## セットアップ

[uv](https://docs.astral.sh/uv/) と [bun](https://bun.sh/) を使う。

```bash
uv sync
```
