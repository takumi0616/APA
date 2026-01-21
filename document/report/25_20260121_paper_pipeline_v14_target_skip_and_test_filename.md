---
title: paper_pipeline_v14 改善（target改悪スキップ / test命名規則拡張）
date: 2026-01-21
script: APA/paper_pipeline_v14.py
---

# 概要

`paper_pipeline_v14.py` に対して、以下2点の改善を実施した。

1. **image/target の入力は改悪生成（degrade）を行わず、そのままパイプラインへ投入**する。
2. **image/test の命名規則を拡張**し、`A_3_1.png` のような形式からも GT（フォーム/テンプレ番号）を復元できるようにする。

# 改善1: image/target を改悪なしで投入

## 背景

`APA/image/target/` は実運用（現場撮影）に近い入力を置く想定で、
合成改悪（warp + shadow + bend）をかける必要がない。

## 実装

- `list_target_images()` を追加し、`APA/image/target/` 直下の画像（png/jpg/jpeg）を列挙。
- `main()` の「改悪事前生成フェーズ」にて、target 画像を以下で `DegradedCaseInput` 化して `degraded_inputs` に追加。
  - `source_dataset="target"`
  - `source_form="target"`
  - `degraded_bgr = src_bgr`（改悪なし）
  - `H_src_to_degraded = eye(3)`
  - `degrade_meta={"mode":"target_skip_degrade"}`
- `process_one_case()` の成功判定:
  - `source_dataset == "target"` の場合、**warp まで到達（stage=done）したら ok=True** とする。

## 補足

現在この環境では `APA/image/target/` が空のため、実データでの確認は未実施。

# 改善2: image/test の命名規則拡張（A_3_1.png 対応）

## 仕様

従来:

- `A_3.png` のように `{form}_{template}.png`

拡張後:

- `A_3_1.png` のように `{form}_{template}_{id}.png`
  - **先頭2要素（form と template）だけを GT として使用**
  - 3要素目以降（id 等）は GT 判定には使用しない

## 実装

- `parse_test_filename()` を更新し、stem を `_` split した先頭2要素のみを解釈する。

# 動作確認

## 実行コマンド

```
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v14.py \
  --src-forms A,C --degrade-n 1 --limit 1 --save-images none
```

## 結果（抜粋）

- A（合成）: `A_1_deg00` が `done` で終了（form/template 正解）
- C（合成）: `C_1_deg00` が `form_unknown` で終了（期待動作扱い）
- test（image/test）:
  - `test_A_3_deg00` / `test_A_4_deg00` / `test_A_5_deg00` は `done`
  - `test_A_6_deg00` は `homography_unstable`（inliers=58 のため warp 不許可）

出力: `APA/output_pipeline/run_20260121_121155/`

# 今後の課題

- `image/test` の一部ケース（例: A_6 系）で `homography_unstable` が発生するため、
  改悪条件や inlier 閾値の再調整、もしくは補正後画像の前処理強化を検討する。

