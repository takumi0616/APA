# 29_20260125_paper_pipeline_v16_docaligner_multi_and_summary_fix

日付: 2026-01-25

## 概要

`paper_pipeline_v16.py` の安定性改善（特に target 画像で「紙がフレーム端ギリギリ」のケース）と、
`summary.json` 書き出しが途中で落ちて完走できない問題の修正を行った。

今回の変更は、**DocAligner の推論を1発勝負にしない**こと（複数条件で実行して最良候補を選ぶ）と、
**JSON 出力の安全化**が主目的。

## 背景 / 課題

- target 画像で紙がフレーム端に近い場合、
  DocAligner の入力パディングが小さいと角が欠け、
  退化 polygon（線/三角形に近い形状）になって後段（rectify〜フォーム判定）が全滅することがあった。
- `summary.json` の書き出し時に、辞書内に `numpy.ndarray` 等が混ざると `json.dump` が落ち、
  run が完走できないことがあった。

## 変更点

### 1) DocAligner 改善（multi 実行 + 候補選択）

- pad_px を画像サイズ比から自動推定するロジックを追加
  - `pad_px_auto = int(min(h,w) * ratio)`（min/max で clamp）
  - `pad_px` 未指定時は `max(pad_base, pad_auto)` を使用
- DocAligner を複数条件で実行して候補 polygon を生成し、候補を評価して最良を採用
  - model/type/pad/scale の候補を生成
  - 退化 quad（面積が小さい、辺長が短い、edge_ratio が大きすぎる）を除外
  - margin 候補も含めて rectify → 既存フォーム判定（0/180）スコアで最良を採用

### 2) polygon expand の安全化

- 従来の中心放射型 expand は透視歪みが強いと方向がズレやすいので、
  **辺法線方向の polygon offset（miter join）** を優先する実装へ変更。
  - 交点計算が不安定な場合のみ旧方式へフォールバック。

### 3) summary.json の安全化

- `json.dump(..., default=_json_default_for_dump)` を導入。
  - `numpy.ndarray` / numpy scalar / Path / set / bytes 等が混じっても落ちない
  - 巨大 ndarray は shape と preview のみ残す

### 4) 実行性改善（環境依存エラーの誘導）

- `capybara` / `docaligner` import 失敗時に、
  `.venv/bin/python ...` を使うべきことが分かるエラーメッセージを追加。

### 5) CLI（件数制限）

- `--test-limit` / `--target-limit` を追加し、デバッグ実行で件数を絞れるようにした。

## 動作確認

最小構成で run が最後まで完走し、`summary.json` / `summary.csv` が生成されることを確認。

実行コマンド（例）:

```bash
.venv/bin/python paper_pipeline_v16.py \
  --src-forms A,B,C \
  --limit 1 \
  --degrade-n 1 \
  --test-limit 1 \
  --target-limit 1 \
  --save-images fail
```

出力例:

- `output_pipeline/run_20260125_014754/summary.json` が生成され、JSON load 可能
- 同 run に `summary.csv` / `run.log` が生成

## メモ（今後）

- DocAligner multi は推論回数が増えるため重い。
  `PIPELINE_DEFAULTS["docaligner"]["multi"]["max_infer_runs"]` の調整で速度/精度をチューニングする。
