# 31_20260127_paper_pipeline_v18_fix_taberror_and_smoketest

## 背景

`paper_pipeline_v18.py` に対して高速化・改善の途中で、
XFeat matching〜warp 周辺のコードが破損し、`IndentationError/TabError` により起動できない状態になっていた。

また、実行環境によっては `python` が conda を指しており、DocAligner 依存（capybara/docaligner）が読み込めないケースがある。

## 対応内容

### 1) 構文エラー修復（最優先）

- `process_one_case()` の XFeat matching〜warp で混入していた不正なインデント/重複処理を修正。
- `process_one_observed_case()` 内に残っていた壊れた古い処理ブロック（タブ混入）を削除。
  - `process_one_observed_case()` は `DegradedCaseInput` を組み立てて `process_one_case()` に委譲する薄いラッパーに変更し、
    将来の二重実装の破綻を防止。
- ファイル中の `\t`（タブ文字）混入が 0 であることを確認。

### 2) QR前処理共有のための引数追加（互換維持）

- `detect_qr_codes_wechat_multiscale(..., preprocessed_variants=None)` を追加。
  - 既存呼び出しは引数を渡さなくても動作（デフォルトは従来通り内部生成）。
  - 速度改善の下地として「同一画像に対し fast/robust を両方呼ぶ場合に前処理生成を共有」できる。

## 動作確認

### 1) コンパイル

```bash
python -m py_compile paper_pipeline_v18.py
```

→ OK

### 2) 最小スモークテスト

DocAligner 依存が入った venv を使って実行する必要がある。

```bash
.venv/bin/python paper_pipeline_v18.py \
  --src-forms A --limit 1 --degrade-n 1 \
  --test-limit 0 --target-limit 0 \
  --save-images none
```

結果（抜粋）:

- `stage=done`
- `ok_expected=TRUE` / `ok_warp=TRUE`
- 時間内訳（1case）
  - docaligner が支配的（今回の実行では約13.8s）
  - matching 約3.5s

## 補足（実行環境）

- `python paper_pipeline_v18.py ...` だと conda の Python が選ばれる場合があり、
  `capybara` が無い/期待するシンボルが無い等で DocAligner import が失敗する。
- 本リポジトリでは `.venv/bin/python ...` を推奨。
