# paper_pipeline_v4 summary 分析（analyze_v4.py 追加）レポート

## 実施日時

2026 年 1 月 9 日

## 目的

`paper_pipeline_v4.py` 実行結果（`output_pipeline/run_xxx/summary.csv`）から、

- **期待動作 KPI（expected_behavior）**
- **warp 成否（ok_warp）**
- **失敗ステージ内訳**（`homography_unstable` / `form_unknown` など）
- **フォーム別の精度/失敗要因**

を **自動集計**し、原因調査を高速化する。

## 対象

- 追加/更新: `APA/analyze_v4.py`
- 入力例: `APA/output_pipeline/run_20260109_112246/summary.csv`

## 背景（発生していた問題）

1. `APA/analyze_v4.py` が **0 bytes（空ファイル）**で、実行・レビューができなかった

2. Windows 環境で `print()` に日本語が含まれると、
   コンソールの既定エンコーディング（cp1252 等）により
   `UnicodeEncodeError` で落ちる可能性があった

## 対応内容

### 1) `analyze_v4.py` を実装（pandas 非依存）

- `summary.csv` を `csv.DictReader` で読み取り
- 以下を集計して Markdown レポート出力
  - 全体 KPI（`ok_expected_behavior` / `ok_warp`）
  - stage counts
  - フォーム別（A/B 精度、C の reject 成功率、C false positive）
  - 失敗内訳
    - `homography_unstable` の reject reason（`homography_inversion_reject_reason`）
    - B 誤判定（`is_predicted_form_correct=FALSE`）の内訳
    - C の false positive（`form_unknown` にならなかったケース）

### 2) 出力ファイル

- `analysis.md`（Markdown 集計レポート）
- `failures_expected_behavior_false.csv`（expected_behavior != TRUE の行だけ抽出）

### 3) Windows コンソールの文字化け/例外対策

- `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` を試行し、
  日本語 `print()` 起因の `UnicodeEncodeError` を回避
- `--output-md` 指定時は、標準出力ではなくファイル出力を優先

## 実行確認

### 実行コマンド（例）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\analyze_v4.py ^
  --input APA\output_pipeline\run_20260109_112246\summary.csv ^
  --output-md APA\output_pipeline\run_20260109_112246\analysis.md ^
  --output-failures-csv APA\output_pipeline\run_20260109_112246\failures_expected_behavior_false.csv
```

### 生成物

- `APA/output_pipeline/run_20260109_112246/analysis.md`
- `APA/output_pipeline/run_20260109_112246/failures_expected_behavior_false.csv`

## 分析結果（run_20260109_112246）

`analysis.md` 抜粋：

- total_cases: **54**
- ok_expected_behavior: **47 (87.0%)**
- ok_warp: **34 (63.0%)**

stage counts:

- done: 34
- form_unknown: 17
- homography_unstable: 3

フォーム別:

- A: form_acc 18/18 (100%), template_acc 18/18 (100%)
- B: form_acc 13/18 (72.2%), template_acc 13/18 (72.2%)
- C（reject 想定）: reject_success 16/18 (88.9%), false_positive_as_A 2

`homography_unstable` の主因:

- inlier_ratio<0.150 (0.137)
- inliers<10 (4)
- inliers<10 (7)

## まとめ

- [x] `analyze_v4.py` を実装し、`summary.csv` の自動集計（Markdown/失敗抽出 CSV）を可能にした
- [x] Windows コンソールのエンコーディング問題による `UnicodeEncodeError` を回避
- [x] run_20260109_112246 に対して分析結果を生成し、B の弱点（誤判定/unstable）と C の false positive を定量化できた
