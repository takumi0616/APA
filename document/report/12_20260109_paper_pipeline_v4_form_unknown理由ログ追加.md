#+#+#+#+----------------------------------------------------------------

# paper_pipeline_v4.py 改善レポート（form_unknown の理由をログ/CSV に出力）

## 実施日時

2026 年 1 月 9 日

## 目的

`APA/paper_pipeline_v4.py` において、`stage=form_unknown`（本来 A のはずが Unknown 扱い等）となったケースで、

- 「どの判定チェックに引っかかったのか」

を **コンソールログ**と **summary.csv** に明示的に残し、原因調査を容易にする。

## 背景（問題）

以下のように `stage=form_unknown` で止まっているが、
どの判定（no_detection / below_threshold / ambiguous 等）で Unknown になったかがログ 1 行から分からず、
調査が難しい状態だった。

例（抜粋）:

```text
... stage=form_unknown gt_form=A pred_form= ...
```

## 対応内容

### 1) Unknown reason を決定ロジック側で保持

`decide_form_by_rotations()` の「Unknown で返す」経路に、`detail["reason"]` を必ず入れるように整理。

主な reason:

- `no_detection`（A/B とも検出ゼロ）
- `below_threshold`（スコア最大値がしきい値未満）
- `ambiguous`（A/B のスコア差が小さすぎる）
- `coarse_all_skipped` / `b_fast_no_qr_and_rescue_failed` / `no_decision_final_fallback` など

### 2) ログ 1 行と CSV で reason を独立カラムとして出力

#### 追加: `extract_form_unknown_reason()`

`FormDecision`（dict 化された form_decision）から、

- `form_unknown_reason`（短い文字列）
- `form_unknown_diagnostics_json`（軽量な補助情報）

を抽出する関数を追加。

#### ログ

`[CASE]` 1 行ログに以下を追加：

- `unknown_reason=...`

出力例（実行ログから）:

```text
2026-01-09 ... [WARNING] [CASE] id=A_4_deg01 ok=FALSE ok_warp=FALSE stage=form_unknown unknown_reason=no_detection gt_form=A pred_form= ...
```

#### CSV

`summary.csv` に以下の列を追加：

- `form_unknown_reason`
- `form_unknown_diagnostics_json`

これにより、Excel/スクリプトで `form_unknown_reason == no_detection` のみ抽出、のような分析が容易になる。

## 変更ファイル

- `APA/paper_pipeline_v4.py`
  - `extract_form_unknown_reason()` を追加
  - `log_case_summary()` に `unknown_reason=` を追加
  - `build_csv_row()` に `form_unknown_reason` / `form_unknown_diagnostics_json` を追加

## 実行確認

### 構文チェック

```bat
C:\Users\takumi\develop\miniconda3\python.exe -m py_compile APA\paper_pipeline_v4.py
```

### スモーク実行（例）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v4.py --src-forms A --limit 1 --degrade-n 1 --console-log-level WARNING
```

ログに `unknown_reason=...` が出ること、および `summary.csv` に `form_unknown_reason` 列が出ることを確認。

確認できた `summary.csv` の例（run_20260109_163659 など）:

- CSV ヘッダに `form_unknown_reason,form_unknown_diagnostics_json` が存在

## 補足（今後の調査のしやすさ）

- `form_unknown_reason` で大分類し、
  - `form_decision_detail_json`（決定ロジックの詳細）
  - `4_rectified_rot/`（判定用の回転画像）
    を突き合わせると、「検出そのものがゼロなのか」「しきい値/幾何制約で弾かれたのか」が追いやすくなる。
