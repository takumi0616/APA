# 31_20260127_paper_pipeline_v18_speed_profile_fast

## 概要

`paper_pipeline_v18.py` に **speed profile（auto/fast/accurate）** を追加し、特に **target（現場撮影）向けに大幅高速化する fast モード**を実装した。

目的は「target を少数枚〜全件で回したときに _完走しない/遅すぎる_」問題の解消。

## 変更点（v18: fast profile）

### 1) speed profile の追加

- CLI
  - `--speed-profile {auto,fast,accurate}`
    - `auto`: dataset が target のときだけ fast、それ以外は accurate
  - `--extra-outputs {auto,all,none}`
    - `auto`: accurate=all / fast=none

### 2) DocAligner の fast 経路

- `detect_polygon_docaligner_fast()` を追加。
  - 入力を `--fast-docaligner-input-max-side` へ縮小
  - 試行は **pad 2回のみ（120/240）**
  - multi / advanced fallback を実行しない

### 3) decide の fast 経路

- `decide_form_fast()` を追加。
  - 回転は **0/180 のみ**（従来通り）
  - A: marker は **1種類のみ**（`--marker-preproc` をそのまま適用）
  - B: WeChat QR を **生画像1回のみ**（前処理/マルチスケール無し）
  - morph fallback / robust fallback は行わない

### 4) fast profile 時の後段省略（速度優先）

- `process_one_case()` 内で profile=fast の場合:
  - UVDoc / bgdiv を原則スキップ（`--fast-skip-uvdoc`, `--fast-skip-bgdiv` も用意）
  - matching 入力画像を `--fast-match-input-max-side` で縮小
  - 追加可視化（demo9/debug_matches）を抑制（extra-outputs=auto → fastはnone）

## スモーク結果（target 2枚）

### 実行コマンド

```bash
.venv/bin/python paper_pipeline_v18.py \
  --target-limit 2 \
  --test-limit 0 \
  --src-forms "" \
  --save-images none \
  --speed-profile fast \
  --console-log-level INFO
```

### 出力

- run: `output_pipeline/run_20260127_134519/`

### 時間（平均）

ログより（2ケース平均）:

- **avg per case: 0.353s**
  - docaligner: 0.105s
  - rectify: 0.007s
  - decide: 0.230s

### 参考（従来: 20260127_134118）

以前の target 2枚実行ログでは（ユーザー提示ログより）:

- **avg per case: 17.583s**
  - docaligner: 17.511s

→ target のボトルネックだった DocAligner を fast 経路で強く削減し、
**約 50倍（17.6s → 0.35s）** の短縮を確認。

## 精度面の状況（今回のスモーク）

今回の target 2枚では `stage=form_unknown (no_detection)` のまま。

ただし本レポートは「speed profile 追加と高速化の確認」を主目的としており、
精度改善（フォームAマーカー/QR検出改善、targetのno_detection救済など）は別途継続課題。

## 追加メモ / 今後

- fast profile でも起動時に UVDoc をロードしているため、
  **「起動時間」**（初回のみ）がまだ大きい可能性がある。
  - ※ per-case の計測には含まれないが、運用体験としては重要。
  - 将来的には speed-profile=fast の場合に UVDoc 初期化をスキップする案もあり。
