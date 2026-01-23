# 28_20260123 paper_pipeline_v15 自己完結化（test_recovery_paper 依存排除）

## 概要

`APA/paper_pipeline_v15.py` を **単体実行可能** にするため、従来 `APA/test_recovery_paper.py` に置かれていた補助実装（改悪生成、XFeat、マーカー検出など）を本ファイルへ統合した。

これにより、`paper_pipeline_v15.py` は **`test_recovery_paper.py` を import せずに**動作する。

## 目的 / 背景

- パイプライン本体を 1 ファイルで運用できるようにし、検証コードとの依存関係を解消して保守性を上げる。
- 実験用途だけでなく、現場投入（`image/target`）の入力にも対応しやすくする。

## 変更内容（実装）

対象ファイル: `APA/paper_pipeline_v15.py`

### 1) 自己完結化のための実装を移植

以下の機能群を `test_recovery_paper.py` 相当の実装として、`paper_pipeline_v15.py` 内へ統合した。

- `ensure_portable_git_on_path()` / `now_run_id()`
- 改悪生成: `warp_template_to_random_view()`
- 改悪追加（v15.5）:
  - 紙のしなり（非線形 remap）: `maybe_apply_bend()` / `maybe_apply_bend_with_mask()`
  - 影/照明ムラ（紙領域マスク適用）: `maybe_apply_shadow()`
- フォームAマーカー検出（ベース実装）: `detect_formA_marker_boxes_base()`
- XFeat matching:
  - `XFeatMatcher`（`torch.hub`）
  - Homography least-squares refine: `refine_homography_least_squares()`
  - inlier 可視化: `draw_inlier_matches()`

### 2) docstring（仕様説明）の更新

docstring 内の処理フロー記述を、以下の現状仕様に合わせて更新した。

- 改悪生成は「本ファイル内に統合済みの実装」を使用
- `image/target` は改悪生成を行わず、そのまま処理へ投入
- `image/test` は推奨命名: `{A|B|C}_{template番号}_{id}.png`（先頭2要素のみをGTに使用）

### 3) バグ修正（自己完結化に伴う参照修正）

`detect_formA_marker_boxes()` 内で、未定義名 `_detect_formA_marker_boxes_base` を参照していたため、
本ファイル内の `detect_formA_marker_boxes_base()` を参照するよう修正。

## 動作確認

### 1) 構文チェック

```bat
C:\Users\takumi\develop\miniconda3\python.exe -m py_compile APA\paper_pipeline_v15.py
```

→ OK（構文エラーなし）

### 2) ヘルプ表示（--explain）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v15.py --explain
```

→ OK（主要パラメータ説明を表示して終了）

### 3) スモーク実行（軽量設定）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v15.py --src-forms C --limit 1 --degrade-n 1 --save-images none
```

#### 実行ログ要約（抜粋）

- `C_1_deg00` は `stage=form_unknown` で終了（期待動作: C は棄却）
- `image/test`（A系）で `stage=done` が多数
- 一部 `homography_unstable` が発生（例: `test_A_6_2_deg00`）
- `image/target` は複数 `done` が出る一方、`docaligner_failed` / `form_unknown` も混在

全体サマリ（ログ末尾）:

- `total=25`
- `ok_expected=14 (56.0%)`
- `ok_warp=13 (52.0%)`

## 既知の課題 / メモ

- `homography_unstable`:
  - inlier 数が `PIPELINE_DEFAULTS["warp"]["min_inliers"]` 未満だと warp を抑止する設計のため、境界ケースで失敗が出る。
- `docaligner_failed`（target画像）:
  - 現場画像の状態によって DocAligner が失敗しうる（撮影条件/画角/反射など）。
  - 追加の前処理や `docaligner_model` 切替の検討余地。

## 関連ファイル

- `APA/paper_pipeline_v15.py`
