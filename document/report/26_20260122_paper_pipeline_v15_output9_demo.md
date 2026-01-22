# 26_20260122_paper_pipeline_v15_output9_demo

## 概要

paper_pipeline_v15 に **出力9（デモ画像）** を追加し、処理結果の確認を容易にした。

出力9は **時間計測の対象外** とし、case_total_s / stage_times には含めない。

## 目的

- **入力（degraded）と最終成果（aligned）を1枚で比較**できるようにする。
- DocAligner による紙領域推定と、フォーム判定根拠（マーカー/QR）を **元画像座標へ重畳**し、
  「どこを見て決めたか」を可視化する。

## 変更内容

### 1) 出力ディレクトリ追加

`APA/output_pipeline/run_YYYYmmdd_HHMMSS/9_demo/` を追加。

### 2) 出力9（demo9）生成

`process_one_case()` の warp（8_aligned）成功後に、以下のデモ画像を生成して保存する。

- ファイル名: `9_demo/{case_id}_demo9.jpg`
- 画像構成（左右横並び）
  - **左**: `degraded`（ズーム・クロップ無し）
    - DocAligner polygon（緑、margin 適用済み）
    - フォーム判定根拠
      - Form A: マーカー bbox（赤）
      - Form B: QR polygon（青）
  - **右**: `8_aligned`（最終）

### 3) 座標変換（逆投影）の考え方

フォーム判定根拠は「rectified_landscape → angle(0/180) 回転後（chosen）」座標系上で得られる。
これを degraded 元画像へ逆投影して重畳するため、以下を構成して逆行列を用いる。

- `H_degraded_to_rectified`（polygon rectify）
- `enforce_landscape` により 90deg CW 回転される場合があるため、その行列 `M_rect_to_land` を導入
- `rotate_image_bound(angle)` と整合する変換行列 `M_rect_to_chosen` を追加（0/180 最適化も含む）

最終的に:

```
H_degraded_to_rectified_landscape = M_rect_to_land @ H_degraded_to_rectified
H_degraded_to_chosen = M_rect_to_chosen @ H_degraded_to_rectified_landscape
H_chosen_to_degraded = inv(H_degraded_to_chosen)
```

これを用いて、bbox/points を degraded に変換して描画する。

## 出力例（動作確認）

以下のコマンドで簡易確認を実施。

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v15.py --src-forms A,C --limit 1 --degrade-n 1 --save-images all
```

実行結果（要点）:

- 出力 run: `APA/output_pipeline/run_20260122_155200/`
- `9_demo/` に demo9 が生成されることを確認
  - `A_1_deg00_demo9.jpg`
  - `test_A_3_1_deg00_demo9.jpg`
  - `test_A_3_2_deg00_demo9.jpg`
  - `test_A_4_1_deg00_demo9.jpg`
  - `test_A_4_2_deg00_demo9.jpg`
  - `test_A_5_1_deg00_demo9.jpg`
  - `test_A_5_2_deg00_demo9.jpg`

※ `homography_unstable` のケース（例: `test_A_6_1_deg00`, `test_A_6_2_deg00`）は 8_aligned が生成されないため、demo9 も生成されない。

## 実装ファイル

- `APA/paper_pipeline_v15.py`
  - 画像生成: `_generate_demo9_image()`
  - 回転行列を返す: `rotate_image_bound_with_matrix()`
  - enforce_landscape の回転行列: `_landscape_rotation_matrix_if_applied()`
  - demo 出力追加: `out_dirs["demo"] = .../9_demo`
  - warp 後に demo9 を保存（計測対象外）
