# 23_20260114_paper_pipeline_v13_bgdiv_stage6

## 目的

paper_pipeline_v13 に「背景除算法（Background Division Method）」を stage6 として追加し、
UVDoc 成形後の画像に対して照明ムラ/影の除去を行い、後段の XFeat マッチングの安定性向上を狙う。

また、時間計測の対象を「本処理（DocAligner〜Warp）」に限定し、
改悪生成・途中画像保存・summary 出力は計測対象外とする（v13.8 方針）。

## 変更点（概要）

1. stage6 追加（UVDoc の後）

   - 関数: `apply_background_division(image_bgr)` を追加。
   - 実装:
     - LAB の L（明度）に大きめの GaussianBlur をかけて背景（低周波）を推定。
     - `cv2.divide(L, bg, scale=255)` で背景除算し、照明ムラ/影を軽減。
     - a/b 成分は保持し、L のみを補正して BGR に戻す。

2. パイプラインのステージ番号更新

   - 出力ディレクトリを以下に統一:
     - `5_uvdoc_unwarp/`
     - `6_bgdiv/`（新規）
     - `7_debug_matches/`
     - `8_aligned/`（成果物）

3. test dataset（`image/test`）への適用

   - synthetic と同様に、test の処理でも stage6 を適用。
   - stage6 出力解像度 `bgdiv_w/bgdiv_h` を item に保存し、CSV にも出力。

4. 時間計測の整理（v13.8 方針）

   - 計測対象: docaligner/rectify/decide/uvdoc/bgdiv/match/warp
   - 計測対象外:
     - 改悪生成（事前生成）
     - 途中画像保存（1_degraded〜7_debug_matches）
     - `summary.json`/`summary.csv` の書き出し
   - 例外: 成果物である `8_aligned` 保存のみ warp 時間に含める。

5. 付随修正
   - docstring の誤って混入した `+` 記号を修正。
   - `if __name__ == "__main__": raise SystemExit(main())` を復元し、単体実行可能に。

## スモークテスト

以下で最低限の動作確認を実施（CPU 環境）。

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v13.py --limit 1 --degrade-n 1 --src-forms A,C --save-images none
```

結果（ログ要約）:

- A_1_deg00: `stage=done` / form・template 正解 / warp 成功
- C_1_deg00: `stage=form_unknown`（期待通り）
- test データ（A_3/A_4 等）も `stage=done` を確認

## 出力

- `APA/output_pipeline/run_20260114_121631/`
  - `6_bgdiv/` に背景除算法後の画像が出力される
  - `summary.csv` に `background_division_image_resolution_width_px/height_px` が追加される

## メモ

- Background Division は照明ムラ/影の軽減に有効だが、
  blur sigma を大きくしすぎると局所コントラストが落ちるため、
  `PIPELINE_DEFAULTS["background_division"]` の `sigma_ratio/sigma_min/sigma_max` で調整可能。
