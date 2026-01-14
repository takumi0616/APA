# 2026-01-13 paper_pipeline_v11: UVDoc 組み込み & 7 段階化

## 目的

`paper_pipeline_v11.py` に **UVDoc（Document Unwarping）** を挿入し、
フォーム判定後の「湾曲/しわ補正」を行った上で XFeat マッチング → ワープへ進む **7 段階パイプライン**に更新する。

## 変更概要（要点）

### 1) パイプラインを 7 段階化

従来の 5/6 ステージ構成から、以下の 7 段階に整理。

1. degrade
2. docaligner
3. rectify
4. form decision
5. **UVDoc unwarp（新規）**
6. XFeat matching
7. warp（template 座標へ）

出力フォルダも番号を揃えて生成します：

- `1_degraded/`
- `2_doc/`
- `3_rectified/`
- `4_rectified_rot/`
- `5_uvdoc_unwarp/`
- `6_debug_matches/`
- `7_aligned/`

CSV の所要時間カラムも以下に更新：

- `elapsed_time_stage_5_uvdoc_unwarp_seconds`
- `elapsed_time_stage_6_xfeat_matching_seconds`
- `elapsed_time_stage_7_warp_seconds`

### 2) UVDoc（第三者リポジトリ）を `APA/third_party/UVDoc` に配置して読み込み

- `APA/third_party/UVDoc` の **ソース一式**を参照
- `APA/third_party/UVDoc/model/best_model.pkl` をチェックポイントとして使用
- UVDoc はパッケージ化されていないため、実行時に一時的に `sys.path` へ差し込み import する方針

### 3) フォーム B 取りこぼし修正（v11.6）

スモーク実行で B が `below_threshold` で落ちるケースがあり、
原因は「A 候補が _見つかった_ が閾値未満」でも、その時点で Unknown 確定して B 探索へ進まないことでした。

修正内容：

- A が検出できても **Unknown 閾値未満**の場合は、Unknown 確定せず **B 探索へフォールバック**
- それでも決まらない場合は Unknown を返すが、`detail.rejected` に棄却理由を残して診断しやすくした

## 動作確認（スモーク）

### 実行コマンド例

（画像保存無しで速度確認する例）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v11.py --src-forms A,B,C --limit 1 --degrade-n 1 --save-images none
```

（B のみ確認）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v11.py --src-forms B --limit 1 --degrade-n 1 --save-images none
```

### 結果（抜粋）

- A: `stage=done` 到達、UVDoc→XFeat→warp が通ることを確認
- B: v11.6 修正後、`stage=done` 到達を確認

出力は `APA/output_pipeline/run_YYYYmmdd_HHMMSS/` 配下に作成されます。

## 依存/注意事項

- **WeChat QR**（フォーム B 判定）は `cv2.wechat_qrcode_WeChatQRCode` が必要
  - `opencv-contrib` が必要
  - `APA/models/wechat_qrcode/` に 4 ファイル（detect/sr の prototxt/caffemodel）が必要
- **UVDoc checkpoint**
  - `APA/third_party/UVDoc/model/best_model.pkl` が必須
- 実行は CPU でも動作するが、XFeat/UVDoc が重いので大量ケースでは時間が掛かる

## 今後の改善候補（任意）

- `image/test` の評価を `--run-test-dataset` のようにフラグで ON/OFF できるようにする（現状は常に実行される）
- homography_unstable を減らすため、warp 許可条件（min_inliers/min_ratio）をケース別に調整、または画像側の前処理を検討
