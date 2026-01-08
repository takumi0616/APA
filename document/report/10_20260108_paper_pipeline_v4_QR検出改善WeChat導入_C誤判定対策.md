# paper_pipeline_v4.py 改善レポート（WeChat QR 導入 + C→A 誤判定対策）

## 実施日時

2026 年 1 月 8 日

## 目的

`APA/paper_pipeline_v4.py` に対して、ユーザー要望の以下 2 点を **順番に**反映し、実行確認まで行う。

1. **B で QR コード検出ができない事例が多数**（特に低解像度 QR が読めない）
   - OpenCV 標準 `QRCodeDetector` から **WeChat QR Code Engine** へ変更
2. **C が A と誤判定される事例が多数**（3 つの四角検出ロジックが不十分）
   - 追加制約（マーカーサイズ比 / bbox 面積 ÷ ページ面積 / 三角形の距離比）を導入

## 前提（環境）

- OS: Windows 11
- Python: Miniconda ローカル（PATH を汚さない）
  - `C:\Users\takumi\develop\miniconda3\python.exe`
- OpenCV: 4.12.0
  - `cv2.wechat_qrcode_WeChatQRCode` が利用可能（opencv-contrib 相当）

## 対象ファイル

- `APA/paper_pipeline_v4.py`

## 改善 1: WeChat QR Code Engine を導入（フォーム B の QR 検出強化）

### 背景

OpenCV 標準の `cv2.QRCodeDetector` は、

- 透視補正後
- 回転後
- QR が小さくなった（低解像度化した）

ケースで検出率が下がる事例があった。

### 対応内容

#### 1) WeChat QR detector のラッパ実装

`WeChatQRDetector` クラスを `paper_pipeline_v4.py` 内に追加。

- `cv2.wechat_qrcode_WeChatQRCode` を **一度だけ初期化**して使い回し（初期化が重いため）
- `detect.prototxt / detect.caffemodel / sr.prototxt / sr.caffemodel` の存在チェック

#### 2) モデルファイル配置

既に以下に配置済みであることを確認。

```
APA/models/wechat_qrcode/
  detect.prototxt
  detect.caffemodel
  sr.prototxt
  sr.caffemodel
```

#### 3) CLI パラメータ追加

- `--wechat-model-dir`（デフォルト: `APA/models/wechat_qrcode`）

#### 4) フォーム B 判定（score_formB）で WeChat を優先

`score_formB()` 内で

1. WeChat で検出（成功したらそれを採用）
2. 失敗したら従来の robust（OpenCV QRCodeDetector + 前処理 + マルチスケール）

の順でフォールバックするように変更。

## 改善 2: C→A 誤判定対策（マーカー 3 点の制約追加）

### 背景

フォーム C が「偶然 3 つの四角っぽい領域」を拾うことで、フォーム A と誤判定されるケースがあった。

### 追加した制約

`MarkerGeometryConfig` と `validate_formA_marker_geometry()` を追加し、
`score_formA()` で **3 点マーカー検出ができた後**に追加判定するようにした。

#### 1) マーカー bbox 面積比

- 3 個の bbox 面積が極端に違う場合は誤検出とみなす
- 判定: `max(area)/min(area) <= max_marker_area_ratio`
  - デフォルト `max_marker_area_ratio = 3.0`

#### 2) bbox 面積 ÷ ページ面積の範囲

- `mean(bbox_area) / page_area` が極端に小さい/大きい場合は誤検出とみなす
  - デフォルト:
    - `min_marker_area_page_ratio = 5e-5`
    - `max_marker_area_page_ratio = 5e-3`

#### 3) 三角形の形（距離比）

- `dist(TL,TR) / dist(TL,BL)` がページの縦横比 `page_w/page_h` に近いはず、という制約
- 相対誤差で判定:
  - `abs(ratio - expected) / expected <= max_dist_ratio_relative_error`
  - デフォルト `max_dist_ratio_relative_error = 0.35`

この 3 条件をすべて満たした場合のみ FormA としてスコアを返し、
満たさない場合は FormA 判定自体を落とす（= C→A になりにくくする）。

## 実行確認

### 構文チェック

```bat
C:\Users\takumi\develop\miniconda3\python.exe -m py_compile APA\paper_pipeline_v4.py
```

### スモークテスト（A,B,C 各 1 枚、改悪 1）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v4.py --limit 1 --src-forms A,B,C --degrade-n 1 --template-topn 3 --log-level INFO --console-log-level INFO
```

出力ディレクトリ:

- `APA/output_pipeline/run_20260108_171605/`

ログ（`run.log`）主要結果（抜粋）:

- A: `ok=TRUE ok_warp=TRUE stage=done`（フォーム/テンプレ一致）
- B: `pred_form=B` で QR 判定は成功（warp は成功）が、今回の 1 ケースではテンプレ一致が外れ `template_ok=FALSE` となった
- C: `stage=form_unknown` になり `ok=TRUE ok_warp=FALSE`（**C は A/B に分類されないことが期待動作**）

### C→A 誤判定対策の確認

このスモークでは C が `form_unknown` になり、
ログ末尾統計でも `false_positive_as_A = 0` になっていることを確認。

## 変更点まとめ

- [x] **WeChat QR Code Engine を導入**し、フォーム B 判定で優先使用（失敗時は robust fallback）
- [x] **フォーム A 判定に幾何/面積制約を追加**し、C→A の誤判定を抑制
- [x] 動作確認（A/B/C 各 1 枚）を実施し、C が `form_unknown` となることを確認

## 補足（今後の改善候補）

- B の「テンプレ一致」精度は改悪条件・探索候補数に依存するため、
  - `--template-topn` を増やす（精度 ↑/速度 ↓）
  - `--match-max-side` を増やす（精度 ↑/速度 ↓）
    といった調整余地がある。
