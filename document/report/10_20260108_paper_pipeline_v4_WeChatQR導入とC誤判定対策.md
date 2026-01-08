# paper_pipeline_v4.py 改善レポート（WeChat QR 導入 / C→A 誤判定対策）

## 実施日時

2026 年 1 月 8 日

## 目的

`APA/paper_pipeline_v4.py` に対して、ユーザー要望の以下 2 点を**順番に確実に**改善する。

1. **フォーム B の QR コード検出率向上**

   - 低解像度 QR が OpenCV `QRCodeDetector` で読めない事例が多数
   - WeChat QR Code Engine を導入して検出性能を上げる

2. **フォーム C がフォーム A と誤判定される問題の抑制**
   - 3 点マーカー検出ロジックが甘く、偶然の 3 つの四角で A と判定される
   - 追加制約（マーカーサイズ比 / ページ面積比 / 三角形距離比）を導入する

## 前提（環境）

- OS: Windows 11
- Python: Miniconda ローカルインストール
  - `C:\Users\takumi\develop\miniconda3\python.exe`
- OpenCV: 4.12.0

## 実施内容

### 1) WeChat QR Code Engine の導入（フォーム B）

#### 実装方針

- OpenCV contrib 機能 `cv2.wechat_qrcode_WeChatQRCode` を利用
- WeChat エンジンは初期化が重いため、**1 回だけ初期化して使い回す**
- モデルファイルが無い／機能が無い場合は **OpenCV QRCodeDetector にフォールバック**して処理継続

#### 追加したもの

- `WeChatQRDetector` クラス
  - `detect.prototxt / detect.caffemodel / sr.prototxt / sr.caffemodel` を読み込み
  - `detectAndDecode` を実行し、`[{data, points, engine="wechat"}]` の形で返す
- `--wechat-model-dir` 引数
  - デフォルト: `APA/models/wechat_qrcode`
- `main()` 起動時に WeChat エンジンを初期化し、`score_formB` にバインド
  - `setattr(score_formB, "_wechat", wechat)`

#### モデル配置確認

既にリポジトリ内に配置済み：

```
APA/models/wechat_qrcode/
  detect.caffemodel
  detect.prototxt
  sr.caffemodel
  sr.prototxt
```

### 2) C→A 誤判定抑制（フォーム A 側の追加制約）

#### 追加した制約

ユーザー指定の制約を `validate_formA_marker_geometry()` として実装。

1. **マーカーサイズ比**
   - 3 個の bbox 面積の `max(area)/min(area)` が極端に大きい場合は除外
2. **bbox 面積 / ページ面積 の範囲**
   - マーカーが小さすぎる（ノイズ）／大きすぎる（別の枠）ケースを除外
3. **三角形の形（距離比）**
   - `dist(TL,TR) / dist(TL,BL)` が `ページ縦横比 (w/h)` と大きくズレていれば除外

これらを `score_formA()` 内の「A 判定 OK（markers=3）」の後段で必須チェックとして挿入し、
満たさない場合は **A 判定そのものを False** として扱う。

#### 実装した構造

- `MarkerGeometryConfig`（dataclass）
  - `max_marker_area_ratio`
  - `min_marker_area_page_ratio`
  - `max_marker_area_page_ratio`
  - `max_dist_ratio_relative_error`

## 動作確認

### 構文チェック

```bat
C:\Users\takumi\develop\miniconda3\python.exe -m py_compile APA\paper_pipeline_v4.py
```

### スモークテスト（A,B,C 各 1 枚）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v4.py --limit 1 --src-forms A,B,C --degrade-n 1
```

実行ログ抜粋（`run_20260108_161031`）：

- A: `done`（フォーム A 判定 OK）
- B: `done`（WeChat QR で QR 検出できたことを CSV で確認）
- C: `form_unknown`（A への誤判定は発生せず）

`summary.csv` の B 行では `form_decision_detail_json` 内に `"engine":"wechat"` が出力され、
WeChat エンジンが実際に使用されたことを確認。

---

## 追加修正（unicodeescape SyntaxError 対応）

ユーザー実行時に以下のエラーが発生した。

```text
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position ...: truncated \\UXXXXXXXX escape
```

### 原因

`paper_pipeline_v4.py` 冒頭の docstring に Windows パス `C:\\Users\\...` をそのまま記述していたため、
Python が `\\U` を Unicode エスケープ（`\\UXXXXXXXX`）として解釈し、
8 桁が続かないことで SyntaxError になっていた。

### 対応

docstring 内のパス表記を `C:/Users/...` のように **スラッシュ区切り**へ変更し、
`python -m py_compile APA\\paper_pipeline_v4.py` が通ることを確認した。

## 変更ファイル

- `APA/paper_pipeline_v4.py`
  - WeChat QR Code Engine の導入（初期化＋フォールバック）
  - フォーム A 判定への幾何・面積制約追加（C→A 誤判定抑制）
- `APA/document/report/10_20260108_paper_pipeline_v4_WeChatQR導入とC誤判定対策.md`
  - 本レポート（新規）

## 結論

- フォーム B の QR 検出は WeChat エンジン（超解像込み）を利用できるようになり、
  低解像度 QR の検出率改善が期待できる構成へ変更した。
- フォーム A 判定に「面積比・ページ面積比・三角形距離比」の制約を追加し、
  フォーム C をフォーム A と誤判定するケースを減らすためのガードを実装した。
