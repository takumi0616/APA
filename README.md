# APA - Automatic Paper Alignment（カメラ画像から紙フォームを自動整列）

本ディレクトリ（`APA/`）は、**撮影した紙書類画像（静止画）から紙領域を検出し、フォーム判定（A/B）→テンプレ照合→テンプレ座標へ整列**するための運用向けコードです。

## このディレクトリの「正」

現在のメインプログラムは **`APA/APA.py`** です。

- `APA.py` は薄いランチャーです（引数パース→実行）。
- 実処理は **`APA/APA_back.py`** に集約されています。

旧READMEは `paper_pipeline_v18.py` 前提の説明になっていましたが、**本ディレクトリの現状（`apa_input/` と `apa_template/` を処理する軽量運用版）**に合わせて、本READMEはそれを正として記述しています。

---

## 目次

- [ディレクトリ構造](#ディレクトリ構造)
- [入出力と運用ルール](#入出力と運用ルール)
- [処理フロー（ステージ）](#処理フローステージ)
- [実行方法（Quickstart）](#実行方法quickstart)
- [主要CLIオプション](#主要cliオプション)
- [依存関係・注意点（重要）](#依存関係注意点重要)
- [トラブルシュート](#トラブルシュート)

---

## ディレクトリ構造

```text
APA/
├── APA.py
├── APA_back.py
├── README.md
├── .gitignore
├── apa_input/          # 入力画像（jpg/png）
├── apa_template/       # テンプレ画像（A/B 各1.jpg〜6.jpg）
├── apa_output/         # 出力（demo9画像のみ）
├── apa_log/            # ログ（run.log / summary.csv）
├── models/             # WeChat QRモデル
├── third_party/        # UVDoc同梱など
├── document/           # 作業ログ・資料（実行必須ではない）
└── trash/              # 旧版・実験コード
```

---

## 入出力と運用ルール

### 入力: `apa_input/`

- `APA/apa_input/` 直下の **`.png/.jpg/.jpeg`** をファイル名に依存せず **上から順に処理**します。
- 例: `A_1_1.png` のような名前でも構いません（現状は評価用途の名残で入っているだけで、処理ロジックは名前に依存しません）。

### テンプレ: `apa_template/`

- `APA/apa_template/A/*.jpg` と `APA/apa_template/B/*.jpg` を使用します。
- 現状は `1.jpg`〜`6.jpg` の6枚を想定しています（ディレクトリ内の `.jpg` を列挙）。

### 出力: `apa_output/`

`APA/apa_output/run_YYYYmmdd_HHMMSS/` に **demo9相当の画像のみ**を出力します。

- 出力ファイル例: `0001_A_1_1_demo9.jpg`
- demo9画像の構成
  - 左: 入力画像 + 推定した紙polygon（緑） + フォーム判定根拠（赤=マーカー/青=QR） + stageラベル
  - 右: テンプレ座標へ整列した aligned 画像（失敗時は NO_OUTPUT）

### ログ: `apa_log/`

`APA/apa_log/run_YYYYmmdd_HHMMSS/` に以下を保存します。

- `run.log` : 実行ログ
- `summary.csv` : 画像ごとの結果一覧

---

## 処理フロー（ステージ）

`APA_back.py` の `run_apa_pipeline()` が概ね以下の順に処理します（静止画一括）。

1. **Stage 2: DocAligner（紙領域検出）**
   - `detect_polygon_docaligner_multi()` により paper polygon（4点quad）を推定
   - 退化quad（重複点/面積ゼロ等）は正規化し、必要なら OpenCVエッジから修復
2. **Stage 3: rectify（透視補正） + 横長統一**
   - polygon margin を付けて rectify（marginは ratio→px へ自動換算。固定px指定も可）
   - rectify前に padding を入れ、margin取りたいのにclipされる問題を緩和
3. **Stage 4: フォーム判定（A/B） + 向き確定（0°/180°）**
   - フォームA: 3点マーカー
   - フォームB: QR（WeChat QR エンジンのみ）
   - 0°/180°の2方向のみ探索（rectify後は横長に統一しているため）
4. **Stage 5: UVDoc（湾曲補正 / unwarp）**
5. **Stage 6: Background Division（照明ムラ/影の軽減）**
6. **Stage 7: XFeat（テンプレ照合）**
   - フォーム確定後、該当フォーム（AまたはB）のテンプレ全てと照合
7. **Stage 8: Homography安定性チェック → テンプレ座標へワープ**
   - inliers / inlier_ratio / det / cond で安定性チェック
8. **Stage 9: demo9画像生成 → 保存**（出力はこれだけ）

---

## 実行方法（Quickstart）

### Windows（このリポジトリ構成の想定）

リポジトリルート（`C:/Users/takumi/develop`）から：

```bat
:: ヘルプ
C:/Users/takumi/develop/miniconda3/python.exe APA/APA.py --help

:: まずは1枚だけ
C:/Users/takumi/develop/miniconda3/python.exe APA/APA.py --limit 1 --log-level INFO --console-log-level INFO
```

### macOS/Linux（例）

```bash
.venv/bin/python APA/APA.py --help
.venv/bin/python APA/APA.py --limit 1
```

---

## 主要CLIオプション

`APA.py` は `APA_back.build_arg_parser()` の引数をそのまま受けます。

### 入出力

- `--input-dir` : 入力画像ディレクトリ（既定: `APA/apa_input`）
- `--template-dir` : テンプレディレクトリ（既定: `APA/apa_template`）
- `--output-dir` : 出力ディレクトリ（既定: `APA/apa_output`）
- `--log-dir` : ログディレクトリ（既定: `APA/apa_log`）
- `--limit N` : 先頭N枚だけ処理（0=全て）

### ログ

- `--log-level {DEBUG,INFO,WARNING,ERROR}`
- `--console-log-level {DEBUG,INFO,WARNING,ERROR}`

### WeChat QR

- `--wechat-model-dir` : `detect/sr` の4モデルファイルを置いたディレクトリ（既定: `APA/models/wechat_qrcode`）

### DocAligner

- `--docaligner-model {lcnet050,lcnet100,fastvit_t8,fastvit_sa24}`
- `--docaligner-type {point,heatmap}`
- `--docaligner-max-side` : rectify後最大辺
- `--polygon-margin-ratio` / `--polygon-margin-min-px` / `--polygon-margin-max-px`
- `--polygon-margin-px` : 固定px（>0で ratio を上書き）

### フォーム判定

- `--rotation-max-workers` : 0°/180°スキャン並列数
- `--marker-preproc {none,basic,morph}`
- `--unknown-score-threshold` / `--unknown-margin`

### XFeat / 実行デバイス

- `--device {auto,cpu,cuda}` : `auto` は CUDA利用可なら cuda、それ以外 cpu
- `--top-k`
- `--match-max-side`

### warp許可（Homography安定性）

- `--min-inliers-for-warp`
- `--min-inlier-ratio-for-warp`
- `--max-h-cond`

---

## 依存関係・注意点（重要）

### 必須の主要依存

- Python 3.x
- OpenCV（**opencv-contrib 必須**：`cv2.wechat_qrcode_WeChatQRCode` を使うため）
- PyTorch（XFeat と UVDoc に使用）
- DocAligner（紙領域検出）
- capybara（DocAligner依存。想定: **capybara-docsaid**）
- Pillow（日本語ラベル描画）

### WeChat QRモデル

`APA/models/wechat_qrcode/` に以下が必要です。

- `detect.prototxt`
- `detect.caffemodel`
- `sr.prototxt`
- `sr.caffemodel`

### XFeat（torch.hub）と git

XFeat は `torch.hub.load("verlab/accelerated_features", "XFeat", ...)` でロードします。
環境によっては `torch.hub` が内部で `git` を呼ぶため、Windowsでは `ensure_portable_git_on_path()` が
`C:\Users\takumi\develop\git\bin`（Portable Git）を一時的に PATH に追加します。

### JPEG保存の高速化（任意）

`python-turbojpeg` が import できる場合はそれを使い、できない場合は `cv2.imwrite` にフォールバックします。

### 日本語描画フォント

OpenCVの `cv2.putText` は日本語が描けないため Pillow で描画します。
`APA_FONT_PATH` 環境変数を設定すると任意フォントを優先します。

---

## トラブルシュート

### `cv2.wechat_qrcode_WeChatQRCode is not available`

- `opencv-python` ではなく **`opencv-contrib-python`** が必要です。

### `WeChat QR model files not found`

- `--wechat-model-dir` が正しいか、4ファイルが揃っているか確認してください。

### `Missing 'capybara' module (expected: capybara-docsaid)` / `No module named docaligner`

- DocAligner一式が未導入です。環境の依存関係を入れてください。

### XFeatロードでgit関連のエラーが出る

- `git` が PATH に無い可能性があります。
- Windowsの場合、`C:/Users/takumi/develop/git/bin` に Portable Git を置くと自動でPATHへ追加されます。

### `UVDoc checkpoint not found`

- `APA/third_party/UVDoc/model/best_model.pkl` が存在するか確認してください。
