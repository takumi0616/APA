# APA - Automatic Paper Alignment（カメラ映像から紙フォームを自動整列）

本ディレクトリ（`APA/`）は、**撮影した紙書類画像を自動で検出・補正し、テンプレート座標へ整列（alignment）する**ための検証/運用用コード一式です。
中心となる最終プログラムは `APA/paper_pipeline_v15.py` で、以下の機能を **静止画像の一括処理パイプライン**として統合しています。

- **DocAligner** による紙領域検出（4点 polygon 推定）
- 透視補正（rectify）と回転正規化
- フォーム判定（フォームA=マーカー / フォームB=QR、フォームC=Unknownとして棄却）
- **UVDoc** による紙の湾曲補正（unwarp）
- 背景除算法（Background Division）による照明ムラ/影の軽減
- **XFeat** によるテンプレ照合 + Homography によるテンプレ座標へのワープ

---

## 目次

- [ディレクトリ構造](#ディレクトリ構造)
- [各ディレクトリの内容](#各ディレクトリの内容)
  - [`document/`（資料・設計・ログ）](#document資料設計ログ)
  - [`image/`（入力画像・テンプレ・評価データ）](#image入力画像テンプレ評価データ)
  - [`models/`（WeChat QRモデル）](#modelswechat-qrモデル)
  - [`third_party/`（外部リポジトリ）](#third_party外部リポジトリ)
  - [`output_pipeline/`（パイプライン出力）](#output_pipelineパイプライン出力)
  - [`trash/`（旧版・実験コード）](#trash旧版実験コード)
  - [その他のトップレベルファイル](#その他のトップレベルファイル)
- [`paper_pipeline_v15.py` の説明](#paper_pipeline_v15py-の説明)
  - [目的・設計方針](#目的設計方針)
  - [入力データセットの扱い](#入力データセットの扱い)
  - [処理フロー（ステージ別）](#処理フローステージ別)
  - [出力（runディレクトリ）](#出力runディレクトリ)
  - [実行方法（Quickstart）](#実行方法quickstart)
  - [主要CLIオプション](#主要cliオプション)
  - [依存関係・注意点（重要）](#依存関係注意点重要)

---

## ディレクトリ構造

```text
APA/
├── .gitignore
├── .git_disabled/
├── document/
├── image/
├── models/
├── output_pipeline/
├── paper_pipeline_v15.py
├── README.md
├── third_party/
├── trash/
└── __pycache__/
```

---

## 各ディレクトリの内容

### `document/`（資料・設計・ログ）

実装・検証の背景や、作業ログ、依存関係メモなどを保存しています（コード実行に必須ではありません）。

- `document/environment/`
  - 例: `windowsCompany.md`, `macbookair.md`
  - **実行環境の仕様書**（OS/ツール/Python/依存関係/再現手順など）
- `document/order/`
  - タスク要件の PDF 等
- `document/paper/`
  - `DocAligner.md`, `XFeatMatching.md` とその PDF
  - アルゴリズム/論文要約や検証メモ
- `document/plan/`
  - 設計メモや計画書
- `document/prompt/`
  - `DeepResearch/` / `Editer/` など、調査/編集用のプロンプト履歴
- `document/report/`
  - `paper_pipeline_v2`〜`v15` の改善ログ（時系列メモ）

> 「どの改善がいつ入ったか」を追う場合は、`document/report/` が最も有用です。

---

### `image/`（入力画像・テンプレ・評価データ）

パイプラインの入力となる画像群と、テンプレート画像を保存します。

```text
APA/image/
├── A/        # フォームAのテンプレ（1.jpg〜6.jpg）
├── B/        # フォームBのテンプレ（1.jpg〜6.jpg）
├── C/        # フォームCのテンプレ（1.jpg〜6.jpg）※基本はUnknownとして棄却されるべき
├── test/     # 観測画像（png/jpg）: ファイル名からGT推定して評価
└── target/   # 現場想定画像（jpg/png）: 改悪生成なしで処理
```

- `image/A`, `image/B`, `image/C`
  - `1.jpg`〜`6.jpg` を前提（対象番号は `PIPELINE_DEFAULTS["template_numbers"]` で変更可）
  - **フォームA/B判定後のテンプレ照合（XFeat）**に使用
- `image/test`
  - `.png/.jpg/.jpeg` を列挙して処理します
  - **ファイル名からGT（正解フォーム/テンプレ番号）を推定して評価**します
    - 推奨命名: `{A|B|C}_{template番号}_{id}.png` 例: `A_3_1.png`
      - 先頭2要素（`A` と `3`）が GT
      - 3要素目以降は識別子で、GT判定には使いません
- `image/target`
  - 現場画像想定。
  - **改悪生成（degrade）を行わず、そのままパイプラインへ投入**します
  - 命名規則は固定せず、拡張子で列挙します

---

### `models/`（WeChat QRモデル）

フォームB判定に使用する **WeChat QRCode エンジン**用モデルを置きます。

```text
APA/models/
└── wechat_qrcode/
    ├── detect.prototxt
    ├── detect.caffemodel
    ├── sr.prototxt
    ├── sr.caffemodel
    ├── README.md
    └── qrcode.py
```

- `detect.*` : QRコード検出（CNN detector）
- `sr.*` : 小さいQR向けの超解像（QRSR）
- `README.md` : WeChatCV による説明（同梱）
- `qrcode.py` : OpenCV wechat_qrcode のデモスクリプト（同梱）

`paper_pipeline_v15.py` は OpenCV の `cv2.wechat_qrcode_WeChatQRCode` を **WeChat-only** で使用します。
したがって **opencv-contrib-python が必須**で、かつ上記 4 モデルファイルが必要です。

---

### `third_party/`（外部リポジトリ）

外部プロジェクト（サブモジュール相当）を配置しています。

```text
APA/third_party/
└── UVDoc/
    ├── model/best_model.pkl
    ├── demo.py
    ├── model.py
    ├── utils.py
    └── ...
```

- `UVDoc/`
  - 書類の湾曲補正（unwarp）を行う **UVDoc** を同梱
  - `paper_pipeline_v15.py` は `third_party/UVDoc/model/best_model.pkl` を読み、ステージ5で使用します

参考: UVDoc公式 README は `APA/third_party/UVDoc/README.md` を参照してください。

---

### `output_pipeline/`（パイプライン出力）

`paper_pipeline_v15.py` 実行時に、以下の形式で結果を出力します。

```text
APA/output_pipeline/
└── run_YYYYmmdd_HHMMSS/
    ├── 1_degraded/
    ├── 2_doc/
    ├── 3_rectified/
    ├── 4_rectified_rot/
    ├── 5_uvdoc_unwarp/
    ├── 6_bgdiv/
    ├── 7_debug_matches/
    ├── 8_aligned/
    ├── 9_demo/
    ├── summary.json
    ├── summary.csv
    └── run.log
```

`.gitignore` により `output_pipeline/` は Git 管理対象外です。

---

### `trash/`（旧版・実験コード）

過去のパイプライン・実験スクリプトが残されています。

- `paper_pipeline_v2.py`〜`paper_pipeline_v14.py` : 旧版の履歴
- `test_*.py` : カメラキャプチャ、DocAligner検証、フォーム検出検証など
- `analyze_*.py` : runログや summary の分析用スクリプト

基本的に **最終版は `paper_pipeline_v15.py`** を参照してください。

---

### その他のトップレベルファイル

- `paper_pipeline_v15.py`
  - 本READMEの主対象。静止画像を一括処理する統合パイプライン。
- `.gitignore`
  - `__pycache__/`, `.venv/`, `output_pipeline/`, `.vscode/` 等を除外。
- `__pycache__/`
  - Python のバイトコードキャッシュ。
- `.git_disabled/`
  - 目的は環境により異なります（Gitの一時無効化・退避用途など）。
  - 実行に必須ではありません。

---

## 主要ファイル一覧（「中身/役割」の要約）

> `document/` 配下はファイル数が多いため、ここでは「用途の分類」と「代表的なファイル」を中心に要約します。

| パス                                         | 種別     | 何が書かれている/何をする                                                                                                   |
| -------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------- |
| `APA/paper_pipeline_v15.py`                  | Python   | **最終パイプライン本体**。改悪生成→紙検出→透視補正→フォーム判定→UVDoc→背景除算→XFeat→warp→サマリ出力。CLI引数もここで定義。 |
| `APA/README.md`                              | Markdown | 本ドキュメント。リポジトリの使い方/構造/注意点。                                                                            |
| `APA/.gitignore`                             | 設定     | `output_pipeline/` や `.venv/`、`__pycache__/` 等を Git 管理対象外にする設定。                                              |
| `APA/models/wechat_qrcode/README.md`         | Markdown | WeChat QRCode detector の説明（CNN detector + 超解像 QRSR 等）。                                                            |
| `APA/models/wechat_qrcode/qrcode.py`         | Python   | OpenCV の `wechat_qrcode_WeChatQRCode` を使った **公式デモ**。モデル4ファイルを同一ディレクトリに置いて実行する想定。       |
| `APA/third_party/UVDoc/README.md`            | Markdown | UVDoc の公式 README（demo/train/eval の手順）。                                                                             |
| `APA/third_party/UVDoc/model/best_model.pkl` | モデル   | UVDoc 推論で使う学習済みモデル。                                                                                            |
| `APA/document/environment/windowsCompany.md` | Markdown | Windows（社用PC）での再現手順・依存関係（OpenCV contrib 必須等）を整理した仕様書（v4向けだが考え方はv15にも近い）。         |
| `APA/document/report/*.md`                   | Markdown | バージョンごとの改善ログ（v2〜v15）。どの変更がどの背景で入ったかの履歴。                                                   |
| `APA/trash/paper_pipeline_v2.py`〜`v14.py`   | Python   | 旧版パイプライン（参照用）。                                                                                                |
| `APA/trash/test_*.py`                        | Python   | カメラ/検出/キャプチャ等の実験コード（参照用）。                                                                            |

---

## `paper_pipeline_v15.py` の説明

### 目的・設計方針

既存の検証コード（DocAligner / フォーム判定 / XFeat Homography）をベースに、
**静止画像の一括処理パイプライン**として統合・運用しやすくすることが目的です。

重視している点：

- **解像度差に強い**：polygon margin を紙サイズ比（ratio）で計算
- **検出率向上**：マーカー/QR に前処理バリアントを用意
- **現実寄せの改悪**：紙のしなり（非線形）/ 撮影時の影（照明ムラ）
- **高速化**：テンプレ特徴のキャッシュ、ターゲット特徴の使い回し
- **安定性**：Unknown 判定、Homography の信頼度チェック（inliers/cond/det）

---

### 入力データセットの扱い

本スクリプトは **複数の入力ソース**をまとめて処理します。

1. **synthetic（改悪あり）**: `APA/image/{A,B,C}/`
   - デフォルトは `1.jpg`〜`6.jpg`
   - 対象フォームは `--src-forms` で指定
   - `--degrade-n` 枚の改悪画像を生成してから本処理へ投入

2. **test（改悪あり / GTあり）**: `APA/image/test/`
   - `.png/.jpg/.jpeg` を列挙
   - ファイル名から GT を推定（例: `A_3_1.png` → フォームA・テンプレ3が正解）
   - `--degrade-n` 枚の改悪画像を生成してから本処理へ投入

3. **target（改悪なし / GTなし）**: `APA/image/target/`
   - `.png/.jpg/.jpeg` を列挙
   - 改悪生成を行わずそのまま本処理へ投入（現場画像の想定）

> v15.7 以降：改悪生成（degrade）は「最初に全ケース分をまとめて生成」し、
> その後に本処理（DocAligner〜warp）を実行します。
> 改悪生成の所要時間は計測対象外です。

---

### 処理フロー（ステージ別）

1 case = 1枚の入力画像から生成した 1枚の（改悪）画像、という単位で処理します。

#### Stage 1: 改悪生成（degrade）

`warp_template_to_random_view()` を使って、テンプレ画像をランダムな四角形へ射影し背景に合成します。
v15.5 以降は追加改悪として以下も入ります。

- **bend（しなり）**：サイン波の変位場で `cv2.remap` により非線形歪み
- **shadow（影/照明ムラ）**：紙領域マスクに対して斜めグラデーション + 周辺減光 + ぼかし

#### Stage 2: DocAligner（紙領域検出）

`detect_polygon_docaligner()` により紙領域 polygon（4点）を推定します。
失敗した場合は `stage=docaligner_failed` で終了します。

#### Stage 3: 透視補正（rectify）

- polygon を外側に拡張（margin）してから `polygon_to_rectified()` で透視補正
- margin はデフォルトで **紙サイズ比（ratio）**から自動計算
  - `--polygon-margin-px > 0` を指定すると固定pxで上書き
- 透視補正後は `enforce_landscape()` で **横長に統一**します（縦長入力でも後段の扱いを簡単にするため）

#### Stage 4: フォーム判定（0°/180°のみ）

`decide_form_by_rotations()` が、rectify後画像を **0° と 180° の2方向だけ**評価し、フォームと向きを確定します。

- フォームA：左上/右上/左下の **3点マーカー**検出で判定
  - `--marker-preproc {none,basic,morph}` で前処理強度を変更
  - フォームCの誤判定抑制として、マーカー周辺の「白地」チェック等の制約を持つ
- フォームB：右上の **QRコード**検出で判定
  - **WeChat QR エンジンのみ**利用（OpenCV標準 `QRCodeDetector` は使わない）
  - fast→robust の2段階（robustは必要時に最大1回）
- 閾値未満/曖昧なら Unknown として `stage=form_unknown` で終了
  - `--unknown-score-threshold`：スコアが低い場合のUnknown
  - `--unknown-margin`：A/Bの差が小さい場合のUnknown（曖昧）

#### Stage 5: UVDoc（湾曲補正 / unwarp）

`UVDocUnwrapper` が `third_party/UVDoc/model/best_model.pkl` を読み、
回転確定後の紙画像を unwarp して **より平坦な紙画像**を得ます。

#### Stage 6: 背景除算法（Background Division）

`apply_background_division()` が LAB 色空間の L（明度）に対して背景（低周波）を推定し、
`cv2.divide` で照明ムラ/影/周辺減光を軽減します。

#### Stage 7: XFeat によるテンプレ照合

フォーム確定後、

- フォームAなら `image/A/1.jpg..6.jpg` の **全テンプレ**
- フォームBなら `image/B/1.jpg..6.jpg` の **全テンプレ**

に対して XFeat で局所特徴マッチングを行い、最良テンプレを選びます。

> v15 では旧版の「グローバル特徴での候補絞り込み」は廃止し、常に全探索です。

#### Stage 8: Homography 安定性チェック → テンプレ座標へワープ

`safe_invert_homography()` により以下を満たす場合のみ逆行列化して warp します。

- inliers 数が `--min-inliers-for-warp` 以上
- inlier_ratio が `--min-inlier-ratio-for-warp` 以上
- 行列の `det` が小さすぎない / `cond` が大きすぎない（`--max-h-cond`）

成功すると最終成果物として `8_aligned/` に保存されます。

---

### 出力（runディレクトリ）

`APA/output_pipeline/run_YYYYmmdd_HHMMSS/` 配下に、処理順が分かるように番号付きで出力します。

| ディレクトリ       | 内容                                                   |
| ------------------ | ------------------------------------------------------ |
| `1_degraded/`      | 改悪画像（synthetic/testは生成、targetは入力そのまま） |
| `2_doc/`           | DocAligner polygon 可視化                              |
| `3_rectified/`     | 透視補正後（横長統一）                                 |
| `4_rectified_rot/` | フォーム確定に使った回転後画像（根拠も描画）           |
| `5_uvdoc_unwarp/`  | UVDoc unwarp 結果                                      |
| `6_bgdiv/`         | 背景除算法（照明ムラ補正）後                           |
| `7_debug_matches/` | best template のマッチ可視化                           |
| `8_aligned/`       | **最終成果物**（テンプレ座標にワープ）                 |
| `9_demo/`          | デモ用並列可視化（左=degraded+逆投影、右=aligned）     |

また、以下を出力します。

- `summary.json` : 全ケースの詳細ログ（機械向け）
- `summary.csv` : 解析しやすいフラットな表（フルパスは出さない方針）
- `run.log` : 実行ログ

#### `--save-images` による保存量制御

デバッグ画像の保存量を制御できます（IOがボトルネックになりやすいため）。

- `--save-images all` : 常に保存（従来通り）
- `--save-images fail` : `stage!=done` のケースのみ保存（成功ケースは保存しない）
- `--save-images none` : 一切保存しない（速度計測向け）

ただし以下は成果物/解析用として **設定に関わらず保存されます**。

- `8_aligned/`（最終成果物）
- `7_debug_matches/`（マッチ可視化）
- `9_demo/`（デモ可視化）

---

### 実行方法（Quickstart）

#### Windows（このリポジトリ構成の想定）

リポジトリルート（`.../develop`）から：

```bat
C:/Users/takumi/develop/miniconda3/python.exe APA/paper_pipeline_v15.py --limit 1
```

#### macOS/Linux（例）

リポジトリルートから（`.venv` を使用する例）：

```bash
.venv/bin/python APA/paper_pipeline_v15.py --limit 1
```

#### パラメータ説明だけ表示

```bat
C:/Users/takumi/develop/miniconda3/python.exe APA/paper_pipeline_v15.py --explain
```

#### target 画像だけを処理したい（改悪生成をしない）

`image/target/` は改悪生成なしで投入されますが、
スクリプトの都合上 `--src-forms` は空にできないため、以下のようにします。

```bat
:: synthetic/test の改悪生成を抑止して target だけ流す例
C:/Users/takumi/develop/miniconda3/python.exe APA/paper_pipeline_v15.py --src-forms A --degrade-n 0
```

> 注意: target 内にフォームB相当の画像が含まれる場合、QR検出に WeChat QR が必要です。
> `opencv-contrib-python` と `models/wechat_qrcode/` の 4ファイルが無いと、B判定できず Unknown になり得ます。

---

### 主要CLIオプション

> 全引数は `--explain` および `paper_pipeline_v15.py` 冒頭docstringも参照してください。

#### 入力/件数

- `--src-forms A,B,C` : synthetic 入力として処理するフォームフォルダ
- `--limit N` : デバッグ用に各フォームの先頭N枚だけ処理（0=全て）
- `--degrade-n N` : 1枚の入力から生成する改悪画像枚数（synthetic/testに適用）

#### 改悪生成（難易度調整）

- `--degrade-w`, `--degrade-h` : 改悪画像のキャンバスサイズ
- `--max-rot` : 回転強度（>=180 で 0..360 の一様回転モード）
- `--perspective` : 射影ゆがみ量
- `--min-visible-area-ratio` : 紙の見えている最小比率
- `--seed` : 再現性のための乱数シード

#### DocAligner（紙検出）

- `--docaligner-model` : `lcnet050/lcnet100/fastvit_t8/fastvit_sa24`
- `--docaligner-type` : `point/heatmap`
- `--docaligner-max-side` : rectify 後の最大辺（大きいほど精度↑/遅い）
- `--polygon-margin-ratio` : polygon を外側に広げる比率（解像度差に強い）
- `--polygon-margin-px` : 固定pxマージン（>0で ratio を上書き）

#### フォーム判定

- `--marker-preproc none|basic|morph` : フォームAマーカー検出の前処理
- `--unknown-score-threshold` : フォーム判定スコアが低い場合のUnknown閾値
- `--unknown-margin` : A/Bスコア差が小さい場合のUnknown閾値

#### WeChat QR（フォームB）

- `--wechat-model-dir` : `detect/sr` の 4 モデルファイルを置いたディレクトリ

#### XFeat（テンプレ照合）

- `--device auto|cpu|cuda` : XFeat/UVDoc の実行デバイス
- `--top-k` : 特徴点数（増やすと精度↑/遅い）
- `--match-max-side` : マッチング用の最大辺（増やすと精度↑/遅い）

#### Homography（warp可否）

- `--min-inliers-for-warp` : 逆行列化を許容する最小 inliers
- `--min-inlier-ratio-for-warp` : 逆行列化を許容する最小 inlier_ratio
- `--max-h-cond` : 行列条件数の上限（大きいと不安定）

#### 出力/ログ

- `--out` : 出力先ルート（その下に `run_...` を作成）
- `--save-images none|fail|all` : デバッグ画像の保存量制御
- `--log-level`, `--console-log-level` : ログレベル

---

### 依存関係・注意点（重要）

#### 必須の主要依存

- Python 3.x
- OpenCV（**opencv-contrib** 必須：`cv2.wechat_qrcode_WeChatQRCode` を使うため）
- PyTorch（XFeat と UVDoc に使用）
- DocAligner（紙領域検出）
- capybara（DocAligner 依存）
- Pillow（日本語ラベル描画）

#### WeChat QRについて

- `--src-forms` に `B` を含む場合、WeChat QR が利用できないと **起動時にエラー終了**します。
- モデルファイルは既定で `APA/models/wechat_qrcode/` を参照します。

#### XFeat（torch.hub）と git について

XFeat は `torch.hub.load("verlab/accelerated_features", "XFeat", ...)` でロードします。
環境によっては `torch.hub` が内部で `git` を呼ぶため、
`ensure_portable_git_on_path()` が Portable Git を一時的に PATH に追加します。

#### 画像保存（高速化の観点）

実運用や速度計測では `--save-images none` を推奨します。
（IO がボトルネックになりやすい）

#### 日本語描画フォント

- OpenCV の `cv2.putText` は日本語描画に非対応のため、Pillow で描画します。
- `APA_FONT_PATH` 環境変数を設定すると任意フォントを優先できます。
