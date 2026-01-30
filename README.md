# APA - Automatic Paper Alignment（カメラ画像から紙フォームを自動整列）

本ディレクトリ（`APA/`）は、**撮影した紙書類画像を自動で検出・補正し、テンプレート座標へ整列（alignment）する**ための検証/運用用コード一式です。

中心となる最終プログラムは **`APA/paper_pipeline_v18.py`** で、以下の機能を **静止画像の一括処理パイプライン**として統合しています。

- **改悪生成（degrade）**：テンプレをランダム四角形へ射影して背景合成（synthetic/test向け）
  - v18.5: **bend（しなり）** / **shadow（影・照明ムラ）** を追加
- **DocAligner** による紙領域検出（polygon→4点quadへ正規化／退化quadの修復）
  - v18.10〜: model/type/pad/scale を変えた **multi推論 + 候補評価**
  - no_detection 時の **advanced fallback（輪郭/エッジベースでの再推定）**
- 透視補正（rectify）と横長統一（`enforce_landscape`）
  - v18.17+: rectify前に padding を入れて「marginを取りたいのにclipされる」問題を緩和
- フォーム判定（フォームA=マーカー / フォームB=QR、フォームC=Unknownとして棄却）
  - 回転探索は **0°/180°の2方向のみ**（rectify後は横長統一しているため）
  - フォームBのQR検出は **WeChat QR エンジンのみ**（opencv-contrib必須）
- **UVDoc** による紙の湾曲補正（unwarp）
- **背景除算法（Background Division）** による照明ムラ/影の軽減（LABのLのみ補正）
- **XFeat** によるテンプレ照合（局所特徴） + Homography 安定性チェック + テンプレ座標へのワープ
  - v18では **テンプレ候補の事前絞り込みは廃止**し、フォーム確定後は **全テンプレ照合**
- `summary.json` / `summary.csv` / `run.log` の出力（デバッグと評価向け）

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
- [`paper_pipeline_v18.py` の説明](#paper_pipeline_v18py-の説明)
  - [目的・設計方針](#目的設計方針)
  - [入力データセットの扱い（v18）](#入力データセットの扱いv18)
  - [処理フロー（ステージ別）](#処理フローステージ別)
  - [出力（runディレクトリ）](#出力runディレクトリ)
  - [実行方法（Quickstart）](#実行方法quickstart)
  - [主要CLIオプション（v18）](#主要cliオプションv18)
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
├── paper_pipeline_v18.py
├── README.md
├── third_party/
├── trash/
└── __pycache__/
```

---

## 各ディレクトリの内容

### `document/`（資料・設計・ログ）

実装・検証の背景や作業ログ、依存関係メモなどを保存しています（コード実行に必須ではありません）。

- `document/environment/` : 実行環境の仕様書（OS/ツール/Python/依存関係/再現手順など）
- `document/report/` : パイプライン改善ログ（時系列メモ）

---

### `image/`（入力画像・テンプレ・評価データ）

パイプラインの入力となる画像群と、テンプレート画像を保存します。

```text
APA/image/
├── A/            # フォームAのテンプレ（1.jpg〜6.jpg）
├── B/            # フォームBのテンプレ（1.jpg〜6.jpg）
├── C/            # フォームCのテンプレ（1.jpg〜6.jpg）※基本はUnknownとして棄却されるべき
├── test/         # 観測画像（png/jpg）: ファイル名からGT推定して評価（改悪生成あり）
├── target/       # 現場想定画像（jpg/png）: 改悪生成なしで処理
└── hard_target/  # 難例の現場画像（jpg/png）: 改悪生成なしで処理
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
  - `--degrade-n` 枚の改悪画像を生成してから本処理へ投入します（v18.7+）

- `image/target`
  - 現場画像想定（GTなし）
  - **改悪生成（degrade）を行わず、そのままパイプラインへ投入**します
  - 命名規則は固定せず、拡張子で列挙します

- `image/hard_target`
  - target と同様に **改悪なし** で投入します（難例想定）

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

`paper_pipeline_v18.py` は OpenCV の `cv2.wechat_qrcode_WeChatQRCode` を **WeChat-only** で使用します。
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
  - `paper_pipeline_v18.py` は `third_party/UVDoc/model/best_model.pkl` を読み、ステージ5で使用します

参考: UVDoc公式 README は `APA/third_party/UVDoc/README.md` を参照してください。

---

### `output_pipeline/`（パイプライン出力）

`paper_pipeline_v18.py` 実行時に、以下の形式で結果を出力します。

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

過去のパイプライン・実験スクリプトが残されています（参照用）。

---

### その他のトップレベルファイル

- `paper_pipeline_v18.py`
  - 本READMEの主対象。静止画像を一括処理する統合パイプライン。
- `.gitignore`
  - `__pycache__/`, `.venv/`, `output_pipeline/`, `.vscode/` 等を除外。

---

## `paper_pipeline_v18.py` の説明

### 目的・設計方針

既存の検証コード（DocAligner / フォームA・B判定 / XFeat Homography）をベースに、
**静止画像の一括処理パイプライン**として統合・運用しやすくすることが目的です。

重視している点：

- **解像度差に強い**：polygon margin を紙サイズ比（ratio）で計算
- **検出率向上**：マーカー/QR に前処理バリアント、DocAligner multi + fallback
- **現実寄せの改悪**：bend（非線形）/ shadow（照明ムラ）
- **高速化**：テンプレ特徴キャッシュ、ターゲット特徴の使い回し
- **安定性**：Unknown 判定、Homography の信頼度チェック（inliers/cond/det）

---

### 入力データセットの扱い（v18）

本スクリプトは **複数の入力ソース**をまとめて処理します。

v18 のデフォルトは重い（synthetic + test + target + hard_target を処理）ため、
まずは **件数制限オプション**で絞って実行することを推奨します。

1. **synthetic（改悪あり）**: `APA/image/{A,B,C}/`
   - 対象フォームは `--src-forms` で指定
   - `--degrade-n` 枚ぶんの改悪画像を生成してから本処理へ投入

2. **test（改悪あり / GTあり）**: `APA/image/test/`
   - ファイル名から GT 推定（例: `A_3_1.png` → フォームA・テンプレ3が正解）
   - `--degrade-n` 枚ぶんの改悪画像を生成してから本処理へ投入

3. **target（改悪なし / GTなし）**: `APA/image/target/`
   - そのまま投入（現場画像想定）

4. **hard_target（改悪なし / GTなし）**: `APA/image/hard_target/`
   - そのまま投入（難例の現場画像想定）

#### `--test-limit / --target-limit / --hard-target-limit` のルール（v18.12+）

各データセットの処理件数は以下のルールで制御します。

- `0` : そのデータセットを **処理しない（skip）**
- `N > 0` : 先頭 `N` 枚だけ処理
- `N < 0` : **全件処理**

---

### 処理フロー（ステージ別）

1 case = 1枚の入力画像から生成した 1枚の（改悪）画像、という単位で処理します。

> v18.7: 改悪生成（degrade）は **最初に全ケース分をまとめて生成**し、以降の本処理へ投入します。
> 改悪生成の所要時間は計測対象外です。

#### Stage 1: 改悪生成（degrade）

`warp_template_to_random_view()` を使って、テンプレ画像をランダムな四角形へ射影し背景に合成します。
追加改悪（v18.5）として以下も入ります。

- **bend（しなり）**：サイン波の変位場で `cv2.remap` により非線形歪み
- **shadow（影/照明ムラ）**：紙領域マスクに対して斜めグラデ + 周辺減光 + ぼかし

#### Stage 2: DocAligner（紙領域検出）

- `detect_polygon_docaligner_multi()` により紙領域 polygon（4点）を推定します
- 退化quad（線/三角形/重複点）を検出し、必要ならエッジから **修復**します
- 失敗した場合は `stage=docaligner_failed` で終了します

#### Stage 3: 透視補正（rectify）

- polygon を外側に拡張（margin）してから透視補正
- margin はデフォルトで **紙サイズ比（ratio）**から自動計算
  - `--polygon-margin-px > 0` を指定すると固定pxで上書き
- 透視補正後は `enforce_landscape()` で **横長に統一**します

#### Stage 4: フォーム判定（0°/180°のみ）

`decide_form_by_rotations()` が、rectify後画像を **0° と 180° の2方向**だけ評価し、フォームと向きを確定します。

- フォームA：左上/右上/左下の **3点マーカー**検出で判定
  - `--marker-preproc {none,basic,morph}` で前処理強度を変更
- フォームB：右上の **QRコード**検出で判定
  - **WeChat QR エンジンのみ**利用（OpenCV標準 `QRCodeDetector` は使いません）
  - fast → robust の2段階（robustは必要時に最大1回）
- 判定不能/曖昧なら Unknown として `stage=form_unknown` で終了
  - `--unknown-score-threshold`：スコアが低い場合のUnknown
  - `--unknown-margin`：A/Bの差が小さい場合のUnknown（曖昧）

#### Stage 5: UVDoc（湾曲補正 / unwarp）

`UVDocUnwrapper` が `third_party/UVDoc/model/best_model.pkl` を読み、回転確定後の紙画像を unwarp します。

#### Stage 6: 背景除算法（Background Division）

`apply_background_division()` が LAB 色空間の L（明度）に対して背景（低周波）を推定し、
`cv2.divide` で照明ムラ/影/周辺減光を軽減します。

#### Stage 7: XFeat によるテンプレ照合

フォーム確定後、

- フォームAなら `image/A/1.jpg..6.jpg` の **全テンプレ**
- フォームBなら `image/B/1.jpg..6.jpg` の **全テンプレ**

に対して XFeat で局所特徴マッチングを行い、最良テンプレを選びます。

#### Stage 8: Homography 安定性チェック → テンプレ座標へワープ

`safe_invert_homography()` により以下を満たす場合のみ逆行列化して warp します。

- inliers 数が `--min-inliers-for-warp` 以上
- inlier_ratio が `--min-inlier-ratio-for-warp` 以上
- 行列の `det` が小さすぎない / `cond` が大きすぎない（`--max-h-cond`）

成功すると最終成果物として `8_aligned/` に保存されます。

---

### 出力（runディレクトリ）

`APA/output_pipeline/run_YYYYmmdd_HHMMSS/` 配下に、処理順が分かるように番号付きで出力します。

| ディレクトリ       | 内容                                                                    |
| ------------------ | ----------------------------------------------------------------------- |
| `1_degraded/`      | 改悪画像（synthetic/testは生成、target/hard_targetは入力そのまま）      |
| `2_doc/`           | DocAligner polygon 可視化                                               |
| `3_rectified/`     | 透視補正後（横長統一）                                                  |
| `4_rectified_rot/` | フォーム確定に使った回転後画像（根拠も描画）                            |
| `5_uvdoc_unwarp/`  | UVDoc unwarp 結果                                                       |
| `6_bgdiv/`         | 背景除算法（照明ムラ補正）後                                            |
| `7_debug_matches/` | best template のマッチ可視化（※extra出力が有効な場合）                  |
| `8_aligned/`       | **最終成果物**（テンプレ座標にワープ）                                  |
| `9_demo/`          | デモ可視化（左=degraded+逆投影、右=aligned） （※extra出力が有効な場合） |

また、以下を出力します。

- `summary.json` : 全ケースの詳細ログ（機械向け）
- `summary.csv` : 解析しやすいフラットな表（**フルパスは出さない方針**：基本はファイル名のみ）
- `run.log` : 実行ログ

#### `--save-images` による保存量制御

デバッグ画像の保存量を制御できます（IOがボトルネックになりやすいため）。

- `--save-images all` : 常に保存
- `--save-images fail` : `stage!=done` のケースのみ保存（成功ケースは保存しない）
- `--save-images none` : 一切保存しない

ただし以下は成果物/解析用のため **save-images の設定に関わらず保存されます**。

- `8_aligned/`（最終成果物）

また `7_debug_matches/` と `9_demo/` は **`--extra-outputs all`（または auto で accurate の場合）** に限り生成され、
生成される場合は save-images に関わらず保存されます。

---

### 実行方法（Quickstart）

#### Windows（このリポジトリ構成の想定）

リポジトリルート（`.../develop`）から：

```bat
:: まずは “最小実行” を推奨（デフォルトは重いので limit を明示）
C:/Users/takumi/develop/miniconda3/python.exe APA/paper_pipeline_v18.py --src-forms A --limit 1 --degrade-n 1 --test-limit 0 --target-limit 0 --hard-target-limit 0
```

#### macOS/Linux（例）

```bash
.venv/bin/python APA/paper_pipeline_v18.py --src-forms A --limit 1 --degrade-n 1 --test-limit 0 --target-limit 0 --hard-target-limit 0
```

#### パラメータ説明だけ表示

```bat
C:/Users/takumi/develop/miniconda3/python.exe APA/paper_pipeline_v18.py --explain
```

#### target だけを処理したい（改悪生成をしない）

```bat
:: synthetic/test をスキップして target だけ流す例
C:/Users/takumi/develop/miniconda3/python.exe APA/paper_pipeline_v18.py --src-forms "" --test-limit 0 --target-limit -1 --hard-target-limit 0
```

#### hard_target だけを処理したい

```bat
C:/Users/takumi/develop/miniconda3/python.exe APA/paper_pipeline_v18.py --src-forms "" --test-limit 0 --target-limit 0 --hard-target-limit -1
```

---

### 主要CLIオプション（v18）

> 全引数は `--explain` および `paper_pipeline_v18.py` 冒頭docstringも参照してください。

#### 入力/件数

- `--src-forms A,B,C` : synthetic入力として処理するフォームフォルダ（空文字で synthetic を skip）
- `--limit N` : synthetic向け。各フォームの先頭N枚だけ処理（0=全て）
- `--test-limit N` : test向け。`0=skip, N>0=先頭N枚, N<0=全件`
- `--target-limit N` : target向け。`0=skip, N>0=先頭N枚, N<0=全件`
- `--hard-target-limit N` : hard_target向け。`0=skip, N>0=先頭N枚, N<0=全件`

#### 改悪生成（synthetic/testのみ）

- `--degrade-n N` : 1枚の入力から生成する改悪画像枚数
- `--degrade-w`, `--degrade-h` : 改悪画像のキャンバスサイズ
- `--max-rot` : 回転強度（>=180 で 0..360 の一様回転モード）
- `--rotation-mode uniform|snap`, `--snap-step-deg` : 回転角の生成方法
- `--perspective` : 射影ゆがみ量
- `--min-visible-area-ratio` : 紙の見えている最小比率
- `--seed` : 再現性のための乱数シード

#### 速度プロファイル（任意）

- `--speed-profile auto|fast|accurate` : 速度/精度の切替
  - **現行v18では auto は accurate 相当（精度優先がデフォルト）**です
- `--extra-outputs auto|all|none` : debug_matches / demo9 を出すか
- fast向け微調整:
  - `--fast-docaligner-input-max-side`
  - `--fast-rectified-max-side`
  - `--fast-match-input-max-side`
  - `--fast-skip-uvdoc`
  - `--fast-skip-bgdiv`

#### DocAligner（紙検出）

- `--docaligner-model` : `lcnet050/lcnet100/fastvit_t8/fastvit_sa24`
- `--docaligner-type` : `point/heatmap`
- `--docaligner-max-side` : rectify後の最大辺（大きいほど精度↑/遅い）
- `--polygon-margin-ratio` : polygon を外側に広げる比率（解像度差に強い）
- `--polygon-margin-px` : 固定pxマージン（>0で ratio を上書き）

#### フォーム判定

- `--marker-preproc none|basic|morph` : フォームAマーカー検出の前処理
- `--unknown-score-threshold` : フォーム判定スコアが低い場合のUnknown閾値
- `--unknown-margin` : A/Bスコア差が小さい場合のUnknown閾値（曖昧）

#### WeChat QR（フォームB）

- `--wechat-model-dir` : `detect/sr` の 4 モデルファイルを置いたディレクトリ

#### XFeat（テンプレ照合）

- `--device auto|cpu|cuda` : XFeat/UVDoc の実行デバイス
- `--top-k` : 特徴点数（増やすと精度↑/遅い）
- `--match-max-side` : マッチング用の最大辺（増やすと精度↑/遅い）

#### Homography（warp可否）

- `--min-inliers-for-warp`
- `--min-inlier-ratio-for-warp`
- `--max-h-cond`

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
- capybara（DocAligner 依存。想定は **capybara-docsaid**）
- Pillow（日本語ラベル描画）

#### WeChat QRについて（重要）

- フォームB判定は WeChat QR のみです。
- 以下のいずれかを処理する場合、WeChat QR が利用できないと **起動時にエラー終了**します。
  - `--src-forms` に `B` を含む
  - `--test-limit != 0`（testはA/B混在し得る）
  - `--target-limit != 0` / `--hard-target-limit != 0`（現場画像はA/B混在し得る）

#### XFeat（torch.hub）と git について

XFeat は `torch.hub.load("verlab/accelerated_features", "XFeat", ...)` でロードします。
環境によっては `torch.hub` が内部で `git` を呼ぶため、
`ensure_portable_git_on_path()` が Portable Git を一時的に PATH に追加します（Windows向け）。

#### JPEG保存の高速化（任意）

- 可能なら `python-turbojpeg` を優先して JPEG 保存を高速化します。
- import に失敗した場合は `cv2.imwrite` へフォールバックします。

#### 日本語描画フォント

- OpenCV の `cv2.putText` は日本語描画に非対応のため、Pillow で描画します。
- `APA_FONT_PATH` 環境変数を設定すると任意フォントを優先できます。
