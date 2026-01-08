# 改悪 → DocAligner → フォーム判定 → XFeat → warp パイプライン実装レポート

## 実施日時

2026 年 1 月 8 日

## 目的

ユーザー要求の以下処理を**静止画像一括処理**として実装する。

1. `APA/image/A`, `APA/image/B`, `APA/image/C` の画像を入力として取得
2. 各画像に対して改悪処理（`APA/test_recovery_paper.py` の改悪生成実装を流用）
3. 改悪画像に対して DocAligner を行い、紙を囲う四角枠（polygon）を得る
   - polygon が得られなければその画像は処理終了
4. polygon 内の紙領域を透視補正して正面画像を得る
5. 透視補正画像を **0〜350 度（10 度刻み）** で回転させ、フォーム種別（A/B）を判定
   - A: 3 点マーク（左上・右上・左下）が検出できる
   - B: QR コードが検出できる
   - 確定できなければその画像は処理終了
6. 確定したフォームに応じて、該当テンプレ（`image/A` or `image/B`）全てと XFeat matching を行い、
   **対応点一致度（inliers）最大**のテンプレを採用して Homography を推定
7. 採用 Homography の逆で warped（テンプレ座標へ整列した画像）を作成して保存

本レポートは、2026/01/08 の追加改善（フォーム判定スコア・可視化・横長統一・日本語文字化け対策）を含め、
`paper_pipeline.py` の仕様・実装内容を **1 本に統合した詳細版**としてまとめる。

## 追加・作成したファイル

- `APA/paper_pipeline.py`
  - 本タスクのメインスクリプト（改悪 →DocAligner→ フォーム判定 →XFeat→warp）
- `APA/README.md`
  - `paper_pipeline.py` の実行例、出力説明を追記

## 実行環境・前提（重要）

- OS: Windows 11
- Python: Miniconda ローカルインストール（`C:\Users\takumi\develop\miniconda3`）
- Git: Portable Git（torch.hub が内部で git を呼ぶ場合があるため `ensure_portable_git_on_path()` を使用）

### 主要依存ライブラリ

- OpenCV: 画像処理全般
- docaligner-docsaid: 紙領域（四角形 polygon）の検出
- capybara-docsaid: DocAligner 周辺ユーティリティ（Windows 互換のため export patch を実施）
- torch / kornia: XFeat + LightGlue matching
- Pillow (PIL): **画像への日本語ラベル描画（cv2.putText では不可のため）**

※ 実行時に `TurboJPEG not available...` の警告が出る場合があるが、OpenCV フォールバックのため致命的ではない。

## 実装の要点

### 1) 入力画像とテンプレ画像

- 入力（改悪元）: `APA/image/{A,B,C}/*.jpg`（基本は 1〜6）
- テンプレ（位置合わせ先）:
  - フォーム A: `APA/image/A/*.jpg`
  - フォーム B: `APA/image/B/*.jpg`

フォーム判定で A/B が確定した時点で、該当するテンプレ集合に対して全探索する。

### 2) 改悪生成

`APA/test_recovery_paper.py` 内の `warp_template_to_random_view()` を流用。

主な挙動:

- ランダム背景を生成（グラデ + ノイズ + 線）
- テンプレをランダム四角形に射影して合成
- ブラー/ノイズを軽く付与
- 「紙が写っていない」改悪が出ないよう、テンプレマスク面積で **最小可視面積比**を満たすまでリトライ

主なパラメータ（`paper_pipeline.py` の CLI から指定）:

- `--max-rot`:
  - 改悪生成の回転強度
  - `>=180` の場合は **0..360 の一様乱数**モードに入り、上下逆・横向きが混ざる
- `--perspective`:
  - 射影ゆがみ量（大きいほど難しい）
- `--degrade-w/--degrade-h`:
  - 改悪画像のキャンバスサイズ
- `--min-visible-area-ratio`:
  - 改悪画像内で紙が占める最小面積比
- `--max-attempts`:
  - 条件を満たす改悪生成の最大リトライ回数

### 3) DocAligner（紙枠 polygon 取得）と透視補正

- `docaligner-docsaid` を使用し polygon（4 点）を取得
- polygon が紙端ギリギリだと QR/マーカーが欠ける場合があるため、
  `--polygon-margin`（デフォルト 80px）で polygon を外側に拡張してから透視補正

透視補正は `polygon_to_rectified()` で以下を行う:

- polygon の点順を `TL/TR/BR/BL` に整列（`order_quad_tl_tr_br_bl`）
- 辺長から出力サイズを推定（長辺を `--docaligner-max-side` で上限）
- `cv2.getPerspectiveTransform` で補正し、紙領域を正面画像へワープ

**追加改善:** 透視補正後の紙画像は `enforce_landscape()` により **横長（landscape）** に統一。

DocAligner 取得失敗時の扱い:

- `polygon` が `None` の場合、`stage=docaligner_failed` としてそのケースの処理を終了

（summary に残るため、後から失敗率や傾向を解析できる）

### 4) フォーム判定（回転スキャン）

ユーザー指定に従い **0〜350 度を 10 度刻み（36 通り）** でスキャン。

フォーム B の QR 検出は、透視補正＋回転後は条件が厳しくなるため、
以下の二段階方式に変更して安定化/高速化した。

- FAST 検出（回転スキャン用）: 最低限の前処理 + 軽いマルチスケールで候補角度を絞る
- ROBUST 検出（確定用）: gray/CLAHE/Otsu + マルチスケール + decodeMulti/Single を試行

さらに FAST で角度候補が全く得られない場合でも救済できるよう、
0/90/180/270 の 4 方向のみ ROBUST を試す「rescue」も追加。

回転スキャンの設定:

- `--rotation-step`（デフォルト 10.0）で `0..350` を生成
- `--rotation-max-workers`（デフォルト 8）で ThreadPool 並列

失敗時:

- A/B いずれも確定できない場合 `stage=form_not_found` で終了

#### 判定スコア（A/B の決め方）

**A 判定:**

- `detect_formA_marker_boxes()` により 3 点マーカー（top_left / top_right / bottom_left）が揃えば OK
- さらに 2026/01/08 の改善で、各マーカーの bbox 中心が
  **期待位置（左上/右上/左下）に近いほど加点**する `pos_score` を導入
  - 正規化座標で距離を評価し、`base_score + pos_score*2.0` を最終スコア化

**B 判定:**

- QR が検出できれば OK
- QR の中心が「右上」に近いほどスコア加点（位置スコア + 相対面積）

#### 回転後画像の可視化（4_rectified_rot）

`4_rectified_rot/` は「フォーム確定に使った回転後画像」を保存するディレクトリ。
ここに **判定に使った特徴を描画**して保存することで、目視でデバッグ可能にした。

- フォーム A: 赤枠で 3 点マーカー bbox を描画し、`top_left(左上)` 等のラベルを表示
- フォーム B: 青枠で QR を描画し、`右上` を表示

※ **日本語ラベルの文字化け対策**として、OpenCV の `cv2.putText()` を使わず、
Pillow + `C:\Windows\Fonts\meiryo.ttc` を使用して描画する（`draw_text_pil()`）。

補足:

- `cv2.putText` は ASCII 前提のため、日本語（UTF-8/Unicode）を直接描画できない
- Pillow の `ImageDraw.text` + TrueType フォントを使用することで解決

### 5) XFeat matching（テンプレ全探索）

- `APA/test_recovery_paper.py` の `XFeatMatcher` を利用
- 全テンプレとマッチングし、**inliers 最大**のテンプレを採用
- 推定 Homography の逆で、回転・透視補正済み画像をテンプレ座標へ warp

採用基準:

1. `inliers`（RANSAC/USAC で残った対応点数）最大
2. 同点の場合 `inlier_ratio` が高い方

※ `match-max-side` はマッチング前に最大辺を縮小して安定化するためのパラメータ。

XFeat の Homography 推定:

- 内部では XFeat で特徴点抽出 → LightGlue で対応付け → `cv2.findHomography(USAC_MAGSAC)`
- `inliers` はマッチングの信頼度の最重要指標として採用

失敗時:

- 全テンプレで `res.ok=False`（マッチ不足/推定失敗）の場合 `stage=xfeat_failed`

## 実行方法

README にも記載。

### スモークテスト（各フォーム先頭 1 枚のみ）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline.py --limit 1
```

### A/B のみを対象にしたスモークテスト（例）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline.py --limit 1 --src-forms A,B --degrade-n 1
```

### パラメータ説明の表示

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline.py --explain
```

（`--explain` にはデフォルト値も出力されるため、実験パラメータの確認に使える）

## 追加対応（2026/01/08）

### 1) 出力フォルダを処理順の番号付きに変更

`APA/output_pipeline/run_.../` 配下が処理順に見えるよう、
サブフォルダを `1_...` のような番号付きに変更。

### 2) 引数を減らし、説明を標準出力できるように変更

デフォルト値を「おすすめ設定」に寄せ、
スモークテストは `--limit 1` だけで実行できるように変更。

また、以下を追加：

- `--explain` : 主要パラメータの説明（日本語）を出力して終了
  - `--explain` の出力には **デフォルト値**も含める
- 起動時に `[CONFIG]` として主要設定を一覧表示

### 3) フォーム判定の改善（位置スコア / 可視化 / 横長統一）

以下を `paper_pipeline.py` に反映：

- フォーム A 判定スコアへ「期待位置に近いほど加点」を追加
- `3_rectified/` を横長（landscape）に統一
- `4_rectified_rot/` に判定特徴（A=3 点マーカー、B=QR）の可視化を追加
- 日本語ラベルが `???` になる問題に対し、Pillow + Meiryo フォントで描画するよう変更

## 実行結果（スモークテスト）

実行ディレクトリ例（環境によりタイムスタンプは変わる）:

- `APA/output_pipeline/run_20260108_110621/`（初期版の例）
- `APA/output_pipeline/run_20260108_112138/`（日本語描画対応後の例）

`summary.csv` 抜粋:

| case      | source_form |    ok | decided_form | decided_angle | best_template | best_inliers |
| --------- | ----------- | ----: | ------------ | ------------: | ------------- | -----------: |
| A_1_deg00 | A           |  True | A            |         180.0 | image/A/5.jpg |            6 |
| B_1_deg00 | B           |  True | B            |         180.0 | image/B/1.jpg |           17 |
| C_1_deg00 | C           | False | -            |             - | -             |            - |

※ `C` は今回の判定ロジック（A の 3 点マーク / B の QR）では該当せず、`form_not_found` となる。

### 実行ログ例（抜粋）

```text
[INFO] Processing sources from form A: 1 images
  [OK] A_1_deg00: form=A angle=0.0 best=1.jpg inliers=14

[INFO] Processing sources from form B: 1 images
  [OK] B_1_deg00: form=B angle=0.0 best=1.jpg inliers=420
```

※ 改悪の乱数・DocAligner の推定・XFeat マッチ数により、角度や inliers は実行ごとに変動し得る。

## 出力物

`APA/output_pipeline/run_YYYYmmdd_HHMMSS/` 配下（処理順が分かるように番号付き）:

- `1_degraded/` : 改悪画像
- `2_doc/` : DocAligner polygon 可視化
- `2_doc/*_doc.jpg` は polygon（緑）と頂点ラベル（TL/TR/BR/BL）を重ねて保存する
- `3_rectified/` : 紙領域透視補正
- `3_rectified/*_rect.jpg` は透視補正後の紙画像（横長統一）
- `4_rectified_rot/` : フォーム確定に使用した回転後画像
- `4_rectified_rot/*_rot.jpg` はフォーム判定の根拠（A=マーカー / B=QR）を可視化して保存
- `5_debug_matches/` : マッチ可視化
- `6_aligned/` : best template に warp した結果
- `summary.json`, `summary.csv`

### summary.csv の見方（重要カラム）

- `ok`: 最終的に warp まで到達したか
- `stage`: 失敗した場合にどこで止まったか（例: `docaligner_failed`, `form_not_found`, `xfeat_failed`）
- `decided_form` / `decided_angle`: フォーム判定結果
- `best_template` / `best_inliers`: 最終採用テンプレと一致度

## 今後の課題

- フォーム C の様式定義が決まり次第、判定ロジックを追加する（現状は A/B のみ確定可能）
- 処理速度最適化（テンプレ数増加に備え、XFeat の事前特徴量キャッシュ等）

追加の改善候補:

- フォーム A の「三点の幾何（直角・辺比）」もスコアに取り入れる（誤検出耐性）
- テンプレ側の XFeat 特徴を事前キャッシュし、テンプレ全探索を高速化
- 失敗ケースの再現性のため、seed/中間画像をまとめて保存するモードを追加

## 変更履歴（ファイル）

- `APA/paper_pipeline.py`
  - パイプライン本体
  - 2026/01/08: 位置スコア・横長統一・rot 可視化・日本語描画対応を追加
- `APA/README.md`
  - 実行方法と改善点を追記
