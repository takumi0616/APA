# test_recovery_paper.py 詳細設計書（レビュー用）

作成日: 2026-01-07  
対象: `APA/test_recovery_paper.py`

---

## 0. 目的（このテストプログラムで「確認」したいこと）

既存の資産を使い、**テンプレート（正解画像）に対して改悪画像を自動補正（位置合わせ）**できるかを検証する。

対象は以下の 2 系統。

- **フォーム A**: 3 点マーク（黒い四角形）で識別し、整列用の対応点として使う。
- **フォーム B**: QR コードで識別し、整列用の対応点（QR 四隅）として使う。

補正の最終目的は「OCR 工程に渡せるほどきれいな正規化画像を得る」だが、今回のテストではまず

1. **フォーム種別が正しく判定できる**
2. **テンプレ座標系に透視変換（warp）できる**
3. **改悪データ（回転・背景・角度・奥行き）でも破綻しにくい**
4. **XFeat matching による Homography の性能（inlier 比、安定性）を確認できる**

までを安定して達成することを狙う。

---

## 1. 入力データ

### 1.1 静止画像モード（まずはこれを主戦場）

- テンプレ（正解）として `APA/image/A/*.jpg` および `APA/image/B/*.jpg` を利用（A/B それぞれ）
- 改悪データは **bad ディレクトリを使わず**、テンプレ（正解）から本プログラムが生成する

---

## 2. 前提・制約

- Windows + Miniconda のローカル環境（PATH 汚染は避ける）
- CPU 中心・軽量優先（myplan.txt の前提）
- フォーム自体の変更不可
- 特徴点は「検出できた情報（QR 四隅／マーカー角点など）」を保持し、整列用対応点として使える形にする

---

## 3. 出力（成果物）

### 3.1 画像出力

入力画像ごとに、下記を保存する。

- `output_recovery/<run_id>/debug/<name>_paper_detect.jpg`
  - 紙検出結果の可視化（検出枠、候補スコア等）
- `output_recovery/<run_id>/debug/<name>_features.jpg`
  - QR/マーカー検出結果の可視化（ポリライン/バウンディングボックス）
- `output_recovery/<run_id>/warped/<name>_warped.jpg`
  - テンプレ座標系へワープした結果（最重要）

※ 品質改善（2 値化、影補正等）は **本タスクでは実装しない**

### 3.2 ログ/集計

- `output_recovery/<run_id>/summary.json`
  - 入力ごとの結果（フォーム判定、検出成功、ホモグラフィ inlier 比、処理時間など）
- `output_recovery/<run_id>/summary.csv`
  - Excel 等で見れる集計表

---

## 4. 処理パイプライン（静止画像モード）

静止画像（改悪含む）から、テンプレへ整列するまでの流れ。

### 4.1 前処理（共通）

1. `cv2.imread`
2. サイズが極端に大きい場合は、
   - **検出用**に縮小画像を作成（速度改善）
   - 座標は元画像へスケール復元
   - ※フォーム B の QR 検出は `test_capture_formB.py` のマルチスケール戦略を踏襲

### 4.2 紙領域の推定

※ 本タスク（静止画像・改悪生成 → 評価）では行わない。

### 4.3 フォーム種別判定（A/B/Other）

紙の正面画像 `paper_crop` に対して以下。

#### 4.3.1 フォーム B 判定（QR）

- `test_capture_formB.py` の `detect_qr_codes()` 相当を再利用・統合
- 成功条件:
  - `qr_count >= 1`

保持する特徴点:

- `qr_corners`（QR 4 点）

#### 4.3.2 フォーム A 判定（3 点マーク）

- `test_capture_formA.py` の `detect_filled_square_markers()` 相当を再利用・統合
- 成功条件:
  - `top_left, top_right, bottom_left` の 3 点が揃う（理想）
  - ただし初期は「2 点以上でも暫定成功」など、テスト用に段階的に緩める余地あり

保持する特徴点:

- 3 点マーク（上左・上右・下左）について、
  - 各マーカーの **四角の角点（4 点）** を推定して保持する
  - 少なくとも `bbox` の 4 隅（矩形角）を角点近似として使える形にする
  - 可能なら輪郭近似 (`approxPolyDP`) の 4 点を使う

#### 4.3.3 その他

- QR もマーカーも見つからない場合は `Other`

### 4.4 位置合わせ（Homography）

基本方針:

- 「紙四隅」だけのホモグラフィは簡単だが、改悪（傾き/奥行き/背景）で誤差が出やすい。
- **フォーム固有の特徴点（QR/マーカー）を対応点として優先し**、紙四隅は補助的に使う。

実装候補（段階導入）:

#### 4.4.1 ルールベース（最初の到達点）

- フォーム B:
  - QR 四隅（4 点）→ テンプレ内の QR 四隅（4 点） で `cv2.getPerspectiveTransform()`
- フォーム A:
  - 3 点マークの 3 点はホモグラフィに不足（4 点必要）
  - ここは選択肢がある：
    1. 3 点 + 紙のもう 1 点（例えば bottom_right を紙四隅から補完）で 4 点を作る
    2. 3 点から affine を推定し、残りはテンプレ比率で補う（誤差は出る）
  - 初期版は (1) を採用し「最低限ワープできる」ことを優先

#### 4.4.2 特徴点マッチング（XFeat / ORB）による改善

ルールベースでワープできるようになったら、次に改悪耐性を上げる。

- `XFeat_matching.py` の流れを踏襲し、
  - テンプレ画像 vs 入力紙正面画像 で特徴点マッチング
  - `cv2.findHomography(..., cv2.USAC_MAGSAC, ...)` で頑健推定
  - inlier 比が閾値以上なら、推定 H でワープ

注意:

- XFeat は torch/kornia 等の依存が重く、CPU だと遅い可能性がある
- そのため **デフォルトは ORB**（OpenCV だけで完結）にして、XFeat はオプション化する案もある
  - `--matcher orb|xfeat` の切り替え

### 4.5 品質改善（本タスクでは実装しない）

myplan には品質改善が含まれるが、今回の目的は **XFeat matching による Homography 性能評価**のため、
品質改善（影補正、2 値化等）は実装しない。

---

## 6. テンプレート設計（重要）

テンプレ画像（正解）の管理方法。

### 6.1 テンプレ置き場

案:

- `APA/image/template/A.jpg`（フォーム A の正解）
- `APA/image/template/B.jpg`（フォーム B の正解）

※ 現状の指示では「フォーム A の正解（APA\image\A）フォーム B の正解（APA\image\B）」とあるが、
`image/A` と `image/B` の中に「正解」も混在している可能性があるため、
**“テンプレ” を明示的に別パスに切り出す**ことを推奨。

### 6.2 テンプレ特徴点

フォーム B:

- テンプレ画像から QR を検出し、`template_qr_corners` を得る
- 入力側の `qr_corners` と対応付け、ホモグラフィを推定

フォーム A:

- テンプレ画像から 3 点マークを検出し、`template_marker_centers` を得る
- 入力側と対応付け
- 4 点目は紙四隅（テンプレでも紙四隅推定）から補完する案

---

## 7. 成功判定（テストの合格条件）

静止画像テストの合格条件（暫定）:

- フォーム A 画像: 6/6 でフォーム A 判定できる
- フォーム B 画像: 6/6 でフォーム B 判定できる
- 改悪画像（bad）: まずは成功率を計測し、改善対象を明確化する
- ワープ画像が保存され、目視でテンプレと概ね一致している

定量指標（優先度順）:

1. `form_id` 正答率
2. `homography_inlier_ratio`（XFeat/特徴点ベースの場合）
3. 再投影誤差（RMS）
4. `warp_iou`（テンプレ枠と重なりを測る、必要なら後で）
5. 処理時間（ms/枚）

---

## 8. CLI（コマンドライン引数）案

```bash
# 静止画像一括（A/B/C/bad を走査）
python APA/test_recovery_paper.py --mode images --input APA/image --out APA/output_recovery

# フォームAだけ
python APA/test_recovery_paper.py --mode images --glob "APA/image/A/*.jpg" --out APA/output_recovery



# matcher 切替（将来）
python APA/test_recovery_paper.py --mode images --matcher orb
python APA/test_recovery_paper.py --mode images --matcher xfeat
```

引数候補:

- `--mode {images}`
- `--input <dir>` or `--glob <pattern>`
- `--out <dir>`
- `--templateA <path>` `--templateB <path>`（テンプレを明示的に指定できるようにする）
- `--paper-detector {opencv}`（将来の拡張用。現状は未実装）
- `--matcher {rule,orb,xfeat}`（まずは rule だけ実装でも OK）
- `--debug-save`（中間画像を保存するか）

---

## 9. モジュール構成（1 ファイルで始め、後で分割可能）

`test_recovery_paper.py` の内部構造（案）。

- `load_image_paths()`
- `detect_paper_polygon_opencv()`（将来の拡張用。現状は未実装）
- `warp_by_polygon()`（将来の拡張用。現状は未実装）
- `detect_formB_qr()`（= test_capture_formB のロジック）
- `detect_formA_markers()`（= test_capture_formA のロジック）
- `classify_form()`
- `estimate_homography_rule_based()`
- `enhance_image()`
- `save_outputs()`
- `run_images_mode()`
- （紙検出/正面化は将来の拡張）

---

## 10. リスク/不確定要素（先に共有）

1. **フォーム A は 3 点マークだけではホモグラフィに不足（4 点必要）**
   - 4 点目の定義（紙四隅から補う等）が精度を左右
2. 改悪データ（bad）の内容次第で、
   - 紙検出が破綻する
   - QR/マーカーが潰れて検出できない
   - 特徴点マッチングに移行が必要
3. XFeat は依存が重く、環境により導入/実行コストが高い

---

## 11. 確定済み方針（ユーザー指示）

- 改悪データはテンプレ（正解）画像から生成する。`APA/image/bad/` は無視する
- フォーム A の対応点は「上左・上右・下左の 3 箇所にあるマーカー」で、これはそれぞれ **四角の角点**
- テンプレート（正解）画像は A/B それぞれ使う
- （リアルタイム入力は本タスクでは扱わない）
- 品質改善は実装しない（XFeat homography の性能確認が目的）

---

## 付録: 既存資産の再利用元（対応表）

- フォーム A マーカー検出: `APA/test_capture_formA.py`
- フォーム B QR 検出: `APA/test_capture_formB.py`
- （リアルタイム入力は別タスクで検討）
- ホモグラフィ推定例: `XFeat_matching.py`
