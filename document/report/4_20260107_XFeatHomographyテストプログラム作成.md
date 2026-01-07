# XFeat Homography テストプログラム作成・動作確認レポート

## 実施日時

2026 年 1 月 7 日

---

## 目的

フォーム A / フォーム B の **正解（テンプレ）画像**を基準にして、

- テンプレ画像から自動生成した「改悪データ」（回転・射影・背景合成・軽微なノイズ/ブラー）
  に対し、**XFeat matching による Homography 推定の性能**（マッチ数、inlier 比、再投影誤差、ワープ結果）を評価できるテストプログラムを作成する。

※ myplan の「品質改善（2 値化等）」は今回実装しない（ユーザー指示）。

---

## 実施内容

### 1. 設計書の更新

`APA/document/plan/test_recovery_paper_design.md` をユーザー指示に合わせて更新。

- `bad/` は使用しない（テンプレから改悪を生成）
- フォーム A の対応点は 3 点マーク（上左・上右・下左）の **四角の角点**を想定
- 品質改善は実装しない
- XFeat matching による Homography 性能確認を主目的化

---

### 2. XFeat 実行環境の準備

当初、Miniconda base（Python 3.13.11）に `torch` がなく、XFeat が動かない状態だったため導入。

導入したパッケージ:

- `torch`（CPU 版, cp313）
- `kornia==0.7.2`（内部で `kornia-rs` を利用）
- `imageio`（XFeat 既存デモの依存）

スモークテスト:

- 既存の `XFeat_matching.py` を実行し、LightGlue + Homography 推定まで成功することを確認。

---

### 3. テストプログラム作成

新規作成:

- `APA/test_recovery_paper.py`

主な機能:

#### (A) 静止画像モード（images）

- 入力: `APA/image/A/*.jpg`, `APA/image/B/*.jpg` をテンプレとして利用
- 改悪データ生成（プログラム内）
  - ランダム背景生成（色＋グラデ＋ノイズ＋線）
  - テンプレをランダム四角形へ射影して合成
  - 追加で軽微な blur / noise
- XFeat matching により Homography 推定
  - `cv2.findHomography(..., cv2.USAC_MAGSAC, ...)`
  - 指標: `matches`, `inliers`, `inlier_ratio`, `reproj_rms`
- 推定した Homography の逆で、改悪画像 → テンプレ座標へワープして保存

---

### 4. 改善（実行結果に基づく試行錯誤）

初回実行では、テンプレが高解像度（約 3509x2480）である影響で XFeat のマッチがほぼ 0 となり失敗。

対応:

- XFeat に渡す前に、テンプレ/改悪画像を `--match-max-side`（デフォルト 1200px）にリサイズしてマッチング
- 推定 Homography を元解像度へスケール補正して戻す

これにより、マッチが安定しワープ画像の保存もできるようになった。

---

## 実行例

### フォーム A（静止画像・改悪生成）

```bash
C:\Users\takumi\develop\miniconda3\python.exe APA\test_recovery_paper.py --mode images --form A --degrade-n 2 --top-k 1024
```

出力例:

- `APA/output_recovery/run_YYYYmmdd_HHMMSS_A_images/summary.csv`
- `.../degraded/*.jpg`（生成した改悪画像）
- `.../warped/*_warped.jpg`（テンプレへ整列した結果）
- `.../debug/*_matches.jpg`（マッチ可視化）

### フォーム B（静止画像・改悪生成）

```bash
C:\Users\takumi\develop\miniconda3\python.exe APA\test_recovery_paper.py --mode images --form B --degrade-n 2 --top-k 1024 --match-max-side 1400
```

---

## 結果（抜粋）

フォーム A（degrade-n=2 の例）:

- 多くのケースで `matches` が得られ、`inlier_ratio` も概ね 0.6〜0.9 程度
- 一部ケースで `matches=0` の失敗が残る（改悪条件が厳しい or テンプレ特徴が少ない等）

フォーム B（match-max-side を大きめにした例）:

- ほとんどのケースで `matches` が得られ、Homography を推定できる

---

## 今後の改善案（次の試行錯誤ポイント）

1. **失敗ケースの再現性確保**
   - seed 固定 + 失敗ケースのみ保存
2. **改悪生成の難易度調整**
   - perspective jitter / 回転角 / blur/noise の強さを段階的に設定
3. **フォーム A の「角点」利用を Homography 推定へ反映**
   - 現状は XFeat のみで推定しているため、マーカー角点を補助対応点として組み合わせる余地あり
4. （削除）

---

## 追加修正（フィードバック対応）

### 1. マッチ可視化画像（debug）の左右サイズを統一

問題:

- `debug/*_matches.jpg` の左（テンプレ）と右（改悪）が大きく異なるサイズで表示され、
  見た目として「全然マッチしていない」ように見える状態だった。

対応:

- マッチングは内部で `--match-max-side` にリサイズして行っているため、
  **可視化も同じリサイズ画像（matching 座標系）で描画**するように修正。

これにより、左右の解像度/スケールが揃い、マッチ状況が正しく確認できるようになった。

### 2. 改悪生成を「優しめ」に調整

問題:

- 回転/射影/背景ノイズが強すぎ、現実的なテストとして厳しすぎる。

対応:

- デフォルト値を弱めた
  - `--max-rot`: 25 → 12
  - `--perspective`: 0.18 → 0.08
- 背景ノイズ・線の量を減らした
- 追加 blur/noise も弱めた

### 3. 動作確認（修正後）

- フォーム A: `--degrade-n 1` で 6/6 成功
- フォーム B: `--degrade-n 1` で 6/6 成功

### 4. 追加検証（高回転＆改悪解像度アップ）

ユーザー要望により、以下を追加対応。

#### (1) 改悪画像の解像度アップ

- `--degrade-w/--degrade-h` のデフォルトを引き上げ
  - 1600x1200 → **2400x1800**
- さらに「紙（テンプレ）が写る領域」を大きめにするように調整
  - out_w に対して紙幅が 70〜92% 程度になるよう変更

#### (2) 高回転条件でのテスト

例:

```bash
C:\Users\takumi\develop\miniconda3\python.exe APA\test_recovery_paper.py --mode images --form A --degrade-n 1 --top-k 2048 --match-max-side 1600 --max-rot 35 --perspective 0.08
```

結果（抜粋）:

- フォーム A: 6 ケース中 **5 ケース成功**（1 ケース失敗）
- フォーム B: 6 ケース中 **5 ケース成功**（1 ケース失敗）

※ `--max-rot 35` は現実より厳しめの条件なので、失敗ケースが残るのは自然。次の改善候補は、
`--match-max-side` をさらに上げる / `--top-k` 増 / マーカー/QR 由来の点を補助として入れる、など。

---

## 追加検証（360 度回転 + 各テンプレ 5 個のランダム改悪）

ユーザー要望:

- 「360 度どれでも回転」
- 「フォーム A / B の 1〜6 それぞれ 5 個ずつ改悪」

対応内容:

1. 改悪生成の回転角を **0〜360 度の一様ランダム**に変更（`--max-rot 180` を指定した場合にフル回転モード）
2. `--degrade-n 5` で各テンプレから 5 枚ずつ改悪生成
3. summary.json に `degrade.angle_deg`（生成した回転角）を記録

### 実行コマンド

フォーム A:

```bash
C:\Users\takumi\develop\miniconda3\python.exe APA\test_recovery_paper.py --mode images --form A --degrade-n 5 --top-k 2048 --match-max-side 1600 --max-rot 180 --perspective 0.08 --seed 42
```

フォーム B:

```bash
C:\Users\takumi\develop\miniconda3\python.exe APA\test_recovery_paper.py --mode images --form B --degrade-n 5 --top-k 2048 --match-max-side 1600 --max-rot 180 --perspective 0.08 --seed 42
```

### 結果（サマリー）

出力ディレクトリ:

- フォーム A: `APA/output_recovery/run_20260107_133534_A_images/`
- フォーム B: `APA/output_recovery/run_20260107_133759_B_images/`

成功数:

- フォーム A: **25/30 成功**（83.3%）
- フォーム B: **25/30 成功**（83.3%）

回転角のレンジ（summary.json の `degrade.angle_deg` から集計）:

- A/B 共通（seed=42 のため）
  - min: 9.33°
  - max: 356.62°

※ 360 度回転は難易度が大きく上がるため、失敗（matches=0）が残るのは自然です。
この条件で成功率を上げたい場合は、

- `--match-max-side` をさらに上げる
- `--top-k` を増やす
- フォーム A は 3 点マーカー角点を補助点として混ぜる（XFeat が落ちるケースの保険）

などが次の改善候補になります。

---

## 追加修正（改悪画像に紙が写らないケースの防止）

ユーザーから「degraded 内に、紙がほぼ写っていない画像がある」という指摘があり、改悪生成ロジックを改善。

### 原因

従来は、テンプレを回転・射影した後に

- 画面外にはみ出した頂点を `np.clip()` で強制的に画面内に押し込む

という処理をしていたため、四角形が潰れてしまい

- 紙領域が極端に小さくなる
- 最悪、紙がほぼ消える

というケースが発生していた。

### 対応

`warp_template_to_random_view()` を以下の方針に変更。

1. 四角形が画面外にはみ出した場合は **クリップではなく中心へスケール**して収める
2. テンプレマスクをワープして、紙の可視面積（マスク面積）が `--min-visible-area-ratio` 以上になるまで **リトライ**

これにより、degraded 画像に「紙が写っていない」ケースが出にくくなる。

### 追加されたパラメータ

- `--min-visible-area-ratio`（デフォルト 0.25）
  - 改悪画像内で紙が占める最小面積比
- `--max-attempts`（デフォルト 50）
  - 条件を満たすまでの最大リトライ回数

---

## 変更ファイル

- `APA/test_recovery_paper.py`（新規作成）
- `APA/document/plan/test_recovery_paper_design.md`（設計更新）
- `APA/document/report/4_20260107_XFeatHomographyテストプログラム作成.md`（本レポート）
