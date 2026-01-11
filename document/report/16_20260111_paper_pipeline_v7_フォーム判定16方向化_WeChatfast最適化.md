#+#+#+#+---------------------------------------------------------------------

# 16 20260111 paper_pipeline_v7 改善（フォーム判定 16 方向化 / WeChat fast マルチスケール最適化）

## 実施日時

2026 年 1 月 11 日

## 目的

`paper_pipeline_v7.py` に対して、以下 2 点の改善を順番に実施する。

1. **改善 1**: フォーム A/B 判定の回転探索を **Coarse-to-Fine から「16 方向のみ」**に変更し、
   **最上位角度を角度確定として採用**する。
2. **改善 2**: WeChat QR の `mode="fast"` におけるマルチスケールを見直し、
   **大きい画像での縮小スケール試行（0.75/0.5 等）を廃止して高速化**する。

## 変更内容

### 改善 1: フォーム判定の Coarse-to-Fine 廃止 → 16 方向スキャン

#### 変更前

`decide_form_by_rotations()` が

- coarse: 0/45/90/.../315（8 方向）
- fine: coarse 上位角度近傍（±window）を `angles(0..350 step)` から抽出

の **Coarse-to-Fine** 構造になっていた。

#### 変更後

以下の仕様に変更。

- **scan16**: 0..360 を 16 等分（22.5 度刻み）で固定スキャン
- scan16 の中で **スコア最上位（A vs B_fast の比較）**をフォーム判定と角度確定に採用
- フォーム B が勝った場合のみ、同一角度で `score_formB()`（robust）を **1 回だけ**実行して確度を上げる
  - 近傍探索は行わない（「最上位角度を使って確定」方針のため）

実装上の変更点：

- `PIPELINE_DEFAULTS["rotation_scan"]["scan_angles_16_deg"]` を追加
- `decide_form_by_rotations()` を scan16 ロジックに置換
- `extract_form_unknown_reason()` の診断キーを coarse 前提から scan 対応へ（`scan_max_*`）

### 改善 2: WeChat QR fast のマルチスケール見直し

#### 背景

WeChat QR は内部で入力を縮小して NN 推論する（面積 160000px 相当など）ため、
**大きい入力に対して 0.75 / 0.5 などの縮小を何度も試しても、NN 入力サイズがほぼ同じになりがち**で、
効果が薄いわりに推論回数だけ増える可能性がある。

#### 変更後の方針

- `mode="fast"` では **縮小スケール（<1.0）を廃止**し、基本は `[1.0]` のみ
- **小さい画像のときだけ** up-scale を追加
  - 既定: `up_scales_small_image = [1.25, 1.5]`
  - 判定条件: `max(h,w) < up_scale_enable_max_side_px`

実装上の変更点：

- `PIPELINE_DEFAULTS["qr"]["wechat"]["fast"]` を
  - `scales=[1.0]`
  - `up_scales_small_image=[1.25,1.5]`
  - `up_scale_enable_max_side_px=1600`
    に更新
- `detect_qr_codes_wechat_multiscale(mode="fast")` のスケール生成を上記方針へ変更

## 実行確認

`.venv/bin/python` を用いて、A/B/C 各 1 枚（`--limit 1`）× `--degrade-n 1` で実行し、
ステージ遷移と出力生成が期待通りであることを確認した。

### フォーム A

```bash
.venv/bin/python paper_pipeline_v7.py --src-forms A --limit 1 --degrade-n 1 --log-level INFO --console-log-level INFO
```

- `stage=done`
- `pred_form=A`
- `template_ok=TRUE`

出力例:

- `output_pipeline/run_20260111_140742/`

### フォーム B

```bash
.venv/bin/python paper_pipeline_v7.py --src-forms B --limit 1 --degrade-n 1 --log-level INFO --console-log-level INFO
```

- `stage=done`
- `pred_form=B`
- `template_ok=TRUE`

出力例:

- `output_pipeline/run_20260111_140822/`

### フォーム C（棄却されるべき）

```bash
.venv/bin/python paper_pipeline_v7.py --src-forms C --limit 1 --degrade-n 1 --log-level INFO --console-log-level INFO
```

- `stage=form_unknown`
- `form_unknown_reason=no_detection`
- `ok_expected=TRUE`（フォーム C は Unknown が成功扱い）

出力例:

- `output_pipeline/run_20260111_140848/`

## 変更ファイル

- `paper_pipeline_v7.py`
  - フォーム判定: Coarse-to-Fine → scan16
  - WeChat fast multiscale: `[1.0]` + 小画像時のみ up-scale

## 補足 / 期待効果

- 回転探索が「8 方向 + 近傍複数角度」から「固定 16 方向」に単純化されるため、
  **探索回数が読みやすくなり、上限時間が安定**する。
- WeChat fast の縮小スケール試行を削除したことで、
  **B の decide（回転 × スケール）の実行回数が減り、速度改善が見込める**。

---

## 追記（v7.2）: フォーム判定の回転探索を 16 方向 → 2 方向（0/180）へ

### 背景

本パイプラインでは 3) の rectify の直後に

```python
rectified, _ = enforce_landscape(rectified)
```

としており、透視補正後の紙画像は **常に横長（landscape）** になるよう統一されている。

この前提があるため、フォーム判定の回転探索で検討すべき角度は

- 90 度回転（縦横入れ替え）

ではなく、

- **上下反転（180 度）**

だけで十分になる。

### 変更内容

- `decide_form_by_rotations()` の探索角度を **固定 2 方向**に変更
  - `scan_angles_2_deg = [0.0, 180.0]`
  - 0 度と 180 度それぞれでフォーム A スコア / フォーム B_fast スコアを計算し、最上位を採用
- Unknown（no_detection）の救済処置は行わない方針は維持
- フォーム B が勝った場合のみ、採用角度で robust 検出（`score_formB()`）を **1 回だけ**実行して確度を上げる

### 期待効果

- decide ステージの回転スキャンが **16 回 → 2 回** になるため、
  QR 検出（特に WeChat fast）を含むコストが大幅に下がる。
  - 実測例（フォーム C 1 枚 ×degrade1）では `decide_s` が **2.57s → 0.56s** まで短縮。

### 実行確認

scan2 変更後に `.venv/bin/python` で A/B/C 各 1 枚（`--limit 1`）× `--degrade-n 1` を実行し、
期待通り完走することを確認。

- A: `stage=done`, `ok_expected=TRUE`
  - 出力例: `output_pipeline/run_20260111_143344/`
- B: `stage=done`, `ok_expected=TRUE`
  - 出力例: `output_pipeline/run_20260111_143406/`
- C: `stage=form_unknown(no_detection)`, `ok_expected=TRUE`
  - 出力例: `output_pipeline/run_20260111_143424/`

### 変更ファイル

- `paper_pipeline_v7.py`
  - `PIPELINE_DEFAULTS["rotation_scan"]`: `scan_angles_2_deg` を追加
  - `decide_form_by_rotations()`: scan16 → scan2（0/180）

---

## 詳細比較解析（v6 / v7-scan16 / v7-scan2）

比較対象ログ：

- v6: `output_pipeline/run_20260110_164101/run.log`
- v7（scan16）: `output_pipeline/run_20260111_141423/run.log`
- v7（scan2）: `output_pipeline/run_20260111_143550/run.log`

### 1. まず前提（ログ比較の注意点）

- 3 つのログは **全て total=180（A/B/C 各 60、degrade 10 枚相当）**で比較条件は揃っている。
- ただし、v6 → v7 の間には「回転探索」以外にも、
  DocAligner/ログ出力/実装詳細の変更が入っており、
  `degrade_s`・`docaligner_s`・`match_s` が v7 側で増えている（≒ 重くなっている）ため、
  **純粋に回転探索だけの差ではない**点に注意する。
  - そこで結論は「探索戦略の優劣」を中心にしつつ、
    **KPI（成功率）とボトルネック（stage 時間）を分解**して判断する。

---

### 2. KPI（全体成功率 / warp 成功率 / 総時間）

ログ末尾の `[SUMMARY]` / `[STATS]` より：

| バージョン | total | ok_expected | ok_expected(%) | ok_warp | ok_warp(%) | per-case mean(s) | run elapsed(s) |
| ---------- | ----: | ----------: | -------------: | ------: | ---------: | ---------------: | -------------: |
| v6         |   180 |         171 |           95.0 |     111 |       61.7 |            3.584 |          646.0 |
| v7 scan16  |   180 |         172 |           95.6 |     112 |       62.2 |            4.342 |          783.3 |
| v7 scan2   |   180 |         168 |           93.3 |     108 |       60.0 |            3.293 |          594.2 |

**読み取り：**

- **成功率（ok_expected）だけ見ると v7 scan16 が最良（95.6%）**。
- **速度（per-case mean / run elapsed）だけ見ると v7 scan2 が最速**。
- v6 は成功率は scan16 に僅差で負けるが、scan16 より速い。

---

### 3. 入力フォーム別（A/B/C）の精度・失敗内訳

#### A（マーカー 3 点）

| バージョン | cases | form_accuracy | template_accuracy | A の失敗件数（form_unknown） |
| ---------- | ----: | ------------: | ----------------: | ---------------------------: |
| v6         |    60 | 88.3% (53/60) |     88.3% (53/60) |                            7 |
| v7 scan16  |    60 | 88.3% (53/60) |     88.3% (53/60) |                            7 |
| v7 scan2   |    60 | 88.3% (53/60) |     88.3% (53/60) |                            7 |

**読み取り：**

- A は探索戦略を変えても **成功率が全く改善していない**。
- 失敗は全て `stage=form_unknown reason=no_detection` で、
  **根本原因は「角度探索」よりも「マーカー検出の不安定さ（前処理/閾値/マーカーが潰れる）」側**にある。
- scan2 は失敗時の `decide_s` が小さく（=早く諦める）なるが、
  **成功/失敗の境界自体は変わらない**。

#### B（QR）

| バージョン | cases | form_accuracy | template_accuracy | B の失敗内訳                          |
| ---------- | ----: | ------------: | ----------------: | ------------------------------------- |
| v6         |    60 | 98.3% (59/60) |     96.7% (58/60) | form_unknown=1, homography_unstable=1 |
| v7 scan16  |    60 | 98.3% (59/60) |     98.3% (59/60) | form_unknown=1                        |
| v7 scan2   |    60 | 98.3% (59/60) |     91.7% (55/60) | form_unknown=1, homography_unstable=4 |

**読み取り：**

- B の「フォーム判定」自体は 3 方式で同等（98.3%）。
- 差が出ているのは **テンプレ一致（=最終的な warp 成功の安定性）**。
  - v7 scan16 は `homography_unstable` が 0 で安定。
  - v7 scan2 は `homography_unstable` が 4 件発生し、その分 template_accuracy が悪化。

scan2 で増えた `homography_unstable`（代表 4 件）は以下（ログから抽出）：

- `B_3_deg02` inliers=16
- `B_3_deg07` inliers=12
- `B_4_deg00` inliers=6
- `B_4_deg03` inliers=7

いずれも inlier が極端に少なく、Homography の安定性チェックで弾かれている。

**原因仮説（重要）：**

- scan2 は 0/180 しか試さないため、
  rectify 後に残った微小な回転ズレ（数度）を吸収できない。
- 一方 scan16 は 22.5 度刻みで、
  「QR の右上」「マーカーの幾何」スコアにより、
  **残差回転を相対的に打ち消す角度を選べる余地**がある。
- この差が、特定ケースでの XFeat マッチの質（inlier 数）に直結し、
  scan2 では `homography_unstable` が増えたと考えられる。

#### C（棄却）

| バージョン | cases | reject_success | false_positive |
| ---------- | ----: | -------------: | -------------: |
| v6         |    60 | 100.0% (60/60) |              0 |
| v7 scan16  |    60 | 100.0% (60/60) |              0 |
| v7 scan2   |    60 | 100.0% (60/60) |              0 |

**読み取り：**

- C はどの方式でも 100% 棄却できており、差は「速度」だけ。

---

### 4. ボトルネック（stage 時間）を「平均」と「割合」で比較

#### 4.1 stage 時間（平均/1 ケース）

`[STATS] stage time mean per case (s)` より：

| バージョン | degrade | docaligner | rectify | decide | match |  warp |
| ---------- | ------: | ---------: | ------: | -----: | ----: | ----: |
| v6         |   0.498 |      0.141 |   0.003 |  2.317 | 0.281 | 0.022 |
| v7 scan16  |   0.928 |      0.280 |   0.006 |  1.843 | 0.613 | 0.046 |
| v7 scan2   |   0.986 |      0.353 |   0.008 |  0.527 | 0.683 | 0.050 |

**読み取り：**

- scan16 → scan2 の変更で、
  **decide が 1.843s → 0.527s（約 71%削減）**。
- ただし v7 は v6 に比べて decide 以外（degrade/doc/match/warp）が重く、
  scan16 では decide 短縮分を他ステージ増加が相殺してしまい、結果として全体が遅くなっている。

#### 4.2 どの stage が支配的か（割合）

（平均 stage 時間を合計したものに対する割合。ログから算出）

- v6: decide が **71%**（圧倒的ボトルネック）
- v7 scan16: decide が **50%**（まだ最大だが支配率は低下）
- v7 scan2: decide が **20%** に低下し、match（26%）/degrade（38%）が支配的へ

**読み取り：**

- scan2 は「回転探索」をほぼ無視できる水準まで落とせる。
- その結果、今後の高速化の主戦場は `degrade` / `docaligner` / `match` に移る。

---

### 5. 結論：どの処理（探索戦略）が優秀か？

目的別に結論が変わる。

#### (A) 正確さ・最終 warp 成功を最優先するなら

- **v7 scan16 が最有力**。
  - ok_expected が最良（95.6%）
  - B の template_accuracy が最良（98.3%）
  - `homography_unstable` が 0（安定）

特にフォーム B は「角度が少しズレただけで局所特徴のマッチ品質が落ちる」ため、
scan16 のように **角度の自由度を残すことが、最終 warp の安定性に効く**。

#### (B) 処理時間（スループット）を最優先するなら

- **v7 scan2 が最速**（run elapsed 594s、per-case mean 3.293s）。
  - decide の大幅短縮（2.317s→0.527s）が効いている

ただし scan2 は B で `homography_unstable` が増え、template_accuracy が落ちるため、
**「とにかく速いが、稀に warp まで落ちる」**という性格になる。

#### (C) 実務上のおすすめ（総合）

ログ結果だけから判断すると、

- 「安定性重視」→ **scan16**
- 「速度重視」→ **scan2**

が妥当。

ただ、scan2 の失敗は少数（B 60 件中 4 件の `homography_unstable`）で、
原因も「微小回転ズレを吸収できない」可能性が高い。

そのため最終的に最も優秀になり得るのは、次の **ハイブリッド戦略**：

1. まず scan2（0/180）で高速に角度を仮決定
2. その角度で XFeat が弱い（inlier 低い/cond 高い）場合だけ、
   **追加で小さな近傍探索（例: ±10 度を数点、もしくは scan16 にフォールバック）**を行う

この方式なら、普段は scan2 の速度で回しつつ、
scan2 で落ちる少数ケースだけを救済できるため、
**「速度と安定性の両立」が期待できる**。
