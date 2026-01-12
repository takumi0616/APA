#+#+#+#+---------------------------------------------------------------------

# 17 20260112 paper_pipeline_v8 改善レポート（画像保存最適化 / 回転最適化 / フォーム B 判定改善）

## 実施日時

2026 年 1 月 12 日

## 目的

`APA/paper_pipeline_v8.py` の精度・速度改善として、ユーザー要望の以下 4 点を **順番に**反映し、実行確認まで行う。

1. **改善 1（B-5）**: JPEG 書き出しを TurboJPEG（libjpeg-turbo 系）へ寄せる（保存する場合の高速化）
2. **改善 2（B-4）**: 0 度 / 180 度の回転で `rotate_image_bound()`（= warpAffine）を使わない
3. **改善 3（B-1）**: デバッグ画像保存を「常時」から「必要時だけ」に切り替え可能にする
4. **改善 4**: フォーム B（QR）向けの「向き判定」精度改善（スコア重み、enforce_landscape 重複の排除、pos_score 改良）

## 前提（環境）

- OS: Windows 11
- Python: Miniconda ローカル
  - `C:\Users\takumi\develop\miniconda3\python.exe`
- OpenCV: 4.12.0
- turbojpeg: import 可能（`python-turbojpeg`）

## 対象ファイル

- `APA/paper_pipeline_v8.py`

---

## 改善 1（B-5）: JPEG 保存を TurboJPEG 優先に変更

### 背景

保存（`cv2.imwrite`）は FPS を大きく下げる要因なので、**本命は保存を止めること**。
ただし「保存が必要なとき」に備えて、OpenCV より速いケースがある TurboJPEG 系を優先して使えるようにする。

### 対応

- `turbojpeg` を optional import
- `write_image(path, image_bgr)` を追加
  - `.jpg/.jpeg` の場合は TurboJPEG.encode() を優先
  - 失敗時は `cv2.imwrite` にフォールバック

---

## 改善 2（B-4）: 0/180 回転で rotate_image_bound を重いまま使わない

### 背景

`rotate_image_bound()` は内部で `warpAffine` を実行するため、0°/180° でもオーバーヘッドが発生する。

### 対応

`rotate_image_bound()` 冒頭に特例を追加。

- 0°: そのまま返す
- 180°: `cv2.rotate(img, cv2.ROTATE_180)`

---

## 改善 3（B-1）: デバッグ画像保存を必要時だけに制御

### 背景

現状は done ケースでも毎回 5〜6 枚保存しており、IO オーバーヘッドが大きい。

### 対応

CLI 引数を追加。

- `--save-images {none,fail,all}`
  - `none`: 一切保存しない（FPS 計測用）
  - `fail`: `stage!=done` のケースだけ保存
  - `all`: 従来通り常時保存

実装は `process_one_case()` 内で、

- `all`: 即時保存
- `fail`: いったんメモリ保持して、最終 `stage` 決定後に必要なら保存
- `none`: 保存しない

となるように統一。

---

## 改善 4: フォーム B 判定（QR の向き）改善

### 変更点

1. **score_best_qr_candidate の重み修正**

- 位置（右上）を支配的に
- 面積（rel）は補助（ノイズ除外程度）へ

変更後：

- `pos_score` を右上(1,0)からの距離で計算
- `pos_score ** 2` を用いて「端に寄っている」ものをより高評価
- スコア重みを `位置 15.0 >> 面積 2.0` に変更

2. **decide_form_by_rotations のループ内で enforce_landscape を重複適用しない**

rectify 直後に `enforce_landscape()` 済みのため、スキャン中に再度回すと座標系が崩れる可能性がある。
ループ内の `enforce_landscape` を削除し、純粋に角度評価のみ実施。

---

## 実行確認

### 1) 構文チェック

```bat
C:\Users\takumi\develop\miniconda3\python.exe -m py_compile APA\paper_pipeline_v8.py
```

### 2) スモーク実行（A,B,C 各 1 枚 × degrade 1）

保存を止めた状態で確認（`--save-images none`）。

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v8.py --src-forms A,B,C --limit 1 --degrade-n 1 --log-level INFO --console-log-level INFO --save-images none
```

実行結果（ログ抜粋）:

- A: `stage=done`, `ok_expected=TRUE`
- B: `stage=done`, `ok_expected=TRUE`
- C: `stage=form_unknown(no_detection)`, `ok_expected=TRUE`（C は reject が成功扱い）

出力ディレクトリ例:

- `APA/output_pipeline/run_20260112_093624/`

---

## まとめ

- [x] **改善 1**: JPEG 保存を TurboJPEG 優先（失敗時 OpenCV フォールバック）に対応
- [x] **改善 2**: 0°/180° 回転で `warpAffine` を避ける（0°=そのまま、180°=cv2.rotate）
- [x] **改善 3**: `--save-images {none,fail,all}` を追加し、必要時のみ保存可能に
- [x] **改善 4**: フォーム B（QR）向けの向き判定を、位置優先スコアへ改善 + enforce_landscape 重複適用を除去
- [x] スモーク実行で A/B/C が想定通りに完走することを確認

---

# v7 vs v8 の結果比較（run.log ベース）

本項では、同一データセット（A/B/C 各 6 枚 × degrade 10 = 計 180 cases）の実行ログを比較し、
**v7→v8 で何が変わり、何が改善され、結果にどう影響したか**を整理する。

- v7 実行ログ: `APA/output_pipeline/run_20260112_110811/run.log`
- v8 実行ログ: `APA/output_pipeline/run_20260112_131554/run.log`

## 1) 全体 KPI の比較（最重要）

| 指標                                |          v7 |          v8 |                  差分 |
| ----------------------------------- | ----------: | ----------: | --------------------: |
| total_cases                         |         180 |         180 |                    ±0 |
| ok_expected_behavior (ユーザー KPI) | 172 (95.6%) | 170 (94.4%) | **-2 cases (-1.2pt)** |
| ok_warp(done_aligned_generated)     | 112 (62.2%) | 110 (61.1%) | **-2 cases (-1.1pt)** |
| run_elapsed_total_seconds           |    563.029s |    529.050s |  **-33.979s (-6.0%)** |
| avg_elapsed_per_case_seconds        |      3.128s |      2.939s |   **-0.189s (-6.0%)** |

**要点**

- **速度は v8 が改善**（全体で約 6%短縮）。
- 一方で、このログ条件では **ユーザー KPI(期待動作)と ok_warp は v7 がわずかに良い**。
  - v8 は誤った warp へ進まずに `form_unknown` 停止するケースが増え、
    結果として **homography_unstable は 2→0 に減った**が、
    **B を取りこぼして form_unknown になっているケースが増えた**、という構図。

## 2) 入力フォーム別の精度比較（A/B/C）

| 入力フォーム | 指標                               |             v7 |             v8 |         差分 |
| ------------ | ---------------------------------- | -------------: | -------------: | -----------: |
| A            | form_accuracy                      |  57/60 (95.0%) |  56/60 (93.3%) |      -1 case |
| A            | template_accuracy                  |  57/60 (95.0%) |  56/60 (93.3%) |      -1 case |
| B            | form_accuracy                      |  57/60 (95.0%) |  54/60 (90.0%) | **-3 cases** |
| B            | template_accuracy                  |  55/60 (91.7%) |  54/60 (90.0%) |      -1 case |
| C            | reject_success(stage=form_unknown) | 60/60 (100.0%) | 60/60 (100.0%) |           ±0 |
| C            | false_positive_as_A / B            |          0 / 0 |          0 / 0 |           ±0 |

**読み取り**

- **C の棄却（誤検出抑制）は v7/v8 ともに 100%**。
- 差が出ているのは **主に B**。
  - v8 は v7 と比べて **form_unknown が増え**（B の取りこぼしが増え）、B の form_accuracy が低下。
- A も 1 ケースだけ悪化している（A の取りこぼしが 1 増）。

## 3) ステージ内訳（件数）の比較

| stage               |  v7 |  v8 |   差分 |
| ------------------- | --: | --: | -----: |
| done                | 112 | 110 |     -2 |
| form_unknown        |  66 |  70 |     +4 |
| homography_unstable |   2 |   0 | **-2** |

**ポイント**

- v8 は **homography_unstable が消えた**（安全側に倒れて form_unknown 停止になる／フォーム確定が変わって warp まで到達しない）。
- ただし **done が減って form_unknown が増えている**ため、KPI としてはトレードオフになっている。

## 4) ステージ別時間の比較（合計/平均）

### 合計時間（s）

| stage        | v7 合計(s) | v8 合計(s) |       差分 |
| ------------ | ---------: | ---------: | ---------: |
| degrade_s    |     166.92 |     168.21 |      +1.29 |
| docaligner_s |      37.23 |      37.33 |      +0.10 |
| rectify_s    |       1.52 |       1.58 |      +0.06 |
| decide_s     |      83.32 |      52.65 | **-30.67** |
| match_s      |     153.30 |     148.49 |      -4.81 |
| warp_s       |       7.97 |       8.13 |      +0.16 |

### 平均（1 ケース当たり, s）

| stage        | v7 平均(s) | v8 平均(s) |                差分 |
| ------------ | ---------: | ---------: | ------------------: |
| degrade_s    |      0.927 |      0.934 |              +0.007 |
| docaligner_s |      0.207 |      0.207 |              ±0.000 |
| rectify_s    |      0.008 |      0.009 |              +0.001 |
| decide_s     |      0.463 |      0.293 | **-0.170 (-36.7%)** |
| match_s      |      0.852 |      0.825 |              -0.027 |
| warp_s       |      0.044 |      0.045 |              +0.001 |

**要点**

- v8 の改善効果が最も大きく出ているのは **フォーム判定(decide)の高速化**。
- degrade/docaligner/rectify はほぼ同等（ここは今回の変更の主戦場ではない）。

## 5) v7→v8 の主な実装差分と、結果への影響

### (A) 画像保存の最適化（TurboJPEG + 保存モード）

- v8 で `write_image()` が追加され、JPEG 保存が TurboJPEG 優先になった。
- v8 で `--save-images {none,fail,all}` が追加され、保存量の抑制が可能になった。

**今回の比較ログへの影響**

- v8 の実行ログには `save-images` の表示が無いため、既定値 `all` で実行された可能性が高い。
  - その場合、「保存枚数そのもの」は v7 と近くなり、
    IO 削減（fail/none）の効果はこの比較では十分に出ていない。
- ただし、実運用で **`--save-images fail/none` を選べるようになった**こと自体が大きい。
  - 特に `done` が大半を占める運用では `fail` が効きやすい。

### (B) 回転処理の最適化（0/180 の特例）

- v8: `rotate_image_bound()` が 0°/180° を特別扱い
  - 0° はそのまま
  - 180° は `cv2.rotate(..., ROTATE_180)`

**影響**

- v8 は回転探索が 0/180 のみのため、この最適化は「効きやすい」。
- decide_s が大きく短縮していることから、周辺の最適化（次項含む）と合わせて実行時間へ寄与している。

### (C) フォーム B 判定の方針変更（WeChat-only 化 + fast→robust）

- v7: WeChat が無い場合は OpenCV QRCodeDetector にフォールバック（fast/robust あり）
- v8: **フォーム B は WeChat-only**（OpenCV QRCodeDetector を使用しない）
  - さらに v8 は WeChat-only でも fast→robust の 2 段階を導入

**影響（精度/安定性の両面）**

- WeChat-only により、環境依存性は上がる（opencv-contrib + モデル必須）が、
  QR が小さい/低解像度のケースでは理論上有利。
- 一方で今回のログでは、B の form_accuracy が v7 より低下している。
  - v8 では `unknown_reason=below_threshold` が B で発生している（例: `B_3_deg01`, `B_4_deg00`, `B_5_deg04`）。
  - スコア設計変更（位置優先化）により、
    **従来よりスコア値が小さくなりやすく、unknown_score_threshold(既定 1.2) を割りやすい**可能性がある。

### (D) フォーム B（向き判定）のスコア改善（位置優先）

- v8: `score_best_qr_candidate()` のスコア設計を変更
  - 右上(1,0)からの距離 → `pos_score`
  - `pos_score**2` による強調
  - 重み: **位置 15.0 >> 面積 2.0**

**影響**

- 「右上に寄っている QR を選ぶ」という向き判定の目的には合致。
- ただし、スコアの絶対値スケールが v7 と変わったことで、
  `--unknown-score-threshold`（既定 1.2）との整合が崩れ、`below_threshold` が増えた可能性がある。

### (E) decide_form_by_rotations の高速化（enforce_landscape 重複排除 + A が強い時に B スキップ）

- v8: decide ループ内で `enforce_landscape()` を重複適用しない（rectify 直後に統一済みのため）
- v8: **A 判定が十分強い場合は B 判定を省略**（WeChat が重いので枝刈り）

**影響**

- decide_s が大きく短縮している主要因。
- ただし「A が強いと判断して B を見ない」ため、
  条件によっては B の検出チャンスを減らす可能性がある（要チューニング）。

## 6) 結論と次アクション（おすすめ）

- **速度面は v8 が明確に改善**（約 6%短縮、特に decide_s）。
- **精度面（特に B）は v7 が良い**結果になっており、v8 は取りこぼし（form_unknown）が増えた。

次の打ち手（優先順）:

1. `--unknown-score-threshold` の再キャリブレーション
   - v8 の B スコア設計変更に合わせて、閾値を下げる（例: 1.0 や 0.8）
   - もしくは `score_formB` の返す `score` のスケールを v7 相当に再調整する
2. v8 の WeChat robust の探索範囲（variants/scales）を v7 に近づけて取りこぼしを減らす
   - 例: robust の scales を増やす、gray 以外の前処理を追加する等
3. 実運用を意識した速度測定
   - `--save-images none` と `--save-images fail` で IO 抑制の効果を別途測る

---

## 7) 追加の詳細分析（case 単位の内訳）

上記の KPI/サマリに加えて、`run.log` の各 `[CASE]` 行を集計して、
「どの入力フォームで」「どの stage/理由で」落ちているかをもう一段詳しく比較する。

（解析スクリプト例: `APA/trash/analyze_runlogs_v7_v8.py`）

### 7.1 入力フォーム別 × stage の件数

#### v7

| 入力フォーム | done | form_unknown | homography_unstable |
| ------------ | ---: | -----------: | ------------------: |
| A            |   57 |            3 |                   0 |
| B            |   55 |            3 |                   2 |
| C            |    0 |           60 |                   0 |

#### v8

| 入力フォーム | done | form_unknown | homography_unstable |
| ------------ | ---: | -----------: | ------------------: |
| A            |   56 |            4 |                   0 |
| B            |   54 |            6 |                   0 |
| C            |    0 |           60 |                   0 |

**読み取り**

- v7→v8 の差分は、主に **B の落ち方が変わった**こと。
  - v7: B は「フォームは通るが warp が不安定」(`homography_unstable`) が 2 件
  - v8: B は「フォーム確定の時点で止まる」(`form_unknown`) が 3 件増（3→6）
- A も `form_unknown` が 1 件増（3→4）。
- C は v7/v8 ともに **全件 form_unknown**（期待動作）で、誤検出は無し。

### 7.2 form_unknown の reason 分布

#### v7

| 入力フォーム | no_detection | below_threshold |
| ------------ | -----------: | --------------: |
| A            |            3 |               0 |
| B            |            3 |               0 |
| C            |           60 |               0 |

#### v8

| 入力フォーム | no_detection | below_threshold |
| ------------ | -----------: | --------------: |
| A            |            4 |               0 |
| B            |            3 |               3 |
| C            |           60 |               0 |

**ポイント**

- v8 では v7 に無かった **`below_threshold` が新規に発生**している。
  - 該当ケース（v8）: `B_3_deg01`, `B_4_deg00`, `B_5_deg04`
    - ログ上は `unknown_reason=below_threshold` で、フォーム確定まで行かず停止
- これは実装上、
  - `score_best_qr_candidate()` のスコア設計を「位置支配」に変更したこと
  - その一方で `--unknown-score-threshold`（既定 1.2）は **v7 のスコア感のまま**
    という組み合わせにより、**スコアの絶対値が閾値を割りやすくなった**可能性が高い。

### 7.3 v8 の「速くなったが、B が落ちやすくなった」構図の整理

- v8 は **フォーム判定(decide)を高速化**できている（83.32s → 52.65s）。
  - 0/180 回転の特例（0° はコピー、180° は `cv2.rotate`）
  - `enforce_landscape()` 重複排除
  - A が強いときの B 判定スキップ
  - WeChat fast→robust の導入（robust を最大 1 回に抑制）
- ただし、その代償として「B を拾えない」方向に倒れるケースが増えた。
  - `below_threshold` は **v8 のスコア/閾値の不整合**が主因の疑い
  - `no_detection` は **WeChat の探索範囲（variants/scales）が v7 より狭い**ことが影響している可能性

### 7.4 次アクション（より具体）

1. **`below_threshold` ケースのスコア確認**

   v8 の `summary.csv` には `form_unknown_diagnostics_json` が出ているため、
   まずは該当 3 件の `top_score`（閾値 1.2 と比較する値）を確認し、
   どれくらい下げれば救えるかを定量化する。

2. **`unknown_score_threshold` の再調整**

   - 例: `1.2 → 1.0` または `0.8`
   - ただし、閾値を下げすぎると C の誤検出が増える可能性があるため、
     「C の false positive 0%」を維持できる範囲で調整する。

3. **WeChat robust の探索範囲を段階的に広げる**

   v8 は速度のために `variants/scales` を絞っているため、
   `no_detection` の B ケース（3 件）が実運用で問題になるなら、
   robust 側だけ段階的に増やして取りこぼしを減らす。
