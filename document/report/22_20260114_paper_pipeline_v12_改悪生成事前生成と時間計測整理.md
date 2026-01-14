# paper_pipeline_v12 改善レポート（改悪生成の事前生成 + 時間計測の整理）

## 実施日時

2026 年 1 月 14 日

## 目的

`APA/paper_pipeline_v12.py` の精度改善（改悪生成の現実寄せ）を維持しつつ、
ユーザー要望に従い **処理フロー**と **時間計測**を次の仕様へ変更する。

### 改善 1（改悪生成の扱い）

- `APA/test_recovery_paper.py` の改悪生成を流用しつつ、
  **改悪生成は処理時間としてカウントしない**
- 最初に必要な改悪処理済み画像を **全件まとめて生成**し、
  それを 1 枚ずつ改善処理（DocAligner→ 判定 →UVDoc→XFeat→warp）へ投入する
- v12.5: 紙のしなり（非線形ワープ）と影（照明ムラ）も改悪生成フェーズで確実に適用する

### 改善 2（時間計測から外す処理）

- 途中途中の画像保存処理は時間計測しない（ただし画像生成・保存自体は行う）
  - **計測に入れる画像保存は `7_aligned/` の作成（保存）だけ**
- `6_debug_matches/`（best template のマッチ可視化作業）は本番ではないため **時間計測から除外**

---

## 対象ファイル

- `APA/paper_pipeline_v12.py`

---

## 実施内容（実装概要）

### 1) 改悪生成を「最初に全件生成」へ変更（計測除外）

`main()` 内に **Pre-generating degraded images (NOT timed)** のフェーズを新設。

- `DegradedCaseInput`（dataclass）を追加
  - `degraded_bgr` / `H_src_to_degraded` / `degrade_meta` を保持
- `_generate_one(...)` で改悪生成し `degraded_inputs` に積む
- その後、本処理は `for di in degraded_inputs:` で `process_one_case(...)` に投入

改悪生成フェーズのログ例：

```text
[INFO] Pre-generating degraded images (NOT timed)...
[DEGRADE] form A: 1 images
[DEGRADE] form B: 1 images
...
[OK] Pre-generated degraded inputs: 7
```

### 2) v12.5（紙のしなり/影）を改悪生成フェーズへ統合

`_apply_extra_degrade_v12_5(...)` を追加し、改悪生成フェーズで

- `maybe_apply_bend_with_mask()`
- `maybe_apply_shadow()`

を適用するように整理。

これにより「改悪生成を先にまとめる」仕様へ変更しても、
v12.5 の追加改悪が確実に入る。

### 3) 時間計測の定義を整理

#### (A) stage time（計測対象）

各 case の `StageTimes` は以下のみを計測対象とした。

- `docaligner_s`
- `rectify_s`
- `decide_s`
- `uvdoc_s`
- `match_s`
- `warp_s`（ここに **aligned 保存時間**も含める）

`degrade_s` は v12.7 仕様として **常に 0.0**。

#### (B) 計測対象外にしたもの

- 改悪生成（degrade）全体
- 途中画像保存（`1_degraded/`〜`6_debug_matches/`）
- `6_debug_matches/` マッチ可視化画像生成
- `summary.json / summary.csv` の書き出し

#### (C) `7_aligned/` の保存は計測対象

- `7_aligned/` は成果物なので `--save-images` に関係なく必ず保存
- 保存時間も `warp_s` に含める

---

## 動作確認

### 構文チェック

```bat
C:\Users\takumi\develop\miniconda3\python.exe -m py_compile APA\paper_pipeline_v12.py
```

### スモーク実行

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v12.py --src-forms A,B,C --limit 1 --degrade-n 1 --save-images none --log-level INFO --console-log-level INFO
```

### 結果（例）

- `Pre-generating degraded images (NOT timed)` が先に走ることを確認
- 各 `[CASE]` で `1_degrade=0.000000` になり、改悪生成が計測から除外されていることを確認
- `7_aligned/` と `6_debug_matches/` が生成されることを確認

出力例：

- `APA/output_pipeline/run_20260114_111433/7_aligned/`
- `APA/output_pipeline/run_20260114_111433/6_debug_matches/`

---

## 変更点まとめ

- [x] 改悪生成（bend/shadow 含む）を最初に全件生成し、以後 1 枚ずつ本処理へ投入する構造へ変更
- [x] 改悪生成の時間を計測対象外にした（`degrade_s=0`）
- [x] 中間画像保存・`6_debug_matches` の可視化生成を計測対象外へ変更
- [x] 計測対象に含める画像保存は `7_aligned/` の保存のみへ整理

---

## 改善前後の結果比較と考察（run.log 比較）

ユーザー提示の以下 2 ログを比較し、性能（精度/KPI）と処理時間の変化を考察する。

- 改善前: `APA/output_pipeline/run_20260114_094551/run.log`
- 改善後: `APA/output_pipeline/run_20260114_112040/run.log`

### 1) 前提（重要）：時間計測の定義が異なる

今回の改善では、ユーザー要望に従い **計測対象から除外する処理を増やした**ため、
`run_elapsed_total_seconds` や `avg_elapsed_per_case_seconds` は「改善前後で同じ定義」ではない。

具体的には改善後（v12）は、

- 改悪生成（degrade）を **計測対象外**
- 中間画像保存（`1_degraded`〜`6_debug_matches`）を **計測対象外**

としている。

したがって「体感の総所要時間」は別途として、
ログ上は **本処理（DocAligner→rectify→decide→UVDoc→match→warp）に近い指標**として読む必要がある。

### 2) KPI（期待動作 / warp 到達率）の比較

ログ末尾の `[SUMMARY]` より：

| 指標                            |     改善前 |     改善後 |                    差分 |
| ------------------------------- | ---------: | ---------: | ----------------------: |
| total_cases                     |        110 |        110 |                      ±0 |
| ok_expected_behavior(user_KPI)  | 96 (87.3%) | 83 (75.5%) | **-13 cases (-11.8pt)** |
| ok_warp(done_aligned_generated) | 70 (63.6%) | 58 (52.7%) | **-12 cases (-10.9pt)** |

**読み取り**

- 改善後は **期待動作 KPI と warp 到達率が低下**している。
- 一方で C の誤検出（false positive）は両者とも 0 で、誤検出抑制は維持されている。

フォーム別の精度（`[STATS]` より）：

| 入力フォーム | 指標                               |        改善前 |        改善後 |     差分 |
| ------------ | ---------------------------------- | ------------: | ------------: | -------: |
| A            | template_accuracy                  | 43/50 (86.0%) | 39/50 (78.0%) | -4 cases |
| B            | template_accuracy                  | 27/30 (90.0%) | 23/30 (76.7%) | -4 cases |
| C            | reject_success(stage=form_unknown) | 26/30 (86.7%) | 25/30 (83.3%) |  -1 case |
| test(A のみ) | template_accuracy                  | 19/20 (95.0%) | 17/20 (85.0%) | -2 cases |

**読み取り**

- A/B が特に悪化している（=本来 done に行くべきケースが落ちている）。
- C の reject は僅差でほぼ維持だが、改善後も **docaligner_failed が増えている**ため、
  「C を form_unknown で止める」以前に DocAligner で落ちるケースが一定数ある。

### 3) stage counts（どこで落ちているか）

`[SUMMARY] stage counts` より：

| stage               | 改善前 | 改善後 |   差分 |
| ------------------- | -----: | -----: | -----: |
| done                |     70 |     58 |    -12 |
| form_unknown        |     29 |     30 |     +1 |
| docaligner_failed   |     11 |     18 | **+7** |
| homography_unstable |      0 |      4 | **+4** |

**読み取り**

- 改善後は `docaligner_failed` と `homography_unstable` が増えている。
  - つまり「フォーム判定以前」や「warp 手前」で落ちるケースが増加している。

### 4) 処理時間（ログ上の変化）

`[STATS] run_elapsed_total_seconds` より：

| 指標                         |  改善前 | 改善後 |                  差分 |
| ---------------------------- | ------: | -----: | --------------------: |
| run_elapsed_total_seconds    | 1613.0s | 185.0s | **-1428.0s (-88.5%)** |
| avg_elapsed_per_case_seconds | 14.664s | 1.682s | **-12.982s (-88.5%)** |

`[SUMMARY] stage time totals (s)` より：

| stage time totals | 改善前 | 改善後 | 主な差                 |
| ----------------- | -----: | -----: | ---------------------- |
| degrade_s         | 206.35 |   0.00 | **改善後は計測対象外** |
| decide_s          |  99.84 |  32.32 | -67.52                 |
| uvdoc_s           |  33.73 |  24.24 | -9.49                  |
| match_s           | 998.18 |  79.32 | **-918.86**            |

**読み取り（重要）**

- 改善後の大幅な短縮は、
  1. degrade を計測から外したこと
  2. **match_s が極端に短くなった**こと
     の影響が大きい。

特に `match_s: 998.18s → 79.32s` は約 **12.6 倍の短縮**で、
主ボトルネックがほぼ解消された形になっている。

### 5) なぜ精度が下がった可能性があるか（仮説）

今回の変更は「計測方法・フロー変更」が主目的であり、
理想的には精度（KPI）が変わらないのが望ましい。
しかしログ上は悪化しているため、主に次の可能性が考えられる。

1. **改悪生成（入力分布）が変化した**
   - v12.5 の bend/shadow を含む改悪が本格適用され、より難しいサンプルが増えた
   - seed や生成順序の差により、改善前 run と改善後 run で「同一入力」ではない可能性がある
2. **DocAligner が落ちるケースが増えている（docaligner_failed 増加）**
   - 影（照明ムラ）や非線形ワープ（bend）で輪郭が不安定になり、紙検出が難化した可能性
3. **Homography が不安定になるケースが新規に出た（homography_unstable）**
   - bend により「射影変換だけでは説明できない歪み」が入るため、
     XFeat の対応点が取れても射影モデルが破綻しやすいケースがあり得る

### 6) 結論（現状の評価）

- **速度（ログ上の計測値）は大幅改善**している一方で、
  **KPI は悪化**している。
- 今回は「計測対象の整理」を優先した変更なので、
  次の改善としては「bend/shadow の強さ（確率や振幅、影強度）を段階的に調整して、
  KPI が改善前水準に戻る点」を探索するのが合理的。
