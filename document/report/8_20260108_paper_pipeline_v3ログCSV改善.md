# paper_pipeline_v3.py ログ/CSV 改善レポート

## 実施日時

2026 年 1 月 8 日

## 目的

`APA/paper_pipeline_v3.py`（`paper_pipeline_v2.py` のコピー）に対し、ユーザー要望の以下 2 点を**優先して**改善する。

1. **ログが読みづらい**
   - 用紙検出不可（DocAligner 失敗等）でもログに出す
   - 正解/不正解をログにも出す（A/B の様式判定 + テンプレ一致）
   - 処理 1〜6 のそれぞれの実行時間
   - 1 枚（1 case）の総実行時間
2. **CSV をさらにリッチにする**
   - 改悪パラメータ
   - 入出力画像の解像度
   - 全実行時間、各ステージ時間
   - DocAligner / XFeat / 判定に関するあらゆる情報
   - 予測したテンプレが正解かどうか（正解/不正解）
   - 解析のために**列名は長くても良いので意味が分かる名前**にする

※ v2 の改善点メモには他の項目もあるが、本レポートでは上記 2 点に絞って対応。

---

## 変更対象

- `APA/paper_pipeline_v3.py`

---

## 実装内容

### 1) ログ改善（全ケース 1 行サマリ + 時間計測）

**追加した仕様**

- 成功/失敗に関わらず、各 case ごとに必ず 1 行ログを出す
  - `stage`（どこで止まったか）
  - `ok`（warp まで到達したか）
  - `gt_form / pred_form`（真の様式 / 推定様式）
  - `form_ok / template_ok`（正誤。A/B 入力のみ）
  - `best_template / inliers / inlier_ratio`
  - `time_total_s` と `1〜6 のステージ時間`

**追加した関数**

- `log_case_summary(logger, row)`
  - `build_csv_row()` が返す 1 行分の情報を使って、コンパクトなログ行を生成
  - 成功時は `INFO`、失敗時は `WARNING` で出力

**ログ例（実行結果より）**

```text
2026-01-08 15:21:20 [INFO] [CASE] id=A_1_deg00 ok=TRUE stage=done gt_form=A pred_form=A form_ok=TRUE template_ok=TRUE best_template=1.jpg inliers=417 inlier_ratio=0.8458 time_total_s=14.014151 (1_degrade=1.581138,2_doc=0.287090,3_rectify=0.011855,4_decide=9.251648,5_match=1.768380,6_warp=0.085635) src=...\APA\image\A\1.jpg

2026-01-08 15:22:13 [WARNING] [CASE] id=C_1_deg00 ok=FALSE stage=homography_unstable gt_form= pred_form=A best_template=2.jpg inliers=9 inlier_ratio=0.642857 time_total_s=12.705387 (1_degrade=0.773203,2_doc=0.255223,3_rectify=0.007656,4_decide=7.941871,5_match=3.504195,6_warp=0.000000) src=...\APA\image\C\1.jpg
```

※ `C` は ground truth を定義しないため `gt_form=`（空）として扱い、正誤列も空にしている。

---

### 2) CSV リッチ化（103 列 + JSON カラム）

**方針**

- CSV は「後から原因分析ができる」ことを最重要視
- 列名は長くてもよいので、見ただけで意味が分かるようにする
- 1 行に収まりにくい情報は JSON 文字列として 1 セルに格納（`*_json` カラム）

**主な追加列（カテゴリ）**

- 正誤（ground truth がある A/B のみ）
  - `is_predicted_form_correct`
  - `is_predicted_best_template_correct`
- 時間
  - `elapsed_time_total_one_case_seconds`
  - `elapsed_time_stage_1_degrade_seconds`〜`elapsed_time_stage_6_warp_seconds`
  - `run_elapsed_time_total_seconds`（run 全体）
- 解像度
  - source / degraded / rectified / rotated / template / aligned の各 `width/height`
- 改悪パラメータ
  - `degradation_*` 系
  - `degradation_parameters_json`
- DocAligner
  - `docaligner_polygon_xy_json`
  - `docaligner_polygon_margin_*`
- フォーム判定
  - `form_decision_score`
  - `form_decision_detail_json`
- XFeat
  - `xfeat_best_inliers / xfeat_best_matches / xfeat_best_inlier_ratio`
  - `xfeat_best_ref_keypoints_count / xfeat_best_tgt_keypoints_count`
  - `xfeat_best_reprojection_rms_px`
  - `xfeat_all_template_candidate_results_json`（候補テンプレの全結果）
- Homography 安定性
  - `homography_inversion_ok`
  - `homography_matrix_condition_number`
  - `homography_matrix_determinant`

**CSV 出力方式**

- `build_csv_row()` で「1 case = 1 dict(row)」を生成
- `csv.DictWriter` で書き出し
- `fieldnames` は実際に生成された row の key を順次収集して決定
  - これにより、列追加に強い（列が増えても落ちない）

---

## 動作確認

### 1) 構文チェック

```bat
C:\Users\takumi\develop\miniconda3\python.exe -m py_compile APA\paper_pipeline_v3.py
```

### 2) スモークテスト（A,B,C 各 1 枚）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v3.py --limit 1 --src-forms A,B,C --degrade-n 1 --log-level INFO --console-log-level INFO
```

**出力ディレクトリ例**

- `APA/output_pipeline/run_20260108_152057/`
  - `run.log`
  - `summary.json`
  - `summary.csv`（リッチ版）
  - `1_degraded/`〜`6_aligned/`

**実行結果サマリー（ログより）**

- total=3, ok=2
- stage 内訳:
  - done: 2
  - homography_unstable: 1

---

## 補足（今回のスモーク結果について）

- フォーム A は `template_ok=TRUE` だった
- フォーム B は `template_ok=FALSE`（例: `1.jpg` 入力が `5.jpg` とマッチ）
  - これは **「改善した CSV/ログにより、後から原因分析しやすくなった」ことを示す例**
  - `summary.csv` の以下の列で、どこが遅い/誤ったかを掘れる
    - `form_decision_detail_json`
    - `xfeat_all_template_candidate_results_json`
    - `degradation_parameters_json`
    - `elapsed_time_stage_*`

---

## 変更点まとめ

- [x] **失敗ケースも含めた読みやすいログ**（全 case で 1 行サマリ）
- [x] **正誤（A/B のみ）をログと CSV に追加**
- [x] **処理 1〜6 の実行時間 + 1 case 総時間**をログ/CSV に追加
- [x] **CSV をリッチ化**（103 列、JSON セルも活用）

---

## 次の改善候補（参考）

- B のテンプレ誤一致を減らす
  - `--template-topn` の増加（候補数を増やす）
  - `--match-max-side` の増加（精度 ↑/速度 ↓）
  - 事前フィルタ（global descriptor）の設計見直し
- フォーム判定の高速化（`elapsed_time_stage_4_form_decision_seconds` が支配的になりやすい）
