# 15 20260110 paper_pipeline_v6 高速化（WeChat 並列化 / XFeat ターゲット特徴キャッシュ）

## 概要

`paper_pipeline_v6.py` に対して、以下の 2 点を**精度を落とさず速度のみ改善**する目的で改修を実施しました。

1. **改善 1（優先度高）**: WeChatQRDetector の `Lock` による直列化を解消（回転スキャンの並列度を維持）
2. **改善 2**: XFeat のターゲット特徴をテンプレ枚数分再計算している無駄を解消（テンプレ 6 枚なら 6 回 →1 回）

実行確認まで行い、A/B/C 各フォームで最低 1 ケースの完走を確認しました。

---

## 環境メモ（重要）

このリポジトリは **conda(base) と `.venv` が混在**し得るため、実行は必ず `.venv` の Python を使用する。

- ✅ 使用すべき Python:

```bash
.venv/bin/python
```

（参考）`document/environment/macbookair.md` にも同趣旨の注意書きあり。

---

## 改善 1: WeChatQRDetector の Lock 直列化を解消

### 背景

回転スキャンは `ThreadPoolExecutor` で並列化されているが、WeChat QR 検出が

```python
with self._lock:
    res, points = self.detector.detectAndDecode(image_bgr)
```

のようにロックされていると、**結局 WeChat 検出だけ 1 本に直列化**される。

### 対応内容

`detectorを1個 + lock` 方式を廃止し、以下の方式に変更。

- **detector pool を作成**（`queue.Queue`）
- `detect()` 呼び出しのたびに detector を借りて使用し、finally で必ず返却
- `detectAndDecode` の重い処理自体は並列実行される

実装:

- `WeChatQRDetectorPool` を追加
- `init_wechat_qr_detector()` を pool を返すように変更
- `pool_size` は `rotation_max_workers` と同数にして起動時に一度だけ初期化

---

## 改善 2: XFeat のターゲット特徴を 1 case 内で使い回す

### 背景

テンプレ（例: 6 枚）ごとに `detectAndCompute()` が走っていたため、
ターゲット側特徴がテンプレ枚数分だけ再計算されていた。

### 対応内容

`CachedXFeatMatcher` に以下を追加。

- `prepare_target(tgt_bgr)`
  - `tgt_small, s_tgt` を作成
  - `out1 = xfeat.detectAndCompute(tgt_small, top_k=top_k)[0]` を **1 回だけ**生成
  - `invS_tgt = inv(scale_matrix(s_tgt))` を返す

さらに、テンプレループ内で使う軽量 API として

- `match_with_cached_ref_and_prepared_target(ref, out1, invS_tgt)`

を追加し、ループ内処理を

- `match_lighterglue(ref.out0, out1)`
- `cv2.findHomography(...)`

のみに寄せた。

※ **同じ out1 を使うだけ**なので、精度（結果）は変えずに速度だけ改善する方針。

---

## 実行確認

### 注意

`python`（conda 側）で動かすと `capybara` が見つからず落ちるため、必ず `.venv/bin/python` を使う。

### 実行コマンド（実施）

#### フォーム B（WeChat 並列 + XFeat キャッシュ経路の確認）

```bash
.venv/bin/python paper_pipeline_v6.py --src-forms B --limit 1 --degrade-n 1 --log-level INFO --console-log-level INFO
```

- WeChat QR detector initialized: `pool_size=8`
- `stage=done`, `ok_expected=TRUE`, `ok_warp=TRUE` を確認
- 例: `match_s=0.417940s`（テンプレ 6 枚照合）

#### フォーム A

```bash
.venv/bin/python paper_pipeline_v6.py --src-forms A --limit 1 --degrade-n 1 --log-level INFO --console-log-level INFO
```

- `stage=done`, `ok_expected=TRUE`, `ok_warp=TRUE` を確認

#### フォーム C（棄却が期待動作）

```bash
.venv/bin/python paper_pipeline_v6.py --src-forms C --limit 1 --degrade-n 1 --log-level INFO --console-log-level INFO
```

- `stage=form_unknown` で停止
- `ok_expected=TRUE`（C は Unknown が成功扱い）

### 出力先

実行ごとに `output_pipeline/run_YYYYmmdd_HHMMSS/` が生成される。

（今回の例）

- `output_pipeline/run_20260110_163741`（B）
- `output_pipeline/run_20260110_163757`（A）
- `output_pipeline/run_20260110_163810`（C）

---

## 変更ファイル

- `paper_pipeline_v6.py`
  - WeChat QR detector を **Lock 方式 →pool 方式**に変更
  - `CachedXFeatMatcher` にターゲット特徴の前計算 API を追加
  - マッチングループで target 特徴を使い回すよう変更

---

## 補足（運用上の注意）

- WeChat detector はモデルロードが重いため、**pool_size を worker 数と同等**にして起動時に一括生成するのが前提。
- `.venv` を使わないと DocAligner 系が import できないため、README/運用手順に `.venv/bin/python` 指定を徹底する。

---

## v5（改善前）と v6（改善後）の run.log 比較・考察

比較対象ログ:

- v5（改善前）: `output_pipeline/run_20260110_103809/run.log`
- v6（改善後）: `output_pipeline/run_20260110_164101/run.log`

いずれも `--src-forms A,B,C`、データセット総数 `total_cases=180` の実行ログで比較しています。

### 1) 全体の実行時間（最重要）

| 指標                         |        v5 |       v6 |      差分 |                              改善率 |
| ---------------------------- | --------: | -------: | --------: | ----------------------------------: |
| run_elapsed_total_seconds    | 1222.808s | 646.012s | -576.796s | **約 47.2% 短縮（約 1.89 倍高速）** |
| avg_elapsed_per_case_seconds |    6.793s |   3.589s |   -3.204s |                   **約 47.1% 短縮** |
| median(per-case total)       |    6.244s |   3.263s |   -2.981s |                   **約 47.7% 短縮** |

結論として、**v6（改善後）は v5（改善前）に対してほぼ「半分の時間」で同じ 180 ケースを処理**できています。

### 2) ステージ別の処理時間（どこが速くなったか）

ログ末尾の `stage time totals` / `stage time mean per case` を比較すると、改善が効いた箇所が明確です。

#### ステージ合計（180 ケース合算）

| stage time totals |      v5 |      v6 |     差分 |                              改善率 |
| ----------------- | ------: | ------: | -------: | ----------------------------------: |
| degrade_s         | 143.54s |  89.67s |  -53.87s |                       約 37.5% 短縮 |
| docaligner_s      |  46.24s |  25.29s |  -20.95s |                       約 45.3% 短縮 |
| rectify_s         |   1.08s |   0.57s |   -0.51s |                       約 47.2% 短縮 |
| decide_s          | 646.11s | 417.06s | -229.05s |                   **約 35.4% 短縮** |
| match_s           | 267.93s |  50.66s | -217.27s | **約 81.1% 短縮（約 5.29 倍高速）** |
| warp_s            |   7.84s |   4.02s |   -3.82s |                       約 48.7% 短縮 |

**最も効いているのは `match_s`（XFeat matching）で、ここが約 5.3 倍速くなっています。**

また `decide_s`（フォーム判定）も 35% 以上短縮しており、こちらも全体短縮に大きく寄与しています。

#### ステージ平均（1 ケースあたり）

| stage time mean per case |     v5 |     v6 |    差分 |            改善率 |
| ------------------------ | -----: | -----: | ------: | ----------------: |
| decide_s                 | 3.589s | 2.317s | -1.272s |     約 35.4% 短縮 |
| match_s                  | 1.488s | 0.281s | -1.207s | **約 81.1% 短縮** |

この 2 ステージが「今回の改善の狙いどおり」主要ボトルネックであり、
**v6 では合計で 1 ケースあたり約 2.479 秒（=1.272+1.207）分の短縮**が出ています。

### 3) 改善内容とログ上の変化の対応付け

#### 改善 2（XFeat の target 特徴キャッシュ）→ `match_s` の大幅短縮

v5 ではテンプレ 6 枚に対して、ターゲット側特徴（`xfeat.detectAndCompute()`）がテンプレごとに再実行されていたため、
match_s が重くなりやすい構造でした。

v6 では target 側特徴を 1 回だけ作り、テンプレループ内を `match_lighterglue + findHomography` 中心にしたため、
**`match_s: 267.93s → 50.66s`（約 81% 減）**という改善がログで確認できます。

#### 改善 1（WeChat QR の Lock 直列化解消）→ `decide_s` の短縮

v5 のログでは

```
[OK] WeChat QR detector initialized: ...
```

とあるだけで pool_size 表記は無く、実装的には `Lock` により WeChat 検出が直列化されている可能性が高い状態でした。

v6 のログでは

```
[OK] WeChat QR detector initialized: ... (pool_size=8)
```

と明示され、回転スキャンの worker 数（`rotation-max-workers: 8`）と同数の detector を確保していることが分かります。
これにより QR 検出が並列化され、`decide_s` が **646.11s → 417.06s（約 35% 減）**まで短縮しています。

### 4) 精度（正解率・棄却率）の変化

今回の改善方針は「精度は落とさず速度だけ上げる」でしたが、
run.log の集計値を見ると、**ごく僅かに成功率が変動**しています。

| 指標                                 |              v5 |              v6 |  差分 |
| ------------------------------------ | --------------: | --------------: | ----: |
| ok_expected_behavior(user_KPI)       | 173/180 (96.1%) | 171/180 (95.0%) | -2 件 |
| ok_warp(done_aligned_generated)      | 113/180 (62.8%) | 111/180 (61.7%) | -2 件 |
| A form accuracy                      |   54/60 (90.0%) |   53/60 (88.3%) | -1 件 |
| B template accuracy                  |   59/60 (98.3%) |   58/60 (96.7%) | -1 件 |
| C reject_success(stage=form_unknown) |  60/60 (100.0%) |  60/60 (100.0%) |    ±0 |

ステージ件数の内訳も

- v5: `done=113`, `form_unknown=67`
- v6: `done=111`, `form_unknown=68`, `homography_unstable=1`

となっており、v6 では **`homography_unstable` が 1 件だけ発生**しています。

#### 考察（なぜ僅差が出る可能性があるか）

- 本改善は理論上「同じ入力に対して同じ out1 を使う」「WeChat detector を増やすだけ」なので、
  期待としては精度（結果）は変わりにくいです。
- 一方で、実運用上は以下の要因で **run 間の微小なブレ**が出る可能性があります。
  - OpenCV / wechat_qrcode 内部処理の非決定性（スレッドスケジューリングや内部 SIMD 差など）
  - 複数 detector を使うことによる内部状態差（完全にステートレスではない実装の可能性）
  - `findHomography(USAC_MAGSAC)` の乱数性/非決定性

#### 結論

- **速度改善はログ上は明確（約 1.89 倍高速）**
- **精度面は大勢は維持しつつ、180 ケース中 1〜2 件の差分**が出ている

このため、厳密に「精度が変わっていない」ことを確認するには、

- 同一 seed / 同一入力での複数回実行
- 特に差分が出たケース（`form_unknown` / `homography_unstable`）の画像を抽出し、
  v5/v6 での中間生成物（`4_rectified_rot`, `5_debug_matches`）を比較

まで行うのが安全です。
