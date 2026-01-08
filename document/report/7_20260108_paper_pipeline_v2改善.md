# paper_pipeline_v2.py 改善レポート（1〜9 対応）

## 実施日時

2026 年 1 月 8 日

## 目的

`APA/paper_pipeline_v2.py`（`paper_pipeline.py` のコピー）について、ユーザー指定の改善 1〜9 を順番に反映し、
大量処理・運用を想定して **安定性/速度/デバッグ容易性** を向上させる。

## 前提（環境）

資料に従い、システム環境を汚さないローカル構成を前提とした。

- Python: Miniconda ローカルインストール
  - `C:\Users\takumi\develop\miniconda3\python.exe`
- Git: Portable Git（torch.hub が内部で呼ぶケース対策）
  - `C:\Users\takumi\develop\git\bin\git.exe`

## 対象ファイル

- `APA/paper_pipeline_v2.py`

## 改善内容（1〜9）

### 1. DocAligner の polygon マージン設定を比率ベースに変更

固定 px ではなく「紙サイズ（polygon 辺長）に対する比率」でマージンを決定。
解像度の違いに対して安定しやすい。

- 追加 CLI
  - `--polygon-margin-ratio`（デフォルト 0.03）
  - `--polygon-margin-min-px` / `--polygon-margin-max-px`
  - 互換用 `--polygon-margin-px`（>0 の場合 ratio を上書き）

### 2. 進捗/失敗要因のログ強化（logging 化）

- `print` ベースから `logging` ベースへ移行
- 出力先
  - コンソール（レベル指定可）
  - `run_.../run.log`
- ステージ別集計・所要時間集計を追加
  - stage counts（例: `done`, `form_unknown`, `homography_unstable`）
  - stage time totals（degrade/docaligner/rectify/decide/match/warp）
- 例外発生時に traceback を summary に残す

### 3. マーカー/QR 検出の追加前処理オプション

- フォーム A マーカー検出に前処理のバリアントを追加
  - `--marker-preproc` = `none/basic/morph`
  - `basic`: gray/CLAHE/adaptive threshold
  - `morph`: adaptive + morphology close/open
- QR 検出（robust）にも adaptive+morph を追加し、照明ムラや回転後の検出落ちを軽減

### 4. テンプレート側の前処理・キャッシュ

- テンプレの XFeat 特徴量を事前計算してキャッシュ（`CachedXFeatMatcher`）
  - 各ケースでターゲット側だけ特徴抽出する方式に変更
- グローバル特徴（軽量ヒストグラム）でテンプレ候補を上位 N 枚に絞り込み
  - `--template-topn`（デフォルト 3、0 で全探索）

### 5. 回転スキャンの効率化（Coarse-to-Fine）

従来の「全角度（36 回）並列」を、

1. 0/90/180/270 の 4 回で粗探索
2. 上位 2 候補の周辺のみを細探索（angles リストに沿ってサブセットを作る）

に変更。

※ FAST で全く拾えない場合は、robust 側で 0/90/180/270 を救済（rescue）。

### 6. Unknown（未知フォーム）判定ロジック追加

- `--unknown-score-threshold`（デフォルト 1.2）
- `--unknown-margin`（デフォルト 0.15）

スコアが低すぎる/差が僅差の場合は **Unknown** として明確に処理を打ち切る。

### 7. 逆ホモグラフィの安定性向上

`np.linalg.inv(H)` をそのまま呼ぶのではなく、

- `min_inliers_for_warp`
- `min_inlier_ratio_for_warp`
- `max_h_cond`（条件数）

でフィルタし、信頼度が低い場合は `homography_unstable` として破棄。

### 8. 日本語フォント処理の汎用化

従来の Windows フォントパス固定を廃止し、

1. `APA_FONT_PATH` 環境変数（ユーザー指定）
2. OS 別の既知候補パス（Windows/Mac/Linux）
3. matplotlib.font_manager による best effort
4. 最後は Pillow default（日本語が出ない場合は ASCII fallback）

の順に解決するようにした。

### 9. 巨大 main 関数の分割

1 ケース（1 改悪画像）処理を `process_one_case()` に分離し、
読みやすさ・保守性を改善。

## 実行確認

### 構文チェック

```bat
C:\Users\takumi\develop\miniconda3\python.exe -m py_compile APA\paper_pipeline_v2.py
```

### スモークテスト（A,B 各 1 枚）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v2.py --limit 1 --src-forms A,B --degrade-n 1
```

結果例（run ディレクトリ）:

- `APA/output_pipeline/run_20260108_134606/`
  - `A_1_deg00`: done
  - `B_1_deg00`: done

### スモークテスト（A,B,C 各 1 枚）

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v2.py --limit 1 --src-forms A,B,C --degrade-n 1
```

結果例:

- `APA/output_pipeline/run_20260108_134726/`
  - `A_1_deg00`: done
  - `B_1_deg00`: homography_unstable（条件数が大きく、安定性優先で破棄）
  - `C_1_deg00`: form_unknown（A/B に該当しない）

※ `homography_unstable` は改善(7)の意図通りで、低信頼結果を無理にワープしないための安全策。

## 生成物（出力）

各 run 配下に以下が生成される：

- `1_degraded/`（改悪画像）
- `2_doc/`（DocAligner polygon 可視化）
- `3_rectified/`（透視補正）
- `4_rectified_rot/`（フォーム確定に使用した回転＋根拠可視化）
- `5_debug_matches/`（XFeat マッチ可視化）
- `6_aligned/`（テンプレ座標への warp）
- `summary.json` / `summary.csv` / `run.log`

## 結論

指定の改善 1〜9 を順番に反映し、

- 解像度差に強い polygon margin
- 運用向け logging + 集計 + 例外詳細
- 検出の前処理オプション追加
- テンプレキャッシュ＋事前絞り込みで高速化
- 回転スキャンの計算量削減
- Unknown 判定の明確化
- 逆ホモグラフィの安全策
- 日本語フォント解決の汎用化
- 関数分割による可読性向上

を実装できた。

---

## 追加対応（2026/01/08）: `--explain` の表示改善

ユーザー指摘により、`--explain` 出力で **選択肢がある引数は選択肢を明示**するように追記。

例:

- `--docaligner-model` の選択肢を出力に含める
  - `(lcnet050/lcnet100/fastvit_t8/fastvit_sa24)`
- `--docaligner-type` / `--device` / `--rotation-mode` / `--marker-preproc` / `--log-level` なども
  - `(a/b/c)` 形式で表示

確認コマンド:

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v2.py --explain
```

---

## 追加対応（2026/01/08）: ファイル冒頭コメント（目的/概要/出力/注意）の整合性確認と修正

ユーザー指摘により、`APA/paper_pipeline_v2.py` の冒頭 docstring（「目的」「パイプライン概要」「出力」「注意」）が
現状実装と一致しているかを点検し、不整合を修正。

主な修正点:

- ファイル名表記を `paper_pipeline.py` → `paper_pipeline_v2.py` に修正
- 実装上の差分を反映
  - 回転探索が Coarse-to-Fine であること
  - Unknown 判定（`stage=form_unknown`）があること
  - polygon margin が比率ベースであること
  - テンプレが `--template-topn` により事前絞り込みされること
  - `run.log` が出力されること
- import 注意書きの実行例も `paper_pipeline_v2.py` に合わせて修正

構文チェック:

```bat
C:\Users\takumi\develop\miniconda3\python.exe -m py_compile APA\paper_pipeline_v2.py
```
