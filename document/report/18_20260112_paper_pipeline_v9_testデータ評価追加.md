# 18 20260112 paper_pipeline_v9 改善レポート（image/test の評価追加）

## 実施日時

2026 年 1 月 12 日

## 目的

`paper_pipeline_v9.py` に対して、通常の処理（`image/A,B,C` を入力として改悪生成 → 推定 →warp）に加えて、
新規に追加された **`image/test/`** データセットを同時に評価し、

- **ファイル名から正解（GT）を読み取り**
- **パイプラインを実行して精度を算出**

できるようにする。

`image/test` の命名規則：

- `{A|B|C}_{number}.png`（例: `A_3.png`）
- 意味：テンプレ `image/A/3.jpg` に手書き文字が追加された画像

## 実施内容

### 1) test データ（image/test）の列挙とファイル名パースを追加

- `list_test_images()`
  - `image/test` 配下の `.png/.jpg/.jpeg` を列挙
- `parse_test_filename()`
  - `A_3.png` → `(form="A", template_number="3")` に変換

### 2) test データは「改悪生成をスキップ」してパイプラインに投入

test データはすでに「手書き追加済み」の観測画像であり、
従来の `warp_template_to_random_view()`（改悪生成）は不要なため、以下の方針に変更。

- `process_one_observed_case()` を新規追加
  - degrade を行わず、入力画像をそのまま `degraded` として扱う
  - 以降は通常通り
    - DocAligner → rectify → (0/180)フォーム判定 → XFeat → warp
  - 正解フォーム/正解テンプレはファイル名から決定して item に保存

### 3) ground truth の扱いを拡張（test 用に上書き可能に）

従来は `image/A,B` 入力時のみ「入力ファイル名＝正解テンプレ」として ground truth を決めていたが、
test では `A_3.png` のようにテンプレ名と一致しない。

そのため、CSV 生成時に以下を優先するように変更：

- item に `ground_truth_form / ground_truth_template_path / ground_truth_template_number` があれば、それを優先
- 無ければ従来ロジック（`_case_truth`）へフォールバック

（実装）

- `_truth_from_item(item)` を追加
- `build_csv_row()` で `_truth_from_item(item)` を使用

### 4) 集計（run.log / STATS）に test データセット指標を追加

`summarize_results()` 内で `source_dataset` が `test` の要素を抽出し、

- test A の template_accuracy
- test B の template_accuracy
- test C の reject_success（`stage=form_unknown`）

をログに出力。

## 実行確認

### 構文チェック

```bash
.venv/bin/python -m py_compile paper_pipeline_v9.py
```

### スモーク実行（A/B/C 各 1 枚 + test も自動で評価）

```bash
.venv/bin/python paper_pipeline_v9.py \
  --src-forms A,B,C \
  --limit 1 \
  --degrade-n 1 \
  --save-images none \
  --log-level INFO \
  --console-log-level INFO
```

確認できたログ（抜粋）：

- `Processing test dataset (image/test): 1 images`
- `id=test_A_3 ... stage=done ... best_template=3.jpg ... template_ok=TRUE`

出力ディレクトリ例：

- `output_pipeline/run_20260112_230646/`

## 変更ファイル

- `paper_pipeline_v9.py`

## まとめ

- [x] `image/test` の画像を自動列挙し、ファイル名から GT（フォーム/テンプレ番号）を解釈
- [x] test データは改悪生成をスキップし、通常パイプラインで処理・精度評価
- [x] `summary.csv` / `run.log` の集計に test データセットの精度指標を追加

---

## 補足（運用メモ）

- test データは `--src-forms` に関係なく実行される実装になっているため、
  test だけ回したい場合は `--src-forms` を `A` のみにして `--degrade-n 0` 相当の挙動が欲しくなる可能性があります。
  必要なら次の改善として `--enable-test` / `--test-only` のような CLI を追加可能です。
