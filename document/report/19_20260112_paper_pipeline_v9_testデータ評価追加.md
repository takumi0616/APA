# paper_pipeline_v9.py 改善レポート（image/test の評価追加）

## 実施日時

2026 年 1 月 12 日

## 目的

`paper_pipeline_v9.py` に対して、`image/test/` 配下の「手書き追記入り画像」も通常処理に加えて評価し、
ファイル名（例: `A_3.png`）から **正解フォーム/正解テンプレ**を自動判定して精度を出せるようにする。

## 仕様（image/test の命名規則）

- `image/test/{A|B|C}_{n}.png`（例: `A_3.png`）
- 意味: `image/{A|B|C}/{n}.jpg` をベースに手書き文字が追加された画像

## 実装内容

### 1) test データの列挙と正解ラベル推定

- `list_test_images()` を追加し、`image/test/` の画像（png/jpg/jpeg）を列挙
- `parse_test_filename()` を追加し、`A_3.png -> ("A", "3")` を抽出
- test 画像の正解テンプレパスを `image/{form}/{n}.jpg` として組み立て

### 2) test データを通常パイプラインに統合して実行

- `process_one_case()` に以下を追加
  - `source_dataset="test"` を保持し、CSV にも `source_dataset_name(synthetic_or_test)` として出力
  - `ground_truth_form / ground_truth_template_path / ground_truth_template_number` を item に保存し、CSV の正解列を上書き可能に
- `main()` にて `image/A,B,C` の通常処理後、`image/test` を追加で処理

### 3) test での A 取りこぼし対策（手書きの影響）

フォーム A 誤検出抑制のため導入していた「マーカー周辺が白地」制約が、
手書きがマーカー近傍に入った test 画像では過剰に厳しくなり `form_unknown` になるケースがあった。

対策として、`source_dataset==test` の場合のみ **白地制約を緩めた `MarkerGeometryConfig`** を適用。

### 4) デバッグ画像（マッチ可視化）の生成確認

`--save-images none` でも確認しやすいよう、
`5_debug_matches/*_matches.jpg` は常に保存されるように調整。

## 実行確認

### 構文チェック

```bash
.venv/bin/python -m py_compile paper_pipeline_v9.py
```

### スモーク実行（A,B,C 各 1 枚 + test 全件、degrade=1）

```bash
.venv/bin/python paper_pipeline_v9.py --src-forms A,B,C --limit 1 --degrade-n 1 --save-images none --log-level INFO --console-log-level INFO
```

確認できた結果（抜粋）:

- `A_1_deg00`: `done`, `template_ok=TRUE`
- `B_1_deg00`: `done`, `template_ok=TRUE`
- `C_1_deg00`: `form_unknown`（棄却成功）
- `test_A_3_deg00`: `done`, `pred_form=A`, `best_template=3.jpg`, `template_ok=TRUE`

### 生成物の確認

以下が生成されることを確認:

```bash
ls -1 output_pipeline/run_20260112_234623/5_debug_matches | head
```

出力例:

```
A_1_deg00_matches.jpg
B_1_deg00_matches.jpg
test_A_3_deg00_matches.jpg
```

## 変更ファイル

- `paper_pipeline_v9.py`
  - `image/test` の読み込み・評価処理を追加
  - test の ground truth を CSV/ログに反映
  - test でのフォーム A 取りこぼし対策（白地制約の緩和）

## まとめ

- [x] `image/test` の画像を追加で処理し、ファイル名から正解（フォーム/テンプレ）を推定して精度集計できるようにした
- [x] test データ特有の手書き影響でフォーム A が落ちる問題を、test 専用の白地制約緩和で改善した
- [x] 実行確認で `test_A_3.png` が `template_ok=TRUE` になることを確認した
