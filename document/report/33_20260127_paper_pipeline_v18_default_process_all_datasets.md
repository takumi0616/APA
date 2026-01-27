# 33_20260127_paper_pipeline_v18_default_process_all_datasets

## 目的

ユーザー要望:

> paper_pipeline_v18.pyにおいてA、B、Cやtestも処理するようにしてください

従来は `src_forms=[]` / `test-limit=0` などの既定で **target-only**（現場画像のみ）に寄っていたため、
デフォルト実行で synthetic（A/B/C）や `image/test` が処理されない状態だった。

## 変更内容（paper_pipeline_v18.py）

### デフォルトで処理するデータセットを拡張

- `PIPELINE_DEFAULTS["src_forms"]`
  - 変更前: `[]`（synthetic を処理しない）
  - 変更後: `["A", "B", "C"]`

- `PIPELINE_DEFAULTS["test_limit"]`
  - 変更前: `0`（skip）
  - 変更後: `-1`（all）

- `PIPELINE_DEFAULTS["target_limit"]`
  - 既に `-1`（all）なので維持

これにより、引数なし実行（デフォルト）で:

- synthetic（A/B/C）
- test（image/test）
- target（image/target）

をまとめて処理する。

### 先頭ドキュメントの記述を更新

ファイル冒頭の「デフォルトは target-only」という説明を削除し、
デフォルトが **synthetic + test + target** であることを明記した。

また、コード内のコメント（v18.13 target-only 前提の説明）を
「過去の運用の話」として意味が通るように調整した。

## スモークテスト

大量実行を避けるため、最小限の件数で起動確認。

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -u paper_pipeline_v18.py \
  --limit 1 --degrade-n 1 --test-limit 1 --target-limit 1 \
  --save-images none --log-level WARNING --console-log-level WARNING
```

結果:

- `output_pipeline/run_20260127_153833/summary.json` / `summary.csv` が生成されることを確認
  - `run.log` は WARNING 以上のみを出す設定のため、警告が無い場合は空になり得る

## 補足（運用メモ）

デフォルト実行はデータ量次第で重くなるため、用途に応じて以下で絞る:

- synthetic のみ: `--target-limit 0 --test-limit 0`
- target のみ: `--src-forms '' --test-limit 0`
- test のみ: `--src-forms '' --target-limit 0`
