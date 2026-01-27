# 32_20260127_paper_pipeline_v18_docaligner_norm_meta_reuse

## 概要

`paper_pipeline_v18.py` の target 運用（精度優先）を継続しつつ、
DocAligner multi 内での **`normalize_polygon_to_quad_with_meta()` の二重実行を削減**して、
無駄な計算を減らす修正を入れた。

本レポートは「処理時間が厳しい target の no_detection ケース」を中心に、
何が時間を使っているかも合わせて整理する。

## 変更内容

### 1) DocAligner multi で normalize を二重に回していた箇所を削減

対象: `detect_polygon_docaligner_multi()`

従来:

- `_run_docaligner_once()` 内部で正規化（`normalize_polygon_to_quad_with_meta`）を実施
- さらに multi 側で「診断のため」に **同じ polygon に対して再度 `normalize_polygon_to_quad_with_meta()` を実行**していた

修正:

- `_run_docaligner_once_with_meta()` を利用し、
  **正規化済み polygon と `norm_meta` を1回の実行で返す**ように統一
- multi 側では `norm_meta` をそのまま `all_polys/candidates` に保持し、再正規化しない

狙い:

- `normalize_polygon_to_quad_with_meta()` は「退化quad修復」でエッジベース復旧が走り得るため、
  パスによっては軽くない
- multi の推論回数が増えると二重実行の積み上げが無視できなくなる

## 実行ログ（target-limit=2）

コマンド:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -u paper_pipeline_v18.py \
  --target-limit 2 --save-images fail --log-level INFO --console-log-level INFO
```

### run_20260127_150427 の結果（抜粋）

- `target_A_1_1` は成功（warpまで到達）
  - total ≈ **11.60s**
  - 内訳: docaligner ≈ **6.56s**, decide ≈ **0.44s**, bgdiv ≈ **1.74s**, match ≈ **2.32s**

- `target_A_1_10` は `form_unknown(no_detection)`
  - total ≈ **66.29s**
  - 内訳: docaligner ≈ **39.77s**, decide ≈ **26.42s**

全体:

- total=2
- ok_expected=1 (50%)
- elapsed avg per case ≈ **39.14s**
- stage time totals:
  - docaligner_s ≈ **46.34s**
  - decide_s ≈ **26.86s**

## 所感 / 次の論点

- 今回の修正は「normalize 二重実行の削減」だが、
  target の no_detection ケースでは
  - DocAligner multi の推論回数
  - さらに no_detection 時の advanced fallback（polygon再推定 + margin試行 + decide再試行）
    により **docaligner/decide が桁違いに重くなる**

- 実運用で **10s/枚** を目指すには、
  no_detection 時の advanced fallback を
  - target では制限する（eval_max_candidates/margins をさらに絞る、試行回数上限、タイムアウト）
  - あるいは profile=fast を明示指定する
    といった “重い救済ルートの制御” が必要

（補足）今回の変更は correctness を崩さずに無駄を削る方向であり、
残る支配項は DocAligner 推論 + no_detection 救済ループ。
