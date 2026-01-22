# 27_20260122_full_rotation_degrade_and_inframe_constraint

## 概要

改悪生成（degrade）で **0〜360度回転**をデフォルト有効化しつつ、
「紙が画像内に収まる（全頂点がin-frame）」制約を追加して、
極端に崩れた学習/評価データが出ないようにした。

あわせて、full-rotation（`max_rot>=180`）時に
生成が **0度近傍に偏る / max_attempts を使い切って遅くなる**問題を避けるため、
面積・縮小率の閾値を現実的な値へ調整した。

## 目的

- 改悪生成で「上下逆」「横向き」を含む **回転バリエーション**を確保する。
- ただし、
  - 紙がフレーム外に逃げる
  - 紙が極端に小さくなる
  - 透視が強すぎて不自然な台形になる
    といったケースはデータとして不適切なので抑制する。

## 変更点

### 1) paper_pipeline_v15 のデフォルトを full-rotation に

- `PIPELINE_DEFAULTS["degrade"]["max_rot_deg"] = 180.0`
  - `warp_template_to_random_view()` の仕様により、`max_rot>=180` の場合は 0〜360一様回転モード。

### 2) full-rotation 時の可視面積閾値を緩和

- `PIPELINE_DEFAULTS["degrade"]["min_visible_area_ratio"]`
  - 斜め回転（例: 30〜60°）では、紙をフレーム内に収めるために **紙の見かけサイズが短辺基準に縮小**しやすい。
  - `out_w*out_h` に対する占有率を高くしすぎると、物理的に成立しないケースが増えて
    - 生成が遅い
    - 0°/90°/180°/270°付近に偏る
      という副作用が出る。
  - そのため「紙が写っていること」だけを保証する目的に留め、デフォルトを **0.25** に。

### 3) test_recovery_paper: full-rotation 時の fit-to-frame 閾値を緩和

`warp_template_to_random_view()` 内:

- `min_fit_scale`
  - `max_rotation_deg >= 180` のとき **0.70**
  - それ以外は従来通り **0.78**

狙い:

- full-rotation では「斜め回転 + in-frame」を満たすために、
  一定の縮小（fit-to-frame）が必須。
- 閾値が厳しすぎると生成が成立せず、結果として回転が偏る。

## 動作確認（パイプライン実行）

以下で実行し、回転バリエーションが混ざること、およびパイプラインが最後まで流れることを確認。

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v15.py --src-forms A --limit 1 --degrade-n 12 --save-images none
```

出力（run）:

- `APA/output_pipeline/run_20260122_164533/`

サマリ（run.log より）:

- total=108
- ok_expected=40 (37.0%) / ok_warp=40 (37.0%)
- stage counts
  - docaligner_failed: 62
  - done: 40
  - form_unknown: 3
  - homography_unstable: 3

ステージ時間合計（s）:

- docaligner_s: 31.63
- decide_s: 32.74
- uvdoc_s: 25.00
- bgdiv_s: 11.42
- match_s: 71.15
- warp_s: 3.49

補足:

- full-rotation + in-frame 制約の導入により、難しいケースで **DocAligner 失敗**が増える傾向が見られた。
  - ここは改悪強度（perspective、影/しなり等）と DocAligner のロバスト性のトレードオフ。

## 変更ファイル

- `APA/paper_pipeline_v15.py`
  - `PIPELINE_DEFAULTS["degrade"]["max_rot_deg"]` を 180.0 に
  - `PIPELINE_DEFAULTS["degrade"]["min_visible_area_ratio"]` を 0.25 に
- `APA/test_recovery_paper.py`
  - full-rotation 時の `min_fit_scale` を 0.70 に緩和
