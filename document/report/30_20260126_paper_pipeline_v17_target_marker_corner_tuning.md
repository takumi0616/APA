<!--
作業レポート: paper_pipeline_v17 target改善（DocAligner安定化＋Aマーカー取りこぼし対策）
日付: 2026-01-26
-->

# 30_20260126_paper_pipeline_v17_target_marker_corner_tuning

## 背景 / 目的

target（現場撮影）画像で DocAligner が「三角形/線」っぽい polygon を返し、rectify 後にフォーム判定（特にフォームA）が `no_detection` で落ちるケースがあった。

今回の目的は以下。

- DocAligner の失敗（角欠け・退化 quad）を **後段へ伝播させない**
- target でのフォームA取りこぼし（cornerの影/枠線の誤検出など）を **減らす**
- `image/target` を **改悪なしでそのまま処理**し、aligned を出力できること

## 実施内容（主な変更）

### 1) DocAligner: multi 推論 + 退化 quad 除外 + フォールバック

- DocAligner の返す polygon を `normalize_polygon_to_quad()` で quad 化
  - N点 polygon / 3点 / 点順不定でも `convexHull -> minAreaRect` で quad を得る
- `_is_valid_quad()` で **面積比・最短辺・辺比**から退化形状を早期除外
- `detect_polygon_docaligner_multi()` で
  - model/type/pad/scale を変えて複数回推論
  - 候補は rectify→フォーム判定スコアで選択（ただしフォーム判定を必須条件にはしない）
  - strict 全滅時は「面積が大きいもの」を救済候補として評価に回す
  - それでもダメなら OpenCV 輪郭ベース `detect_polygon_fallback_opencv()` を利用

### 2) フォームA（target）取りこぼし対策: マーカー検出の corner 絞り込み

target の rectify 後画像ではマーカーが「かなり端」にある一方、探索範囲が広いと枠線/文字を誤検出しやすい。

そこで `detect_formA_marker_boxes_base()` を調整。

- corner探索範囲（`corner_margin_ratio`）を導入し、既定値を 0.12 に設定
- マーカー想定サイズ（min/max ratio）を設定化
- corner への近さ（pos_score）をスコアに加点し、端の正しいマーカーを優先

さらに `process_one_case()` 側で target の A-geometry を recall 寄りに調整。

- `surround_min_mean_gray` を 150 に緩和（影でcornerが暗いケース救済）

### 3) 更新履歴の追記

- `paper_pipeline_v17.py` 冒頭コメントに v17.15（2026-01-26）を追記。

### 4) v17.16: target の no_detection をさらに救済（A-geometry recall寄り）

target の一部ケースでは、マーカー自体は拾えているが「周辺白地制約（ink_ratio）」「最小面積比」の閾値で弾かれて `no_detection` になることがあった。

そのため target（`source_dataset==target`）のみ、A-geometry を追加で緩めた。

- `surround_max_ink_ratio`: **0.05 → 0.08**
- `min_marker_area_page_ratio`: **5e-5 → 3.5e-5**

※ C->A 誤判定抑制よりも「実フォームを落とさない」を優先。

### 5) v17.16: advanced fallback の診断性改善（attempts記録）

`form_unknown(no_detection)` のときに polygon を再推定して救済する _advanced fallback_ について、
「なぜ採用されなかったのか（margin探索の結果）」が summary に残らず追いづらかった。

そこで各 margin 試行の結果（decision / unknown_reason / scan_max 等の診断）を `attempts` として summary に記録するようにした。

## 実行結果

以下コマンドで `image/target` の先頭10枚を処理。

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v17.py --src-forms "" --test-limit 0 --target-limit 10 --save-images fail
```

出力:

- `APA/output_pipeline/run_20260126_124047/`

サマリ（summary.jsonより）:

- total=10
- done=5
- form_unknown(no_detection)=5

失敗は残ったが、targetで **aligned まで到達するケースが確認できた**。

## 今後の改善案（残課題）

- `no_detection` になった target ケースについて、
  - `2_doc` / `3_rectified` の見た目（マーカーが切れていないか）
  - `form_decision_detail_json`（scan内のAスコア推移）
    を見て原因を切り分ける。
- corner_margin_ratio / marker_size_ratio を target 画像のばらつきに合わせて微調整。
  （端ギリギリ過ぎる撮影・トリミングが強い場合は corner 範囲を少し拡大する等）

---

## 参考: form_unknown の内訳（run_20260126_124047）

`summary.json` 集計:

- `form_unknown`: 5
- `unknown_reasons`: `no_detection` のみ
- cases:
  - target_A_1_10_001
  - target_A_1_13_004
  - target_A_1_15_006
  - target_A_1_16_007
  - target_A_1_17_008
