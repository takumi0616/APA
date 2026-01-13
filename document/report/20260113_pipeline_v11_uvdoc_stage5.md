# 2026-01-13: paper_pipeline_v11 へ UVDoc(stage5) 組み込み + 改悪の安定化

## 目的

- パイプラインを 7 段階に整理し、**フォーム判定後に UVDoc による unwarp(成形) を挿入**する。
- 改悪生成で「紙が小さくなりすぎる」「奥行き(透視)が強すぎる」ケースを抑制し、
  form_unknown / homography_unstable を減らす。

## 実施内容（主な変更点）

### 1) 改悪生成（warp_template_to_random_view）の安全制約追加

対象: `APA/test_recovery_paper.py`

- 生成 quadrilateral の **縮小率**（fit-to-frame のスケール）が小さすぎる場合を棄却
  - `min_fit_scale = 0.78`
- **透視が強すぎる**（上下/左右の辺長比が極端）場合を棄却
  - `max_perspective_edge_ratio = 1.55`
- **潰れすぎ**（最短辺/最長辺比が小さい）を棄却
  - `min_edge_len_ratio = 0.58`
- 最終 fallback は clipping ではなく **安全な正面矩形**を返す
- meta に以下を追加し、CSV/JSON で追跡可能にした
  - `visible_area_ratio`, `fit_scale`, `edge_ratio_top_bottom`, `edge_ratio_left_right`, `quad_area_ratio_to_base` など

### 2) フォーム A マーカー検出の探索領域を拡張

対象: `APA/test_recovery_paper.py`

- corner 領域を 15% → 20% に拡大
  - DocAligner の polygon margin / rectify の影響で「角から少し内側」にマーカーが来るケースの取りこぼしを減らす。

### 3) paper_pipeline_v11 の改悪デフォルトを「常識的」に調整

対象: `APA/paper_pipeline_v11.py`

- `max_rot_deg` を過度な 180 → 25 に変更
- `perspective_jitter` を 0.04 → 0.03 に変更
- `min_visible_area_ratio` を 0.35 → 0.55 に変更

### 4) UVDoc(stage5) をパイプラインへ組み込み

対象: `APA/paper_pipeline_v11.py`

- UVDoc リポジトリ（`APA/third_party/UVDoc/`）を sys.path 経由で import する仕組みを追加
- `UVDocUnwrapper` を実装し、
  - 4. フォーム確定後の回転済み rectified 画像を入力
  - 5. UVDoc で unwarp
  - 6. XFeat matching は **unwarp 後の画像**で実施

### 5) form_unknown / homography_unstable 対策

対象: `APA/paper_pipeline_v11.py`

- A の取りこぼし原因が `marker_surrounding_not_blank`（照明ムラで暗くなる）だったため
  - `surround_min_mean_gray` を 190 → 175 に緩和
- `homography_unstable` の一因が `inliers<100` だったため
  - `warp.min_inliers` を 100 → 70
  - `warp.min_inlier_ratio` を 0.15 → 0.12
  - det/cond のチェックは維持（破綻ケースの抑制は継続）

## 動作確認

### 実行コマンド

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v11.py --limit 1 --degrade-n 1 --save-images fail
```

### 結果（run_20260113_162706）

- `ok_expected_behavior(user_KPI)` : **7/7 (100%)**
- `ok_warp(done_aligned_generated)` : 6/7 (85.7%)
  - C は form_unknown が期待動作なので ok_warp は FALSE

主なログ抜粋:

- `A_1_deg00` : **done**（従来 form_unknown だったが改善）
- `B_1_deg00` : done
- `C_1_deg00` : form_unknown（期待動作）
- `test_A_6_deg00` : **done**（従来 homography_unstable だったが改善）

## 今後の課題（任意）

- 速度ボトルネックは `match_s`（XFeat 全テンプレ照合）。
  - 速度が必要なら `--save-images none` を推奨。
  - さらに必要なら、A/B 内でのテンプレ候補絞り込み（ただしユーザー要望で現状は無効）を再検討。
