#+#+#+#+---------------------------------------------------------------------

# 17 20260112 paper_pipeline_v8 改善レポート（画像保存最適化 / 回転最適化 / フォームB判定改善）

## 実施日時

2026 年 1 月 12 日

## 目的

`APA/paper_pipeline_v8.py` の精度・速度改善として、ユーザー要望の以下 4 点を **順番に**反映し、実行確認まで行う。

1. **改善1（B-5）**: JPEG 書き出しを TurboJPEG（libjpeg-turbo 系）へ寄せる（保存する場合の高速化）
2. **改善2（B-4）**: 0 度 / 180 度の回転で `rotate_image_bound()`（= warpAffine）を使わない
3. **改善3（B-1）**: デバッグ画像保存を「常時」から「必要時だけ」に切り替え可能にする
4. **改善4**: フォームB（QR）向けの「向き判定」精度改善（スコア重み、enforce_landscape 重複の排除、pos_score 改良）

## 前提（環境）

- OS: Windows 11
- Python: Miniconda ローカル
  - `C:\Users\takumi\develop\miniconda3\python.exe`
- OpenCV: 4.12.0
- turbojpeg: import 可能（`python-turbojpeg`）

## 対象ファイル

- `APA/paper_pipeline_v8.py`

---

## 改善1（B-5）: JPEG保存を TurboJPEG 優先に変更

### 背景

保存（`cv2.imwrite`）は FPS を大きく下げる要因なので、**本命は保存を止めること**。
ただし「保存が必要なとき」に備えて、OpenCV より速いケースがある TurboJPEG 系を優先して使えるようにする。

### 対応

- `turbojpeg` を optional import
- `write_image(path, image_bgr)` を追加
  - `.jpg/.jpeg` の場合は TurboJPEG.encode() を優先
  - 失敗時は `cv2.imwrite` にフォールバック

---

## 改善2（B-4）: 0/180回転で rotate_image_bound を重いまま使わない

### 背景

`rotate_image_bound()` は内部で `warpAffine` を実行するため、0°/180°でもオーバーヘッドが発生する。

### 対応

`rotate_image_bound()` 冒頭に特例を追加。

- 0°: そのまま返す
- 180°: `cv2.rotate(img, cv2.ROTATE_180)`

---

## 改善3（B-1）: デバッグ画像保存を必要時だけに制御

### 背景

現状は done ケースでも毎回 5〜6 枚保存しており、IO オーバーヘッドが大きい。

### 対応

CLI 引数を追加。

- `--save-images {none,fail,all}`
  - `none`: 一切保存しない（FPS計測用）
  - `fail`: `stage!=done` のケースだけ保存
  - `all`: 従来通り常時保存

実装は `process_one_case()` 内で、

- `all`: 即時保存
- `fail`: いったんメモリ保持して、最終 `stage` 決定後に必要なら保存
- `none`: 保存しない

となるように統一。

---

## 改善4: フォームB判定（QRの向き）改善

### 変更点

1) **score_best_qr_candidate の重み修正**

- 位置（右上）を支配的に
- 面積（rel）は補助（ノイズ除外程度）へ

変更後：

- `pos_score` を右上(1,0)からの距離で計算
- `pos_score ** 2` を用いて「端に寄っている」ものをより高評価
- スコア重みを `位置 15.0 >> 面積 2.0` に変更

2) **decide_form_by_rotations のループ内で enforce_landscape を重複適用しない**

rectify 直後に `enforce_landscape()` 済みのため、スキャン中に再度回すと座標系が崩れる可能性がある。
ループ内の `enforce_landscape` を削除し、純粋に角度評価のみ実施。

---

## 実行確認

### 1) 構文チェック

```bat
C:\Users\takumi\develop\miniconda3\python.exe -m py_compile APA\paper_pipeline_v8.py
```

### 2) スモーク実行（A,B,C 各 1 枚 × degrade 1）

保存を止めた状態で確認（`--save-images none`）。

```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v8.py --src-forms A,B,C --limit 1 --degrade-n 1 --log-level INFO --console-log-level INFO --save-images none
```

実行結果（ログ抜粋）:

- A: `stage=done`, `ok_expected=TRUE`
- B: `stage=done`, `ok_expected=TRUE`
- C: `stage=form_unknown(no_detection)`, `ok_expected=TRUE`（C は reject が成功扱い）

出力ディレクトリ例:

- `APA/output_pipeline/run_20260112_093624/`

---

## まとめ

- [x] **改善1**: JPEG 保存を TurboJPEG 優先（失敗時 OpenCV フォールバック）に対応
- [x] **改善2**: 0°/180°回転で `warpAffine` を避ける（0°=そのまま、180°=cv2.rotate）
- [x] **改善3**: `--save-images {none,fail,all}` を追加し、必要時のみ保存可能に
- [x] **改善4**: フォームB（QR）向けの向き判定を、位置優先スコアへ改善 + enforce_landscape 重複適用を除去
- [x] スモーク実行で A/B/C が想定通りに完走することを確認
