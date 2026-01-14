# 24_20260114_paper_pipeline_v13_bgdiv_divide_LAB

## 概要
`paper_pipeline_v13.py` の stage6「背景除算法（Background Division Method）」を、
指定どおり **OpenCV の background division（`cv2.divide`）** を用いた実装に統一した。

目的は、UVDoc 後の画像に残る **影・照明ムラ・周辺減光** を軽減し、
後段の XFeat マッチング（stage7）の安定性を上げること。

## 変更内容

### stage6 実装の方針
- 色空間: **LAB**
- 補正対象: **L（明度）チャネルのみ**
  - a/b は保持（色味が変わりすぎないようにする）
- 背景推定: `bg = GaussianBlur(L)`
  - `sigma = clamp(max(h,w)*sigma_ratio, sigma_min, sigma_max)`
- 除算: `L_corr = cv2.divide(L, bg, scale=255)`
  - `bg` が小さいと除算が発散するため `bg_min` で下限を設定

### デフォルト設定（`PIPELINE_DEFAULTS["background_division"]`）
- `enable: True`
- `sigma_ratio: 0.02`
- `sigma_min: 15.0`
- `sigma_max: 80.0`
- `bg_min: 8.0`

## 動作確認

### 実行コマンド
```bat
C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v13.py --limit 1 --degrade-n 1 --src-forms A,C --save-images none --log-level INFO --console-log-level INFO
```

### 結果（要約）
- stage6 の時間計測が `bgdiv_s` に出力され、パイプラインは完走。
- 例: ログ上 `6_bgdiv=...` が出ていることを確認。

## メモ
- stage6 は **UVDoc の後**に適用し、その出力を stage7（XFeat matching）に入力する。
- 失敗時は処理を止めず、補正なし画像を返す（`applied=False` と reason を meta に記録）。

