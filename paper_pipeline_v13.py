#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper_pipeline_v13.py

[windows]
C:/Users/takumi/develop/miniconda3/python.exe APA/paper_pipeline_v13.py

[mac]
# リポジトリルートから実行する想定（`APA/` 配下のスクリプトを直接指定）
.venv/bin/python paper_pipeline_v13.py

目的
----
既存の検証コード（DocAligner / フォームA・B判定 / XFeat Homography）をベースに、
静止画像の一括処理パイプラインとして統合・運用しやすくする。

特に以下を重視：

- 解像度差に強い処理（polygon margin の比率化）
- 検出率向上（マーカー/QR の前処理オプション）
- 改悪の現実寄せ（紙がしなっているような歪み / 撮影時の影の混入）
- 高速化（テンプレ特徴キャッシュ。※グローバル特徴での候補絞り込みは互換用だが現在は無効）
- 安定性（Unknown 判定、逆ホモグラフィの信頼度チェック）

パイプライン概要
----------------
入力:

- 改悪元画像: `APA/image/{A,B,C}/` 配下（デフォルト実装では `1.jpg`〜`6.jpg` を対象）
  - 対象フォームは `--src-forms` で指定

処理フロー（1 case = 1 枚の入力から生成した 1 枚の改悪画像）:

※ v13.7: 改悪生成（degrade）は **最初に全ケース分をまとめて生成**し、以降の本処理へ投入する。
   改悪生成の所要時間は計測対象外。

1) 改悪生成（`APA/test_recovery_paper.py` の実装を流用）
   - v13.5: 紙のしなり（非線形ワープ）と、撮影時の影（照明ムラ）を追加
2) DocAligner により紙領域 polygon（4点）を推定
   - 失敗したら `stage=docaligner_failed` で終了
3) polygon を（紙サイズ比の margin で）外側に拡張 → 透視補正（rectify）
   - 透視補正後の画像は横長に統一（`enforce_landscape`）
   - `--polygon-margin-px > 0` の場合は固定pxマージンで上書き可能
4) フォーム判定（回転探索）
   - rectify 後は `enforce_landscape` で横長に統一しているため、回転探索は **2方向（0度/180度）** のみ
     - 0 と 180 を比較し、最上位角度を確定に使う
     - 0/180 で何も見つからない場合は Unknown（no_detection）とする（救済処置は行わない）
   - フォームA: 3点マーク（TL/TR/BL）が検出できる（`--marker-preproc` で前処理オプション）
     - v13 では **フォームAスコアが十分高い場合**、速度のため **フォームB判定をスキップ**する
       - 既定の閾値は `PIPELINE_DEFAULTS["formA_strong"]["score_threshold"]`（現状 4.0）
     - v13.6: A が検出できても **Unknown閾値未満** の場合は、その時点で Unknown 確定せず
       **フォームB探索へフォールバック**する（B の取りこぼし回避）
   - フォームB: QRコードが検出できる
     - **WeChat QR エンジンのみ** を使用（OpenCV 標準 `QRCodeDetector` は使わない）
     - `--wechat-model-dir` にモデルが必要（opencv-contrib 必須）
     - v13 では WeChat-only でも v7 と同様に **fast → robust の2段階**で評価する
       - scan中は fast（軽量）で角度候補を評価
       - B が勝った場合のみ、同一角度で robust を **最大1回** 実行して確定
   - 判定不能/曖昧なら `stage=form_unknown`（Unknown）で終了
5) UVDoc による成形（しわ/湾曲の補正）
   - https://github.com/tanguymagne/UVDoc
   - 4) で確定した回転後画像を UVDoc で unwarp し、より平坦な紙画像を得る
6) 背景除算法（Background Division Method）による照明ムラ/影の除去（新規）
   - OpenCV の background division（divide）を用いて、影・周辺減光などを軽減する
   - 目的: 書類の白地ができるだけ均一な白に近づくよう補正し、後段の特徴点マッチングを安定化させる
7) XFeat matching によるテンプレ照合
   - テンプレは `APA/image/A` または `APA/image/B`（`1.jpg`〜`6.jpg`）
   - フォームAなら `APA/image/A` の全テンプレ、フォームBなら `APA/image/B` の全テンプレに対して局所特徴（XFeat）で照合する。
8) Homography を信頼度チェックの上で逆行列化し、テンプレ座標へ warp
   - 不安定なら `stage=homography_unstable` で終了

出力
----
`APA/output_pipeline/run_YYYYmmdd_HHMMSS/` 配下に（処理順が分かるように番号付き）：

- `1_degraded/`       : 改悪画像
- `2_doc/`            : DocAligner polygon 可視化
- `3_rectified/`      : 透視補正した紙画像
- `4_rectified_rot/`  : フォーム確定に使った回転後画像（根拠も描画）
- `5_uvdoc_unwarp/`   : UVDoc による成形（unwarp）後の紙画像
- `6_bgdiv/`          : 背景除算法（Background Division）後の画像
- `7_debug_matches/`  : best template のマッチ可視化
- `8_aligned/`        : best template にワープした結果
- `summary.json` / `summary.csv`
- `run.log`           : 実行ログ（logging）

※v13 ではデバッグ画像の保存量を `--save-images {all,fail,none}` で制御できる。

- `all` : 従来通り、常に保存
- `fail`: `stage!=done` のケースだけ保存（成功ケースは保存しない）
- `none`: 一切保存しない（速度計測向け）

ディレクトリ自体は作られるが、`fail/none` の場合は中身が空になることがある。

※ v13.7: `--save-images` の設定に関わらず、`7_aligned/` は成果物として必ず保存される（※v13.8 で `8_aligned/` に移動）。
※ v13.8: 背景除算法（stage6）を追加したため、成果物は `8_aligned/` に移動。
   `--save-images` の設定に関わらず、`8_aligned/` は成果物として必ず保存される。

注意
----
- torch.hub 経由の XFeat 読み込みで git が必要になることがあるため、
  portable git を PATH に追加する処理を `test_recovery_paper` から流用する。
- QR 検出は WeChat QR エンジン（`cv2.wechat_qrcode_WeChatQRCode`）のみ利用する。
  - WeChat を使うには opencv-contrib のビルドと、4つのモデルファイル
    （detect/sr の prototxt/caffemodel）が必要
  - **src-forms に B を含む場合、WeChat が利用できないと起動時にエラー終了**する
- JPEG 保存は可能なら python-turbojpeg（libjpeg-turbo）を優先する（失敗時は `cv2.imwrite` にフォールバック）
- 日本語ラベル描画は Pillow を使用（OpenCV putText は日本語非対応のため）。
  - `APA_FONT_PATH` を設定すると任意フォントを優先可能

改善点メモ
----

v13.8（本タスク）
----------------
- 改悪生成（degrade）は **最初に全ケース分をまとめて生成**し、その所要時間は計測対象外とする。
  - 生成した改悪画像を 1 枚ずつパイプライン本体へ投入する。
- 時間計測（case_total / stage_time / run_elapsed）は「本処理」のみを対象とし、
  以下は **計測対象外** とする。
  - 途中画像保存（1_degraded/〜7_debug_matches/）
  - `7_debug_matches/` のマッチ可視化生成
  - `summary.json` / `summary.csv` の書き出し
- 計測に含める画像保存は **`8_aligned/` の保存だけ** とする。
v13.8.1
------
- v13.8 の stage6（背景除算法）を **test dataset（image/test）側の処理にも適用**する。

- stage6 の出力解像度（bgdiv_w/bgdiv_h）も item に保存して CSV に反映する。
 
- 追加: UVDoc の後に **背景除算法（Background Division Method）** を stage6 として挿入する。

"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import json
import os
import platform
import queue
import random
import sys
import threading
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

try:
    # python-turbojpeg
    from turbojpeg import TJPF_BGR, TurboJPEG

    _TURBOJPEG_IMPORT_OK = True
except Exception:
    TJPF_BGR = None  # type: ignore
    TurboJPEG = None  # type: ignore
    _TURBOJPEG_IMPORT_OK = False

from PIL import Image, ImageDraw, ImageFont


# ------------------------------------------------------------
# UVDoc（Neural Grid-based Document Unwarping）
# ------------------------------------------------------------


_UVDOCDIR = Path(__file__).resolve().parent / "third_party" / "UVDoc"


def _try_import_uvdoc(logger: Optional[logging.Logger] = None) -> tuple[bool, dict[str, Any]]:
    """UVDoc のモジュール群を import 可能にする。

    NOTE:
      UVDoc リポジトリはパッケージ化されていない（__init__.py が無い）ため、
      sys.path にディレクトリを差し込んで import する。
      `utils.py` / `model.py` のような一般名モジュールが sys.modules に入る点には注意。

    戻り値:
      (ok, meta)
    """

    meta: dict[str, Any] = {
        "uvdoc_dir": str(_UVDOCDIR),
        "exists": bool(_UVDOCDIR.exists()),
    }
    if not _UVDOCDIR.exists():
        if logger:
            logger.error("UVDoc repo not found: %s", _UVDOCDIR)
        return False, meta

    try:
        sys.path.insert(0, str(_UVDOCDIR))
        import utils as uvdoc_utils  # type: ignore
        import model as uvdoc_model  # type: ignore

        meta["IMG_SIZE"] = getattr(uvdoc_utils, "IMG_SIZE", None)
        meta["GRID_SIZE"] = getattr(uvdoc_utils, "GRID_SIZE", None)
        meta["has_bilinear_unwarping"] = hasattr(uvdoc_utils, "bilinear_unwarping")
        meta["has_UVDocnet"] = hasattr(uvdoc_model, "UVDocnet")
        return True, meta
    except Exception as e:
        meta["error"] = str(e)
        if logger:
            logger.error("UVDoc import failed: %s", e)
        return False, meta
    finally:
        # 先頭に挿入したパスだけ除去
        try:
            if sys.path and str(sys.path[0]) == str(_UVDOCDIR):
                sys.path.pop(0)
        except Exception:
            pass


class UVDocUnwrapper:
    """UVDoc のモデルを使って、紙の湾曲を補正（unwarp）するラッパー。"""

    def __init__(self, *, ckpt_path: Path, device: str, logger: Optional[logging.Logger] = None):
        ok, meta = _try_import_uvdoc(logger=logger)
        if not ok:
            raise RuntimeError(f"UVDoc import failed: {meta}")

        sys.path.insert(0, str(_UVDOCDIR))
        try:
            import utils as uvdoc_utils  # type: ignore
            import model as uvdoc_model  # type: ignore

            self.uvdoc_utils = uvdoc_utils
            self.uvdoc_model = uvdoc_model
            self.img_size = tuple(int(x) for x in getattr(uvdoc_utils, "IMG_SIZE"))
            self.device = torch.device(device)

            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"UVDoc checkpoint not found: {ckpt_path}")

            # UVDoc は dict{'model_state'} を読む。ckpt が CUDA で保存されているケースがあるため map_location 必須。
            ckpt = torch.load(str(ckpt_path), map_location=self.device)
            if not isinstance(ckpt, dict) or "model_state" not in ckpt:
                raise RuntimeError(f"Unexpected UVDoc checkpoint format: type={type(ckpt)} keys={getattr(ckpt, 'keys', lambda: [])()}")

            model = uvdoc_model.UVDocnet(num_filter=32, kernel_size=5)
            model.load_state_dict(ckpt["model_state"])
            model.to(self.device)
            model.eval()
            self.model = model
        finally:
            try:
                if sys.path and str(sys.path[0]) == str(_UVDOCDIR):
                    sys.path.pop(0)
            except Exception:
                pass

    @torch.no_grad()
    def unwarp_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        """入力(BGR, uint8)を UVDoc で unwarp して(BGR, uint8)で返す。"""

        if image_bgr is None:
            raise ValueError("image_bgr is None")

        # UVDoc demo.py に合わせて RGB / 0..1 float で推論する
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 入力を IMG_SIZE にリサイズして推論
        inp = cv2.resize(img_rgb, self.img_size).transpose(2, 0, 1)
        inp_t = torch.from_numpy(inp).unsqueeze(0).to(self.device)

        point_positions2D, _ = self.model(inp_t)

        # unwarp は「元解像度」に戻す（demo と同じ）
        out_w = int(img_rgb.shape[1])
        out_h = int(img_rgb.shape[0])
        warped_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        unwarped = self.uvdoc_utils.bilinear_unwarping(
            warped_img=warped_t,
            point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
            img_size=(out_w, out_h),
        )
        out_rgb = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


# ============================================================
# 変更しやすい設定値（デフォルトのハイパーパラメータ集約）
# ============================================================
#
# 方針:
# - argparse の default 値や、コード中のハードコード値（しきい値/スケール/角度など）をここに集約する。
# - まとまりのあるものは配列（list/tuple）や辞書にまとめ、上部だけ見れば調整できるようにする。
# - ここを変えると「CLI引数のデフォルト」も同時に変わる（引数で上書きも可能）。


PIPELINE_DEFAULTS: dict[str, Any] = {
    # 入力
    "src_forms": ["A", "B", "C"],  # 入力元フォーム（カンマ区切りで指定される想定）
    "limit": 0,  # デバッグ用：各フォームで先頭N枚だけ処理（0=全て）
    "template_numbers": [1, 2, 3, 4, 5, 6],  # テンプレ/入力画像の対象番号（例: 1.jpg〜6.jpg）

    # 改悪生成（degrade）
    "degrade": {
        "n": 5,  # 1枚の入力から何枚の改悪画像を作るか
        "out_size_wh": [2400, 1800],  # 改悪画像の出力サイズ（幅, 高さ）
        # ユーザー要望: 過度な改悪（極端な傾き/奥行き/縮小）が出ないよう、デフォルトを「常識的」にする。
        # 必要なら CLI 引数で上げられる。
        "max_rot_deg": 25.0,  # 改悪生成の回転強度（度）
        "min_abs_rot_deg": 0.0,  # 最小回転量（0なら小さな回転も許可）
        "rotation_mode": "uniform",  # 回転角の出し方（"uniform" または "snap"）
        "snap_step_deg": 90.0,  # rotation_mode="snap" の場合の角度刻み
        # 歪みが強すぎるとのフィードバックがあったため弱めに調整
        "perspective_jitter": 0.03,  # 射影ゆがみ量（大きいほど難しい）
        "min_visible_area_ratio": 0.55,  # 生成画像でテンプレが見えている最小比率（小さすぎ防止）
        "max_attempts": 50,  # 改悪生成の最大試行回数
        "seed": 45,  # 乱数シード（再現性）

        # v13.5: 紙がしなっているような歪み（非線形ワープ）
        # 目的:
        #   - 撮影で起きる「紙のたわみ」により、単純な射影変換だけでは表現できない歪みを追加する。
        #   - ただし難しすぎると全滅するため、弱め〜中程度をデフォルトとする。
        "bend": {
            "enable": True,
            "prob": 0.60,  # この確率でしなりを入れる（0..1）
            "amplitude_ratio": 0.008,  # 画面短辺に対する振幅比（例: 0.008 -> 1800pxなら約14px）
            "amplitude_min_px": 3.0,
            "amplitude_max_px": 22.0,
            "freq_choices": [1, 2],  # 波の周期数
        },

        # v13.5: 撮影時の影（照明ムラ）の混入
        # 目的:
        #   - 斜め方向の影/周辺減光を軽く入れて、現実の撮影に近づける。
        "shadow": {
            "enable": True,
            "prob": 0.90,  # この確率で影を入れる（0..1）
            # 影の強さ（0..1）: 強すぎると「汚れ」に見えるので弱めに調整
            "strength_min": 0.10,
            "strength_max": 0.30,
            # 光が当たっている感じ（明るい側）
            "highlight_min": 0.06,
            "highlight_max": 0.16,
            # 影マスク平滑化 sigma（大きいほど滑らか）
            "blur_sigma_min": 25.0,
            "blur_sigma_max": 80.0,
            # 周辺減光の強さ（0..1）
            "vignette_strength": 0.25,
            # bend と連動した薄い帯（折れ陰影）
            "band_strength": 0.05,
            # 紙境界を馴染ませるためのソフトマスクぼかし
            "edge_blur_sigma": 6.0,
        },

        # NOTE:
        # `warp_template_to_random_view()` 側でも軽い blur/noise を入れている。
        # v13.5 では「紙のしなり」「影」を主目的として追加し、
        # post の追加劣化は行わない（過度な難化・二重劣化を避ける）。
    },

    # WeChat QRモデル
    "wechat": {
        "model_dir": str(Path(__file__).resolve().parent / "models" / "wechat_qrcode"),  # WeChat QRモデル配置ディレクトリ
    },

    # UVDoc（成形 / unwarp）
    "uvdoc": {
        "repo_dir": str(_UVDOCDIR),
        "ckpt_path": str(_UVDOCDIR / "model" / "best_model.pkl"),
    },

    # 背景除算法（Background Division Method）
    # 目的:
    #   - 影/照明ムラ/周辺減光を軽減し、紙の白地をできるだけ均一にする
    #   - XFeat のマッチングを安定化させる
    # 実装:
    #   - LAB 色空間の L（明度）に対して大きめの GaussianBlur をかけて背景を推定
    #   - cv2.divide(L, bg, scale=255) で背景除算
    #   - a/b（色成分）は保持し、L のみを補正して BGR に戻す
    "background_division": {
        "enable": True,
        # sigma = max(h,w) * sigma_ratio
        "sigma_ratio": 0.02,
        "sigma_min": 15.0,
        "sigma_max": 80.0,
        # bg が極端に小さいと divide が発散するため、下限を設ける
        "bg_min": 8.0,
    },

    # XFeat（テンプレマッチング）
    "xfeat": {
        "device_default": "cpu",  # 既定の実行デバイス（auto/cpu/cuda のうち default に使う）
        # 高精度優先（時間はかかってよい想定）
        # - top_k を増やすと対応点候補が増え、Homography の安定性が上がりやすい
        # - match_max_side_px を増やすと細部が残り、マッチング精度が上がりやすい
        #   （ただしメモリ/時間コストが増える）
        "top_k": 1024,
        "match_max_side_px": 1024,
    },

    # フォーム判定（回転スキャン）
    "rotation_scan": {
        "max_workers": 8,  # 回転スキャンの並列数（スレッド）
        # v13.2 改善: rectify 後は enforce_landscape で横長に統一されているため、
        # 追加で見るべきは「上下反転（180度）」のみ。
        "scan_angles_2_deg": [0.0, 180.0],
    },

    # フォームAが「十分強い」場合の枝刈り（B判定をスキップ）
    # NOTE:
    # score_formA は概ね 0..(base_score~3 + pos_score*2~2) 程度なので、
    # 4.0 前後を "ほぼ間違いなくA" の目安として使う。
    "formA_strong": {
        "score_threshold": 4.0,
    },

    # DocAligner（紙領域検出）
    "docaligner": {
        "model": "fastvit_sa24",  # DocAlignerのモデル名
        "type": "heatmap",  # 推論タイプ（"point" / "heatmap"）
        # 透視補正後の紙画像が小さすぎると、マーカー/QRの判定や UVDoc の精度が落ちる。
        "rectified_max_side_px": 2048,
        "pad_px": 100,  # DocAligner入力前に周囲へ足すパディング(px)
        "polygon_margin": {
            "ratio": 0.1,  # polygonを外側に広げる比率（紙の長辺に対する割合）
            "min_px": 10.0,  # ratio計算の下限(px)
            "max_px": 200.0,  # ratio計算の上限(px)（0以下なら無制限）
            "fixed_px": 0.0,  # 固定pxマージン（>0の場合 ratio を上書き）
        },
    },

    # マーカー検出（フォームA）向け前処理
    "marker": {
        # 高精度優先: morph は多少重いが、照明ムラ/影/ノイズに対して頑健になりやすい
        "preproc_mode": "morph",  # マーカー検出前処理の強さ（"none" / "basic" / "morph"）
        "clahe": {"clipLimit": 2.0, "tileGridSize": [8, 8]},  # CLAHE設定（照明ムラ対策）
        "adaptive_threshold": {"block_size": 51, "C": 5},  # 自適応二値化の設定
        "morph": {
            # 画像短辺に対する比率でカーネルサイズを決める
            "kernel_ratio": 0.004,  # カーネルサイズ = 短辺 * 比率（概算）
            "kernel_min": 3,  # カーネルサイズの最小値
        },
    },

    # フォームA判定の追加制約（C->A誤判定の抑制）
    "formA": {
        "geometry": {
            # 既存: 面積/アスペクト等の幾何制約
            "max_marker_area_ratio": 3.0,  # max(area)/min(area) が大きすぎるケースを除外
            "min_marker_area_page_ratio": 5e-5,  # マーカーが小さすぎる場合を除外（ノイズ対策）
            "max_marker_area_page_ratio": 5e-3,  # マーカーが大きすぎる場合を除外（誤検出対策）
            "max_dist_ratio_relative_error": 0.35,  # 三角形の距離比がページ比率から外れすぎる場合を除外

            # 追加: マーカー周辺が「ほぼ白地」であること
            # 目的: フォームCの文字（例: 「記」など四角っぽい漢字）がマーカー誤検出になるのを抑える。
            # 考え方:
            # - 正しいフォームAのマーカー周辺は、ほぼ何もなく真っ白に近い
            # - マーカー周辺に文字/線（=濃い画素）が多い場合はフォームAではない可能性が高い
            "surround_pad_ratio": 2.0,  # bbox外側に見る幅 = max(w,h)*ratio（周辺の評価範囲）
            "surround_pad_px_min": 8,  # 周辺評価の最小パディング(px)
            "surround_pad_px_max": 120,  # 周辺評価の最大パディング(px)
            # NOTE:
            # 2026/01/09: A 正解なのに `marker_surrounding_not_blank` で弾かれるケースが発生したため、
            # 誤検出抑制は維持しつつ「A の取りこぼし」を減らす方向で閾値を少し緩める。
            # v13.5 で shadow(照明ムラ) を入れると、
            # 正しいフォームAでも右上マーカー周辺が 180前後まで暗くなるケースがある。
            # C->A 誤判定抑制を維持しつつ取りこぼしを減らすため、少し緩める。
            "surround_min_mean_gray": 175.0,  # 周辺領域の平均輝度がこの値未満なら「汚れている」とみなす
            "surround_max_ink_ratio": 0.05,  # 周辺領域の「非白（インク）」比率の上限
            "surround_adaptive_block_size": 41,  # 周辺領域のインク抽出（二値化）のblock size（奇数）
            # THRESH_BINARY_INV の場合、C を大きくすると閾値が下がり「インク扱い」が減る傾向があるため、
            # 取りこぼしを減らす方向で C を少し増やす。
            "surround_adaptive_C": 9,
        }
    },

    # QR 検出（フォームB）向け設定
    "qr": {
        "min_test_side_px": 120,  # QR検出で試す画像サイズの最小辺(px)
        "wechat": {
            # v13.2 改善:
            # v7並みの精度を確保するため、前処理バリエーションを増やす。
            # fast→robust の2段階で評価する。
            "fast": {
                # fast: 角度候補の絞り込み用
                # v13.2: 前処理を増やして精度向上
                # 高精度優先: 角度選択で取りこぼすと復帰できないため、fast 側も少し厚めにする。
                "variants": ["bgr", "gray"],
                "scales": [1.0],
            },
            "robust": {
                # robust: 最終確定用（多少重くてもよいが、呼ぶのは最大1回）
                # v13.2: v7並みの前処理バリエーションに拡張
                # 注意:
                # 以前の実装では variants に "adaptive_threshold" を書いていたが、
                # 実装側が未対応で無視されるケースがあった。
                # 本ファイルでは "adaptive_threshold" を明示的にサポート（下の関数修正）する。
                "variants": ["bgr", "gray", "clahe", "adaptive_threshold", "adaptive_morph"],
                # 高精度優先: スケール探索を厚めにする（時間は増える）
                "scales": [0.5,1.5],
            },
            "up_scale_enable_max_side_px": 1200,  # 最大辺がこの値以上なら拡大は無効化
            "max_test_side_px": 6500,  # WeChat で試す画像の最大辺(px)
            "adaptive_morph_kernel": [5, 5],
        },
        # CLAHE設定（照明ムラ対策）
        "clahe": {"clipLimit": 2.0, "tileGridSize": [8, 8]},
        # 自適応二値化の設定
        "adaptive_threshold": {"block_size": 51, "C": 5},
    },

    # Homography（特徴点マッチングの射影変換）
    "homography": {
        "find": {
            # 高精度優先:
            # - iters/confidence を上げて収束率を上げる（時間は増える）
            # - reproj threshold は僅かに厳しめにして外れ値混入を抑える
            "ransac_reproj_threshold_px": 5.0,
            "max_iters": 1000,
            "confidence": 0.995,
        },
        "invert": {
            "det_abs_min": 1e-12,  # 逆行列化を許可する最小 |det|（小さいと不安定）
        },
    },

    # 可視化（デバッグ画像）
    "visual": {
        "polygon_line_thickness": 4,  # polygon枠線の太さ
        "polygon_point_radius": 10,  # 角点の半径
        "polygon_label_font_scale": 1.0,  # 角ラベル（TL/TR...）のフォント倍率
        "polygon_label_thickness": 2,  # 角ラベルの太さ
    },

    # 画像保存（デバッグ用）
    # NOTE: FPS計測や実運用では IO がボトルネックになるため、基本は none 推奨。
    "save_images": {
        "mode": "all",  # all/none/fail
        "jpeg_quality": 95,
    },

    # Unknown 判定（フォームA/Bのどちらでもない扱い）
    "unknown": {
        "score_threshold": 1.2,  # 最大スコアがこの値未満なら Unknown 扱い
        "margin": 0.15,  # A/Bスコア差がこの値未満なら Unknown 扱い（曖昧）
    },

    # warp 許可条件（テンプレ座標へのワープを行うための条件）
    "warp": {
        # 取りこぼし（特に test_A_6 など）を減らすため、既定を少し緩める。
        # ただし、cond/det チェックは残るため破綻ケースは弾かれる。
        "min_inliers": 70,  # warpを許可する最小inlier数
        "min_inlier_ratio": 0.07,  # warpを許可する最小inlier_ratio
        "max_h_cond": 1e6,  # Homographyの条件数上限（大きいと不安定）
    },
}


# ------------------------------------------------------------
# 画像保存（JPEGは libjpeg-turbo / python-turbojpeg を優先）
# ------------------------------------------------------------


_TURBOJPEG: Optional[Any] = None


def _get_turbojpeg() -> Optional[Any]:
    """TurboJPEG インスタンスを lazy 初期化する。"""

    global _TURBOJPEG
    if _TURBOJPEG is not None:
        return _TURBOJPEG
    if not _TURBOJPEG_IMPORT_OK or TurboJPEG is None:
        return None
    try:
        _TURBOJPEG = TurboJPEG()
        return _TURBOJPEG
    except Exception:
        _TURBOJPEG = None
        return None


def write_image(
    path: Path,
    image_bgr: np.ndarray,
    *,
    jpeg_quality: int = 95,
) -> bool:
    """画像保存。

    - JPEG は可能なら TurboJPEG.encode() を使う（高速なケースがある）
    - 失敗したら cv2.imwrite にフォールバック

    NOTE:
      速度最適化の本命は「保存しない」こと。
      本関数は "保存が必要なとき" のオーバーヘッド低減用。
    """

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    ext = str(path.suffix).lower()
    if ext in (".jpg", ".jpeg"):
        tj = _get_turbojpeg()
        if tj is not None and TJPF_BGR is not None:
            try:
                buf = tj.encode(
                    image_bgr,
                    quality=int(jpeg_quality),
                    pixel_format=TJPF_BGR,
                )
                with open(path, "wb") as f:
                    f.write(buf)
                return True
            except Exception:
                # fall back
                pass
        try:
            return bool(
                cv2.imwrite(
                    str(path),
                    image_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
            )
        except Exception:
            return False

    # other formats
    try:
        return bool(cv2.imwrite(str(path), image_bgr))
    except Exception:
        return False


# ------------------------------------------------------------
# 背景除算法（Background Division Method）
# ------------------------------------------------------------


def apply_background_division(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """背景除算法（Background Division Method）で照明ムラ/影を軽減する。

    実装:
      - LAB に変換し、L（明度）だけを補正する
      - L の大きいガウシアンぼかしで背景（低周波）を推定
      - cv2.divide(L, bg, scale=255) で背景除算

    期待効果:
      - 紙の白地が均一になりやすい
      - 後段の XFeat マッチングが安定しやすい
    """

    cfg = dict((PIPELINE_DEFAULTS.get("background_division") or {}))
    if not bool(cfg.get("enable", True)):
        return image_bgr, {"applied": False, "reason": "disabled"}

    if image_bgr is None:
        return image_bgr, {"applied": False, "reason": "image_is_none"}

    h, w = image_bgr.shape[:2]
    if h < 16 or w < 16:
        return image_bgr, {"applied": False, "reason": "too_small", "h": h, "w": w}

    sigma_ratio = float(cfg.get("sigma_ratio", 0.02) or 0.02)
    sigma_min = float(cfg.get("sigma_min", 15.0) or 15.0)
    sigma_max = float(cfg.get("sigma_max", 80.0) or 80.0)
    bg_min = float(cfg.get("bg_min", 8.0) or 8.0)

    sigma = float(max(h, w)) * sigma_ratio
    sigma = _clamp(sigma, sigma_min, sigma_max)

    try:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        # 背景（低周波）推定
        bg = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma, sigmaY=sigma)
        bg = np.maximum(bg.astype(np.float32), bg_min)

        # 背景除算（白地を均一化）
        L_corr = cv2.divide(L.astype(np.float32), bg, scale=255.0)
        L_corr = np.clip(L_corr, 0, 255).astype(np.uint8)

        lab2 = cv2.merge([L_corr, A, B])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out, {"applied": True, "sigma": float(sigma), "bg_min": float(bg_min)}
    except Exception as e:
        # 失敗しても処理は継続（補正なしで返す）
        return image_bgr, {"applied": False, "reason": f"exception:{e}"}


# ------------------------------------------------------------
# 追加の改悪（v13.5）: 紙のしなり / 影（照明ムラ）
# ------------------------------------------------------------


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def maybe_apply_bend(
    image_bgr: np.ndarray,
    rng: random.Random,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """紙のしなり（非線形ワープ）を軽く入れる。

    実装方針:
      - 画像全体を「サイン波の変位場」で remap する
      - 射影変換では表現できない「たわみ」を擬似的に作る

    注意:
      - ここは *改悪生成専用* のため、物理的正確さより「それっぽさ」を優先
      - 破綻しやすいので振幅を小さめに制限
    """

    if image_bgr is None:
        return image_bgr, {"applied": False, "reason": "image_is_none"}

    enable = bool(cfg.get("enable", True))
    prob = float(cfg.get("prob", 1.0))
    if (not enable) or (prob <= 0.0) or (rng.random() > prob):
        return image_bgr, {"applied": False, "enable": enable, "prob": prob}

    h, w = image_bgr.shape[:2]
    if h < 16 or w < 16:
        return image_bgr, {"applied": False, "reason": "too_small", "h": h, "w": w}

    amp_ratio = float(cfg.get("amplitude_ratio", 0.0) or 0.0)
    amp_min = float(cfg.get("amplitude_min_px", 0.0) or 0.0)
    amp_max = float(cfg.get("amplitude_max_px", 1e9) or 1e9)
    amp = float(min(h, w)) * amp_ratio
    amp = _clamp(amp, amp_min, amp_max)

    freq_choices = cfg.get("freq_choices") or [1, 2]
    try:
        freq_choices = [int(x) for x in list(freq_choices) if int(x) >= 1]
    except Exception:
        freq_choices = [1, 2]
    freq = int(rng.choice(freq_choices)) if freq_choices else 1

    # 曲げ方向をランダムに選ぶ
    mode = str(cfg.get("mode") or "auto")
    if mode == "auto":
        mode = rng.choice(["x_to_y", "y_to_x"])  # xに応じてyが揺れる / yに応じてxが揺れる
    if mode not in ("x_to_y", "y_to_x"):
        mode = "x_to_y"

    phase = rng.uniform(0.0, 2.0 * math.pi)

    # remap 用の座標場
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(xs, ys)

    if mode == "x_to_y":
        # x方向にサイン波 → yに変位
        disp = amp * np.sin((2.0 * math.pi * float(freq) * map_x / float(max(1, w))) + phase)
        map_y = map_y + disp.astype(np.float32)
    else:
        # y方向にサイン波 → xに変位
        disp = amp * np.sin((2.0 * math.pi * float(freq) * map_y / float(max(1, h))) + phase)
        map_x = map_x + disp.astype(np.float32)

    warped = cv2.remap(
        image_bgr,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    meta = {
        "applied": True,
        "prob": prob,
        "mode": mode,
        "amplitude_px": float(amp),
        "freq": int(freq),
        "phase_rad": float(phase),
    }
    return warped, meta


def maybe_apply_bend_with_mask(
    image_bgr: np.ndarray,
    mask_u8: Optional[np.ndarray],
    rng: random.Random,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, Any]]:
    """bend を画像とマスクに同一の変位場で適用する。

    目的:
      - bend（非線形歪み）と、紙領域マスクの整合性を保つ
      - 後段の shadow/light を "紙の上だけ" に適用しやすくする
    """

    img2, meta = maybe_apply_bend(image_bgr, rng=rng, cfg=cfg)
    if not meta.get("applied"):
        return img2, mask_u8, meta

    if mask_u8 is None:
        return img2, None, meta

    # maybe_apply_bend と同じ変位場を再現する必要があるため、
    # meta のパラメータから map を再生成して remap する。
    try:
        h, w = image_bgr.shape[:2]
        mode = str(meta.get("mode") or "x_to_y")
        amp = float(meta.get("amplitude_px") or 0.0)
        freq = int(meta.get("freq") or 1)
        phase = float(meta.get("phase_rad") or 0.0)

        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        map_x, map_y = np.meshgrid(xs, ys)

        if mode == "x_to_y":
            disp = amp * np.sin((2.0 * math.pi * float(freq) * map_x / float(max(1, w))) + phase)
            map_y = map_y + disp.astype(np.float32)
        else:
            disp = amp * np.sin((2.0 * math.pi * float(freq) * map_y / float(max(1, h))) + phase)
            map_x = map_x + disp.astype(np.float32)

        m = mask_u8
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        m2 = cv2.remap(
            m,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        m2 = np.clip(m2, 0, 255).astype(np.uint8)
        return img2, m2, meta
    except Exception as e:
        meta["mask_warp_error"] = str(e)
        return img2, mask_u8, meta


def maybe_apply_shadow(
    image_bgr: np.ndarray,
    rng: random.Random,
    cfg: dict[str, Any],
    *,
    paper_mask_u8: Optional[np.ndarray] = None,
    bend_meta: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """影/光（紙に対して自然なライティング）を入れる。

    改善点（今回のユーザー指摘に対応）:
      - 画像全体ではなく「紙領域だけ」に適用する
      - 境界はソフトマスク（ぼかし）で馴染ませる
      - 影だけでなくハイライト（明るい側）も入れて“光が当たっている”感じを作る
      - bend のパラメータと弱く連動した「帯状の陰影」を薄く入れる（折れ/たわみの表現）
    """

    if image_bgr is None:
        return image_bgr, {"applied": False, "reason": "image_is_none"}

    enable = bool(cfg.get("enable", True))
    prob = float(cfg.get("prob", 1.0))
    if (not enable) or (prob <= 0.0) or (rng.random() > prob):
        return image_bgr, {"applied": False, "enable": enable, "prob": prob}

    h, w = image_bgr.shape[:2]
    if h < 16 or w < 16:
        return image_bgr, {"applied": False, "reason": "too_small", "h": h, "w": w}

    strength_min = float(cfg.get("strength_min", 0.0) or 0.0)
    strength_max = float(cfg.get("strength_max", 0.0) or 0.0)
    strength = rng.uniform(strength_min, strength_max) if strength_max >= strength_min else float(strength_min)

    # ハイライト（明るい側）の強さ（0..1）
    highlight_min = float(cfg.get("highlight_min", 0.0) or 0.0)
    highlight_max = float(cfg.get("highlight_max", 0.0) or 0.0)
    highlight = rng.uniform(highlight_min, highlight_max) if highlight_max >= highlight_min else float(highlight_min)

    vignette_strength = float(cfg.get("vignette_strength", 0.0) or 0.0)

    blur_sigma_min = float(cfg.get("blur_sigma_min", 0.0) or 0.0)
    blur_sigma_max = float(cfg.get("blur_sigma_max", 0.0) or 0.0)
    blur_sigma = rng.uniform(blur_sigma_min, blur_sigma_max) if blur_sigma_max >= blur_sigma_min else float(blur_sigma_min)

    # 斜め影の方向
    angle_deg = rng.uniform(0.0, 360.0)
    ang = math.radians(angle_deg)
    dx = math.cos(ang)
    dy = math.sin(ang)

    # 正規化座標 [-1..1]
    xs = (np.linspace(-1.0, 1.0, w, dtype=np.float32))[None, :]
    ys = (np.linspace(-1.0, 1.0, h, dtype=np.float32))[:, None]

    # 影/光グラデ（0..1）: 方向ベクトルに投影して正規化
    t = (dx * xs) + (dy * ys)  # [-sqrt(2)..sqrt(2)]
    t01 = (t - float(t.min())) / float(max(1e-6, (t.max() - t.min())))

    # 暗い側（shadow）と明るい側（highlight）を同時に作る
    # - t01 が 1 に近い側を暗く、0 に近い側を明るく
    light = 1.0 - (float(strength) * t01) + (float(highlight) * (1.0 - t01))

    # 周辺減光（中心=1、端=1-v）
    if vignette_strength > 0:
        rr = np.sqrt(xs**2 + ys**2) / 1.41421356  # 0..1
        vig = 1.0 - float(vignette_strength) * (rr**2)
        light = light * vig

    # bend と弱く連動する帯（折れ陰影）: sin/cos を使って薄い縞を入れる
    band_strength = float(cfg.get("band_strength", 0.0) or 0.0)
    if band_strength > 0 and bend_meta and bool(bend_meta.get("applied")):
        try:
            mode_b = str(bend_meta.get("mode") or "x_to_y")
            freq_b = int(bend_meta.get("freq") or 1)
            phase_b = float(bend_meta.get("phase_rad") or 0.0)
            if mode_b == "x_to_y":
                # x方向に帯
                band = np.cos((2.0 * math.pi * float(freq_b) * xs) + phase_b)
                band = np.repeat(band, h, axis=0)
            else:
                band = np.cos((2.0 * math.pi * float(freq_b) * ys) + phase_b)
                band = np.repeat(band, w, axis=1)
            # cos in [-1..1] -> [0.8..1.2] くらいの弱い変調
            light = light * (1.0 + float(band_strength) * band)
        except Exception:
            pass

    # 強い影/ハイライトを許容しつつ、破綻を避けるための安全クリップ
    light = np.clip(light, 0.12, 1.55).astype(np.float32)

    # 滑らかにする
    if blur_sigma > 0.1:
        # ksize=(0,0) を指定すると sigma から適切なカーネルサイズを OpenCV が選ぶ
        light = cv2.GaussianBlur(light, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))

    # 紙マスクが無ければ従来通り「全体に乗算」だが、基本は紙だけに適用
    if paper_mask_u8 is None:
        out = image_bgr.astype(np.float32) * light[:, :, None]
        out = np.clip(out, 0, 255).astype(np.uint8)
        meta = {
            "applied": True,
            "prob": prob,
            "strength": float(strength),
            "highlight": float(highlight),
            "angle_deg": float(angle_deg),
            "blur_sigma": float(blur_sigma),
            "vignette_strength": float(vignette_strength),
            "band_strength": float(band_strength),
            "mask_mode": "none_full_image",
        }
        return out, meta

    # mask を 0..1 の alpha にして、境界をぼかして自然に馴染ませる
    pm = paper_mask_u8
    if pm.ndim == 3:
        pm = cv2.cvtColor(pm, cv2.COLOR_BGR2GRAY)
    pm_f = pm.astype(np.float32) / 255.0
    edge_blur_sigma = float(cfg.get("edge_blur_sigma", 6.0) or 6.0)
    if edge_blur_sigma > 0.1:
        pm_f = cv2.GaussianBlur(pm_f, (0, 0), sigmaX=edge_blur_sigma, sigmaY=edge_blur_sigma)
    pm_f = np.clip(pm_f, 0.0, 1.0)

    # 紙領域のみライティング適用（背景はそのまま）
    img_f = image_bgr.astype(np.float32)
    lit = img_f * light[:, :, None]
    out = (img_f * (1.0 - pm_f[:, :, None])) + (lit * pm_f[:, :, None])
    out = np.clip(out, 0, 255).astype(np.uint8)

    meta = {
        "applied": True,
        "prob": prob,
        "strength": float(strength),
        "highlight": float(highlight),
        "angle_deg": float(angle_deg),
        "blur_sigma": float(blur_sigma),
        "vignette_strength": float(vignette_strength),
        "band_strength": float(band_strength),
        "edge_blur_sigma": float(edge_blur_sigma),
        "mask_mode": "paper_only",
    }
    return out, meta


# ------------------------------------------------------------
# WeChat QRCode エンジン（cv2.wechat_qrcode_WeChatQRCode）
# ------------------------------------------------------------


class WeChatQRDetector:
    """WeChat QRコード検出器の薄いラッパー。

    目的:
      OpenCV標準の QRCodeDetector が「小さい/低解像度QR」で失敗することがあるため、
      CNN検出器 + 超解像モデルを含む WeChat エンジンを使えるようにする。

    注意:
      opencv-contrib のビルドと、4つのモデルファイルが必要。
      重い detector インスタンスは生成を1回に抑える。
    """

    def __init__(self, model_dir: str):
        self.model_dir = str(model_dir)
        self.detector = self._init_detector(self.model_dir)

    @staticmethod
    def _init_detector(model_dir: str) -> Any:
        if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
            raise RuntimeError(
                "cv2.wechat_qrcode_WeChatQRCode is not available. "
                "Install opencv-contrib-python and restart python."
            )

        detect_proto = os.path.join(model_dir, "detect.prototxt")
        detect_caffe = os.path.join(model_dir, "detect.caffemodel")
        sr_proto = os.path.join(model_dir, "sr.prototxt")
        sr_caffe = os.path.join(model_dir, "sr.caffemodel")

        if not all(map(os.path.exists, [detect_proto, detect_caffe, sr_proto, sr_caffe])):
            raise FileNotFoundError(
                "WeChat QR model files not found. Expected: "
                f"{detect_proto}, {detect_caffe}, {sr_proto}, {sr_caffe}"
            )

        return cv2.wechat_qrcode_WeChatQRCode(detect_proto, detect_caffe, sr_proto, sr_caffe)

    @staticmethod
    def _decode_from_detector(detector: Any, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        """detector.detectAndDecode の戻り値を本パイプラインの形式へ整形する。"""

        if image_bgr is None:
            return []

        res, points = detector.detectAndDecode(image_bgr)

        out: list[dict[str, Any]] = []
        if res is None or points is None:
            return out

        # OpenCV は（文字列のタプル/リスト）と（points の配列）を返すことがある
        try:
            res_list = list(res)
        except Exception:
            res_list = [str(res)]

        pts_arr = np.asarray(points, dtype=np.float32)
        if pts_arr.ndim == 2:
            pts_arr = pts_arr.reshape(1, -1, 2)

        for i, data in enumerate(res_list):
            if not data:
                continue
            if i >= len(pts_arr):
                continue
            pts = pts_arr[i].reshape(-1, 2)
            out.append({"data": str(data), "points": pts.tolist(), "engine": "wechat"})

        return out

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        """QRコードを検出してデコードする。

        戻り値:
          dict のリスト: [{data, points, engine}]
        """

        # NOTE:
        # OpenCV の wechat_qrcode_WeChatQRCode はスレッドセーフが保証されない。
        # v13 改善: 呼び出し側で「detector をスレッド数分用意」し、ここでは Lock を使わない。
        return self._decode_from_detector(self.detector, image_bgr)


class WeChatQRDetectorPool:
    """WeChat QR detector のプール。

    目的:
      回転スキャン（ThreadPoolExecutor）時に `detectAndDecode` を Lock で直列化しない。

    方針:
      - detector を pool_size 個だけ事前生成して queue に入れる
      - detect() の都度 detector を借りて使い、返却する

    備考:
      queue 自体の lock は極小で、`detectAndDecode` の重い処理は並列に走る。
    """

    def __init__(self, model_dir: str, pool_size: int):
        self.model_dir = str(model_dir)
        self.pool_size = int(pool_size)
        if self.pool_size <= 0:
            raise ValueError("pool_size must be >= 1")

        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=self.pool_size)
        for _ in range(self.pool_size):
            det = WeChatQRDetector._init_detector(self.model_dir)
            self._q.put(det)

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        if image_bgr is None:
            return []
        det = self._q.get()
        try:
            return WeChatQRDetector._decode_from_detector(det, image_bgr)
        finally:
            # 例外時も必ず返却する
            self._q.put(det)


_WECHAT_QR: Optional[Any] = None


def init_wechat_qr_detector(
    model_dir: str,
    logger: Optional[logging.Logger] = None,
    *,
    pool_size: int = 1,
) -> Optional[Any]:
    """グローバルな WeChat QR detector（重い）を1回だけ初期化する。

    利用できない場合は None を返す。
    """

    global _WECHAT_QR
    try:
        # v13 改善: detector をスレッド数ぶん用意し、Lock による直列化を避ける。
        _WECHAT_QR = WeChatQRDetectorPool(model_dir=model_dir, pool_size=int(pool_size))
        if logger:
            logger.info("[OK] WeChat QR detector initialized: %s (pool_size=%d)", model_dir, int(pool_size))
        return _WECHAT_QR
    except Exception as e:
        _WECHAT_QR = None
        if logger:
            logger.warning("[WARN] WeChat QR detector disabled: %s", e)
        return None


# --- 既存実装の流用 ---
# 注意: このスクリプトは `python APA/paper_pipeline_v13.py ...` の形で実行される想定。
# その場合 sys.path[0] は `.../APA` になるため、同ディレクトリのモジュールは
# `from test_recovery_paper import ...` の形で import する（`import APA.xxx` は失敗しやすい）。
from test_recovery_paper import (
    XFeatMatcher,
    detect_formA_marker_boxes as _detect_formA_marker_boxes_base,
    draw_inlier_matches,
    ensure_portable_git_on_path,
    now_run_id,
    refine_homography_least_squares,
    resize_keep_aspect,
    scale_matrix,
    warp_template_to_random_view,
)


os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Windows のコンソールは既定で cp932/cp1252 になることがあり、
# 日本語を print すると UnicodeEncodeError になる場合がある。
# そのため stdout/stderr を UTF-8 に寄せる。
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr:
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ------------------------------------------------------------
# ロギング
# ------------------------------------------------------------


def setup_logging(
    out_root: Optional[Path],
    level: str = "INFO",
    console_level: Optional[str] = None,
) -> logging.Logger:
    """logging の設定。

    - console: デフォルトINFO（または console_level）
    - file   : level と同じレベルで out_root/run.log に保存
    """

    logger = logging.getLogger("paper_pipeline")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, (console_level or level).upper(), logging.INFO))
    logger.addHandler(ch)

    if out_root is not None:
        try:
            fh = logging.FileHandler(str(out_root / "run.log"), encoding="utf-8")
            fh.setFormatter(fmt)
            fh.setLevel(getattr(logging, level.upper(), logging.INFO))
            logger.addHandler(fh)
        except Exception:
            # ファイルハンドラの作成に失敗しても処理は継続する
            pass

    return logger


# ------------------------------------------------------------
# DocAligner 補助関数（test_docaligner_camera_v3.py を元に調整）
# ------------------------------------------------------------


def patch_capybara_exports() -> None:
    """capybara の namespace package に期待されるシンボルを追加する（Windows回避策）。"""

    import capybara as cb

    from capybara.mixins import EnumCheckMixin
    from capybara.onnxengine.engine import ONNXEngine
    from capybara.onnxengine.enum import Backend
    from capybara.structures.polygons import Polygons
    from capybara.utils.custom_path import Path as CbPath, get_curdir
    from capybara.utils.utils import download_from_google
    from capybara.vision.functionals import centercrop, imbinarize, pad
    from capybara.vision.geometric import imresize
    from capybara.vision.improc import is_numpy_img

    for name, obj in {
        "Path": CbPath,
        "get_curdir": get_curdir,
        "download_from_google": download_from_google,
        "EnumCheckMixin": EnumCheckMixin,
        "Backend": Backend,
        "ONNXEngine": ONNXEngine,
        "pad": pad,
        "centercrop": centercrop,
        "imresize": imresize,
        "imbinarize": imbinarize,
        "is_numpy_img": is_numpy_img,
        "Polygons": Polygons,
    }.items():
        if not hasattr(cb, name):
            setattr(cb, name, obj)


def order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """4点を TL/TR/BR/BL の順に並べる。"""

    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def expand_polygon(polygon_xy: np.ndarray, margin_px: float, img_w: int, img_h: int) -> np.ndarray:
    """polygon を margin_px だけ外側に広げる（可能な範囲で）。"""

    poly = np.asarray(polygon_xy, dtype=np.float32).reshape(4, 2)
    if margin_px <= 0:
        return poly
    center = poly.mean(axis=0)
    out = []
    for pt in poly:
        v = pt - center
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            out.append(pt)
        else:
            out.append(pt + (v / n) * float(margin_px))
    out = np.asarray(out, dtype=np.float32)
    out[:, 0] = np.clip(out[:, 0], 0, max(0, img_w - 1))
    out[:, 1] = np.clip(out[:, 1], 0, max(0, img_h - 1))
    return out


def polygon_margin_px_from_ratio(
    polygon_xy: np.ndarray,
    ratio: float,
    min_px: float,
    max_px: float,
) -> float:
    """polygon サイズ比からマージン(px)を計算する。

    polygon から推定した紙サイズ（辺長の最大）を基準にすることで、
    入力解像度が変わっても挙動が安定しやすい。
    """

    poly = order_quad_tl_tr_br_bl(polygon_xy)
    w_top = float(np.linalg.norm(poly[1] - poly[0]))
    w_bottom = float(np.linalg.norm(poly[2] - poly[3]))
    h_left = float(np.linalg.norm(poly[3] - poly[0]))
    h_right = float(np.linalg.norm(poly[2] - poly[1]))
    ref = max(w_top, w_bottom, h_left, h_right)
    px = float(ref) * float(ratio)
    px = max(float(min_px), px)
    if max_px > 0:
        px = min(float(max_px), px)
    return float(px)


def polygon_to_rectified(
    image_bgr: np.ndarray,
    polygon_xy: np.ndarray,
    out_max_side: int = 1800,
) -> tuple[np.ndarray, np.ndarray]:
    """polygon 領域を正面視（透視補正済み）の画像へワープする。

    戻り値:
        rectified_bgr, H_src_to_rect
    """

    poly = order_quad_tl_tr_br_bl(polygon_xy)

    # polygon の辺長から出力サイズを概算
    w_top = np.linalg.norm(poly[1] - poly[0])
    w_bottom = np.linalg.norm(poly[2] - poly[3])
    h_left = np.linalg.norm(poly[3] - poly[0])
    h_right = np.linalg.norm(poly[2] - poly[1])
    out_w = int(round(max(w_top, w_bottom)))
    out_h = int(round(max(h_left, h_right)))
    out_w = max(320, out_w)
    out_h = max(320, out_h)

    # 速度のため、最大辺を制限
    scale = 1.0
    m = max(out_w, out_h)
    if m > out_max_side:
        scale = float(out_max_side) / float(m)
        out_w = int(round(out_w * scale))
        out_h = int(round(out_h * scale))

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(poly.astype(np.float32), dst)
    rectified = cv2.warpPerspective(image_bgr, H, (out_w, out_h))
    return rectified, H


def rotate_image_bound(image_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """切り取りが起きないようにキャンバスを拡張して回転する（imutils.rotate_bound 相当）。"""

    # 改善2:
    # 0°/180° の回転で warpAffine を使うのは無駄なので、特別扱いする。
    # - 0°  : そのまま
    # - 180°: cv2.rotate
    a = float(angle_deg) % 360.0
    if abs(a - 0.0) < 1e-6:
        return image_bgr
    if abs(a - 180.0) < 1e-6:
        return cv2.rotate(image_bgr, cv2.ROTATE_180)

    h, w = image_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(round((h * sin) + (w * cos)))
    new_h = int(round((h * cos) + (w * sin)))
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(image_bgr, M, (new_w, new_h))


def enforce_landscape(image_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """長辺が横になるように統一する（横長化）。戻り値: (image, rotated_flag)"""

    h, w = image_bgr.shape[:2]
    if w >= h:
        return image_bgr, False
    # 90度（時計回り）回転
    return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE), True


def _thickness_params(image_bgr: np.ndarray) -> tuple[int, float, int]:
    """画像サイズに応じた (thickness, font_scale, font_thickness) を返す。"""

    h, w = image_bgr.shape[:2]
    scale = min(w, h) / 1000.0
    thickness = max(6, int(scale * 10))
    font_scale = max(0.8, scale * 1.2)
    font_thickness = max(2, int(scale * 4))
    return thickness, font_scale, font_thickness


def _get_japanese_font(size_px: int) -> ImageFont.FreeTypeFont:
    """Pillow 描画用の「日本語対応フォント」を取得する。

    OpenCV の cv2.putText は日本語描画ができないため、Pillow を使う。
    """

    # 注意:
    # 以前は Windows フォントパスをハードコードしていたが、
    # Linux/Mac/Docker では存在しないため、可能な限り OS 非依存で解決する。

    # 1) ユーザー指定（環境変数）
    font_path = os.environ.get("APA_FONT_PATH")
    if font_path:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size=int(size_px))
        except Exception:
            pass

    # 2) OSでよくあるフォント
    candidates: list[str] = []
    sysname = platform.system().lower()
    if "windows" in sysname:
        candidates += [
            r"C:\Windows\Fonts\meiryo.ttc",
            r"C:\Windows\Fonts\meiryob.ttc",
            r"C:\Windows\Fonts\msgothic.ttc",
            r"C:\Windows\Fonts\msyh.ttc",
        ]
    elif "darwin" in sysname or "mac" in sysname:
        candidates += [
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:
        candidates += [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    for p in candidates:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size=int(size_px))
        except Exception:
            pass

    # 3) matplotlib font_manager（可能なら）
    try:
        import matplotlib.font_manager as fm

        # 日本語対応が期待できるファミリ名を試す（無ければ findfont がフォールバック）
        for fam in ["Meiryo", "MS Gothic", "Noto Sans CJK JP", "Noto Sans CJK", "IPAPGothic", "DejaVu Sans"]:
            try:
                p = fm.findfont(fm.FontProperties(family=fam), fallback_to_default=True)
                if p and os.path.exists(p):
                    return ImageFont.truetype(p, size=int(size_px))
            except Exception:
                continue
    except Exception:
        pass

    # 4) 最後の手段: デフォルト（日本語が出ない可能性はあるが、処理継続を優先）
    return ImageFont.load_default()


def draw_text_pil(
    image_bgr: np.ndarray,
    xy: tuple[int, int],
    text: str,
    color_bgr: tuple[int, int, int],
    font_size: int,
    outline: bool = True,
) -> np.ndarray:
    """Pillow で文字を描画する（OpenCV で日本語が '???' になる問題の回避）。"""

    # BGR -> RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)

    font = _get_japanese_font(font_size)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))

    x, y = int(xy[0]), int(xy[1])
    if outline:
        # 視認性のため黒縁取り
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=color_rgb)

    out_rgb = np.array(pil)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def draw_text_ascii_cv2(
    image_bgr: np.ndarray,
    xy: tuple[int, int],
    text: str,
    color_bgr: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> np.ndarray:
    """日本語フォントが見つからない場合のASCII限定フォールバック。"""

    out = image_bgr.copy()
    cv2.putText(
        out,
        text,
        (int(xy[0]), int(xy[1])),
        cv2.FONT_HERSHEY_COMPLEX,
        float(font_scale),
        color_bgr,
        int(thickness),
        lineType=cv2.LINE_AA,
    )
    return out


def _marker_center_xy(marker: dict[str, Any]) -> tuple[float, float]:
    """bbox からマーカー中心を計算する。"""

    x, y, w, h = marker.get("bbox", [0, 0, 0, 0])
    return float(x) + float(w) * 0.5, float(y) + float(h) * 0.5


def draw_formA_markers_overlay(image_bgr: np.ndarray, markers: list[dict[str, Any]]) -> np.ndarray:
    """フォームAのマーカーを赤枠 + 角ラベル（日本語）で描画する。"""

    out = image_bgr.copy()
    thickness, font_scale, font_thickness = _thickness_params(out)
    font_px = max(18, int(font_scale * 28))
    jp = {"top_left": "左上", "top_right": "右上", "bottom_left": "左下"}
    for m in markers:
        x, y, w, h = m.get("bbox", [0, 0, 0, 0])
        corner = str(m.get("corner", ""))
        label = f"{corner}({jp.get(corner, corner)})"
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), thickness)
        # 可能なら日本語ラベル、無理ならASCIIへフォールバック
        try:
            out = draw_text_pil(
                out,
                (int(x), max(5, int(y) - font_px - 4)),
                label,
                color_bgr=(0, 0, 255),
                font_size=font_px,
                outline=True,
            )
        except Exception:
            out = draw_text_ascii_cv2(
                out,
                (int(x), max(5, int(y) - font_px - 4)),
                corner,
                color_bgr=(0, 0, 255),
                font_scale=float(font_scale),
                thickness=int(font_thickness),
            )
    return out


def draw_formB_qr_overlay(image_bgr: np.ndarray, qrs: list[dict[str, Any]]) -> np.ndarray:
    """フォームBのQRを青枠 + 「右上」ラベルで描画する。"""

    out = image_bgr.copy()
    thickness, font_scale, font_thickness = _thickness_params(out)
    font_px = max(18, int(font_scale * 28))
    if not qrs:
        return out

    pts = np.asarray(qrs[0]["points"], dtype=np.float32).reshape(-1, 2)
    pts_i = pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts_i], True, (255, 0, 0), thickness)
    x, y, w, h = cv2.boundingRect(pts_i)
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), thickness)

    # QR の右上点付近にラベルを置く
    tr_idx = int(np.argmax(pts[:, 0] - pts[:, 1]))
    tr = pts[tr_idx]
    try:
        out = draw_text_pil(
            out,
            (int(tr[0] + 10), max(5, int(tr[1] - font_px - 4))),
            "右上",
            color_bgr=(255, 0, 0),
            font_size=font_px,
            outline=True,
        )
    except Exception:
        out = draw_text_ascii_cv2(
            out,
            (int(tr[0] + 10), max(5, int(tr[1] - font_px - 4))),
            "TOP_RIGHT",
            color_bgr=(255, 0, 0),
            font_scale=float(font_scale),
            thickness=int(font_thickness),
        )
    return out


def draw_polygon_overlay(image_bgr: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    result = image_bgr.copy()
    poly = order_quad_tl_tr_br_bl(polygon_xy).astype(np.int32)
    overlay = result.copy()
    cv2.fillPoly(overlay, [poly], (0, 255, 0))
    cv2.addWeighted(overlay, 0.2, result, 0.8, 0, result)
    vis = PIPELINE_DEFAULTS["visual"]
    cv2.polylines(result, [poly], True, (0, 255, 0), int(vis["polygon_line_thickness"]))
    labels = ["TL", "TR", "BR", "BL"]
    for i, pt in enumerate(poly):
        cv2.circle(result, tuple(pt), int(vis["polygon_point_radius"]), (0, 0, 255), -1)
        cv2.putText(
            result,
            labels[i],
            (int(pt[0] + 10), int(pt[1] + 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(vis["polygon_label_font_scale"]),
            (255, 255, 255),
            int(vis["polygon_label_thickness"]),
        )
    return result


def detect_polygon_docaligner(
    model: Any,
    cb: Any,
    image_bgr: np.ndarray,
    pad_px: int = int(PIPELINE_DEFAULTS["docaligner"]["pad_px"]),
) -> Optional[np.ndarray]:
    padded = cb.pad(image_bgr, pad_px)
    poly = model(img=padded, do_center_crop=False)
    if poly is None:
        return None
    poly = np.asarray(poly, dtype=np.float32)
    if poly.shape[0] < 4:
        return None
    poly = poly[:4] - float(pad_px)
    return poly


# ------------------------------------------------------------
# マーカー検出ラッパー（前処理オプション付き）
# ------------------------------------------------------------


def _preprocess_variants_for_markers(image_bgr: np.ndarray, mode: str) -> list[tuple[str, np.ndarray]]:
    """マーカー検出を安定させるための前処理バリエーションを作る。"""

    if mode == "none":
        return [("bgr", image_bgr)]

    variants: list[tuple[str, np.ndarray]] = [("bgr", image_bgr)]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

    if mode in ("basic", "morph"):
        # 照明ムラに強いコントラスト補正（CLAHE）
        try:
            clahe_cfg = PIPELINE_DEFAULTS["marker"]["clahe"]
            clahe = cv2.createCLAHE(
                clipLimit=float(clahe_cfg["clipLimit"]),
                tileGridSize=tuple(int(x) for x in clahe_cfg["tileGridSize"]),
            )
            g2 = clahe.apply(gray)
            variants.append(("clahe", cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass
        # 自適応二値化（輪郭ベースの検出で効くことがある）
        try:
            at = PIPELINE_DEFAULTS["marker"]["adaptive_threshold"]
            bw = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                int(at["block_size"]),
                int(at["C"]),
            )
            variants.append(("adaptive", cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass

    if mode == "morph":
        try:
            # モルフォロジー処理でノイズ除去 + ブロブ結合を狙う
            morph_cfg = PIPELINE_DEFAULTS["marker"]["morph"]
            k = max(int(morph_cfg["kernel_min"]), int(round(min(image_bgr.shape[:2]) * float(morph_cfg["kernel_ratio"]))))
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            at = PIPELINE_DEFAULTS["marker"]["adaptive_threshold"]
            bw = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                int(at["block_size"]),
                int(at["C"]),
            )
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            variants.append(("adaptive_morph", cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass

    return variants


def detect_formA_marker_boxes(image_bgr: np.ndarray, preproc_mode: str = "none") -> list[dict[str, Any]]:
    """前処理バリエーションを試しながらマーカー検出を行う。"""

    best: list[dict[str, Any]] = []
    best_score = -1.0
    for name, var in _preprocess_variants_for_markers(image_bgr, preproc_mode):
        markers = _detect_formA_marker_boxes_base(var)
        # 3点揃ったケースを強く優先
        ok = len(markers) == 3
        score = float(sum(m.get("score", 0.0) for m in markers))
        if ok:
            score += 10.0
        # 前処理ありを僅かに優先（同点回避）
        if name != "bgr":
            score += 0.05
        if score > best_score:
            best_score = score
            best = markers
    return best


def detect_qr_codes_wechat(
    image_bgr: np.ndarray,
    wechat: Optional[Any],
) -> list[dict[str, Any]]:
    """WeChat エンジンによるQR検出（小さい/低解像度QRに強い）。

    detector が利用できない場合は空リストを返す。
    """

    if wechat is None:
        return []
    try:
        return wechat.detect(image_bgr)
    except Exception:
        return []


def _preprocess_variants_for_qr(image_bgr: np.ndarray, variant_names: list[str]) -> list[tuple[str, np.ndarray]]:
    """QR検出のための前処理バリエーションを作る。

    v13.2 改善:
      v7並みの精度を確保するため、clahe / adaptive_morph をサポート。
      これにより照明ムラや低コントラストのQRコードも検出しやすくなる。

    サポートする前処理:
      - bgr: 元画像そのまま
      - gray: グレースケール
      - clahe: CLAHE（照明ムラ対策）
      - adaptive_threshold: 自適応二値化（輪郭/コントラストが弱いQRで効くことがある）
      - adaptive_morph: 自適応二値化 + モルフォロジー
    """

    if image_bgr is None:
        return []

    names = [str(x) for x in (variant_names or [])]
    if not names:
        names = ["bgr"]

    out: list[tuple[str, np.ndarray]] = []

    # gray は複数の前処理で使うので、必要なら1回だけ生成
    gray: Optional[np.ndarray] = None

    def _get_gray() -> Optional[np.ndarray]:
        nonlocal gray
        if gray is None:
            try:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            except Exception:
                return None
        return gray

    for name in names:
        if name == "bgr":
            out.append(("bgr", image_bgr))

        elif name == "gray":
            g = _get_gray()
            if g is not None:
                out.append(("gray", cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)))

        elif name == "clahe":
            # CLAHE（照明ムラ対策）
            g = _get_gray()
            if g is not None:
                try:
                    clahe_cfg = PIPELINE_DEFAULTS.get("qr", {}).get("clahe", {})
                    if not clahe_cfg:
                        clahe_cfg = PIPELINE_DEFAULTS.get("marker", {}).get("clahe", {})
                    clip_limit = float(clahe_cfg.get("clipLimit", 2.0))
                    tile_grid = tuple(int(x) for x in clahe_cfg.get("tileGridSize", [8, 8]))
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
                    g2 = clahe.apply(g)
                    out.append(("clahe", cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)))
                except Exception:
                    continue

        elif name == "adaptive_threshold":
            # 自適応二値化（morph無し）。
            # QRコードは黒白の境界が重要なため、照明ムラが強いときに効くケースがある。
            g = _get_gray()
            if g is not None:
                try:
                    at_cfg = PIPELINE_DEFAULTS.get("qr", {}).get("adaptive_threshold", {})
                    if not at_cfg:
                        at_cfg = PIPELINE_DEFAULTS.get("marker", {}).get("adaptive_threshold", {})
                    block_size = int(at_cfg.get("block_size", 51))
                    c_val = int(at_cfg.get("C", 5))
                    if block_size < 3:
                        block_size = 3
                    if block_size % 2 == 0:
                        block_size += 1
                    bw = cv2.adaptiveThreshold(
                        g,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        block_size,
                        c_val,
                    )
                    out.append(("adaptive_threshold", cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)))
                except Exception:
                    continue

        elif name == "adaptive_morph":
            # 自適応二値化 + モルフォロジー（v7の detect_qr_codes_robust 相当）
            g = _get_gray()
            if g is not None:
                try:
                    at_cfg = PIPELINE_DEFAULTS.get("qr", {}).get("adaptive_threshold", {})
                    if not at_cfg:
                        at_cfg = PIPELINE_DEFAULTS.get("marker", {}).get("adaptive_threshold", {})
                    block_size = int(at_cfg.get("block_size", 51))
                    c_val = int(at_cfg.get("C", 5))

                    kernel_xy = PIPELINE_DEFAULTS.get("qr", {}).get("wechat", {}).get("adaptive_morph_kernel", [5, 5])
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(int(x) for x in kernel_xy))

                    bw = cv2.adaptiveThreshold(
                        g,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        block_size,
                        c_val,
                    )
                    bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
                    bw2 = cv2.morphologyEx(bw2, cv2.MORPH_OPEN, kernel)
                    out.append(("adaptive_morph", cv2.cvtColor(bw2, cv2.COLOR_GRAY2BGR)))
                except Exception:
                    continue

        else:
            # 未対応は無視（設定ミスで落ちないようにする）
            continue

    # 重複排除（順序維持）
    seen: set[str] = set()
    out2: list[tuple[str, np.ndarray]] = []
    for n, img in out:
        if n in seen:
            continue
        seen.add(n)
        out2.append((n, img))
    return out2


def detect_qr_codes_wechat_multiscale(
    image_bgr: np.ndarray,
    wechat: Optional[Any],
    *,
    mode: str = "robust",
) -> list[dict[str, Any]]:
    """WeChatエンジンによるQR検出（前処理 + マルチスケール）。

    v13 改善（ユーザー要望）:
      - B判定を fast→robust の2段階にする
      - robust は必要時のみ 1 回

    mode:
      - fast  : 角度候補絞り込み用（軽量）
      - robust: 最終確定用
    """

    if wechat is None or image_bgr is None:
        return []

    mode = str(mode or "robust")
    cfg_all = PIPELINE_DEFAULTS.get("qr", {}).get("wechat", {})
    cfg = cfg_all.get(mode, {}) if isinstance(cfg_all, dict) else {}

    variant_names = list(cfg.get("variants") or ["bgr"])
    scales = [float(s) for s in (cfg.get("scales") or [1.0])]

    h0, w0 = image_bgr.shape[:2]
    up_enable_max = int(cfg_all.get("up_scale_enable_max_side_px", 0) or 0)
    if up_enable_max > 0 and max(h0, w0) >= up_enable_max:
        # 大きい画像は up-scale を無効化
        scales = [s for s in scales if s <= 1.0 + 1e-9]

    min_side = int(PIPELINE_DEFAULTS.get("qr", {}).get("min_test_side_px", 120))
    max_side = int(cfg_all.get("max_test_side_px", 6500))

    variants = _preprocess_variants_for_qr(image_bgr, variant_names)

    best: list[dict[str, Any]] = []
    best_score = float("-inf")

    for prep_name, src in variants:
        h, w = src.shape[:2]
        for s in scales:
            if abs(s - 1.0) < 1e-9:
                test = src
            else:
                new_w = int(round(w * s))
                new_h = int(round(h * s))
                if new_w < min_side or new_h < min_side:
                    continue
                if new_w > max_side or new_h > max_side:
                    continue
                interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
                test = cv2.resize(src, (new_w, new_h), interpolation=interp)

            qrs = detect_qr_codes_wechat(test, wechat)
            if not qrs:
                continue

            # points を元画像座標に戻す
            if abs(s - 1.0) > 1e-9:
                for q in qrs:
                    try:
                        pts = np.asarray(q.get("points"), dtype=np.float32).reshape(-1, 2)
                        pts = pts / float(s)
                        q["points"] = pts.tolist()
                    except Exception:
                        continue

            # どう見つけたか（前処理・スケール）を記録
            for q in qrs:
                q.setdefault("engine", "wechat")
                q["prep"] = prep_name
                q["scale"] = float(s)

            # fast は "見つけたら即返す"（軽量化）
            if mode == "fast":
                return qrs

            # robust: best を選ぶ
            score = float("-inf")
            try:
                score, _ = score_best_qr_candidate(test if abs(s - 1.0) < 1e-9 else src, qrs)
            except Exception:
                score = 0.0

            if score > best_score:
                best_score = score
                best = qrs

    return best


def score_formB_fast(image_bgr: np.ndarray) -> tuple[bool, float, dict[str, Any]]:
    """回転スキャン中の高速B判定。

    v7の設計（fast→必要ならrobust）を v13(WeChat-only) にも導入するためのもの。
    - fast は見つけたら即返す
    - scale/variant は PIPELINE_DEFAULTS["qr"]["wechat"]["fast"] に従う
    """

    wechat = getattr(score_formB, "_wechat", None)
    if wechat is None:
        return False, 0.0, {"qrs": [], "reason": "wechat_detector_disabled", "phase": "fast"}

    qrs = detect_qr_codes_wechat_multiscale(image_bgr, wechat, mode="fast")
    if not qrs:
        return False, 0.0, {"qrs": [], "reason": "wechat_no_qr", "phase": "fast"}

    best_score, detail = score_best_qr_candidate(image_bgr, qrs)
    score = 1.0 + float(best_score)
    detail["phase"] = "fast"
    return True, float(score), detail


def score_best_qr_candidate(
    image_bgr: np.ndarray,
    qrs: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """複数候補の中から最良のQRを1つ選ぶ。

    目的: 回転角を選んだ後に QR が「右上」に来るようにしたい。
    スコアは以下で構成する:
      - 右上に近いほど高得点（主）
      - QR面積が大きいほど高得点（副：安定性向上）

    v13.1 改善:
      フォームBのQRコードは必ず「右上」にあるべき。
      横長画像において、QRコードが右上象限にあるか左下象限にあるかを
      明確に区別し、右上にある場合に大きなボーナスを与える。
    """

    h, w = image_bgr.shape[:2]
    best = None
    best_score = float("-inf")
    best_detail: dict[str, Any] = {}

    for q in (qrs or []):
        try:
            pts = np.asarray(q.get("points"), dtype=np.float32).reshape(-1, 2)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            # detector によって points 順が違うため、abs() で負の面積を避ける
            area = float(abs(cv2.contourArea(pts.astype(np.float32))))
            rel = area / float(max(1, w * h))

            # 正規化座標（0〜1）
            nx = cx / float(max(1, w))
            ny = cy / float(max(1, h))

            # v13.1 改善:
            # 横長画像において、QRコードが「右上」にあるか「左下」にあるかを
            # 明確に判定する。フォームBでは正しい向きなら必ず右上にQRがある。
            #
            # 右上象限: nx > 0.5 AND ny < 0.5
            # 左下象限: nx < 0.5 AND ny > 0.5
            #
            # 右上にある場合は大きなボーナス（+100）を与え、
            # 左下にある場合はペナルティ（-100）を与える。
            # これにより、0度と180度の比較で右上に来る方が確実に選ばれる。

            is_in_top_right_quadrant = (nx > 0.5) and (ny < 0.5)
            is_in_bottom_left_quadrant = (nx < 0.5) and (ny > 0.5)

            # 位置に基づくボーナス/ペナルティ
            if is_in_top_right_quadrant:
                quadrant_bonus = 100.0
            elif is_in_bottom_left_quadrant:
                quadrant_bonus = -100.0
            else:
                # 中央付近や他の象限（右下、左上）は中立
                quadrant_bonus = 0.0

            # 「右上」(nx=1.0, ny=0.0) からの距離で pos_score を作る。
            dist_from_top_right = math.sqrt((nx - 1.0) ** 2 + (ny - 0.0) ** 2)
            # dist in [0..sqrt(2)] -> score in [0..1]
            pos_score = max(0.0, 1.0 - (dist_from_top_right / 1.41421356))

            # pos_score を2乗して「端っこ」をより強く評価。
            final_pos_score = pos_score**2

            # 最終スコア = 象限ボーナス + 位置スコア + 面積スコア
            score = quadrant_bonus + (final_pos_score * 15.0) + (rel * 2.0)

            if score > best_score:
                best_score = float(score)
                best = q
                best_detail = {
                    "qr_center": [cx, cy],
                    "qr_center_normalized": [nx, ny],
                    "qr_rel_area": rel,
                    "qr_pos_score": pos_score,
                    "qr_pos_score_sq": float(final_pos_score),
                    "qr_is_in_top_right_quadrant": bool(is_in_top_right_quadrant),
                    "qr_is_in_bottom_left_quadrant": bool(is_in_bottom_left_quadrant),
                    "qr_quadrant_bonus": float(quadrant_bonus),
                }
        except Exception:
            continue

    if best is None:
        return 0.0, {"qrs": []}

    # 後段の描画で使いやすいよう、best を先頭にする
    reordered = [best] + [q for q in (qrs or []) if q is not best]
    detail = {
        "qrs": reordered,
        **best_detail,
        "qr_engine": str(best.get("engine", "wechat")),
        "qr_prep": str(best.get("prep", "")),
        "qr_scale": best.get("scale", None),
    }
    return float(best_score), detail


# ------------------------------------------------------------
# フォーム判定
# ------------------------------------------------------------


@dataclass
class FormDecision:
    ok: bool
    form: Optional[str]
    angle_deg: Optional[float]
    score: float
    detail: dict[str, Any]


def extract_form_unknown_reason(decision: Any) -> tuple[str, dict[str, Any]]:
    """フォーム判定が Unknown になった理由を、人間が追える形で抽出する。

    目的:
      - stage=form_unknown のときに「どのチェックに引っかかったか」を 1 行ログに出す
      - CSV にも reason を独立カラムで出して、フィルタ/集計しやすくする

    戻り値:
      (reason, diagnostics)

    備考:
      decide_form_by_rotations() の detail には既に reason を入れているが、
      ここでは 1 行ログ向けに軽量な診断値も併せて取り出す。
    """

    if decision is None:
        return "no_decision", {}

    # asdict(FormDecision) / dict のどちらも許容
    if isinstance(decision, dict):
        ok = bool(decision.get("ok"))
        detail = decision.get("detail") or {}
        score = decision.get("score")
    else:
        ok = bool(getattr(decision, "ok", False))
        detail = getattr(decision, "detail", None) or {}
        score = getattr(decision, "score", None)

    if ok:
        return "", {}

    reason = str(detail.get("reason") or "unknown")

    diag: dict[str, Any] = {}
    if score is not None:
        diag["top_score"] = score

    # 閾値/曖昧チェックの詳細
    for k in ["a_score", "b_score"]:
        if k in detail:
            diag[k] = detail.get(k)

    # no_detection のときは、探索段階の最大スコアだけ抜粋する（全 detail は巨大になり得るため）
    if reason == "no_detection":
        try:
            scan = detail.get("scan") or detail.get("coarse") or []
            max_a = float("-inf")
            max_b = float("-inf")
            for r in scan:
                try:
                    max_a = max(max_a, float(((r.get("A") or {}).get("score") or 0.0)))
                    # v13では scan に B_fast が入る
                    max_b = max(max_b, float(((r.get("B_fast") or r.get("B") or {}).get("score") or 0.0)))
                except Exception:
                    continue
            if max_a != float("-inf"):
                diag["scan_max_A_score"] = max_a
            if max_b != float("-inf"):
                diag["scan_max_B_score"] = max_b
        except Exception:
            pass

    return reason, diag


@dataclass
class MarkerGeometryConfig:
    """フォームA判定の誤検出を減らすための制約（例: フォームCをAと誤認しない）。"""

    # マーカーbboxの面積が似ていること（巨大1つ + 微小ノイズ2つ、のようなケースを避ける）
    max_marker_area_ratio: float = 3.0  # max(area)/min(area)

    # ページ（透視補正後画像）に対するマーカーの相対サイズ
    min_marker_area_page_ratio: float = 5e-5
    max_marker_area_page_ratio: float = 5e-3

    # 三角形の形状制約（TL-TR と TL-BL）
    # dist(TL,TR) / dist(TL,BL) ≒ (page_w / page_h) を期待
    max_dist_ratio_relative_error: float = 0.35

    # --- 追加: マーカー周辺が白地であること ---
    # bbox を少し拡張した領域（bbox自身は除外）に対し、
    # - 平均輝度が高い（白い）
    # - 黒っぽい画素（文字/線）が少ない
    # ことを要求する。
    surround_pad_ratio: float = 2.0
    surround_pad_px_min: int = 8
    surround_pad_px_max: int = 120
    # NOTE:
    # 2026/01/09: A 正解の取りこぼしが出たため、既定値を少し緩める。
    surround_min_mean_gray: float = 190.0
    surround_max_ink_ratio: float = 0.05
    surround_adaptive_block_size: int = 41
    surround_adaptive_C: int = 9


def validate_formA_marker_geometry(
    image_bgr: np.ndarray,
    markers: list[dict[str, Any]],
    cfg: MarkerGeometryConfig,
) -> tuple[bool, dict[str, Any]]:
    """フォームAマーカーに追加の幾何/スケール制約を適用する。

    戻り値:
      (ok, detail)
    """

    detail: dict[str, Any] = {"ok": False, "reasons": [], "cfg": asdict(cfg)}
    if image_bgr is None or len(markers) != 3:
        detail["reasons"].append("markers_not_3")
        return False, detail

    h, w = image_bgr.shape[:2]
    page_area = float(max(1, w * h))

    areas: list[float] = []
    corner_to_center: dict[str, tuple[float, float]] = {}
    for m in markers:
        x, y, bw, bh = m.get("bbox", [0, 0, 0, 0])
        a = float(max(0, bw) * max(0, bh))
        areas.append(a)
        corner = str(m.get("corner", ""))
        corner_to_center[corner] = _marker_center_xy(m)

    if not areas or min(areas) <= 0:
        detail["reasons"].append("invalid_area")
        return False, detail

    max_over_min = float(max(areas) / max(1e-9, min(areas)))
    detail["marker_area_max_over_min"] = max_over_min
    if max_over_min > float(cfg.max_marker_area_ratio):
        detail["reasons"].append("marker_area_ratio_too_large")

    mean_area_ratio = float(np.mean(areas) / page_area)
    detail["marker_area_page_ratio_mean"] = mean_area_ratio
    if mean_area_ratio < float(cfg.min_marker_area_page_ratio):
        detail["reasons"].append("marker_too_small_for_page")
    if mean_area_ratio > float(cfg.max_marker_area_page_ratio):
        detail["reasons"].append("marker_too_large_for_page")

    # 三角形の距離比: TL/TR/BL の中心が揃っていることが前提
    need = ["top_left", "top_right", "bottom_left"]
    if all(k in corner_to_center for k in need):
        tl = np.array(corner_to_center["top_left"], dtype=np.float32)
        tr = np.array(corner_to_center["top_right"], dtype=np.float32)
        bl = np.array(corner_to_center["bottom_left"], dtype=np.float32)
        dist_w = float(np.linalg.norm(tr - tl))
        dist_h = float(np.linalg.norm(bl - tl))
        if dist_h <= 1e-6 or dist_w <= 1e-6:
            detail["reasons"].append("invalid_marker_dist")
        else:
            ratio = dist_w / dist_h
            expected = float(w) / float(max(1, h))
            rel_err = float(abs(ratio - expected) / max(1e-9, expected))
            detail.update(
                {
                    "marker_dist_ratio_w_over_h": ratio,
                    "page_aspect_w_over_h": expected,
                    "marker_dist_ratio_relative_error": rel_err,
                }
            )
            if rel_err > float(cfg.max_dist_ratio_relative_error):
                detail["reasons"].append("marker_triangle_ratio_off")
    else:
        detail["reasons"].append("missing_required_corners")

    # ------------------------------------------------------------
    # 追加制約: マーカー周辺が白地であること
    # ------------------------------------------------------------

    def _check_surrounding_blankness(
        *,
        gray_img: np.ndarray,
        bbox_xywh: tuple[float, float, float, float],
        cfg: MarkerGeometryConfig,
    ) -> tuple[bool, dict[str, Any]]:
        """マーカーbboxの周辺（bboxを除外したリング領域）が白いかどうかを判定する。"""

        x, y, bw, bh = bbox_xywh
        x = float(x)
        y = float(y)
        bw = float(bw)
        bh = float(bh)

        if bw <= 1 or bh <= 1:
            return False, {"ok": False, "reason": "bbox_too_small"}

        pad = float(max(bw, bh)) * float(cfg.surround_pad_ratio)
        pad = max(float(cfg.surround_pad_px_min), pad)
        pad = min(float(cfg.surround_pad_px_max), pad)

        H, W = gray_img.shape[:2]
        x0 = int(max(0, math.floor(x - pad)))
        y0 = int(max(0, math.floor(y - pad)))
        x1 = int(min(W, math.ceil(x + bw + pad)))
        y1 = int(min(H, math.ceil(y + bh + pad)))

        if (x1 - x0) < 10 or (y1 - y0) < 10:
            return False, {"ok": False, "reason": "roi_too_small", "roi": [x0, y0, x1, y1]}

        roi = gray_img[y0:y1, x0:x1]
        if roi.size == 0:
            return False, {"ok": False, "reason": "roi_empty", "roi": [x0, y0, x1, y1]}

        # ROI内のbbox座標
        bx0 = int(max(0, math.floor(x - x0)))
        by0 = int(max(0, math.floor(y - y0)))
        bx1 = int(min(roi.shape[1], math.ceil(x - x0 + bw)))
        by1 = int(min(roi.shape[0], math.ceil(y - y0 + bh)))

        # リング領域（bbox外側）をマスクで作る
        mask = np.ones_like(roi, dtype=np.uint8)
        if bx1 > bx0 and by1 > by0:
            mask[by0:by1, bx0:bx1] = 0

        ring_area = int(mask.sum())
        if ring_area <= 0:
            return False, {"ok": False, "reason": "ring_area_zero", "roi": [x0, y0, x1, y1]}

        # 平均輝度（白地なら高いはず）
        mean_gray = float((roi.astype(np.float32) * mask.astype(np.float32)).sum() / float(ring_area))

        # インク量（文字/線）を推定: adaptive threshold で黒っぽい画素を抽出
        blk = int(cfg.surround_adaptive_block_size)
        if blk < 3:
            blk = 3
        if blk % 2 == 0:
            blk += 1
        bw_img = cv2.adaptiveThreshold(
            roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blk,
            int(cfg.surround_adaptive_C),
        )
        ink = ((bw_img > 0) & (mask > 0)).astype(np.uint8)
        ink_ratio = float(int(ink.sum())) / float(ring_area)

        ok_blank = (mean_gray >= float(cfg.surround_min_mean_gray)) and (ink_ratio <= float(cfg.surround_max_ink_ratio))
        return bool(ok_blank), {
            "ok": bool(ok_blank),
            "pad_px": float(pad),
            "roi": [int(x0), int(y0), int(x1), int(y1)],
            "ring_area": int(ring_area),
            "mean_gray": float(mean_gray),
            "ink_ratio": float(ink_ratio),
            "threshold": {
                "min_mean_gray": float(cfg.surround_min_mean_gray),
                "max_ink_ratio": float(cfg.surround_max_ink_ratio),
                "adaptive_block_size": int(blk),
                "adaptive_C": int(cfg.surround_adaptive_C),
            },
        }

    # 全マーカーに対して周辺チェック
    try:
        gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        surround_details: dict[str, Any] = {}
        for m in markers:
            corner = str(m.get("corner", ""))
            x, y, bw, bh = m.get("bbox", [0, 0, 0, 0])
            ok_blank, sdet = _check_surrounding_blankness(gray_img=gray_img, bbox_xywh=(x, y, bw, bh), cfg=cfg)
            surround_details[corner or "unknown"] = sdet
            if not ok_blank:
                detail["reasons"].append(f"marker_surrounding_not_blank:{corner or 'unknown'}")
        detail["surrounding_blankness"] = surround_details
    except Exception as e:
        # 周辺チェック自体が失敗する場合は、安全側（Aと認めない）に倒す
        detail["reasons"].append(f"surrounding_blankness_check_failed:{e}")

    ok = len(detail["reasons"]) == 0
    detail["ok"] = ok
    return ok, detail


def score_formA(
    image_bgr: np.ndarray,
    marker_preproc: str = "none",
    geom_cfg: Optional[MarkerGeometryConfig] = None,
) -> tuple[bool, float, dict[str, Any]]:
    """フォームA判定。

    3点マーカー検出（TL/TR/BL）ができることに加えて、
    それぞれが「本来の位置（左上/右上/左下）」に近いほどスコア加点する。
    """

    markers = detect_formA_marker_boxes(image_bgr, preproc_mode=marker_preproc)
    ok = len(markers) == 3
    if not ok:
        return False, 0.0, {"markers": markers}

    # 追加制約で C->A の誤検出を抑える
    # （設定値は PIPELINE_DEFAULTS["formA"]["geometry"] で調整可能）
    if geom_cfg is not None:
        cfg = geom_cfg
    else:
        cfg_dict = (PIPELINE_DEFAULTS.get("formA") or {}).get("geometry") or {}
        # dataclass のフィールド判定は __dataclass_fields__ を使う（hasattr(class, field) では取れない）
        allowed = set(getattr(MarkerGeometryConfig, "__dataclass_fields__", {}).keys())
        cfg = MarkerGeometryConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})
    geom_ok, geom_detail = validate_formA_marker_geometry(image_bgr, markers, cfg)
    if not geom_ok:
        return False, 0.0, {"markers": markers, "geometry": geom_detail}

    base_score = float(sum(m.get("score", 0.0) for m in markers))

    h, w = image_bgr.shape[:2]
    expected = {
        "top_left": (0.0, 0.0),
        "top_right": (1.0, 0.0),
        "bottom_left": (0.0, 1.0),
    }
    per_corner: dict[str, float] = {}
    pos_scores: list[float] = []
    for m in markers:
        corner = str(m.get("corner", ""))
        if corner not in expected:
            continue
        cx, cy = _marker_center_xy(m)
        nx = cx / float(max(1, w))
        ny = cy / float(max(1, h))
        ex, ey = expected[corner]
        dist = float(np.hypot(nx - ex, ny - ey))
        # dist in [0..sqrt(2)] -> [0..1] に正規化
        closeness = max(0.0, 1.0 - (dist / 1.41421356))
        per_corner[corner] = float(closeness)
        pos_scores.append(float(closeness))

    pos_score = float(np.mean(pos_scores)) if pos_scores else 0.0

    # pos_score は 0..1。ベーススコア（概ね0..3）に対して少し効くように重み付け。
    score = base_score + pos_score * 2.0

    return True, float(score), {
        "markers": markers,
        "geometry": geom_detail,
        "pos_score": pos_score,
        "pos_score_per_corner": per_corner,
        "base_score": base_score,
        "marker_preproc": marker_preproc,
    }


def score_formB(image_bgr: np.ndarray) -> tuple[bool, float, dict[str, Any]]:
    wechat = getattr(score_formB, "_wechat", None)

    # ユーザー要望:
    # フォームBのQR検出は WeChat QR のみ使用する（OpenCV QRCodeDetector は使用しない）。
    if wechat is None:
        return False, 0.0, {"qrs": [], "reason": "wechat_detector_disabled"}

    # robustは最終確定用（必要時に1回だけ呼ぶ想定）
    qrs = detect_qr_codes_wechat_multiscale(image_bgr, wechat, mode="robust")
    if not qrs:
        return False, 0.0, {"qrs": [], "reason": "wechat_no_qr"}

    best_score, detail = score_best_qr_candidate(image_bgr, qrs)
    # 既存のしきい値（デフォルト>=1.2）と大きくズレないようスケールを合わせる
    score = 1.0 + float(best_score)
    return True, float(score), detail


def decide_form_by_rotations(
    rectified_bgr: np.ndarray,
    max_workers: int = 8,
    marker_preproc: str = "none",
    unknown_score_threshold: float = 0.0,
    unknown_margin: float = 0.0,
    formA_geom_cfg: Optional[MarkerGeometryConfig] = None,
) -> FormDecision:
    """回転スキャンで、最良の判定（A/B/Unknown）を返す。

    v13.3 改善（ユーザー要望）:
      フォールバック構造に変更:
        1. A探索（見つかればA確定）
        2. Aが見つからない → Bのfast探索
        3. fastで見つかればB確定
        4. fastで見つからない → robustで再挑戦
        5. robustでも見つからなければUnknown

    方針:
      - 0/180 のみを評価する（回転ステップ探索はしない）
      - フォームBの判定は WeChat QR のみ（OpenCV QRCodeDetector によるフォールバックはしない）
    """

    scan_angles = [float(a) for a in (PIPELINE_DEFAULTS.get("rotation_scan") or {}).get("scan_angles_2_deg", [])]
    if not scan_angles:
        scan_angles = [0.0, 180.0]

    scan_results: list[dict[str, Any]] = []

    # threshold 未満で棄却された候補の記録（診断用）
    rejected_by_threshold: dict[str, Any] = {}
    thr = float(unknown_score_threshold or 0.0)

    # ----------------------------------
    # Step 1: A探索（見つかればA確定）
    # ----------------------------------

    def _eval_formA(angle: float) -> dict[str, Any]:
        rotated = rotate_image_bound(rectified_bgr, angle)
        h, w = rotated.shape[:2]
        if h > w:
            return {"angle": float(angle), "skip": True}
        okA, scoreA, detA = score_formA(rotated, marker_preproc=marker_preproc, geom_cfg=formA_geom_cfg)
        return {
            "angle": float(angle),
            "skip": False,
            "A": {"ok": bool(okA), "score": float(scoreA), "detail": detA},
        }

    bestA: Optional[FormDecision] = None

    with ThreadPoolExecutor(max_workers=min(int(max_workers), len(scan_angles))) as ex:
        futures = [ex.submit(_eval_formA, a) for a in scan_angles]
        for fut in as_completed(futures):
            r = fut.result()
            if not r or r.get("skip"):
                continue
            scan_results.append(r)
            angle = float(r["angle"])
            if (r.get("A") or {}).get("ok"):
                candA = FormDecision(True, "A", angle, float(r["A"]["score"]), {"A": r["A"]["detail"], "phase": "formA_found"})
                if bestA is None or candA.score > bestA.score:
                    bestA = candA

    # Aが見つかった場合は即座にA確定
    if bestA is not None:
        # v13.6 修正:
        # Aが検出できてもスコアが閾値未満なら、ここで Unknown に確定せず、
        # B探索へフォールバックする（BのQRが見えるケースを取りこぼさない）。
        if thr > 0 and bestA.score < thr:
            rejected_by_threshold["A"] = {"score": float(bestA.score), "phase": str((bestA.detail or {}).get("phase") or "formA_found")}
        else:
            return bestA

    # v13.4 追加:
    # test データや強い改悪条件では marker_preproc=basic で取りこぼすことがあるため、
    # A が全滅した場合のみ「morph」を追加で試す（2角度なのでオーバーヘッドは小さい）。
    if str(marker_preproc) != "morph":

        def _eval_formA_morph(angle: float) -> dict[str, Any]:
            rotated = rotate_image_bound(rectified_bgr, angle)
            h, w = rotated.shape[:2]
            if h > w:
                return {"angle": float(angle), "skip": True}
            okA, scoreA, detA = score_formA(rotated, marker_preproc="morph", geom_cfg=formA_geom_cfg)
            return {
                "angle": float(angle),
                "skip": False,
                "A_morph": {"ok": bool(okA), "score": float(scoreA), "detail": detA},
            }

        bestA_morph: Optional[FormDecision] = None
        with ThreadPoolExecutor(max_workers=min(int(max_workers), len(scan_angles))) as ex:
            futures = [ex.submit(_eval_formA_morph, a) for a in scan_angles]
            for fut in as_completed(futures):
                r = fut.result()
                if not r or r.get("skip"):
                    continue

                # scan_results に追記（angle でマージ）
                for sr in scan_results:
                    if abs(sr.get("angle", -999) - r.get("angle", -999)) < 1e-6:
                        sr["A_morph"] = r.get("A_morph")
                        break
                else:
                    scan_results.append(r)

                angle = float(r["angle"])
                if (r.get("A_morph") or {}).get("ok"):
                    candA = FormDecision(
                        True,
                        "A",
                        angle,
                        float(r["A_morph"]["score"]),
                        {"A": r["A_morph"]["detail"], "phase": "formA_found_fallback_morph"},
                    )
                    if bestA_morph is None or candA.score > bestA_morph.score:
                        bestA_morph = candA

        if bestA_morph is not None:
            if thr > 0 and bestA_morph.score < thr:
                rejected_by_threshold["A_morph"] = {
                    "score": float(bestA_morph.score),
                    "phase": str((bestA_morph.detail or {}).get("phase") or "formA_found_fallback_morph"),
                }
            else:
                return bestA_morph

    # ----------------------------------
    # Step 2: Aが見つからない → Bのfast探索
    # ----------------------------------

    def _eval_formB_fast(angle: float) -> dict[str, Any]:
        rotated = rotate_image_bound(rectified_bgr, angle)
        h, w = rotated.shape[:2]
        if h > w:
            return {"angle": float(angle), "skip": True}
        okBf, scoreBf, detBf = score_formB_fast(rotated)
        return {
            "angle": float(angle),
            "skip": False,
            "B_fast": {"ok": bool(okBf), "score": float(scoreBf), "detail": detBf},
        }

    bestB_fast: Optional[FormDecision] = None

    with ThreadPoolExecutor(max_workers=min(int(max_workers), len(scan_angles))) as ex:
        futures = [ex.submit(_eval_formB_fast, a) for a in scan_angles]
        for fut in as_completed(futures):
            r = fut.result()
            if not r or r.get("skip"):
                continue
            # scan_resultsに追加（B_fast情報をマージ）
            for sr in scan_results:
                if abs(sr.get("angle", -999) - r.get("angle", -999)) < 1e-6:
                    sr["B_fast"] = r.get("B_fast")
                    break
            else:
                scan_results.append(r)

            angle = float(r["angle"])
            if (r.get("B_fast") or {}).get("ok"):
                candB = FormDecision(
                    True,
                    "B",
                    angle,
                    float(r["B_fast"]["score"]),
                    {"B_fast": r["B_fast"]["detail"], "phase": "formB_fast_found"},
                )
                if bestB_fast is None or candB.score > bestB_fast.score:
                    bestB_fast = candB

    # fastで見つかればB確定
    if bestB_fast is not None:
        if thr > 0 and bestB_fast.score < thr:
            rejected_by_threshold["B_fast"] = {"score": float(bestB_fast.score), "phase": str((bestB_fast.detail or {}).get("phase") or "formB_fast_found")}
        else:
            return bestB_fast

    # ----------------------------------
    # Step 3: fastで見つからない → robustで再挑戦
    # ----------------------------------

    def _eval_formB_robust(angle: float) -> dict[str, Any]:
        rotated = rotate_image_bound(rectified_bgr, angle)
        h, w = rotated.shape[:2]
        if h > w:
            return {"angle": float(angle), "skip": True}
        okB, scoreB, detB = score_formB(rotated)
        return {
            "angle": float(angle),
            "skip": False,
            "B_robust": {"ok": bool(okB), "score": float(scoreB), "detail": detB},
        }

    bestB_robust: Optional[FormDecision] = None

    with ThreadPoolExecutor(max_workers=min(int(max_workers), len(scan_angles))) as ex:
        futures = [ex.submit(_eval_formB_robust, a) for a in scan_angles]
        for fut in as_completed(futures):
            r = fut.result()
            if not r or r.get("skip"):
                continue
            # scan_resultsに追加（B_robust情報をマージ）
            for sr in scan_results:
                if abs(sr.get("angle", -999) - r.get("angle", -999)) < 1e-6:
                    sr["B_robust"] = r.get("B_robust")
                    break
            else:
                scan_results.append(r)

            angle = float(r["angle"])
            if (r.get("B_robust") or {}).get("ok"):
                candB = FormDecision(
                    True,
                    "B",
                    angle,
                    float(r["B_robust"]["score"]),
                    {"B": r["B_robust"]["detail"], "phase": "formB_robust_fallback"},
                )
                if bestB_robust is None or candB.score > bestB_robust.score:
                    bestB_robust = candB

    # robustで見つかればB確定
    if bestB_robust is not None:
        if thr > 0 and bestB_robust.score < thr:
            rejected_by_threshold["B_robust"] = {
                "score": float(bestB_robust.score),
                "phase": str((bestB_robust.detail or {}).get("phase") or "formB_robust_fallback"),
            }
        else:
            return bestB_robust

    # ----------------------------------
    # Step 4: robustでも見つからなければUnknown
    # ----------------------------------

    # 閾値未満で棄却された候補があるなら、その情報を含めて Unknown を返す。
    if rejected_by_threshold:
        try:
            best_rejected = max((float(v.get("score", 0.0)) for v in rejected_by_threshold.values()), default=0.0)
        except Exception:
            best_rejected = 0.0
        return FormDecision(
            False,
            None,
            None,
            float(best_rejected),
            {
                "reason": "below_threshold",
                "threshold": float(thr),
                "rejected": rejected_by_threshold,
                "scan": scan_results,
                "scan_angles": scan_angles,
                "note": "candidates_found_but_rejected_by_threshold",
            },
        )

    return FormDecision(
        False,
        None,
        None,
        0.0,
        {
            "reason": "no_detection",
            "scan": scan_results,
            "scan_angles": scan_angles,
            "note": "fallback_all_failed",
        },
    )



"""（template-topn / グローバル特徴による事前絞り込み）

v13 ではユーザー要望により「フォーム確定後は全テンプレを XFeat で照合」します。
そのため、旧版にあったグローバル特徴によるテンプレ候補絞り込み機能は削除しました。
（CSVにも template-topn は出さず空欄にしています）
"""


@dataclass
class CachedRef:
    template_path: str
    s_ref: float
    out0: dict[str, Any]


class CachedXFeatMatcher:
    """テンプレ側の特徴をキャッシュして高速化した XFeat マッチャー。"""

    def __init__(self, base: XFeatMatcher):
        self.base = base
        self.xfeat = base.xfeat
        self.top_k = int(base.top_k)
        self.match_max_side = int(base.match_max_side)
        self.device = str(base.device)

    def prepare_ref(self, template_bgr: np.ndarray, template_path: str) -> CachedRef:
        ref_small, s_ref = resize_keep_aspect(template_bgr, self.match_max_side)
        out0 = self.xfeat.detectAndCompute(ref_small, top_k=self.top_k)[0]
        out0.update({"image_size": (ref_small.shape[1], ref_small.shape[0])})
        return CachedRef(template_path=str(template_path), s_ref=float(s_ref), out0=out0)

    def prepare_target(self, tgt_bgr: np.ndarray) -> tuple[dict[str, Any], float, np.ndarray]:
        """ターゲット画像の特徴を前計算する。

        改善2:
          テンプレ6枚などのループ内で target 側の特徴(out1)を再計算しない。

        戻り値:
          (out1, s_tgt, invS_tgt)
        """

        tgt_small, s_tgt = resize_keep_aspect(tgt_bgr, self.match_max_side)
        out1 = self.xfeat.detectAndCompute(tgt_small, top_k=self.top_k)[0]
        out1.update({"image_size": (tgt_small.shape[1], tgt_small.shape[0])})
        invS_tgt = np.linalg.inv(scale_matrix(float(s_tgt)))
        return out1, float(s_tgt), invS_tgt

    def match_with_cached_ref_and_prepared_target(
        self,
        ref: CachedRef,
        *,
        out1: dict[str, Any],
        invS_tgt: np.ndarray,
    ) -> tuple[Any, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """ターゲット特徴(out1)を使い回して照合する。

        (XFeatHomographyResult相当, H_full, mkpts0, mkpts1)

        H_full = invS_tgt @ H_small @ S_ref
        """

        matches = self.xfeat.match_lighterglue(ref.out0, out1)
        if isinstance(matches, (list, tuple)) and len(matches) >= 2:
            mkpts0, mkpts1 = matches[0], matches[1]
        elif isinstance(matches, dict) and "mkpts0" in matches and "mkpts1" in matches:
            mkpts0, mkpts1 = matches["mkpts0"], matches["mkpts1"]
        else:
            return (
                type("Res", (), {"ok": False, "inliers": 0, "matches": 0, "inlier_ratio": 0.0, "H_ref_to_tgt": None})(),
                None,
                None,
                None,
            )

        mkpts0 = np.asarray(mkpts0, dtype=np.float32)
        mkpts1 = np.asarray(mkpts1, dtype=np.float32)
        if len(mkpts0) < 4:
            return (
                type("Res", (), {"ok": False, "inliers": 0, "matches": int(len(mkpts0)), "inlier_ratio": 0.0, "H_ref_to_tgt": None})(),
                None,
                mkpts0,
                mkpts1,
            )

        H_small, mask = cv2.findHomography(
            mkpts0,
            mkpts1,
            cv2.USAC_MAGSAC,
            float(PIPELINE_DEFAULTS["homography"]["find"]["ransac_reproj_threshold_px"]),
            maxIters=int(PIPELINE_DEFAULTS["homography"]["find"]["max_iters"]),
            confidence=float(PIPELINE_DEFAULTS["homography"]["find"]["confidence"]),
        )
        if H_small is None or mask is None:
            return (
                type("Res", (), {"ok": False, "inliers": 0, "matches": int(len(mkpts0)), "inlier_ratio": 0.0, "H_ref_to_tgt": None})(),
                None,
                mkpts0,
                mkpts1,
            )

        mask = mask.reshape(-1).astype(bool)
        inliers = int(mask.sum())
        matches_n = int(len(mask))
        inlier_ratio = float(inliers) / float(matches_n) if matches_n else 0.0

        # inlier の最小二乗でHを微調整し、ワープ品質を改善する
        reproj = None
        if inliers >= 4:
            try:
                H_refined, rms = refine_homography_least_squares(H_small, mkpts0, mkpts1, mask)
                if H_refined is not None:
                    H_small = H_refined
                reproj = rms
            except Exception:
                reproj = None

        S_ref = scale_matrix(float(ref.s_ref))
        H_full = invS_tgt @ H_small @ S_ref

        return (
            type(
                "Res",
                (),
                {
                    "ok": True,
                    "inliers": inliers,
                    "matches": matches_n,
                    "inlier_ratio": float(inlier_ratio),
                    "reproj_rms": reproj,
                    "H_ref_to_tgt": H_full.astype(float).tolist(),
                },
            )(),
            H_full,
            mkpts0,
            mkpts1,
        )

    def match_with_cached_ref(
        self,
        ref: CachedRef,
        tgt_bgr: np.ndarray,
    ) -> tuple[Any, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """(XFeatHomographyResult相当, H_full, mkpts0, mkpts1) を返す。"""

        # 後方互換API: ターゲット特徴を都度計算（※改善2の高速経路は別メソッド）
        out1, _s_tgt, invS_tgt = self.prepare_target(tgt_bgr)
        return self.match_with_cached_ref_and_prepared_target(ref, out1=out1, invS_tgt=invS_tgt)


def select_top_templates(
    target_desc: np.ndarray,
    templates: list[CachedRef],
    top_n: int,
) -> list[CachedRef]:
    # v13 では prefilter を使わないため、互換用に「そのまま返す」だけにする。
    # （この関数自体は本ファイル内では呼ばれない）
    _ = (target_desc, top_n)
    return templates


# ------------------------------------------------------------
# Homography 逆行列化の安全性チェック
# ------------------------------------------------------------


def safe_invert_homography(
    H: np.ndarray,
    inliers: int,
    inlier_ratio: float,
    min_inliers: int,
    min_inlier_ratio: float,
    max_cond: float,
) -> tuple[bool, Optional[np.ndarray], str, float, float]:
    """Homography の逆行列化を安全に行う。

    - inlier 数が少なすぎる場合は却下
    - inlier_ratio が小さすぎる場合は却下
    - 行列が特異に近い場合（detが小さい / condが大きい）は却下
    """

    if int(inliers) < int(min_inliers):
        return False, None, f"inliers<{min_inliers} ({inliers})", float("nan"), float("nan")
    if float(inlier_ratio) < float(min_inlier_ratio):
        return False, None, f"inlier_ratio<{min_inlier_ratio:.3f} ({inlier_ratio:.3f})", float("nan"), float("nan")

    H = np.asarray(H, dtype=np.float64)
    det = float(np.linalg.det(H))
    if not math.isfinite(det) or abs(det) < float(PIPELINE_DEFAULTS["homography"]["invert"]["det_abs_min"]):
        return False, None, f"det too small ({det:.3e})", float("nan"), float(det)
    try:
        cond = float(np.linalg.cond(H))
        if not math.isfinite(cond) or (max_cond > 0 and cond > float(max_cond)):
            return False, None, f"cond too large ({cond:.3e})", float(cond), float(det)
    except Exception:
        # cond 計算が失敗することがあるため、例外は握りつぶして inversion は試す
        cond = float("nan")

    try:
        H_inv = np.linalg.inv(H)
        return True, H_inv, "ok", float(cond), float(det)
    except Exception as e:
        return False, None, f"inv failed: {e}", float(cond), float(det)


# ------------------------------------------------------------
# CSV補助関数
# ------------------------------------------------------------


def _bool_to_str(v: Any) -> str:
    if v is None:
        return ""
    return "TRUE" if bool(v) else "FALSE"


def _to_json_cell(v: Any) -> str:
    """CSVセルに入れるため、複雑なオブジェクトをJSON文字列化する。"""

    if v is None:
        return ""
    try:
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(v)


def _filename_only(p: Any) -> str:
    """パスからファイル名部分だけを返す（ディレクトリは落とす）。"""

    if not p:
        return ""
    try:
        return Path(str(p)).name
    except Exception:
        return str(p)


def _filenames_only_list(v: Any) -> list[str]:
    """パスのリストを、ファイル名のリストへ変換する。"""

    if not v:
        return []
    out: list[str] = []
    try:
        for x in list(v):
            out.append(_filename_only(x))
        return out
    except Exception:
        return []


def _sanitize_template_candidate_results(v: Any) -> Any:
    """template_match_candidates の template パスを filename のみにする（JSONセル用）。"""

    if not v:
        return []
    try:
        out = []
        for d in list(v):
            if not isinstance(d, dict):
                continue
            dd = dict(d)
            if "template" in dd:
                dd["template"] = _filename_only(dd.get("template"))
            out.append(dd)
        return out
    except Exception:
        return v


def _template_number_from_path(p: str) -> str:
    try:
        stem = Path(p).stem
        return stem
    except Exception:
        return ""


def _template_filename_from_path(p: str) -> str:
    try:
        return Path(p).name
    except Exception:
        return ""


def _case_truth(src_form: str, src_path: Path) -> dict[str, Any]:
    """このパイプラインにおける Ground Truth（正解）の定義。

    - src_form が A/B の場合、正解フォームは同じ
    - 正解テンプレは同じファイル名（stem=1..6）
    - src_form が C（またはそれ以外）の場合、正解は未定義（unknown扱い）
    """

    gt_form = src_form if src_form in ("A", "B") else ""
    gt_template = ""
    gt_template_number = ""
    if gt_form:
        gt_template = str(src_path)
        gt_template_number = str(src_path.stem)
    return {
        "ground_truth_source_form(A_or_B)": gt_form,
        "ground_truth_source_template_path(if_A_or_B)": gt_template,
        "ground_truth_source_template_number(if_A_or_B)": gt_template_number,
    }


def _truth_from_item(item: dict[str, Any]) -> dict[str, Any]:
    """item 内の ground truth を優先して返す。

    v13.4:
      - image/test の評価では、入力画像名とテンプレ番号が一致しないため、
        item 側で ground truth を明示して上書きできるようにする。
    """

    gt_form = str(item.get("ground_truth_form") or "")
    gt_tpl_path = str(item.get("ground_truth_template_path") or "")
    gt_tpl_num = str(item.get("ground_truth_template_number") or "")

    if gt_form:
        # A/B/C のいずれでも入れられるが、既存 CSV 互換のためキー名は A_or_B を維持
        return {
            "ground_truth_source_form(A_or_B)": gt_form if gt_form in ("A", "B") else "",
            "ground_truth_source_template_path(if_A_or_B)": gt_tpl_path if gt_form in ("A", "B") else "",
            "ground_truth_source_template_number(if_A_or_B)": gt_tpl_num if gt_form in ("A", "B") else "",
        }

    # 既存データセット（image/A,B,C）
    src_form = str(item.get("source_form") or "")
    src_path_s = str(item.get("source_path") or "")
    return _case_truth(src_form, Path(src_path_s) if src_path_s else Path(""))


def build_csv_row(
    *,
    args: argparse.Namespace,
    item: dict[str, Any],
    times: "StageTimes",
) -> dict[str, Any]:
    """解析しやすいCSV行を構築する。

    カラム名は「誰が見ても分かる」ことを優先して冗長にしている。
    """

    dec = (item.get("form_decision") or {})
    best = (item.get("best_match") or {})
    degrade = (item.get("degrade") or {})
    poly_margin = (item.get("polygon_margin") or {})
    inv = (item.get("homography_inv") or {})
    xfeat_best = (item.get("xfeat_best") or {})

    predicted_form = str(dec.get("form") or "")
    predicted_angle = dec.get("angle_deg")
    best_template_path = str(best.get("template") or "")
    best_template_filename = _template_filename_from_path(best_template_path)
    best_template_number = _template_number_from_path(best_template_path)

    src_form = str(item.get("source_form") or "")
    src_path = Path(str(item.get("source_path") or ""))
    truth = _truth_from_item(item)
    gt_form = str(truth["ground_truth_source_form(A_or_B)"])
    gt_template_path = str(truth["ground_truth_source_template_path(if_A_or_B)"])
    gt_template_filename = _template_filename_from_path(gt_template_path)
    gt_template_number = str(truth["ground_truth_source_template_number(if_A_or_B)"])

    is_form_correct = bool(gt_form) and (predicted_form == gt_form)
    is_template_correct = False
    if bool(gt_form) and gt_template_path and best_template_path:
        try:
            is_template_correct = (Path(best_template_path).name == Path(gt_template_path).name) and (predicted_form == gt_form)
        except Exception:
            is_template_correct = False

    # 注意: CSV は「フルパス禁止」の要望に従い、原則 filename のみ出力する。
    src_filename = _filename_only(item.get("source_path"))

    source_dataset = str(item.get("source_dataset") or "synthetic")

    expected_behavior_label = ""
    if src_form == "C":
        expected_behavior_label = "C_should_be_rejected_as_form_unknown"
    elif src_form in ("A", "B"):
        expected_behavior_label = "A_or_B_should_be_correct_form_and_template_and_warp"
    else:
        expected_behavior_label = "unknown_source_form"

    # form_unknown の理由を独立カラムで出す（ログ/CSVで追えるように）
    form_unknown_reason, form_unknown_diag = extract_form_unknown_reason(dec)

    row: dict[str, Any] = {
        # ---- 識別情報（短く・人間向け） ----
        "case_id": str(item.get("case") or ""),
        "source_dataset_name(synthetic_or_test)": source_dataset,
        "source_form_folder_name(A_or_B_or_C)": src_form,
        "source_image_filename": src_filename,
        "source_image_filename_stem": str(src_path.stem) if src_path else "",
        "degraded_variant_index": str(item.get("degraded_variant_index") or ""),

        # ---- 正解ラベル（A/Bのみ） ----
        "ground_truth_source_form(A_or_B)": gt_form,
        "ground_truth_source_template_filename(if_A_or_B)": gt_template_filename,
        "ground_truth_source_template_number(if_A_or_B)": gt_template_number,

        # ---- 予測 ----
        "predicted_decided_form(A_or_B_or_empty)": predicted_form,
        "predicted_decided_rotation_angle_deg": "" if predicted_angle is None else str(predicted_angle),
        "predicted_best_template_filename": best_template_filename,
        "predicted_best_template_number": best_template_number,

        # ---- 正誤（A/Bのみ） ----
        "is_predicted_form_correct": _bool_to_str(is_form_correct) if gt_form else "",
        "is_predicted_best_template_correct": _bool_to_str(is_template_correct) if gt_form else "",

        # ---- パイプライン状態 ----
        "pipeline_final_ok(warp_done)": _bool_to_str(item.get("ok_warp")),
        "pipeline_final_ok(expected_behavior)": _bool_to_str(item.get("ok")),
        "pipeline_stop_stage": str(item.get("stage") or ""),
        "pipeline_expected_behavior_label": expected_behavior_label,
        "pipeline_predicted_form_raw(A_or_B_or_empty)": str(item.get("predicted_form") or ""),

        # ---- form_unknown の理由（独立カラム） ----
        "form_unknown_reason": form_unknown_reason,
        "form_unknown_diagnostics_json": _to_json_cell(form_unknown_diag),

        # ---- 所要時間 ----
        "elapsed_time_total_one_case_seconds": f"{float(item.get('case_total_s', 0.0)):.6f}",
        "elapsed_time_stage_1_degrade_seconds": f"{times.degrade_s:.6f}",
        "elapsed_time_stage_2_docaligner_seconds": f"{times.docaligner_s:.6f}",
        "elapsed_time_stage_3_rectify_seconds": f"{times.rectify_s:.6f}",
        "elapsed_time_stage_4_form_decision_seconds": f"{times.decide_s:.6f}",
        "elapsed_time_stage_5_uvdoc_unwarp_seconds": f"{times.uvdoc_s:.6f}",
        "elapsed_time_stage_6_background_division_seconds": f"{times.bgdiv_s:.6f}",
        "elapsed_time_stage_7_xfeat_matching_seconds": f"{times.match_s:.6f}",
        "elapsed_time_stage_8_warp_seconds": f"{times.warp_s:.6f}",

        # ---- 実行メタ情報（フルパスなし） ----
        "run_id": str(item.get("run_id") or ""),
        "run_output_root_directory_name": _filename_only(item.get("run_output_root_directory")),
        "run_elapsed_time_total_seconds": str(item.get("run_elapsed_time_total_seconds") or ""),

        # ---- 出力ファイル名（フルパスなし） ----
        "output_degraded_image_filename": _filename_only(item.get("output_degraded_image_path")),
        "output_doc_overlay_image_filename": _filename_only(item.get("output_doc_overlay_image_path")),
        "output_rectified_image_filename": _filename_only(item.get("output_rectified_image_path")),
        "output_rotated_decision_visualization_image_filename": _filename_only(item.get("output_rotated_decision_visualization_image_path")),
        "output_uvdoc_unwarped_image_filename": _filename_only(item.get("output_uvdoc_unwarped_image_path")),
        "output_background_division_image_filename": _filename_only(item.get("output_background_division_image_path")),
        "output_debug_matches_image_filename": _filename_only(item.get("output_debug_matches_image_path")),
        "output_aligned_image_filename": _filename_only(item.get("output_aligned_image_path")),

        # ---- 画像サイズ（解像度） ----
        "source_image_resolution_width_px": str(item.get("source_w") or ""),
        "source_image_resolution_height_px": str(item.get("source_h") or ""),
        "degraded_image_resolution_width_px": str(item.get("degraded_w") or ""),
        "degraded_image_resolution_height_px": str(item.get("degraded_h") or ""),
        "rectified_paper_image_resolution_width_px": str(item.get("rectified_w") or ""),
        "rectified_paper_image_resolution_height_px": str(item.get("rectified_h") or ""),
        "rectified_rotated_for_decision_image_resolution_width_px": str(item.get("chosen_w") or ""),
        "rectified_rotated_for_decision_image_resolution_height_px": str(item.get("chosen_h") or ""),
        "uvdoc_unwarped_image_resolution_width_px": str(item.get("uvdoc_w") or ""),
        "uvdoc_unwarped_image_resolution_height_px": str(item.get("uvdoc_h") or ""),
        "background_division_image_resolution_width_px": str(item.get("bgdiv_w") or ""),
        "background_division_image_resolution_height_px": str(item.get("bgdiv_h") or ""),
        "best_template_resolution_width_px": str(item.get("best_template_w") or ""),
        "best_template_resolution_height_px": str(item.get("best_template_h") or ""),
        "aligned_output_resolution_width_px": str(item.get("aligned_w") or ""),
        "aligned_output_resolution_height_px": str(item.get("aligned_h") or ""),

        # ---- 改悪パラメータ（詳細） ----
        "degradation_generated_rotation_angle_deg": str(degrade.get("angle_deg") or ""),
        "degradation_rotation_mode(uniform_or_snap)": str(degrade.get("rotation_mode") or ""),
        "degradation_snap_step_deg": str(degrade.get("snap_step_deg") or ""),
        "degradation_output_canvas_width_px": str(degrade.get("out_w") or ""),
        "degradation_output_canvas_height_px": str(degrade.get("out_h") or ""),
        "degradation_perspective_jitter_strength": str(degrade.get("perspective_jitter") or ""),
        "degradation_visible_area_min_ratio": str(degrade.get("min_visible_area_ratio") or ""),
        "degradation_generator_max_attempts": str(degrade.get("max_attempts") or ""),
        "degradation_template_projected_base_width_px": str(degrade.get("base_w") or ""),
        "degradation_template_projected_base_height_px": str(degrade.get("base_h") or ""),
        "degradation_parameters_json": _to_json_cell(degrade),

        # ---- 紙領域検出（DocAligner） ----
        "docaligner_polygon_xy_json": _to_json_cell(item.get("polygon")),
        "docaligner_polygon_margin_mode(ratio_or_fixed_px)": str(poly_margin.get("mode") or ""),
        "docaligner_polygon_margin_computed_px": str(poly_margin.get("computed_px") or poly_margin.get("value") or ""),
        "docaligner_polygon_margin_details_json": _to_json_cell(poly_margin),

        # ---- フォーム判定のデバッグ ----
        "form_decision_score": str(dec.get("score") or ""),
        "form_decision_detail_json": _to_json_cell(dec.get("detail")),

        # ---- XFeat 最良マッチ ----
        "xfeat_best_inliers": str(best.get("inliers") or ""),
        "xfeat_best_matches": str(best.get("matches") or ""),
        "xfeat_best_inlier_ratio": str(best.get("inlier_ratio") or ""),
        "xfeat_best_ref_keypoints_count": str(xfeat_best.get("ref_kpts") or ""),
        "xfeat_best_tgt_keypoints_count": str(xfeat_best.get("tgt_kpts") or ""),
        "xfeat_best_reprojection_rms_px": str(xfeat_best.get("reproj_rms") or ""),
        "xfeat_match_ref_resized_scale": str(xfeat_best.get("s_ref") or ""),
        "xfeat_match_tgt_resized_scale": str(xfeat_best.get("s_tgt") or ""),
        "xfeat_match_ref_resized_resolution_width_px": str(xfeat_best.get("ref_small_w") or ""),
        "xfeat_match_ref_resized_resolution_height_px": str(xfeat_best.get("ref_small_h") or ""),
        "xfeat_match_tgt_resized_resolution_width_px": str(xfeat_best.get("tgt_small_w") or ""),
        "xfeat_match_tgt_resized_resolution_height_px": str(xfeat_best.get("tgt_small_h") or ""),
        "xfeat_template_prefilter_candidate_filenames_json": _to_json_cell(
            _filenames_only_list((item.get("template_prefilter") or {}).get("candidates"))
        ),
        "xfeat_all_template_candidate_results_json": _to_json_cell(_sanitize_template_candidate_results(item.get("template_match_candidates"))),

        # ---- homography の安定性 ----
        "homography_inversion_ok": _bool_to_str(inv.get("ok")),
        "homography_inversion_reject_reason": str(inv.get("reason") or ""),
        "homography_matrix_condition_number": str(inv.get("cond") or ""),
        "homography_matrix_determinant": str(inv.get("det") or ""),

        # ---- 実行設定（主要なものだけ抜粋） ----
        "run_config_rotation_step_deg": str(getattr(args, "rotation_step", "")),
        # v13 では template-topn は廃止（常に全テンプレ照合）
        "run_config_template_topn": "",
        "run_config_xfeat_top_k": str(getattr(args, "top_k", "")),
        "run_config_xfeat_match_max_side_px": str(getattr(args, "match_max_side", "")),
        "run_config_marker_preproc": str(getattr(args, "marker_preproc", "")),
        "run_config_unknown_score_threshold": str(getattr(args, "unknown_score_threshold", "")),
        "run_config_unknown_margin": str(getattr(args, "unknown_margin", "")),
        "run_config_docaligner_model": str(getattr(args, "docaligner_model", "")),
        "run_config_docaligner_type": str(getattr(args, "docaligner_type", "")),
        "run_config_docaligner_max_side_px": str(getattr(args, "docaligner_max_side", "")),
        "run_config_polygon_margin_ratio": str(getattr(args, "polygon_margin_ratio", "")),
        "run_config_polygon_margin_min_px": str(getattr(args, "polygon_margin_min_px", "")),
        "run_config_polygon_margin_max_px": str(getattr(args, "polygon_margin_max_px", "")),
        "run_config_polygon_margin_fixed_px": str(getattr(args, "polygon_margin_px", "")),
        "run_config_degrade_w": str(getattr(args, "degrade_w", "")),
        "run_config_degrade_h": str(getattr(args, "degrade_h", "")),
        "run_config_degrade_max_rot": str(getattr(args, "max_rot", "")),
        "run_config_degrade_min_abs_rot": str(getattr(args, "min_abs_rot", "")),
        "run_config_degrade_perspective": str(getattr(args, "perspective", "")),
        "run_config_degrade_rotation_mode": str(getattr(args, "rotation_mode", "")),
        "run_config_degrade_snap_step_deg": str(getattr(args, "snap_step_deg", "")),
        "run_config_seed": str(getattr(args, "seed", "")),
    }

    # 例外情報（例外が発生した場合のみ）
    if item.get("stage") == "exception":
        row["exception_error_message"] = str(item.get("error") or "")
        row["exception_traceback"] = str(item.get("traceback") or "")
    else:
        row["exception_error_message"] = ""
        row["exception_traceback"] = ""

    return row


# ------------------------------------------------------------
# 入出力（IO）補助
# ------------------------------------------------------------


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(form: str) -> list[Path]:
    base = Path(__file__).resolve().parent / "image" / form
    paths: list[Path] = []
    # 1.jpg〜6.jpg を対象（必要なら PIPELINE_DEFAULTS 側で変更）
    nums = list(PIPELINE_DEFAULTS.get("template_numbers", [1, 2, 3, 4, 5, 6]))
    for i in nums:
        p = base / f"{i}.jpg"
        if p.exists():
            paths.append(p)
    return paths


def list_test_images() -> list[Path]:
    """image/test 配下の画像（A_3.png 等）を列挙する。"""

    base = Path(__file__).resolve().parent / "image" / "test"
    if not base.exists():
        return []

    exts = {".png", ".jpg", ".jpeg"}
    paths = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


def parse_test_filename(p: Path) -> Optional[tuple[str, str]]:
    """test 画像ファイル名から (form, template_number) を推定する。

    規則: {A|B|C}_{number}.(png|jpg)
      例: A_3.png -> ("A", "3")
    """

    try:
        stem = p.stem
        if "_" not in stem:
            return None
        head, num = stem.split("_", 1)
        head = head.strip().upper()
        num = num.strip()
        if head not in ("A", "B", "C"):
            return None
        if not num.isdigit():
            return None
        return head, num
    except Exception:
        return None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument(
        "--explain",
        action="store_true",
        help="主要パラメータの意味（日本語）を出力して終了します",
    )

    # ----------------------------
    # 入力/件数
    # ----------------------------
    p.add_argument(
        "--src-forms",
        type=str,
        default=",".join(PIPELINE_DEFAULTS["src_forms"]),
        help="入力元フォーム（A,B,C をカンマ区切り）",
    )
    p.add_argument(
        "--degrade-n",
        type=int,
        default=int(PIPELINE_DEFAULTS["degrade"]["n"]),
        help="1枚の入力から改悪画像を何枚作るか",
    )
    p.add_argument("--degrade-w", type=int, default=int(PIPELINE_DEFAULTS["degrade"]["out_size_wh"][0]))
    p.add_argument("--degrade-h", type=int, default=int(PIPELINE_DEFAULTS["degrade"]["out_size_wh"][1]))
    p.add_argument(
        "--max-rot",
        type=float,
        default=float(PIPELINE_DEFAULTS["degrade"]["max_rot_deg"]),
        help="改悪生成の回転強度（>=180で0..360一様回転モード）",
    )
    p.add_argument("--min-abs-rot", type=float, default=float(PIPELINE_DEFAULTS["degrade"]["min_abs_rot_deg"]))
    p.add_argument(
        "--rotation-mode",
        choices=["uniform", "snap"],
        default=str(PIPELINE_DEFAULTS["degrade"]["rotation_mode"]),
    )
    p.add_argument("--snap-step-deg", type=float, default=float(PIPELINE_DEFAULTS["degrade"]["snap_step_deg"]))
    p.add_argument("--perspective", type=float, default=float(PIPELINE_DEFAULTS["degrade"]["perspective_jitter"]))
    p.add_argument("--min-visible-area-ratio", type=float, default=float(PIPELINE_DEFAULTS["degrade"]["min_visible_area_ratio"]))
    p.add_argument("--max-attempts", type=int, default=int(PIPELINE_DEFAULTS["degrade"]["max_attempts"]))
    p.add_argument("--seed", type=int, default=int(PIPELINE_DEFAULTS["degrade"]["seed"]))

    # WeChat QRモデル
    p.add_argument(
        "--wechat-model-dir",
        type=str,
        default=str(PIPELINE_DEFAULTS["wechat"]["model_dir"]),
        help="WeChat QRCode Engine のモデルディレクトリ（detect/sr の prototxt/caffemodel を配置）",
    )

    # 回転スキャン: rectify 後は横長に統一されるため、0/180 のみ固定で評価する。
    # （回転ステップ探索は行わない）
    p.add_argument(
        "--rotation-max-workers",
        type=int,
        default=int(PIPELINE_DEFAULTS["rotation_scan"]["max_workers"]),
        help="回転スキャンの並列数（スレッド）",
    )

    p.add_argument(
        "--docaligner-model",
        choices=["lcnet050", "lcnet100", "fastvit_t8", "fastvit_sa24"],
        default=str(PIPELINE_DEFAULTS["docaligner"]["model"]),
    )
    p.add_argument(
        "--docaligner-type",
        choices=["point", "heatmap"],
        default=str(PIPELINE_DEFAULTS["docaligner"]["type"]),
    )
    # 透視補正後の紙画像が小さすぎると QR が潰れて検出しづらいので、デフォルトは少し大きめ。
    p.add_argument(
        "--docaligner-max-side",
        type=int,
        default=int(PIPELINE_DEFAULTS["docaligner"]["rectified_max_side_px"]),
        help="透視補正後の紙画像の最大辺(px)",
    )
    # (1) polygon margin: 解像度差に強い ratio ベース
    p.add_argument(
        "--polygon-margin-ratio",
        type=float,
        default=float(PIPELINE_DEFAULTS["docaligner"]["polygon_margin"]["ratio"]),
        help=(
            "DocAligner polygon を外側に広げるマージン（紙サイズに対する比率）。"
            " 例: 0.03 は紙の長辺の 3%% をマージンにする。"
        ),
    )
    p.add_argument(
        "--polygon-margin-min-px",
        type=float,
        default=float(PIPELINE_DEFAULTS["docaligner"]["polygon_margin"]["min_px"]),
        help="ratio-based マージンの下限(px)",
    )
    p.add_argument(
        "--polygon-margin-max-px",
        type=float,
        default=float(PIPELINE_DEFAULTS["docaligner"]["polygon_margin"]["max_px"]),
        help="ratio-based マージンの上限(px)（0以下で無制限）",
    )
    p.add_argument(
        "--polygon-margin-px",
        type=float,
        default=float(PIPELINE_DEFAULTS["docaligner"]["polygon_margin"]["fixed_px"]),
        help="互換用: 固定pxマージン（>0 の場合 ratio を上書き）",
    )

    # (2) ログ
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    p.add_argument("--console-log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    # (B-1) デバッグ画像保存の抑制
    p.add_argument(
        "--save-images",
        choices=["none", "fail", "all"],
        default=str((PIPELINE_DEFAULTS.get("save_images") or {}).get("mode") or "all"),
        help=(
            "デバッグ画像の保存モード。"
            " none=一切保存しない（FPS測定用） / fail=stage!=done の時だけ保存 / all=常時保存"
        ),
    )

    # (3) 追加の前処理
    p.add_argument(
        "--marker-preproc",
        choices=["none", "basic", "morph"],
        default=str(PIPELINE_DEFAULTS["marker"]["preproc_mode"]),
        help="フォームAマーカー検出の前処理（照明ムラ対策）",
    )

    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=str(PIPELINE_DEFAULTS["xfeat"]["device_default"]),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=int(PIPELINE_DEFAULTS["xfeat"]["top_k"]),
        help="XFeatの特徴点数（大きいほど高精度だが遅い）",
    )
    p.add_argument(
        "--match-max-side",
        type=int,
        default=int(PIPELINE_DEFAULTS["xfeat"]["match_max_side_px"]),
        help="XFeat用にリサイズする最大辺(px)（大きいほど高精度だが遅い）",
    )

    # (6) Unknown 判定しきい値
    p.add_argument(
        "--unknown-score-threshold",
        type=float,
        default=float(PIPELINE_DEFAULTS["unknown"]["score_threshold"]),
        help="フォーム判定スコアがこの値未満なら Unknown 扱い",
    )
    p.add_argument(
        "--unknown-margin",
        type=float,
        default=float(PIPELINE_DEFAULTS["unknown"]["margin"]),
        help="A/B スコア差がこの値未満なら Unknown 扱い（曖昧）",
    )

    # (7) ホモグラフィ安定性
    p.add_argument(
        "--min-inliers-for-warp",
        type=int,
        default=int(PIPELINE_DEFAULTS["warp"]["min_inliers"]),
        help="warp を許可する最小 inlier 数",
    )
    p.add_argument(
        "--min-inlier-ratio-for-warp",
        type=float,
        default=float(PIPELINE_DEFAULTS["warp"]["min_inlier_ratio"]),
        help="warp を許可する最小 inlier_ratio",
    )
    p.add_argument(
        "--max-h-cond",
        type=float,
        default=float(PIPELINE_DEFAULTS["warp"]["max_h_cond"]),
        help="Homography 行列の条件数上限（大きいと不安定）",
    )

    p.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "output_pipeline"))
    p.add_argument(
        "--limit",
        type=int,
        default=int(PIPELINE_DEFAULTS["limit"]),
        help="デバッグ用：各フォームで先頭N枚だけ処理（0=全て）",
    )

    return p.parse_args(argv)


def print_explain() -> None:
    """主要パラメータの意味をまとめて表示する（README代替としても使える）。"""

    # デフォルト値も同時に表示したいので、引数なしで parse した値を参照する
    defaults = parse_args([])

    lines = [
        "=" * 70,
        "paper_pipeline パラメータ説明（要点）",
        "=" * 70,
        "",
        "【入力/件数】",
        f"  --src-forms          入力元フォーム（A,B,C をカンマ区切り） [default: {defaults.src_forms}]",
        f"  --limit              デバッグ用：各フォームで先頭N枚だけ処理（0=全て） [default: {defaults.limit}]",
        f"  --degrade-n           1枚の入力から改悪画像を何枚作るか [default: {defaults.degrade_n}]",
        "",
        "【改悪生成（difficulty調整）】",
        f"  --max-rot             改悪生成の回転強度（>=180で0..360一様回転モード） [default: {defaults.max_rot}]",
        f"  --perspective         射影ゆがみ量（大きいほど難しい） [default: {defaults.perspective}]",
        f"  --degrade-w/--degrade-h  改悪画像の出力サイズ [default: {defaults.degrade_w}x{defaults.degrade_h}]",
        "",
        "【DocAligner】",
        f"  --docaligner-model    使用モデル（精度/速度のトレードオフ） (lcnet050/lcnet100/fastvit_t8/fastvit_sa24) [default: {defaults.docaligner_model}]",
        f"  --docaligner-type     推論タイプ (point/heatmap) [default: {defaults.docaligner_type}]",
        f"  --docaligner-max-side 透視補正後の紙画像の最大辺(px) [default: {defaults.docaligner_max_side}]",
        "  --polygon-margin-ratio 紙サイズ比で polygon を外側に広げる（解像度差に強い）",
        f"    default: {defaults.polygon_margin_ratio} (min={defaults.polygon_margin_min_px}px, max={defaults.polygon_margin_max_px}px)",
        f"  --polygon-margin-px    固定pxで polygon を外側に広げる（>0で ratio を上書き） [default: {defaults.polygon_margin_px}]",
        "",
        "【フォーム判定】",
        "  (回転探索) rectify 後は横長に統一されるため、0度/180度のみでフォーム判定・角度確定します（回転ステップ探索は行いません）",
        f"  --rotation-max-workers 回転スキャンの並列数（スレッド） [default: {defaults.rotation_max_workers}]",
        f"  --rotation-mode       改悪生成の回転モード (uniform/snap) [default: {defaults.rotation_mode}]",
        f"  --marker-preproc      フォームAマーカー前処理 (none/basic/morph) [default: {defaults.marker_preproc}]",
        f"  --unknown-score-threshold スコアが低ければ Unknown 扱い [default: {defaults.unknown_score_threshold}]",
        f"  --unknown-margin      A/B のスコア差が小さければ Unknown 扱い [default: {defaults.unknown_margin}]",
        "",
        "【XFeat（位置合わせ）】",
        f"  --device              XFeatの実行デバイス (auto/cpu/cuda) [default: {defaults.device}]",
        f"  --top-k               特徴点数（大きいほど高精度だが遅い） [default: {defaults.top_k}]",
        f"  --match-max-side      マッチング前にリサイズする最大辺(px)（大きいほど高精度だが遅い） [default: {defaults.match_max_side}]",
        "  (注) v13 ではテンプレ候補絞り込み（template-topn）は廃止し、常に全テンプレ照合します。",
        "",
        "【ログ】",
        f"  --log-level           ログレベル (DEBUG/INFO/WARNING/ERROR) [default: {defaults.log_level}]",
        f"  --console-log-level   コンソールログレベル (DEBUG/INFO/WARNING/ERROR) [default: {defaults.console_log_level}]",
        f"  --save-images         デバッグ画像保存 (none/fail/all) [default: {defaults.save_images}]",
        "",
        "【出力】",
        f"  --out                 出力ディレクトリ（run_... が作成される） [default: {defaults.out}]",
        "",
        "最小コマンド例（おすすめデフォルト使用）:",
        r"  C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v13.py --limit 1",
        "",
    ]
    print("\n".join(lines))


def log_case_summary(logger: logging.Logger, row: dict[str, Any]) -> None:
    """可読性のため、各ケースのサマリを必ず1行でログ出力する。"""

    case_id = str(row.get("case_id") or "")
    # ユーザー向けの ok = 期待動作として成功したか
    ok = str(row.get("pipeline_final_ok(expected_behavior)") or "")
    ok_warp = str(row.get("pipeline_final_ok(warp_done)") or "")
    stage = str(row.get("pipeline_stop_stage") or "")
    src = str(row.get("source_image_filename") or "")

    gt_form = str(row.get("ground_truth_source_form(A_or_B)") or "")
    pred_form = str(row.get("predicted_decided_form(A_or_B_or_empty)") or "")

    form_ok = str(row.get("is_predicted_form_correct") or "")
    template_ok = str(row.get("is_predicted_best_template_correct") or "")

    best_tpl_name = str(row.get("predicted_best_template_filename") or "")
    inliers = str(row.get("xfeat_best_inliers") or "")
    inlier_ratio = str(row.get("xfeat_best_inlier_ratio") or "")

    # Unknown reason（stage=form_unknown のときに特に重要）
    unknown_reason = str(row.get("form_unknown_reason") or "")

    t_total = str(row.get("elapsed_time_total_one_case_seconds") or "")
    t1 = str(row.get("elapsed_time_stage_1_degrade_seconds") or "")
    t2 = str(row.get("elapsed_time_stage_2_docaligner_seconds") or "")
    t3 = str(row.get("elapsed_time_stage_3_rectify_seconds") or "")
    t4 = str(row.get("elapsed_time_stage_4_form_decision_seconds") or "")
    t5 = str(row.get("elapsed_time_stage_5_uvdoc_unwarp_seconds") or "")
    t6 = str(row.get("elapsed_time_stage_6_background_division_seconds") or "")
    t7 = str(row.get("elapsed_time_stage_7_xfeat_matching_seconds") or "")
    t8 = str(row.get("elapsed_time_stage_8_warp_seconds") or "")

    # Ground truth が無い場合（例: C）は、正誤カラムは空欄にする
    truth_part = f"gt_form={gt_form} pred_form={pred_form}"
    if gt_form:
        truth_part += f" form_ok={form_ok} template_ok={template_ok}"

    msg = (
        f"[CASE] id={case_id} ok={ok} ok_warp={ok_warp} stage={stage} "
        f"unknown_reason={unknown_reason} {truth_part} "
        f"best_template={best_tpl_name} inliers={inliers} inlier_ratio={inlier_ratio} "
        f"time_total_s={t_total} (1_degrade={t1},2_doc={t2},3_rectify={t3},4_decide={t4},5_uvdoc={t5},6_bgdiv={t6},7_match={t7},8_warp={t8}) "
        f"src={src}"
    )

    if ok == "TRUE":
        logger.info(msg)
    else:
        # 失敗は後段の解析で重要なので warning にする
        logger.warning(msg)


def _safe_div(n: float, d: float) -> float:
    if d == 0:
        return float("nan")
    return float(n) / float(d)


def _mean(xs: list[float]) -> float:
    xs2 = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs2:
        return float("nan")
    return float(sum(xs2) / len(xs2))


def _median(xs: list[float]) -> float:
    xs2 = sorted([float(x) for x in xs if x is not None and math.isfinite(float(x))])
    if not xs2:
        return float("nan")
    m = len(xs2) // 2
    if len(xs2) % 2 == 1:
        return float(xs2[m])
    return float((xs2[m - 1] + xs2[m]) / 2.0)


def summarize_results(logger: logging.Logger, summary: list[dict[str, Any]], stage_times: dict[str, float], dt_total: float) -> None:
    """ログ末尾に、データセット全体の統計を出力する。

    主な集計観点:
      - expected-behavior 成功率（ユーザー要望の主KPI）
      - A/B のフォーム＋テンプレ正解率
      - C の棄却成功率（stage=form_unknown になるべき）
      - 誤検出分析（CがA/Bに誤判定された回数）
      - ステージ別時間（平均/中央値）
    """

    total = len(summary)
    if total == 0:
        logger.info("[STATS] no cases")
        return

    ok_warp = sum(1 for s in summary if bool(s.get("ok_warp")))
    ok_expected = sum(1 for s in summary if bool(s.get("ok")))

    # 入力フォーム別に集計
    by_src: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": [], "other": []}
    for s in summary:
        sf = str(s.get("source_form") or "")
        if sf in by_src:
            by_src[sf].append(s)
        else:
            by_src["other"].append(s)

    # dataset(test/synthetic) 別にも集計
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for s in summary:
        ds = str(s.get("source_dataset") or "synthetic")
        by_dataset.setdefault(ds, []).append(s)

    # A/B 正解数（フォーム正解・テンプレ正解）
    def _count_true(items: list[dict[str, Any]], key: str) -> int:
        return sum(1 for it in items if bool(it.get(key)))

    a_items = by_src["A"]
    b_items = by_src["B"]
    c_items = by_src["C"]

    a_form_ok = _count_true(a_items, "is_predicted_form_correct")
    b_form_ok = _count_true(b_items, "is_predicted_form_correct")
    a_tpl_ok = _count_true(a_items, "is_predicted_best_template_correct")
    b_tpl_ok = _count_true(b_items, "is_predicted_best_template_correct")

    # C は form_unknown で棄却されるべき
    c_reject_ok = sum(1 for it in c_items if str(it.get("stage")) == "form_unknown")
    c_fp_as_A = sum(1 for it in c_items if str(it.get("predicted_form") or "") == "A")
    c_fp_as_B = sum(1 for it in c_items if str(it.get("predicted_form") or "") == "B")

    # ケース別の処理時間（mean/median）
    t_total_cases = [float(s.get("case_total_s", 0.0)) for s in summary if s.get("case_total_s") is not None]
    t1 = [float(s.get("stage_times", {}).get("degrade_s", 0.0)) for s in summary if isinstance(s.get("stage_times"), dict)]
    # stage_times が埋まっていない場合は、集計値/総数から推定する

    logger.info("=" * 70)
    logger.info("[STATS] overall")
    logger.info("  total_cases                       : %d", total)
    logger.info("  ok_warp(done_aligned_generated)    : %d (%.1f%%)", ok_warp, _safe_div(ok_warp * 100.0, total))
    logger.info("  ok_expected_behavior(user_KPI)     : %d (%.1f%%)", ok_expected, _safe_div(ok_expected * 100.0, total))
    logger.info("  run_elapsed_total_seconds          : %.3f", float(dt_total))
    logger.info("  avg_elapsed_per_case_seconds       : %.3f", float(dt_total) / float(total))

    logger.info("[STATS] A form")
    logger.info("  cases                             : %d", len(a_items))
    logger.info("  form_accuracy                      : %d (%.1f%%)", a_form_ok, _safe_div(a_form_ok * 100.0, len(a_items)))
    logger.info("  template_accuracy                  : %d (%.1f%%)", a_tpl_ok, _safe_div(a_tpl_ok * 100.0, len(a_items)))
    logger.info("[STATS] B form")
    logger.info("  cases                             : %d", len(b_items))
    logger.info("  form_accuracy                      : %d (%.1f%%)", b_form_ok, _safe_div(b_form_ok * 100.0, len(b_items)))
    logger.info("  template_accuracy                  : %d (%.1f%%)", b_tpl_ok, _safe_div(b_tpl_ok * 100.0, len(b_items)))
    logger.info("[STATS] C form (should be rejected)")
    logger.info("  cases                             : %d", len(c_items))
    logger.info("  reject_success(stage=form_unknown) : %d (%.1f%%)", c_reject_ok, _safe_div(c_reject_ok * 100.0, len(c_items)))
    logger.info("  false_positive_as_A                : %d (%.1f%%)", c_fp_as_A, _safe_div(c_fp_as_A * 100.0, len(c_items)))
    logger.info("  false_positive_as_B                : %d (%.1f%%)", c_fp_as_B, _safe_div(c_fp_as_B * 100.0, len(c_items)))

    # test データセット（image/test）精度
    if "test" in by_dataset:
        test_items = by_dataset["test"]
        test_a = [it for it in test_items if str(it.get("source_form") or "") == "A"]
        test_b = [it for it in test_items if str(it.get("source_form") or "") == "B"]
        test_c = [it for it in test_items if str(it.get("source_form") or "") == "C"]

        logger.info("[STATS] test dataset (image/test)")
        logger.info("  total                             : %d", len(test_items))
        logger.info("  A cases                            : %d", len(test_a))
        logger.info(
            "  A template_accuracy                 : %d (%.1f%%)",
            _count_true(test_a, "is_predicted_best_template_correct"),
            _safe_div(_count_true(test_a, "is_predicted_best_template_correct") * 100.0, len(test_a)),
        )
        logger.info("  B cases                            : %d", len(test_b))
        logger.info(
            "  B template_accuracy                 : %d (%.1f%%)",
            _count_true(test_b, "is_predicted_best_template_correct"),
            _safe_div(_count_true(test_b, "is_predicted_best_template_correct") * 100.0, len(test_b)),
        )
        logger.info("  C cases                            : %d", len(test_c))
        logger.info(
            "  C reject_success(stage=form_unknown): %d (%.1f%%)",
            sum(1 for it in test_c if str(it.get("stage")) == "form_unknown"),
            _safe_div(sum(1 for it in test_c if str(it.get("stage")) == "form_unknown") * 100.0, len(test_c)),
        )

    # ステージ別の合計時間
    logger.info("[STATS] stage time totals (s) (same as SUMMARY)")
    for k, v in stage_times.items():
        logger.info("  %-12s : %.2f", k, float(v))

    # 合計から平均を算出
    logger.info("[STATS] stage time mean per case (s)")
    for k, v in stage_times.items():
        logger.info("  %-12s : %.3f", k, float(v) / float(total))

    # 1ケース当たりの総時間（平均/中央値）
    logger.info("[STATS] per-case total time (s)")
    logger.info("  mean  : %.3f", _mean(t_total_cases))
    logger.info("  median: %.3f", _median(t_total_cases))


def print_config(args: argparse.Namespace) -> None:
    """起動時に主要設定を一覧表示する（引数が多い問題への対策）。"""

    print("[CONFIG]")
    print(f"  src-forms          : {args.src_forms}")
    print(f"  limit              : {args.limit}")
    print(f"  degrade-n           : {args.degrade_n}")
    print("  rotation-scan       : fixed [0deg, 180deg] (no step scan)")
    print(f"  rotation-max-workers: {args.rotation_max_workers}")
    if float(getattr(args, "polygon_margin_px", 0.0)) > 0:
        print(f"  polygon-margin      : {args.polygon_margin_px} px (fixed)")
    else:
        print(
            f"  polygon-margin      : ratio={args.polygon_margin_ratio} (min={args.polygon_margin_min_px}px, max={args.polygon_margin_max_px}px)"
        )
    print(f"  marker-preproc      : {args.marker_preproc}")
    print(f"  save-images         : {args.save_images}")
    print("  template-topn       : (removed) always match all templates")
    print(f"  unknown-threshold   : {args.unknown_score_threshold} / margin={args.unknown_margin}")
    print(f"  device              : {args.device}")
    print(f"  top-k               : {args.top_k}")
    print(f"  match-max-side      : {args.match_max_side} px")


def load_docaligner_model(model_name: str, model_type: str) -> tuple[Any, Any]:
    patch_capybara_exports()
    import capybara as cb
    from docaligner import DocAligner, ModelType

    mtype = ModelType.heatmap if model_type == "heatmap" else ModelType.point
    model = DocAligner(model_type=mtype, model_cfg=model_name)
    return model, cb


# ------------------------------------------------------------
# メイン処理（パイプライン本体）
# ------------------------------------------------------------


@dataclass
class StageTimes:
    degrade_s: float = 0.0
    docaligner_s: float = 0.0
    rectify_s: float = 0.0
    decide_s: float = 0.0
    uvdoc_s: float = 0.0
    bgdiv_s: float = 0.0
    match_s: float = 0.0
    warp_s: float = 0.0


@dataclass
class DegradedCaseInput:
    """改悪生成済みの入力（v13.7）。

    - 改悪生成は main 冒頭でまとめて実行し、ここに格納して process_one_case へ渡す。
    - degrade 生成時間は計測しない（times.degrade_s は常に 0）
    """

    source_dataset: str
    source_form: str
    source_path: Path
    source_w: int
    source_h: int
    degraded_variant_index: int
    case_id: str
    degraded_bgr: np.ndarray
    H_src_to_degraded: np.ndarray
    degrade_meta: dict[str, Any]
    output_degraded_image_path: Path
    # ground truth（test dataset 用）
    ground_truth_form: str = ""
    ground_truth_template_path: Optional[Path] = None
    ground_truth_template_number: str = ""


def _apply_extra_degrade_v13_5(
    *,
    src_bgr: np.ndarray,
    degraded_bgr: np.ndarray,
    H_src_to_deg: np.ndarray,
    degrade_meta: dict[str, Any],
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, Any]]:
    """v13.5 の追加改悪（bend/shadow）を適用する。

    NOTE:
      v13.7 では「改悪生成フェーズ」を最初に全件実行するため、
      追加改悪もここでまとめて適用する。
    """

    deg_cfg = (PIPELINE_DEFAULTS.get("degrade") or {})
    bend_cfg = (deg_cfg.get("bend") or {}) if isinstance(deg_cfg, dict) else {}
    shadow_cfg = (deg_cfg.get("shadow") or {}) if isinstance(deg_cfg, dict) else {}

    # まず「紙領域マスク」を作る（src->degraded のホモグラフィから）
    paper_mask_u8: Optional[np.ndarray] = None
    try:
        Hm = np.asarray(H_src_to_deg, dtype=np.float64)
        sh, sw = src_bgr.shape[:2]
        mask_src = np.full((int(sh), int(sw)), 255, dtype=np.uint8)
        if Hm.shape == (3, 3):
            paper_mask_u8 = cv2.warpPerspective(
                mask_src,
                Hm,
                (int(degraded_bgr.shape[1]), int(degraded_bgr.shape[0])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        elif Hm.shape == (2, 3):
            paper_mask_u8 = cv2.warpAffine(
                mask_src,
                Hm,
                (int(degraded_bgr.shape[1]), int(degraded_bgr.shape[0])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            paper_mask_u8 = None
    except Exception:
        paper_mask_u8 = None

    degraded_bgr2, paper_mask_u8, bend_meta = maybe_apply_bend_with_mask(
        degraded_bgr,
        paper_mask_u8,
        rng=rng,
        cfg=dict(bend_cfg),
    )
    degraded_bgr3, shadow_meta = maybe_apply_shadow(
        degraded_bgr2,
        rng=rng,
        cfg=dict(shadow_cfg),
        paper_mask_u8=paper_mask_u8,
        bend_meta=bend_meta,
    )

    degrade_meta["bend"] = bend_meta
    degrade_meta["shadow"] = shadow_meta
    degrade_meta["extra_degrade_v13_5"] = True
    return degraded_bgr3, degrade_meta


def process_one_case(
    *,
    logger: logging.Logger,
    args: argparse.Namespace,
    model: Any,
    cb: Any,
    matcher: XFeatMatcher,
    cached_matcher: Optional[CachedXFeatMatcher],
    templates_A: list[CachedRef],
    templates_B: list[CachedRef],
    degraded_input: DegradedCaseInput,
    out_dirs: dict[str, Path],
) -> tuple[dict[str, Any], StageTimes]:
    """改悪生成済み画像（degraded_input）を処理する（v13.7）。

    重要:
      - 改悪生成（degrade）は main で全件生成済みであり、ここでは行わない。
      - 時間計測は「本処理」のみ（docaligner/rectify/decide/uvdoc/match/warp）とし、
        途中画像の保存や 6_debug_matches の可視化生成は計測から除外する。
      - 計測に含める画像保存は 8_aligned の保存のみ。
    """

    di = degraded_input
    src_form = str(di.source_form)
    src_path = Path(di.source_path)
    case_id = str(di.case_id)

    item: dict[str, Any] = {
        "source_dataset": str(di.source_dataset),
        "source_form": str(di.source_form),
        "source_path": str(di.source_path),
        "case": str(di.case_id),
        # 注意:
        #   ユーザー要望に合わせて ok の意味を変更する：
        #     ok      = 期待動作として成功したか（C は form_unknown が成功）
        #     ok_warp = warp まで到達したか（aligned 出力が生成されたか）
        "ok": False,
        "ok_warp": False,
        "stage": "start",
        "degraded_variant_index": int(di.degraded_variant_index),
    }

    # 入力画像の解像度（CSV向け）
    try:
        item["source_w"] = int(di.source_w)
        item["source_h"] = int(di.source_h)
    except Exception:
        pass

    # ground truth override（image/test 用）
    if di.ground_truth_form:
        item["ground_truth_form"] = str(di.ground_truth_form)
    if di.ground_truth_template_path is not None:
        item["ground_truth_template_path"] = str(di.ground_truth_template_path)
    if di.ground_truth_template_number:
        item["ground_truth_template_number"] = str(di.ground_truth_template_number)

    times = StageTimes()
    # v13.7: 改悪生成は最初に全件作成し、計測対象外
    times.degrade_s = 0.0

    # ------------------------------------------------------------
    # 画像保存モード（B-1）
    # ------------------------------------------------------------

    save_mode = str(getattr(args, "save_images", "all"))
    jpeg_quality = int((PIPELINE_DEFAULTS.get("save_images") or {}).get("jpeg_quality") or 95)

    # fail モード向け: 最終ステージが確定するまで保存を遅延する
    pending_images: list[tuple[str, Path, np.ndarray]] = []

    def _schedule_image(field_name: str, path: Path, img: np.ndarray) -> None:
        """保存の即時/遅延/無効化を統一する。"""

        if save_mode == "all":
            write_image(path, img, jpeg_quality=jpeg_quality)
            item[field_name] = str(path)
        elif save_mode == "fail":
            pending_images.append((field_name, path, img))
            item[field_name] = ""
        else:
            item[field_name] = ""

    def _finalize_images_for_stage(stage: str) -> None:
        """fail モードで必要なら画像を保存する。"""

        if save_mode != "fail":
            return
        do_save = str(stage) != "done"
        if not do_save:
            return
        for field_name, path, img in pending_images:
            write_image(path, img, jpeg_quality=jpeg_quality)
            item[field_name] = str(path)

    degraded_bgr = di.degraded_bgr
    H_src_to_deg = di.H_src_to_degraded
    degrade_meta = di.degrade_meta

    # 改悪画像（1_degraded）は事前生成フェーズで保存済み。
    # ここでは item への参照だけ入れる（時間計測対象外）。
    item["output_degraded_image_path"] = str(di.output_degraded_image_path)
    try:
        hd, wd = degraded_bgr.shape[:2]
        item["degraded_w"] = int(wd)
        item["degraded_h"] = int(hd)
    except Exception:
        pass
    item["stage"] = "degraded"  # 既に生成済み
    item["degrade"] = degrade_meta
    item["H_src_to_degraded"] = H_src_to_deg.astype(float).tolist()

    # 2) DocAligner（計測対象：推論のみ。画像保存は計測外）
    t0 = time.perf_counter()
    poly = detect_polygon_docaligner(model, cb, degraded_bgr)
    times.docaligner_s = time.perf_counter() - t0
    if poly is None:
        item["stage"] = "docaligner_failed"
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(times.degrade_s + times.docaligner_s)
        return item, times

    item["stage"] = "docaligner_ok"
    item["polygon"] = poly.astype(float).tolist()

    # (1) polygon margin: デフォルトは ratio ベース
    if float(getattr(args, "polygon_margin_px", 0.0)) > 0:
        margin_px = float(args.polygon_margin_px)
        item["polygon_margin"] = {"mode": "fixed_px", "value": margin_px}
    else:
        margin_px = polygon_margin_px_from_ratio(
            poly,
            ratio=float(args.polygon_margin_ratio),
            min_px=float(args.polygon_margin_min_px),
            max_px=float(args.polygon_margin_max_px),
        )
        item["polygon_margin"] = {
            "mode": "ratio",
            "ratio": float(args.polygon_margin_ratio),
            "min_px": float(args.polygon_margin_min_px),
            "max_px": float(args.polygon_margin_max_px),
            "computed_px": float(margin_px),
        }

    poly_exp = expand_polygon(
        poly,
        margin_px=float(margin_px),
        img_w=int(degraded_bgr.shape[1]),
        img_h=int(degraded_bgr.shape[0]),
    )
    overlay = draw_polygon_overlay(degraded_bgr, poly_exp)
    out_doc = out_dirs["doc"] / f"{case_id}_doc.jpg"
    _schedule_image("output_doc_overlay_image_path", out_doc, overlay)

    # 3) Rectify（計測対象：rectifyのみ。画像保存は計測外）
    t0 = time.perf_counter()
    rectified, H_deg_to_rect = polygon_to_rectified(
        degraded_bgr,
        poly_exp,
        out_max_side=int(args.docaligner_max_side),
    )
    rectified, _ = enforce_landscape(rectified)
    times.rectify_s = time.perf_counter() - t0
    out_rect = out_dirs["rect"] / f"{case_id}_rect.jpg"
    _schedule_image("output_rectified_image_path", out_rect, rectified)
    try:
        hr, wr = rectified.shape[:2]
        item["rectified_w"] = int(wr)
        item["rectified_h"] = int(hr)
    except Exception:
        pass
    item["stage"] = "rectified"
    item["H_degraded_to_rectified"] = H_deg_to_rect.astype(float).tolist()

    # 4) decide form by rotations（計測対象：判定ロジックのみ。画像保存は計測外）
    t0 = time.perf_counter()

    # test データは「正例（A/B）」の取りこぼしを避けたいので、
    # フォームA誤検出抑制のうち「周辺が白地」制約だけを緩めた config を使う。
    # （手書きがマーカー近傍に入ると、この制約で弾かれてしまうため）
    formA_geom_cfg: Optional[MarkerGeometryConfig] = None
    if str(di.source_dataset) == "test" and str(src_form) in ("A", "B"):
        try:
            base_cfg_dict = (PIPELINE_DEFAULTS.get("formA") or {}).get("geometry") or {}
            allowed = set(getattr(MarkerGeometryConfig, "__dataclass_fields__", {}).keys())
            base_cfg = MarkerGeometryConfig(**{k: v for k, v in base_cfg_dict.items() if k in allowed})

            # 周辺白地制約を緩める（test 専用）
            formA_geom_cfg = MarkerGeometryConfig(
                **{
                    **asdict(base_cfg),
                    "surround_min_mean_gray": 0.0,
                    "surround_max_ink_ratio": 1.0,
                }
            )
        except Exception:
            formA_geom_cfg = None

    decision = decide_form_by_rotations(
        rectified,
        max_workers=int(args.rotation_max_workers),
        marker_preproc=str(args.marker_preproc),
        unknown_score_threshold=float(args.unknown_score_threshold),
        unknown_margin=float(args.unknown_margin),
        formA_geom_cfg=formA_geom_cfg,
    )
    times.decide_s = time.perf_counter() - t0
    item["form_decision"] = asdict(decision)
    # すぐ参照する項目（ログ/集計向け）
    item["predicted_form"] = str(decision.form or "")
    item["predicted_angle_deg"] = "" if decision.angle_deg is None else float(decision.angle_deg)

    if not decision.ok or decision.form not in ("A", "B") or decision.angle_deg is None:
        item["stage"] = "form_unknown"
        # 期待動作:
        # - C: form_unknown になるべき（紙は検出できたが A/B ではない）
        # - A/B: form_unknown になってはいけない
        item["ok"] = bool(str(src_form) == "C")
        item["ok_warp"] = False
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(times.degrade_s + times.docaligner_s + times.rectify_s + times.decide_s)
        return item, times
    item["stage"] = "form_found"

    # Form correctness for A/B
    gt_form_for_scoring = str(di.ground_truth_form or "")
    if not gt_form_for_scoring and str(src_form) in ("A", "B"):
        gt_form_for_scoring = str(src_form)
    item["is_predicted_form_correct"] = bool(decision.form == gt_form_for_scoring) if gt_form_for_scoring in ("A", "B") else None

    chosen = rotate_image_bound(rectified, float(decision.angle_deg))
    try:
        hc, wc = chosen.shape[:2]
        item["chosen_w"] = int(wc)
        item["chosen_h"] = int(hc)
    except Exception:
        pass

    # フォーム判定の根拠を可視化
    if decision.form == "A":
        markers = ((decision.detail or {}).get("A") or {}).get("markers") or []
        rot_vis = draw_formA_markers_overlay(chosen, markers)
    else:
        qrs = ((decision.detail or {}).get("B") or {}).get("qrs")
        if not qrs:
            # 可視化でも WeChat ベース検出のみ
            wechat = getattr(score_formB, "_wechat", None)
            if wechat is not None:
                qrs = detect_qr_codes_wechat_multiscale(chosen, wechat)
        rot_vis = draw_formB_qr_overlay(chosen, qrs)
    out_rot = out_dirs["rot"] / f"{case_id}_rot.jpg"
    _schedule_image("output_rotated_decision_visualization_image_path", out_rot, rot_vis)

    # 5) UVDoc unwarp（成形）（計測対象：推論のみ。画像保存は計測外）
    t0 = time.perf_counter()
    uvdoc: Optional[UVDocUnwrapper] = getattr(process_one_case, "_uvdoc", None)
    if uvdoc is None:
        item["stage"] = "uvdoc_failed"
        item["ok"] = False
        item["ok_warp"] = False
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(times.degrade_s + times.docaligner_s + times.rectify_s + times.decide_s)
        return item, times

    try:
        chosen_unwarped = uvdoc.unwarp_bgr(chosen)
        item["uvdoc"] = {"ok": True}
    except Exception as e:
        item["uvdoc"] = {"ok": False, "error": str(e)}
        item["stage"] = "uvdoc_failed"
        item["ok"] = False
        item["ok_warp"] = False
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(times.degrade_s + times.docaligner_s + times.rectify_s + times.decide_s)
        return item, times

    times.uvdoc_s = time.perf_counter() - t0
    out_uvdoc = out_dirs["uvdoc"] / f"{case_id}_uvdoc.jpg"
    _schedule_image("output_uvdoc_unwarped_image_path", out_uvdoc, chosen_unwarped)
    try:
        hu, wu = chosen_unwarped.shape[:2]
        item["uvdoc_w"] = int(wu)
        item["uvdoc_h"] = int(hu)
    except Exception:
        pass

    # 6) 背景除算法（Background Division）（計測対象：補正のみ。画像保存は計測外）
    t0 = time.perf_counter()
    bgdiv_bgr, bgdiv_meta = apply_background_division(chosen_unwarped)
    times.bgdiv_s = time.perf_counter() - t0
    item["background_division"] = bgdiv_meta
    out_bgdiv = out_dirs["bgdiv"] / f"{case_id}_bgdiv.jpg"
    _schedule_image("output_background_division_image_path", out_bgdiv, bgdiv_bgr)
    try:
        hb, wb = bgdiv_bgr.shape[:2]
        item["bgdiv_w"] = int(wb)
        item["bgdiv_h"] = int(hb)
    except Exception:
        pass

    chosen_for_match = bgdiv_bgr

    # 7) XFeat matching（計測対象：照合のみ。画像保存は計測外）
    #   ユーザー要望: "絞り込みをやめる"。
    #   フォームAなら APA/image/A の全テンプレ、フォームBなら APA/image/B の全テンプレへ
    #   局所特徴（XFeat）で照合して最良を選ぶ。
    t0 = time.perf_counter()
    templates = templates_A if decision.form == "A" else templates_B
    best: Optional[dict[str, Any]] = None

    # 注意: 絞り込みを廃止（常に全探索）
    candidates = list(templates)
    item["template_prefilter"] = {
        "mode": "disabled",
        "topn": 0,
        "candidates": [c.template_path for c in candidates],
        "total": len(templates),
        "note": "global prefilter disabled; matched against all templates in decided form",
    }

    template_candidate_results: list[dict[str, Any]] = []

    # 改善2:
    #   キャッシュ経路（CachedXFeatMatcher）では、ターゲット特徴(out1)はテンプレによらず同じなので
    #   ここで1回だけ計算し、ループ内は match_lighterglue + findHomography のみにする。
    tgt_prepared_out1: Optional[dict[str, Any]] = None
    tgt_prepared_invS: Optional[np.ndarray] = None
    if cached_matcher is not None:
        try:
            tgt_prepared_out1, _s_tgt, tgt_prepared_invS = cached_matcher.prepare_target(chosen_for_match)
        except Exception:
            tgt_prepared_out1, tgt_prepared_invS = None, None

    best_mk0: Optional[np.ndarray] = None
    best_mk1: Optional[np.ndarray] = None

    for ref in candidates:
        tp = Path(ref.template_path)
        if cached_matcher is not None:
            # キャッシュ経路: テンプレ画像の再読込は不要（特徴は事前計算済み）
            if tgt_prepared_out1 is not None and tgt_prepared_invS is not None:
                res, H_tpl_to_img, mk0, mk1 = cached_matcher.match_with_cached_ref_and_prepared_target(
                    ref,
                    out1=tgt_prepared_out1,
                    invS_tgt=tgt_prepared_invS,
                )
            else:
                # 万一前計算に失敗した場合は互換APIへフォールバック
                res, H_tpl_to_img, mk0, mk1 = cached_matcher.match_with_cached_ref(ref, chosen_for_match)
        else:
            tpl_bgr = cv2.imread(str(tp))
            if tpl_bgr is None:
                continue
            res, H_tpl_to_img, mk0, mk1 = matcher.match_and_estimate_h(tpl_bgr, chosen_for_match)

        ok = bool(getattr(res, "ok", False)) and H_tpl_to_img is not None
        cand = {
            "template": str(tp),
            "ok": ok,
            "inliers": int(getattr(res, "inliers", 0)),
            "matches": int(getattr(res, "matches", 0)),
            "inlier_ratio": float(getattr(res, "inlier_ratio", 0.0)),
            "reproj_rms": getattr(res, "reproj_rms", None),
        }
        if ok and getattr(res, "H_ref_to_tgt", None) is not None:
            cand["H_ref_to_tgt"] = getattr(res, "H_ref_to_tgt")

        template_candidate_results.append(cand)
        if best is None:
            best = cand
            best_mk0, best_mk1 = (mk0, mk1)
        else:
            if int(cand.get("inliers", 0)) > int(best.get("inliers", 0)):
                best = cand
                best_mk0, best_mk1 = (mk0, mk1)
            elif int(cand.get("inliers", 0)) == int(best.get("inliers", 0)):
                if float(cand.get("inlier_ratio", 0.0)) > float(best.get("inlier_ratio", 0.0)):
                    best = cand
                    best_mk0, best_mk1 = (mk0, mk1)
                elif float(cand.get("inlier_ratio", 0.0)) == float(best.get("inlier_ratio", 0.0)):
                    # reprojection error が小さい方を優先（取れない場合は無視）
                    try:
                        r0 = best.get("reproj_rms", None)
                        r1 = cand.get("reproj_rms", None)
                        if r0 is None and r1 is not None:
                            best = cand
                            best_mk0, best_mk1 = (mk0, mk1)
                        elif (r0 is not None) and (r1 is not None) and float(r1) < float(r0):
                            best = cand
                            best_mk0, best_mk1 = (mk0, mk1)
                    except Exception:
                        pass

    times.match_s = time.perf_counter() - t0
    item["best_match"] = best
    item["template_match_candidates"] = template_candidate_results
    if best is None or not best.get("ok"):
        item["stage"] = "xfeat_failed"
        item["ok"] = False
        item["ok_warp"] = False
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(
            times.degrade_s
            + times.docaligner_s
            + times.rectify_s
            + times.decide_s
            + times.uvdoc_s
            + times.bgdiv_s
            + times.match_s
        )
        return item, times

    tpl_path = Path(str(best["template"]))
    tpl_bgr = cv2.imread(str(tpl_path))
    if tpl_bgr is None:
        item["stage"] = "template_read_failed"
        item["ok"] = False
        item["ok_warp"] = False
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(
            times.degrade_s
            + times.docaligner_s
            + times.rectify_s
            + times.decide_s
            + times.uvdoc_s
            + times.bgdiv_s
            + times.match_s
        )
        return item, times

    # Template correctness for A/B only
    gt_tpl_path_for_scoring: Optional[Path] = di.ground_truth_template_path
    if gt_tpl_path_for_scoring is None and gt_form_for_scoring in ("A", "B"):
        gt_tpl_path_for_scoring = Path(str(src_path))
    if gt_form_for_scoring in ("A", "B") and gt_tpl_path_for_scoring is not None:
        try:
            item["is_predicted_best_template_correct"] = bool(Path(str(best.get("template", ""))).name == Path(str(gt_tpl_path_for_scoring)).name)
        except Exception:
            item["is_predicted_best_template_correct"] = False
    else:
        item["is_predicted_best_template_correct"] = None

    try:
        ht, wt = tpl_bgr.shape[:2]
        item["best_template_w"] = int(wt)
        item["best_template_h"] = int(ht)
    except Exception:
        pass

    # (8) 逆ホモグラフィ（逆行列）安定性 + warp + 8_aligned 保存
    # 計測対象に含める画像保存は aligned のみ。
    t0 = time.perf_counter()
    H_tpl_to_img = np.asarray(best.get("H_ref_to_tgt"), dtype=np.float64)
    ok_inv, H_img_to_tpl, inv_reason, h_cond, h_det = safe_invert_homography(
        H_tpl_to_img,
        inliers=int(best.get("inliers", 0)),
        inlier_ratio=float(best.get("inlier_ratio", 0.0)),
        min_inliers=int(args.min_inliers_for_warp),
        min_inlier_ratio=float(args.min_inlier_ratio_for_warp),
        max_cond=float(args.max_h_cond),
    )
    item["homography_inv"] = {"ok": bool(ok_inv), "reason": inv_reason, "cond": h_cond, "det": h_det}
    if not ok_inv or H_img_to_tpl is None:
        item["stage"] = "homography_unstable"
        item["ok"] = False
        item["ok_warp"] = False
        _finalize_images_for_stage(item["stage"])
        times.warp_s = time.perf_counter() - t0
        item["case_total_s"] = float(
            times.degrade_s
            + times.docaligner_s
            + times.rectify_s
            + times.decide_s
            + times.uvdoc_s
            + times.bgdiv_s
            + times.match_s
            + times.warp_s
        )
        return item, times

    # Homography でテンプレ座標へ warp（従来どおり）
    warped_final = cv2.warpPerspective(chosen_for_match, H_img_to_tpl, (tpl_bgr.shape[1], tpl_bgr.shape[0]))

    # 最終の aligned を保存（計測対象）
    out_aligned = out_dirs["aligned"] / f"{case_id}_aligned.jpg"
    # aligned は save-images に関わらず必ず保存する（本成果物）
    write_image(out_aligned, warped_final, jpeg_quality=jpeg_quality)
    item["output_aligned_image_path"] = str(out_aligned)
    try:
        ha, wa = warped_final.shape[:2]
        item["aligned_w"] = int(wa)
        item["aligned_h"] = int(ha)
    except Exception:
        pass
    times.warp_s = time.perf_counter() - t0

    # 6_debug_matches（best template のマッチ可視化）
    # ユーザー要望: 本番ではない処理のため、時間計測から除外する。
    try:
        # best の mkpts（ループ中に保持したもの）で可視化する。
        if best_mk0 is not None and best_mk1 is not None:
            dbg = draw_inlier_matches(tpl_bgr, chosen_for_match, best_mk0, best_mk1, args.match_max_side)
            out_dbg = out_dirs["debug_matches"] / f"{case_id}_matches.jpg"
            # debug_matches は保存モードに関係なく必ず保存（ただし計測外）
            write_image(out_dbg, dbg, jpeg_quality=jpeg_quality)
            item["output_debug_matches_image_path"] = str(out_dbg)
    except Exception:
        pass

    item["stage"] = "done"
    item["ok_warp"] = True
    # 期待動作としての成功条件:
    # - A/B: フォーム正解 AND テンプレ正解 AND warp 完了
    # - C  : "done" に到達したら誤検出（本来は棄却されるべき）
    if src_form in ("A", "B"):
        item["ok"] = bool(item.get("is_predicted_form_correct")) and bool(item.get("is_predicted_best_template_correct"))
    else:
        item["ok"] = False
    item["case_total_s"] = float(
        times.degrade_s
        + times.docaligner_s
        + times.rectify_s
        + times.decide_s
        + times.uvdoc_s
        + times.bgdiv_s
        + times.match_s
        + times.warp_s
    )

    # done のケースでは fail モードは保存しない
    _finalize_images_for_stage(item["stage"])
    return item, times


def process_one_observed_case(
    *,
    logger: logging.Logger,
    args: argparse.Namespace,
    model: Any,
    cb: Any,
    matcher: XFeatMatcher,
    cached_matcher: Optional[CachedXFeatMatcher],
    templates_A: list[CachedRef],
    templates_B: list[CachedRef],
    src_form: str,
    src_path: Path,
    src_bgr: np.ndarray,
    out_dirs: dict[str, Path],
    ground_truth_form: str,
    ground_truth_template_path: Path,
    ground_truth_template_number: str,
    source_dataset: str = "test",
) -> tuple[dict[str, Any], StageTimes]:
    """image/test のような「観測画像」を処理する。

    - 改悪生成（degrade）は行わない
    - ファイル名から得た ground truth を item に入れる
    - DocAligner → rectify → decide → XFeat → warp の通常処理を適用
    """

    # v13.7 方針: 改悪生成以外でも「画像保存等の周辺処理」は時間計測に含めない。
    # この関数は現状 main() からは呼ばれていないが、今後使われても整合するよう
    # case_total_s は stage time の合計で計算する。
    case_id = f"test_{src_path.stem}"
    item: dict[str, Any] = {
        "source_dataset": str(source_dataset),
        "source_form": str(src_form),
        "source_path": str(src_path),
        "case": case_id,
        "ok": False,
        "ok_warp": False,
        "stage": "start",
        "degraded_variant_index": "",
        # ground truth override
        "ground_truth_form": str(ground_truth_form),
        "ground_truth_template_path": str(ground_truth_template_path),
        "ground_truth_template_number": str(ground_truth_template_number),
    }
    times = StageTimes()
    times.degrade_s = 0.0

    # ------------------------------------------------------------
    # 画像保存モード（B-1）
    # ------------------------------------------------------------

    save_mode = str(getattr(args, "save_images", "all"))
    jpeg_quality = int((PIPELINE_DEFAULTS.get("save_images") or {}).get("jpeg_quality") or 95)
    pending_images: list[tuple[str, Path, np.ndarray]] = []

    def _schedule_image(field_name: str, path: Path, img: np.ndarray) -> None:
        if save_mode == "all":
            write_image(path, img, jpeg_quality=jpeg_quality)
            item[field_name] = str(path)
        elif save_mode == "fail":
            pending_images.append((field_name, path, img))
            item[field_name] = ""
        else:
            item[field_name] = ""

    def _finalize_images_for_stage(stage: str) -> None:
        if save_mode != "fail":
            return
        if str(stage) == "done":
            return
        for field_name, path, img in pending_images:
            write_image(path, img, jpeg_quality=jpeg_quality)
            item[field_name] = str(path)

    try:
        h0, w0 = src_bgr.shape[:2]
        item["source_w"] = int(w0)
        item["source_h"] = int(h0)
    except Exception:
        pass

    # 1) degrade をスキップし、観測画像をそのまま degraded 扱いにする
    degraded_bgr = src_bgr
    item["stage"] = "degraded"
    item["degrade"] = {"mode": "test_skip"}
    try:
        hd, wd = degraded_bgr.shape[:2]
        item["degraded_w"] = int(wd)
        item["degraded_h"] = int(hd)
    except Exception:
        pass
    out_degraded = out_dirs["degraded"] / f"{case_id}.jpg"
    _schedule_image("output_degraded_image_path", out_degraded, degraded_bgr)

    # 2) DocAligner
    t0 = time.perf_counter()
    poly = detect_polygon_docaligner(model, cb, degraded_bgr)
    times.docaligner_s = time.perf_counter() - t0
    if poly is None:
        item["stage"] = "docaligner_failed"
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(times.degrade_s + times.docaligner_s)
        return item, times

    item["stage"] = "docaligner_ok"
    item["polygon"] = poly.astype(float).tolist()

    # polygon margin
    if float(getattr(args, "polygon_margin_px", 0.0)) > 0:
        margin_px = float(args.polygon_margin_px)
        item["polygon_margin"] = {"mode": "fixed_px", "value": margin_px}
    else:
        margin_px = polygon_margin_px_from_ratio(
            poly,
            ratio=float(args.polygon_margin_ratio),
            min_px=float(args.polygon_margin_min_px),
            max_px=float(args.polygon_margin_max_px),
        )
        item["polygon_margin"] = {
            "mode": "ratio",
            "ratio": float(args.polygon_margin_ratio),
            "min_px": float(args.polygon_margin_min_px),
            "max_px": float(args.polygon_margin_max_px),
            "computed_px": float(margin_px),
        }

    poly_exp = expand_polygon(
        poly,
        margin_px=float(margin_px),
        img_w=int(degraded_bgr.shape[1]),
        img_h=int(degraded_bgr.shape[0]),
    )
    overlay = draw_polygon_overlay(degraded_bgr, poly_exp)
    out_doc = out_dirs["doc"] / f"{case_id}_doc.jpg"
    _schedule_image("output_doc_overlay_image_path", out_doc, overlay)

    # 3) Rectify
    t0 = time.perf_counter()
    rectified, H_deg_to_rect = polygon_to_rectified(degraded_bgr, poly_exp, out_max_side=int(args.docaligner_max_side))
    rectified, _ = enforce_landscape(rectified)
    times.rectify_s = time.perf_counter() - t0
    item["stage"] = "rectified"
    item["H_degraded_to_rectified"] = H_deg_to_rect.astype(float).tolist()
    out_rect = out_dirs["rect"] / f"{case_id}_rect.jpg"
    _schedule_image("output_rectified_image_path", out_rect, rectified)
    try:
        hr, wr = rectified.shape[:2]
        item["rectified_w"] = int(wr)
        item["rectified_h"] = int(hr)
    except Exception:
        pass

    # 4) decide
    t0 = time.perf_counter()
    decision = decide_form_by_rotations(
        rectified,
        max_workers=int(args.rotation_max_workers),
        marker_preproc=str(args.marker_preproc),
        unknown_score_threshold=float(args.unknown_score_threshold),
        unknown_margin=float(args.unknown_margin),
    )
    times.decide_s = time.perf_counter() - t0
    item["form_decision"] = asdict(decision)
    item["predicted_form"] = str(decision.form or "")
    item["predicted_angle_deg"] = "" if decision.angle_deg is None else float(decision.angle_deg)

    if not decision.ok or decision.form not in ("A", "B") or decision.angle_deg is None:
        item["stage"] = "form_unknown"
        # 期待動作:
        # - test/C は A/B として認識されない（form_unknown が成功扱い）
        # - test/A,B は form_unknown になってはいけない
        item["ok"] = bool(str(ground_truth_form) == "C")
        item["ok_warp"] = False
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(times.degrade_s + times.docaligner_s + times.rectify_s + times.decide_s)
        return item, times

    item["stage"] = "form_found"

    # 正解フォーム（test では必ず定義される想定）
    item["is_predicted_form_correct"] = bool(str(decision.form) == str(ground_truth_form))

    chosen = rotate_image_bound(rectified, float(decision.angle_deg))
    try:
        hc, wc = chosen.shape[:2]
        item["chosen_w"] = int(wc)
        item["chosen_h"] = int(hc)
    except Exception:
        pass

    # 判定根拠可視化
    if decision.form == "A":
        markers = ((decision.detail or {}).get("A") or {}).get("markers") or []
        rot_vis = draw_formA_markers_overlay(chosen, markers)
    else:
        qrs = ((decision.detail or {}).get("B") or {}).get("qrs")
        if not qrs:
            wechat = getattr(score_formB, "_wechat", None)
            if wechat is not None:
                qrs = detect_qr_codes_wechat_multiscale(chosen, wechat)
        rot_vis = draw_formB_qr_overlay(chosen, qrs)
    out_rot = out_dirs["rot"] / f"{case_id}_rot.jpg"
    _schedule_image("output_rotated_decision_visualization_image_path", out_rot, rot_vis)

    # 5) UVDoc unwarp（成形）
    t0 = time.perf_counter()
    uvdoc: Optional[UVDocUnwrapper] = getattr(process_one_case, "_uvdoc", None)
    if uvdoc is None:
        item["stage"] = "uvdoc_failed"
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(times.degrade_s + times.docaligner_s + times.rectify_s + times.decide_s)
        return item, times
    try:
        chosen_unwarped = uvdoc.unwarp_bgr(chosen)
        item["uvdoc"] = {"ok": True}
    except Exception as e:
        item["uvdoc"] = {"ok": False, "error": str(e)}
        item["stage"] = "uvdoc_failed"
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(times.degrade_s + times.docaligner_s + times.rectify_s + times.decide_s)
        return item, times
    times.uvdoc_s = time.perf_counter() - t0
    out_uvdoc = out_dirs["uvdoc"] / f"{case_id}_uvdoc.jpg"
    _schedule_image("output_uvdoc_unwarped_image_path", out_uvdoc, chosen_unwarped)
    try:
        hu, wu = chosen_unwarped.shape[:2]
        item["uvdoc_w"] = int(wu)
        item["uvdoc_h"] = int(hu)
    except Exception:
        pass

    # 背景除算法（Background Division）
    t0 = time.perf_counter()
    bgdiv_bgr, bgdiv_meta = apply_background_division(chosen_unwarped)
    times.bgdiv_s = time.perf_counter() - t0
    item["background_division"] = bgdiv_meta
    out_bgdiv = out_dirs["bgdiv"] / f"{case_id}_bgdiv.jpg"
    _schedule_image("output_background_division_image_path", out_bgdiv, bgdiv_bgr)

    try:
        hb, wb = bgdiv_bgr.shape[:2]
        item["bgdiv_w"] = int(wb)
        item["bgdiv_h"] = int(hb)
    except Exception:
        pass

    chosen_for_match = bgdiv_bgr

    # 7) XFeat matching
    t0 = time.perf_counter()
    templates = templates_A if decision.form == "A" else templates_B
    candidates = list(templates)
    item["template_prefilter"] = {
        "mode": "disabled",
        "topn": 0,
        "candidates": [c.template_path for c in candidates],
        "total": len(templates),
        "note": "test dataset; matched against all templates",
    }

    template_candidate_results: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None

    tgt_prepared_out1: Optional[dict[str, Any]] = None
    tgt_prepared_invS: Optional[np.ndarray] = None
    if cached_matcher is not None:
        try:
            tgt_prepared_out1, _s_tgt, tgt_prepared_invS = cached_matcher.prepare_target(chosen_for_match)
        except Exception:
            tgt_prepared_out1, tgt_prepared_invS = None, None

    for ref in candidates:
        tp = Path(ref.template_path)
        if cached_matcher is not None:
            if tgt_prepared_out1 is not None and tgt_prepared_invS is not None:
                res, H_tpl_to_img, mk0, mk1 = cached_matcher.match_with_cached_ref_and_prepared_target(
                    ref,
                    out1=tgt_prepared_out1,
                    invS_tgt=tgt_prepared_invS,
                )
            else:
                res, H_tpl_to_img, mk0, mk1 = cached_matcher.match_with_cached_ref(ref, chosen_for_match)
        else:
            tpl_bgr = cv2.imread(str(tp))
            if tpl_bgr is None:
                continue
            res, H_tpl_to_img, mk0, mk1 = matcher.match_and_estimate_h(tpl_bgr, chosen_for_match)

        ok = bool(getattr(res, "ok", False)) and H_tpl_to_img is not None
        cand = {
            "template": str(tp),
            "ok": ok,
            "inliers": int(getattr(res, "inliers", 0)),
            "matches": int(getattr(res, "matches", 0)),
            "inlier_ratio": float(getattr(res, "inlier_ratio", 0.0)),
            "reproj_rms": getattr(res, "reproj_rms", None),
        }
        if ok and getattr(res, "H_ref_to_tgt", None) is not None:
            cand["H_ref_to_tgt"] = getattr(res, "H_ref_to_tgt")
        template_candidate_results.append(cand)

        if best is None:
            best = cand
        else:
            if int(cand.get("inliers", 0)) > int(best.get("inliers", 0)):
                best = cand
            elif int(cand.get("inliers", 0)) == int(best.get("inliers", 0)):
                if float(cand.get("inlier_ratio", 0.0)) > float(best.get("inlier_ratio", 0.0)):
                    best = cand

    times.match_s = time.perf_counter() - t0
    item["best_match"] = best
    item["template_match_candidates"] = template_candidate_results

    if best is None or not best.get("ok"):
        item["stage"] = "xfeat_failed"
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(
            times.degrade_s
            + times.docaligner_s
            + times.rectify_s
            + times.decide_s
            + times.uvdoc_s
            + times.bgdiv_s
            + times.match_s
        )
        return item, times

    tpl_path = Path(str(best["template"]))
    tpl_bgr = cv2.imread(str(tpl_path))
    if tpl_bgr is None:
        item["stage"] = "template_read_failed"
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(
            times.degrade_s
            + times.docaligner_s
            + times.rectify_s
            + times.decide_s
            + times.uvdoc_s
            + times.bgdiv_s
            + times.match_s
        )
        return item, times

    # test: 正解テンプレは ground_truth_template_path
    try:
        item["is_predicted_best_template_correct"] = bool(tpl_path.name == Path(str(ground_truth_template_path)).name)
    except Exception:
        item["is_predicted_best_template_correct"] = False

    try:
        ht, wt = tpl_bgr.shape[:2]
        item["best_template_w"] = int(wt)
        item["best_template_h"] = int(ht)
    except Exception:
        pass

    # 8) Homography inversion & warp
    t0 = time.perf_counter()
    H_tpl_to_img = np.asarray(best.get("H_ref_to_tgt"), dtype=np.float64)
    ok_inv, H_img_to_tpl, inv_reason, h_cond, h_det = safe_invert_homography(
        H_tpl_to_img,
        inliers=int(best.get("inliers", 0)),
        inlier_ratio=float(best.get("inlier_ratio", 0.0)),
        min_inliers=int(args.min_inliers_for_warp),
        min_inlier_ratio=float(args.min_inlier_ratio_for_warp),
        max_cond=float(args.max_h_cond),
    )
    item["homography_inv"] = {"ok": bool(ok_inv), "reason": inv_reason, "cond": h_cond, "det": h_det}
    if not ok_inv or H_img_to_tpl is None:
        item["stage"] = "homography_unstable"
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(
            times.degrade_s
            + times.docaligner_s
            + times.rectify_s
            + times.decide_s
            + times.uvdoc_s
            + times.bgdiv_s
            + times.match_s
            + times.warp_s
        )
        return item, times

    warped_final = cv2.warpPerspective(chosen_for_match, H_img_to_tpl, (tpl_bgr.shape[1], tpl_bgr.shape[0]))
    out_aligned = out_dirs["aligned"] / f"{case_id}_aligned.jpg"
    # v13.7: aligned は成果物なので save-images に関係なく必ず保存し、保存時間も計測に含める。
    write_image(out_aligned, warped_final, jpeg_quality=jpeg_quality)
    item["output_aligned_image_path"] = str(out_aligned)
    times.warp_s = time.perf_counter() - t0
    item["ok_warp"] = True

    # done
    item["stage"] = "done"
    item["ok"] = bool(item.get("is_predicted_form_correct")) and bool(item.get("is_predicted_best_template_correct"))
    item["case_total_s"] = float(
        times.degrade_s
        + times.docaligner_s
        + times.rectify_s
        + times.decide_s
        + times.uvdoc_s
        + times.bgdiv_s
        + times.match_s
        + times.warp_s
    )
    _finalize_images_for_stage(item["stage"])
    return item, times


def main(argv=None) -> int:
    args = parse_args(argv)

    if getattr(args, "explain", False):
        print_explain()
        return 0

    # 出力ルート（ログファイルを置くため先に作る）
    run_id = now_run_id()
    out_root = mkdir(Path(args.out) / f"run_{run_id}")
    logger = setup_logging(out_root, level=str(args.log_level), console_level=str(args.console_log_level))

    logger.info("=" * 70)
    logger.info("paper_pipeline_v13")
    logger.info("=" * 70)
    logger.info("OpenCV: %s", cv2.__version__)
    logger.info("torch : %s", torch.__version__)
    logger.info("src-forms: %s", args.src_forms)
    print_config(args)

    # XFeat 実行デバイスを決定
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")
    ensure_portable_git_on_path()

    # 出力ディレクトリ（番号付き）
    out_dirs = {
        "degraded": mkdir(out_root / "1_degraded"),
        "doc": mkdir(out_root / "2_doc"),
        "rect": mkdir(out_root / "3_rectified"),
        "rot": mkdir(out_root / "4_rectified_rot"),
        "uvdoc": mkdir(out_root / "5_uvdoc_unwarp"),
        "bgdiv": mkdir(out_root / "6_bgdiv"),
        "debug_matches": mkdir(out_root / "7_debug_matches"),
        "aligned": mkdir(out_root / "8_aligned"),
    }

    # 重いモデルをロード
    logger.info("[INFO] Loading DocAligner...")
    model, cb = load_docaligner_model(args.docaligner_model, args.docaligner_type)
    logger.info("[OK] DocAligner loaded")

    logger.info("[INFO] Loading XFeat...")
    matcher = XFeatMatcher(top_k=args.top_k, device=device, match_max_side=args.match_max_side)
    logger.info("[OK] XFeat loaded")

    src_forms = [s.strip() for s in args.src_forms.split(",") if s.strip()]
    src_forms = [s for s in src_forms if s in ("A", "B", "C")]
    if not src_forms:
        logger.error("src-forms must contain at least one of A,B,C")
        return 1

    # WeChat QR detector を初期化（フォームBは WeChat のみ）
    # v13 改善: 回転スキャンでの直列化を避けるため、ThreadPool の worker 数だけ detector を確保する。
    wechat_pool_size = int(getattr(args, "rotation_max_workers", 1))
    wechat = init_wechat_qr_detector(str(getattr(args, "wechat_model_dir", "")), logger=logger, pool_size=wechat_pool_size)
    # 引数経由でスレッドに流すと取り回しが悪いので、score_formB に属性としてぶら下げる
    setattr(score_formB, "_wechat", wechat)
    if "B" in src_forms and wechat is None:
        logger.error("Form B is enabled but WeChat QR detector is not available. Please install opencv-contrib and set --wechat-model-dir.")
        return 1

    # UVDoc unwarp を初期化（必須）
    try:
        uvdoc_ckpt = Path(str((PIPELINE_DEFAULTS.get("uvdoc") or {}).get("ckpt_path") or ""))
        uvdoc = UVDocUnwrapper(ckpt_path=uvdoc_ckpt, device=str(device), logger=logger)
        # process_one_case / process_one_observed_case から参照できるよう関数属性に保持
        setattr(process_one_case, "_uvdoc", uvdoc)
        logger.info("[OK] UVDoc loaded: ckpt=%s img_size=%s", str(uvdoc_ckpt), getattr(uvdoc, "img_size", None))
    except Exception as e:
        logger.error("UVDoc initialization failed: %s", e)
        return 1

    # (4) テンプレ特徴キャッシュ
    cached_matcher: Optional[CachedXFeatMatcher] = None
    try:
        cached_matcher = CachedXFeatMatcher(matcher)
        logger.info("[OK] CachedXFeatMatcher enabled")
    except Exception as e:
        logger.warning("[WARN] CachedXFeatMatcher disabled: %s", e)
        cached_matcher = None

    # 最終位置合わせ用テンプレ（A/Bのみ）
    template_paths_A = list_images("A")
    template_paths_B = list_images("B")
    if not template_paths_A or not template_paths_B:
        logger.error("templates not found. Expected APA/image/A and APA/image/B")
        return 1

    # テンプレキャッシュをウォームアップ
    templates_A: list[CachedRef] = []
    templates_B: list[CachedRef] = []
    if cached_matcher is not None:
        for pth in template_paths_A:
            img = cv2.imread(str(pth))
            if img is None:
                continue
            templates_A.append(cached_matcher.prepare_ref(img, str(pth)))
        for pth in template_paths_B:
            img = cv2.imread(str(pth))
            if img is None:
                continue
            templates_B.append(cached_matcher.prepare_ref(img, str(pth)))
        logger.info("[OK] template cache built: A=%d B=%d", len(templates_A), len(templates_B))
    else:
        # CachedXFeatMatcher が無い場合でも、テンプレのパス一覧だけは必要。
        templates_A = [CachedRef(template_path=str(p), s_ref=1.0, out0={}) for p in template_paths_A]
        templates_B = [CachedRef(template_path=str(p), s_ref=1.0, out0={}) for p in template_paths_B]
        logger.info("[OK] template list prepared (no feature cache): A=%d B=%d", len(templates_A), len(templates_B))

    summary: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------
    # 改善1: 改悪生成を最初に全件生成（計測対象外）
    # ------------------------------------------------------------

    logger.info("[INFO] Pre-generating degraded images (NOT timed)...")
    degraded_inputs: list[DegradedCaseInput] = []

    def _stable_rng_for_case(key: str, k: int) -> random.Random:
        stable = zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF
        case_seed = (int(args.seed) * 1_000_000) + int(stable) * 100 + int(k)
        return random.Random(case_seed)

    def _make_case_id(dataset: str, form: str, path: Path, k: int) -> str:
        if str(dataset) == "test":
            return f"test_{path.stem}_deg{k:02d}"
        return f"{form}_{path.stem}_deg{k:02d}"

    def _generate_one(
        *,
        dataset: str,
        form: str,
        src_path: Path,
        src_bgr: np.ndarray,
        k: int,
        gt_form: str = "",
        gt_template_path: Optional[Path] = None,
        gt_template_number: str = "",
    ) -> Optional[DegradedCaseInput]:
        case_id = _make_case_id(dataset, form, src_path, k)
        out_degraded = out_dirs["degraded"] / f"{case_id}.jpg"

        try:
            h0, w0 = src_bgr.shape[:2]
        except Exception:
            return None

        rng = _stable_rng_for_case(f"{dataset}/{form}/{src_path.name}", k)
        degraded_bgr, H_src_to_deg, degrade_meta = warp_template_to_random_view(
            src_bgr,
            out_size=(int(args.degrade_w), int(args.degrade_h)),
            rng=rng,
            max_rotation_deg=float(args.max_rot),
            min_abs_rotation_deg=float(args.min_abs_rot),
            rotation_mode=str(args.rotation_mode),
            snap_step_deg=float(args.snap_step_deg),
            perspective_jitter=float(args.perspective),
            min_visible_area_ratio=float(args.min_visible_area_ratio),
            max_attempts=int(args.max_attempts),
        )

        # v13.5 extra degrade (bend/shadow)
        try:
            degraded_bgr, degrade_meta = _apply_extra_degrade_v13_5(
                src_bgr=src_bgr,
                degraded_bgr=degraded_bgr,
                H_src_to_deg=H_src_to_deg,
                degrade_meta=degrade_meta,
                rng=rng,
            )
        except Exception as e:
            if isinstance(degrade_meta, dict):
                degrade_meta.setdefault("extra_degrade_v13_5", False)
                degrade_meta["extra_degrade_error"] = str(e)

        # 改悪画像の保存（計測対象外）
        # - all : ここで保存しておく（後段の本処理とは独立）
        # - fail/none: ここでは保存しない（fail の場合は stage 判定後に必要分だけ保存したい）
        if str(getattr(args, "save_images", "all")) == "all":
            write_image(out_degraded, degraded_bgr, jpeg_quality=int((PIPELINE_DEFAULTS.get("save_images") or {}).get("jpeg_quality") or 95))

        return DegradedCaseInput(
            source_dataset=str(dataset),
            source_form=str(form),
            source_path=Path(src_path),
            source_w=int(w0),
            source_h=int(h0),
            degraded_variant_index=int(k),
            case_id=str(case_id),
            degraded_bgr=degraded_bgr,
            H_src_to_degraded=np.asarray(H_src_to_deg),
            degrade_meta=degrade_meta if isinstance(degrade_meta, dict) else {"meta": str(degrade_meta)},
            output_degraded_image_path=Path(out_degraded),
            ground_truth_form=str(gt_form),
            ground_truth_template_path=Path(gt_template_path) if gt_template_path is not None else None,
            ground_truth_template_number=str(gt_template_number),
        )

    # ステージ別の件数/時間（集計用）
    stage_counts: dict[str, int] = {}
    stage_times: dict[str, float] = {
        "degrade_s": 0.0,
        "docaligner_s": 0.0,
        "rectify_s": 0.0,
        "decide_s": 0.0,
        "uvdoc_s": 0.0,
        "bgdiv_s": 0.0,
        "match_s": 0.0,
        "warp_s": 0.0,
    }

    for sf in src_forms:
        sources = list_images(sf)
        if args.limit and args.limit > 0:
            sources = sources[: int(args.limit)]
        if not sources:
            logger.warning("no sources: APA/image/%s", sf)
            continue

        logger.info("[DEGRADE] form %s: %d images", sf, len(sources))
        for sp in sources:
            src_bgr = cv2.imread(str(sp))
            if src_bgr is None:
                logger.warning("failed to read: %s", sp)
                continue
            for k in range(int(args.degrade_n)):
                di = _generate_one(dataset="synthetic", form=sf, src_path=sp, src_bgr=src_bgr, k=k)
                if di is not None:
                    degraded_inputs.append(di)

    # test dataset も先に改悪生成する（計測対象外）
    test_paths = list_test_images()
    if test_paths:
        logger.info("[DEGRADE] test dataset (image/test): %d images", len(test_paths))
    for tp in test_paths:
        parsed = parse_test_filename(tp)
        if parsed is None:
            logger.warning("skip test image (invalid name): %s", tp.name)
            continue
        gt_form, gt_num = parsed

        src_bgr = cv2.imread(str(tp))
        if src_bgr is None:
            logger.warning("failed to read test image: %s", tp)
            continue

        gt_template_path = Path(__file__).resolve().parent / "image" / gt_form / f"{gt_num}.jpg"
        if not gt_template_path.exists():
            logger.warning("ground truth template not found for test image: %s -> %s", tp.name, gt_template_path)

        for k in range(int(args.degrade_n)):
            di = _generate_one(
                dataset="test",
                form=gt_form,
                src_path=tp,
                src_bgr=src_bgr,
                k=k,
                gt_form=gt_form,
                gt_template_path=gt_template_path,
                gt_template_number=gt_num,
            )
            if di is not None:
                degraded_inputs.append(di)

    logger.info("[OK] Pre-generated degraded inputs: %d", len(degraded_inputs))

    # ------------------------------------------------------------
    # 本処理（計測対象）
    # ------------------------------------------------------------

    t_process0 = time.perf_counter()

    # 生成した改悪画像を 1 枚ずつパイプラインへ投入
    for di in degraded_inputs:
        try:
            item, st = process_one_case(
                logger=logger,
                args=args,
                model=model,
                cb=cb,
                matcher=matcher,
                cached_matcher=cached_matcher,
                templates_A=templates_A,
                templates_B=templates_B,
                degraded_input=di,
                out_dirs=out_dirs,
            )
        except Exception as e:
            item = {
                "source_dataset": str(di.source_dataset),
                "source_form": str(di.source_form),
                "source_path": str(di.source_path),
                "case": str(di.case_id),
                "ok": False,
                "ok_warp": False,
                "stage": "exception",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "degraded_variant_index": int(di.degraded_variant_index),
                "ground_truth_form": str(di.ground_truth_form),
                "ground_truth_template_path": str(di.ground_truth_template_path or ""),
                "ground_truth_template_number": str(di.ground_truth_template_number),
                "source_w": int(di.source_w),
                "source_h": int(di.source_h),
                "output_degraded_image_path": str(di.output_degraded_image_path),
                "degrade": di.degrade_meta,
                "H_src_to_degraded": np.asarray(di.H_src_to_degraded).astype(float).tolist(),
            }
            st = StageTimes()
            logger.error("[ERROR] case failed: %s\n%s", item.get("case"), item.get("traceback"))

        # attach run metadata (for CSV)
        item["run_id"] = str(run_id)
        item["run_output_root_directory"] = str(out_root)

        # attach stage times so we can compute per-case stats later
        item["stage_times"] = {
            "degrade_s": float(st.degrade_s),
            "docaligner_s": float(st.docaligner_s),
            "rectify_s": float(st.rectify_s),
            "decide_s": float(st.decide_s),
            "uvdoc_s": float(st.uvdoc_s),
            "bgdiv_s": float(st.bgdiv_s),
            "match_s": float(st.match_s),
            "warp_s": float(st.warp_s),
        }

        summary.append(item)

        # build per-case rich csv row + log line (ALWAYS)
        try:
            row = build_csv_row(args=args, item=item, times=st)
        except Exception as e:
            row = {
                "case_id": str(item.get("case") or ""),
                "pipeline_final_ok(warp_done)": "FALSE",
                "pipeline_stop_stage": "csv_row_build_failed",
                "exception_error_message": f"csv_row_build_failed: {e}",
                "exception_traceback": traceback.format_exc(),
            }
        csv_rows.append(row)
        log_case_summary(logger, row)

        stage = str(item.get("stage", ""))
        stage_counts[stage] = int(stage_counts.get(stage, 0)) + 1
        stage_times["degrade_s"] += float(st.degrade_s)
        stage_times["docaligner_s"] += float(st.docaligner_s)
        stage_times["rectify_s"] += float(st.rectify_s)
        stage_times["decide_s"] += float(st.decide_s)
        stage_times["uvdoc_s"] += float(st.uvdoc_s)
        stage_times["bgdiv_s"] += float(st.bgdiv_s)
        stage_times["match_s"] += float(st.match_s)
        stage_times["warp_s"] += float(st.warp_s)

    dt_process = time.perf_counter() - t_process0

    # サマリ保存（JSON）
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 注意: summary.json / summary.csv の書き出しは計測対象外
    dt = float(dt_process)

    # Fill run elapsed (same value for all rows so filtering becomes easy)
    for r in csv_rows:
        r["run_elapsed_time_total_seconds"] = f"{dt:.6f}"
        r.setdefault("run_id", str(run_id))
        # 注意: ユーザー要望により CSV にフルパスは出さない。
        # run 出力は run_id / run_output_root_directory_name で特定できる。

    # 詳細な summary.csv を出力
    csv_path = out_root / "summary.csv"
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in csv_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    # (2) ステージ別サマリ
    total_cases = len(summary)
    ok_expected_cases = sum(1 for s in summary if bool(s.get("ok")))
    ok_warp_cases = sum(1 for s in summary if bool(s.get("ok_warp")))
    logger.info("=" * 70)
    logger.info(
        "[SUMMARY] total=%d ok_expected=%d (%.1f%%) ok_warp=%d (%.1f%%)",
        total_cases,
        ok_expected_cases,
        (ok_expected_cases / total_cases * 100.0) if total_cases else 0.0,
        ok_warp_cases,
        (ok_warp_cases / total_cases * 100.0) if total_cases else 0.0,
    )
    if total_cases:
        logger.info("[SUMMARY] elapsed avg per case: %.3fs", float(dt) / float(total_cases))
    logger.info("[SUMMARY] stage counts:")
    for k, v in sorted(stage_counts.items(), key=lambda x: (-x[1], x[0])):
        logger.info("  %-20s : %d", k, v)
    logger.info("[SUMMARY] stage time totals (s):")
    for k, v in stage_times.items():
        logger.info("  %-12s : %.2f", k, float(v))

    # Additional dataset-level stats (requested)
    summarize_results(logger, summary, stage_times, dt)

    logger.info("[DONE] outputs: %s", out_root)
    logger.info("[DONE] elapsed: %.1fs", dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
