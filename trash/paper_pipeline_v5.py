#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper_pipeline_v5.py

[windows]
C:/Users/takumi/develop/miniconda3/python.exe APA/paper_pipeline_v5.py --degrade-n 3

[mac]
# リポジトリルートから実行する想定（`APA/` 配下のスクリプトを直接指定）
.venv/bin/python paper_pipeline_v5.py --degrade-n 3

目的
----
既存の検証コード（DocAligner / フォームA・B判定 / XFeat Homography）をベースに、
静止画像の一括処理パイプラインとして統合・運用しやすくする。

特に以下を重視：

- 解像度差に強い処理（polygon margin の比率化）
- 大量処理時に原因追跡しやすいログ/サマリ（logging + stage集計 + 所要時間）
- 検出率向上（マーカー/QR の前処理オプション）
- 高速化（テンプレ特徴キャッシュ。※グローバル特徴での候補絞り込みは互換用だが現在は無効）
- 安定性（Unknown 判定、逆ホモグラフィの信頼度チェック）

パイプライン概要
----------------
入力:

- 改悪元画像: `APA/image/{A,B,C}/` 配下（デフォルト実装では `1.jpg`〜`6.jpg` を対象）
  - 対象フォームは `--src-forms` で指定

処理フロー（1 case = 1 枚の入力から生成した 1 枚の改悪画像）:

1) 改悪生成（`APA/test_recovery_paper.py` の実装を流用）
2) DocAligner により紙領域 polygon（4点）を推定
   - 失敗したら `stage=docaligner_failed` で終了
3) polygon を（紙サイズ比の margin で）外側に拡張 → 透視補正（rectify）
   - 透視補正後の画像は横長に統一（`enforce_landscape`）
   - `--polygon-margin-px > 0` の場合は固定pxマージンで上書き可能
4) フォーム判定（回転探索）
   - 角度リストは仕様として `0..350` を `--rotation-step` 刻みで作成
   - 実処理は高速化のため Coarse-to-Fine
     - coarse: 0/45/90/.../315 の 8 方向で粗探索し、上位2角度を選ぶ
       - この coarse 段階で QR が 1 度も見つからなければ「フォームBではない」と判断し、以降フォームBの探索は行わない
     - fine  : 上位角度の近傍（±50度）だけ、上記の角度リスト内で細かく探索
     - fine で何も見つからない場合は Unknown（no_detection）とする（救済処置は行わない）
   - フォームA: 3点マーク（TL/TR/BL）が検出できる（`--marker-preproc` で前処理オプション）
   - フォームB: QRコードが検出できる
     - まず高速（軽量）検出で角度候補を絞り、最後に robust 検出で確定
     - `--wechat-model-dir` にモデルがあり、opencv-contrib が入っていれば WeChat QR エンジンを優先（小さいQRに強い）
   - 判定不能/曖昧なら `stage=form_unknown`（Unknown）で終了
5) XFeat matching によるテンプレ照合
   - テンプレは `APA/image/A` または `APA/image/B`（`1.jpg`〜`6.jpg`）
   - フォームAなら `APA/image/A` の全テンプレ、フォームBなら `APA/image/B` の全テンプレに対して局所特徴（XFeat）で照合する。
6) Homography を信頼度チェックの上で逆行列化し、テンプレ座標へ warp
   - 不安定なら `stage=homography_unstable` で終了

出力
----
`APA/output_pipeline/run_YYYYmmdd_HHMMSS/` 配下に（処理順が分かるように番号付き）：

- `1_degraded/`       : 改悪画像
- `2_doc/`            : DocAligner polygon 可視化
- `3_rectified/`      : 透視補正した紙画像
- `4_rectified_rot/`  : フォーム確定に使った回転後画像（根拠も描画）
- `5_debug_matches/`  : best template のマッチ可視化
- `6_aligned/`        : best template にワープした結果
- `summary.json` / `summary.csv`
- `run.log`           : 実行ログ（logging）

注意
----
- torch.hub 経由の XFeat 読み込みで git が必要になることがあるため、
  portable git を PATH に追加する処理を `test_recovery_paper` から流用する。
- QR 検出は OpenCV 標準の `QRCodeDetector` を基本にしつつ、
  条件により WeChat QR エンジン（`cv2.wechat_qrcode_WeChatQRCode`）も利用する。
  - WeChat を使うには opencv-contrib のビルドと、4つのモデルファイル
    （detect/sr の prototxt/caffemodel）が必要
- 日本語ラベル描画は Pillow を使用（OpenCV putText は日本語非対応のため）。
  - `APA_FONT_PATH` を設定すると任意フォントを優先可能

改善点メモ
----

"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import json
import os
import platform
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

from PIL import Image, ImageDraw, ImageFont


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
        "n": 10,  # 1枚の入力から何枚の改悪画像を作るか
        "out_size_wh": [2400, 1800],  # 改悪画像の出力サイズ（幅, 高さ）
        "max_rot_deg": 180.0,  # 改悪生成の回転強度（>=180で0..360の一様回転モード）
        "min_abs_rot_deg": 0.0,  # 最小回転量（0なら小さな回転も許可）
        "rotation_mode": "uniform",  # 回転角の出し方（"uniform" または "snap"）
        "snap_step_deg": 90.0,  # rotation_mode="snap" の場合の角度刻み
        "perspective_jitter": 0.08,  # 射影ゆがみ量（大きいほど難しい）
        "min_visible_area_ratio": 0.25,  # 生成画像でテンプレが見えている最小比率
        "max_attempts": 50,  # 改悪生成の最大試行回数
        "seed": 42,  # 乱数シード（再現性）
    },

    # WeChat QRモデル
    "wechat": {
        "model_dir": str(Path(__file__).resolve().parent / "models" / "wechat_qrcode"),  # WeChat QRモデル配置ディレクトリ
    },

    # XFeat（テンプレマッチング）
    "xfeat": {
        "device_default": "cpu",  # 既定の実行デバイス（auto/cpu/cuda のうち default に使う）
        "top_k": 1024,  # 特徴点数（大きいほど高精度だが遅い）
        "match_max_side_px": 1200,  # マッチング前にリサイズする最大辺(px)
    },

    # フォーム判定（回転スキャン）
    "rotation_scan": {
        "step_deg": 10.0,  # 回転スキャンの角度刻み（0..350 をこの刻みで生成）
        "max_workers": 8,  # 回転スキャンの並列数（スレッド）
        "coarse_angles_deg": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],  # 粗探索の角度（向きの当たりを付ける）
        "fine_window_deg": 50.0,  # 粗探索の上位角度から±何度を細探索するか
    },

    # DocAligner（紙領域検出）
    "docaligner": {
        "model": "fastvit_sa24",  # DocAlignerのモデル名
        "type": "heatmap",  # 推論タイプ（"point" / "heatmap"）
        "rectified_max_side_px": 2400,  # 透視補正後の紙画像の最大辺(px)
        "pad_px": 100,  # DocAligner入力前に周囲へ足すパディング(px)
        "polygon_margin": {
            "ratio": 0.03,  # polygonを外側に広げる比率（紙の長辺に対する割合）
            "min_px": 10.0,  # ratio計算の下限(px)
            "max_px": 200.0,  # ratio計算の上限(px)（0以下なら無制限）
            "fixed_px": 0.0,  # 固定pxマージン（>0の場合 ratio を上書き）
        },
    },

    # マーカー検出（フォームA）向け前処理
    "marker": {
        "preproc_mode": "basic",  # マーカー検出前処理の強さ（"none" / "basic" / "morph"）
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
            "surround_min_mean_gray": 190.0,  # 周辺領域の平均輝度がこの値未満なら「汚れている」とみなす
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
        "max_test_side_px": 6000,  # QR検出で試す画像サイズの最大辺(px)
        "robust": {
            "base_scales": [1.0, 0.75, 0.5, 0.25],  # robust検出で試す基本スケール（縮小中心）
            "up_scales_small_image": [1.5, 2.0],  # 入力が小さいときだけ追加で試す拡大スケール
            "up_scale_enable_max_side_px": 1800,  # 最大辺がこの値未満なら拡大も試す
            "adaptive_morph_kernel": [5, 5],  # 二値化後のモルフォロジーカーネル
        },
        "fast": {
            "scales": [1.0, 0.5, 0.25, 0.75],  # fast検出で試すスケール
            "extra_up_scales_small_image": [1.5],  # 入力が小さい場合のみ追加で試す拡大スケール
            "up_scale_enable_max_side_px": 1400,  # 最大辺がこの値未満なら拡大も試す
        },
        "wechat": {
            "fast": {
                "scales": [1.0, 0.75, 0.5, 1.25, 1.5],  # WeChat fast のスケール
                "up_scale_enable_max_side_px": 1600,  # 最大辺がこの値以上なら拡大は無効化
            },
            "robust": {
                "scales": [1.0, 0.75, 0.5, 0.25, 1.25, 1.5, 2.0],  # WeChat robust のスケール
                "up_scale_enable_max_side_px": 1800,  # 最大辺がこの値以上なら拡大は無効化
            },
            "max_test_side_px": 6500,  # WeChat で試す画像の最大辺(px)
        },
    },

    # Homography（特徴点マッチングの射影変換）
    "homography": {
        "find": {
            "ransac_reproj_threshold_px": 3.5,  # RANSACの再投影誤差しきい値(px)
            "max_iters": 1500,  # RANSACの最大反復回数
            "confidence": 0.999,  # RANSACの信頼度
        },
        "invert": {
            "det_abs_min": 1e-12,  # 逆行列化を許可する最小 |det|（小さいと不安定）
        },
    },

    # 可視化（デバッグ画像）
    "visual": {
        "polygon_line_thickness": 6,  # polygon枠線の太さ
        "polygon_point_radius": 10,  # 角点の半径
        "polygon_label_font_scale": 1.0,  # 角ラベル（TL/TR...）のフォント倍率
        "polygon_label_thickness": 2,  # 角ラベルの太さ
    },

    # Unknown 判定（フォームA/Bのどちらでもない扱い）
    "unknown": {
        "score_threshold": 1.2,  # 最大スコアがこの値未満なら Unknown 扱い
        "margin": 0.15,  # A/Bスコア差がこの値未満なら Unknown 扱い（曖昧）
    },

    # warp 許可条件（テンプレ座標へのワープを行うための条件）
    "warp": {
        "min_inliers": 10,  # warpを許可する最小inlier数
        "min_inlier_ratio": 0.15,  # warpを許可する最小inlier_ratio
        "max_h_cond": 1e6,  # Homographyの条件数上限（大きいと不安定）
    },
}


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
        # 注意: OpenCV の wechat_qrcode_WeChatQRCode はスレッドセーフが保証されない。
        # 本パイプラインは回転スキャンで ThreadPoolExecutor を使うため、Lock で保護する。
        self._lock = threading.Lock()

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

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        """QRコードを検出してデコードする。

        戻り値:
          dict のリスト: [{data, points, engine}]
        """

        if image_bgr is None:
            return []
        # ネイティブ呼び出しをスレッド安全にする
        with self._lock:
            res, points = self.detector.detectAndDecode(image_bgr)

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


_WECHAT_QR: Optional[WeChatQRDetector] = None


def init_wechat_qr_detector(model_dir: str, logger: Optional[logging.Logger] = None) -> Optional[WeChatQRDetector]:
    """グローバルな WeChat QR detector（重い）を1回だけ初期化する。

    利用できない場合は None を返す。
    """

    global _WECHAT_QR
    try:
        _WECHAT_QR = WeChatQRDetector(model_dir=model_dir)
        if logger:
            logger.info("[OK] WeChat QR detector initialized: %s", model_dir)
        return _WECHAT_QR
    except Exception as e:
        _WECHAT_QR = None
        if logger:
            logger.warning("[WARN] WeChat QR detector disabled: %s", e)
        return None


# --- 既存実装の流用 ---
# 注意: このスクリプトは `python APA/paper_pipeline_v5.py ...` の形で実行される想定。
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


# ------------------------------------------------------------
# QR検出（robust / 安定性重視）
# ------------------------------------------------------------


def _try_decode_qr(qr: cv2.QRCodeDetector, img: np.ndarray) -> Optional[tuple[str, np.ndarray]]:
    """検出できたら (data, points) を返す。"""

    try:
        # multi の方が安定するビルドがあるため先に試す
        ok_multi, decoded_info, points, _ = qr.detectAndDecodeMulti(img)
        if ok_multi and decoded_info and points is not None and len(decoded_info) >= 1:
            for data, pts in zip(decoded_info, points):
                if data:
                    return data, np.asarray(pts, dtype=np.float32)
    except Exception:
        pass

    try:
        data, points, _ = qr.detectAndDecode(img)
        if data and points is not None:
            return data, np.asarray(points, dtype=np.float32)
    except Exception:
        pass

    return None


def detect_qr_codes_robust(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """フォームB向け: 回転/透視補正後の画像でも落ちにくい QR 検出。

    方針:
    - 複数の前処理（gray/CLAHE/Otsu/Adaptive + Morphology）
    - マルチスケール（小さすぎる場合は upscale も試す）
    """

    qr = cv2.QRCodeDetector()
    h0, w0 = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    candidates: list[tuple[str, np.ndarray]] = [("bgr", image_bgr)]

    # gray
    candidates.append(("gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

    # CLAHE
    try:
        clahe_cfg = PIPELINE_DEFAULTS["marker"]["clahe"]
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_cfg["clipLimit"]),
            tileGridSize=tuple(int(x) for x in clahe_cfg["tileGridSize"]),
        )
        g2 = clahe.apply(gray)
        candidates.append(("clahe", cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)))
    except Exception:
        pass

    # 大津の二値化
    try:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(("otsu", cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)))
    except Exception:
        pass

    # 自適応二値化 + モルフォロジー（照明ムラ / ブレ対策）
    try:
        at = PIPELINE_DEFAULTS["marker"]["adaptive_threshold"]
        kernel_xy = PIPELINE_DEFAULTS["qr"]["robust"]["adaptive_morph_kernel"]
        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(at["block_size"]),
            int(at["C"]),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(int(x) for x in kernel_xy))
        bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        bw2 = cv2.morphologyEx(bw2, cv2.MORPH_OPEN, kernel)
        candidates.append(("adaptive_morph", cv2.cvtColor(bw2, cv2.COLOR_GRAY2BGR)))
    except Exception:
        pass

    # スケール（透視補正/回転でQRが小さくなることがあるので、必要に応じて拡大も試す）
    qr_cfg = PIPELINE_DEFAULTS["qr"]["robust"]
    base_scales = list(qr_cfg["base_scales"])
    up_scales = list(qr_cfg["up_scales_small_image"])
    enable_up = max(h0, w0) < int(qr_cfg["up_scale_enable_max_side_px"])
    scales = base_scales + (up_scales if enable_up else [])

    for prep_name, img in candidates:
        h, w = img.shape[:2]
        for s in scales:
            if abs(s - 1.0) < 1e-9:
                test = img
            else:
                new_w = int(round(w * s))
                new_h = int(round(h * s))
                if new_w < int(PIPELINE_DEFAULTS["qr"]["min_test_side_px"]) or new_h < int(PIPELINE_DEFAULTS["qr"]["min_test_side_px"]):
                    continue
                if new_w > int(PIPELINE_DEFAULTS["qr"]["max_test_side_px"]) or new_h > int(PIPELINE_DEFAULTS["qr"]["max_test_side_px"]):
                    continue
                interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
                test = cv2.resize(img, (new_w, new_h), interpolation=interp)

            decoded = _try_decode_qr(qr, test)
            if decoded is None:
                continue
            data, pts = decoded
            if abs(s - 1.0) > 1e-9:
                pts = pts / float(s)
            return [
                {
                    "data": data,
                    "points": pts.reshape(-1, 2).tolist(),
                    "prep": prep_name,
                    "scale": float(s),
                }
            ]

    return []


def detect_qr_codes_wechat(
    image_bgr: np.ndarray,
    wechat: Optional[WeChatQRDetector],
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


def _preprocess_variants_for_qr(image_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """QR検出のための前処理バリエーションを作る。

    WeChat QR detector は BGR 入力を受け付けるため、前処理を施した結果も BGR に戻して渡す。
    低コントラストや照明変動のケースで検出率が上がることがある。
    """

    if image_bgr is None:
        return []

    variants: list[tuple[str, np.ndarray]] = [("bgr", image_bgr)]
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        variants.append(("gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

        # CLAHE
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

        # 自適応二値化 + モルフォロジー
        try:
            at = PIPELINE_DEFAULTS["marker"]["adaptive_threshold"]
            kernel_xy = PIPELINE_DEFAULTS["qr"]["robust"]["adaptive_morph_kernel"]
            bw = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                int(at["block_size"]),
                int(at["C"]),
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(int(x) for x in kernel_xy))
            bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            bw2 = cv2.morphologyEx(bw2, cv2.MORPH_OPEN, kernel)
            variants.append(("adaptive_morph", cv2.cvtColor(bw2, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass
    except Exception:
        pass

    return variants


def detect_qr_codes_wechat_multiscale(
    image_bgr: np.ndarray,
    wechat: Optional[WeChatQRDetector],
    *,
    mode: str = "fast",
) -> list[dict[str, Any]]:
    """WeChatエンジンによるQR検出（前処理 + マルチスケール）。

    mode:
      - fast  : スキャン中に使う想定の軽量モード（試すスケールが少ない）
      - robust: 安定性重視（前処理バリエーション + スケール多め）
    """

    if wechat is None or image_bgr is None:
        return []

    h0, w0 = image_bgr.shape[:2]

    # スキャン時は軽量にしつつ、必要なら軽い拡大も試す。
    if mode == "fast":
        variants = [("bgr", image_bgr)]
        cfg = PIPELINE_DEFAULTS["qr"]["wechat"]["fast"]
        scales = [float(s) for s in cfg["scales"]]
        if max(h0, w0) >= int(cfg["up_scale_enable_max_side_px"]):
            scales = [s for s in scales if s <= 1.0]
    else:
        variants = _preprocess_variants_for_qr(image_bgr)
        cfg = PIPELINE_DEFAULTS["qr"]["wechat"]["robust"]
        scales = [float(s) for s in cfg["scales"]]
        if max(h0, w0) >= int(cfg["up_scale_enable_max_side_px"]):
            scales = [s for s in scales if s <= 1.0]

    best: list[dict[str, Any]] = []
    best_score = float("-inf")

    # WeChat detector は複数QRを返すことがあるため、後段でスコアリングする
    for prep_name, src in variants:
        h, w = src.shape[:2]
        for s in scales:
            if abs(s - 1.0) < 1e-9:
                test = src
            else:
                new_w = int(round(w * s))
                new_h = int(round(h * s))
                if new_w < int(PIPELINE_DEFAULTS["qr"]["min_test_side_px"]) or new_h < int(PIPELINE_DEFAULTS["qr"]["min_test_side_px"]):
                    continue
                if new_w > int(PIPELINE_DEFAULTS["qr"]["wechat"]["max_test_side_px"]) or new_h > int(PIPELINE_DEFAULTS["qr"]["wechat"]["max_test_side_px"]):
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

            # 簡単なヒューリスティックで best を選ぶ:
            # - QRの面積が大きいほど安定
            # - 右上に寄っているほど望ましい
            score = float("-inf")
            try:
                score, _ = score_best_qr_candidate(test if abs(s - 1.0) < 1e-9 else src, qrs)
            except Exception:
                score = 0.0

            if score > best_score:
                best_score = score
                best = qrs

            # fast モードでは最初に見つかった時点で即返す
            if mode == "fast":
                return best

    return best


def score_best_qr_candidate(
    image_bgr: np.ndarray,
    qrs: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """複数候補の中から最良のQRを1つ選ぶ。

    目的: 回転角を選んだ後に QR が「右上」に来るようにしたい。
    スコアは以下で構成する:
      - 右上に近いほど高得点（主）
      - QR面積が大きいほど高得点（副：安定性向上）
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

            x_score = cx / float(max(1, w))
            y_score = 1.0 - (cy / float(max(1, h)))
            pos_score = 0.6 * x_score + 0.4 * y_score

            # 右上らしさを強く優先し、次に面積を評価
            score = (pos_score * 3.0) + (rel * 12.0)

            if score > best_score:
                best_score = float(score)
                best = q
                best_detail = {
                    "qr_center": [cx, cy],
                    "qr_rel_area": rel,
                    "qr_pos_score": pos_score,
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


def detect_qr_codes_fast(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """フォーム判定のスキャン用: 速さ優先のQR検出。

    - 前処理なし（BGRのみ）
    - マルチスケールも最小限
    """

    qr = cv2.QRCodeDetector()
    h, w = image_bgr.shape[:2]

    # OpenCV のビルド差で BGR より Gray の方が安定するケースがあるため、
    # FASTでも最低限 gray を試す。
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    candidates: list[tuple[str, np.ndarray]] = [("bgr", image_bgr), ("gray", gray_bgr)]

    # 最小限（fast）。安定性のため、基本は縮小を優先。
    cfg = PIPELINE_DEFAULTS["qr"]["fast"]
    scales = [float(s) for s in cfg["scales"]]
    if max(h, w) < int(cfg["up_scale_enable_max_side_px"]):
        scales = scales + [float(s) for s in cfg["extra_up_scales_small_image"]]

    for prep, src in candidates:
        for s in scales:
            if abs(s - 1.0) < 1e-9:
                test = src
            else:
                new_w = int(round(w * s))
                new_h = int(round(h * s))
                if new_w < int(PIPELINE_DEFAULTS["qr"]["min_test_side_px"]) or new_h < int(PIPELINE_DEFAULTS["qr"]["min_test_side_px"]):
                    continue
                interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
                test = cv2.resize(src, (new_w, new_h), interpolation=interp)

            decoded = _try_decode_qr(qr, test)
            if decoded is None:
                continue
            data, pts = decoded
            if abs(s - 1.0) > 1e-9:
                pts = pts / float(s)
            return [{"data": data, "points": pts.reshape(-1, 2).tolist(), "prep": f"{prep}_fast", "scale": float(s)}]

    return []


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

    # no_detection のときは、coarse の最大スコアだけ抜粋する（全 detail は巨大になり得るため）
    if reason == "no_detection":
        try:
            coarse = detail.get("coarse") or []
            max_a = float("-inf")
            max_b = float("-inf")
            for r in coarse:
                try:
                    max_a = max(max_a, float(((r.get("A") or {}).get("score") or 0.0)))
                    max_b = max(max_b, float(((r.get("B_fast") or {}).get("score") or 0.0)))
                except Exception:
                    continue
            if max_a != float("-inf"):
                diag["coarse_max_A_score"] = max_a
            if max_b != float("-inf"):
                diag["coarse_max_B_fast_score"] = max_b
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
    qrs: list[dict[str, Any]] = []

    # ユーザー要望: 可能なら最初から WeChatQR を使う（微調整も含めて）。
    # WeChat が無い場合は OpenCV の robust 検出にフォールバック。
    if wechat is not None:
        qrs = detect_qr_codes_wechat_multiscale(image_bgr, wechat, mode="robust")
    if not qrs:
        qrs = detect_qr_codes_robust(image_bgr)
    if not qrs:
        return False, 0.0, {"qrs": []}

    best_score, detail = score_best_qr_candidate(image_bgr, qrs)
    # 既存のしきい値（デフォルト>=1.2）と大きくズレないようスケールを合わせる
    score = 1.0 + float(best_score)
    return True, float(score), detail


def score_formB_fast(image_bgr: np.ndarray) -> tuple[bool, float, dict[str, Any]]:
    """回転スキャン時の高速判定用（QRがある角度候補を絞る）。"""

    wechat = getattr(score_formB, "_wechat", None)
    qrs: list[dict[str, Any]] = []
    if wechat is not None:
        qrs = detect_qr_codes_wechat_multiscale(image_bgr, wechat, mode="fast")
    else:
        # フォールバック（opencv-contrib が無い環境向け）
        qrs = detect_qr_codes_fast(image_bgr)

    if not qrs:
        return False, 0.0, {"qrs": []}

    best_score, detail = score_best_qr_candidate(image_bgr, qrs)
    score = 1.0 + float(best_score)
    detail["phase"] = "fast"
    return True, float(score), detail


def decide_form_by_rotations(
    rectified_bgr: np.ndarray,
    angles: list[float],
    max_workers: int = 8,
    marker_preproc: str = "none",
    unknown_score_threshold: float = 0.0,
    unknown_margin: float = 0.0,
) -> FormDecision:
    """Coarse-to-fine の回転スキャンで、最良の判定（A/B/Unknown）を返す。

    注意:
    - まず 0/45/90/.../315 の 8 回で大まかな向き（縦横/上下逆）を推定し、
      その近傍だけ細かく探索する（計算量削減）。
    - フォームB（QR探索）は coarse 段階で一度も QR が見つからなければ「フォームBではない」と判断し、
      以降はフォームB探索（B_fast / robust）を行わない（救済処置もしない）。
    - A/B どちらもスコアが低い（閾値未満） or 近すぎる場合は Unknown 扱い。
    """

    def _eval(angle: float, *, enable_formB: bool) -> dict[str, Any]:
        rotated = rotate_image_bound(rectified_bgr, angle)
        # 回転後も横長に統一（ユーザー要望）
        rotated, _ = enforce_landscape(rotated)
        h, w = rotated.shape[:2]
        if h > w:
            return {"angle": float(angle), "skip": True}

        okA, scoreA, detA = score_formA(rotated, marker_preproc=marker_preproc)
        if enable_formB:
            okBf, scoreBf, detBf = score_formB_fast(rotated)
        else:
            okBf, scoreBf, detBf = (False, 0.0, {"qrs": [], "disabled": True})

        return {
            "angle": float(angle),
            "skip": False,
            "A": {"ok": bool(okA), "score": float(scoreA), "detail": detA},
            "B_fast": {"ok": bool(okBf), "score": float(scoreBf), "detail": detBf},
        }

    def _wrap_angle(a: float) -> float:
        return float(a) % 360.0

    def _circular_dist_deg(a: float, b: float) -> float:
        d = abs((_wrap_angle(a) - _wrap_angle(b)) % 360.0)
        return float(min(d, 360.0 - d))

    def _pick_nearest_angles(base_angles: list[float], all_angles: list[float], k_per_base: int = 2) -> list[float]:
        """fine候補が空になった場合の最小限の救済。

        目的:
          ユーザー要望により「全角度探索フォールバック」は廃止する。
          ただし `angles` の刻みが粗すぎる等で fine_set が空になる可能性はあるため、
          base_angles それぞれに対して近い角度だけを拾う。

        - 各 base_angle について circular distance が近い順に最大 k_per_base 個
        - 全体では base_angles の個数 * k_per_base 程度に抑える
        """

        if not base_angles or not all_angles:
            return []

        picked: list[float] = []
        for ba in base_angles:
            ranked = sorted(all_angles, key=lambda a: _circular_dist_deg(float(a), float(ba)))
            for a in ranked[: max(1, int(k_per_base))]:
                picked.append(float(a))

        # 重複排除しつつ順序をある程度安定させる
        out: list[float] = []
        seen: set[float] = set()
        for a in picked:
            aa = float(a)
            if aa in seen:
                continue
            seen.add(aa)
            out.append(aa)
        return out

    # ----------------------------------
    # coarse: 8方向で粗探索（0/45/..../315）
    # ----------------------------------

    coarse = [float(a) for a in PIPELINE_DEFAULTS["rotation_scan"]["coarse_angles_deg"]]
    coarse_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(coarse))) as ex:
        futures = [ex.submit(_eval, a, enable_formB=True) for a in coarse]
        for fut in as_completed(futures):
            r = fut.result()
            if not r or r.get("skip"):
                continue
            coarse_results.append(r)

    if not coarse_results:
        return FormDecision(False, None, None, 0.0, {"reason": "coarse_all_skipped"})

    # coarse 段階で QR が一度も見つからなければ「フォームBではない」
    has_qr_in_coarse = any(bool((r.get("B_fast") or {}).get("ok")) for r in coarse_results)

    # coarse 結果から上位2角度を選ぶ
    # - フォームB候補が存在する場合: max(A_score, B_fast_score)
    # - フォームB候補が存在しない場合: A_score のみ（B探索を打ち切るため）
    coarse_sorted = sorted(
        coarse_results,
        key=lambda rr: (
            max(float(rr["A"]["score"]), float(rr["B_fast"]["score"]))
            if has_qr_in_coarse
            else float(rr["A"]["score"])
        ),
        reverse=True,
    )
    coarse_top = coarse_sorted[:2]
    base_angles = [float(r["angle"]) for r in coarse_top]

    # ----------------------------------
    # fine: coarse上位の近傍だけ探索
    # ----------------------------------

    window = float(PIPELINE_DEFAULTS["rotation_scan"]["fine_window_deg"])  # 取り逃し防止のため少し広め

    # fine の角度候補は、ユーザー指定の角度リスト（0..350, step=N）に揃える。
    # （355 のように新規生成はしない）
    fine_set: set[float] = set()
    for ba in base_angles:
        for a in angles:
            if _circular_dist_deg(a, ba) <= window:
                fine_set.add(float(a))
    fine = sorted(fine_set)
    if not fine:
        # NOTE:
        # 以前はここで「全角度探索」にフォールバックしていたが、処理時間が大きく伸びるため廃止。
        # 代わりに base_angles の近傍だけを最小限試す。
        fine = _pick_nearest_angles(base_angles=base_angles, all_angles=angles, k_per_base=2)
        if not fine:
            return FormDecision(False, None, None, 0.0, {"reason": "no_detection", "coarse": coarse_results, "fine_angles": []})

    bestA: Optional[FormDecision] = None
    bestB_fast: Optional[FormDecision] = None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_eval, a, enable_formB=has_qr_in_coarse) for a in fine]
        for fut in as_completed(futures):
            r = fut.result()
            if not r or r.get("skip"):
                continue

            angle = float(r["angle"])

            if r["A"]["ok"]:
                candA = FormDecision(True, "A", angle, float(r["A"]["score"]), {"A": r["A"]["detail"], "phase": "fine"})
                if bestA is None or candA.score > bestA.score:
                    bestA = candA

            if r["B_fast"]["ok"]:
                candB = FormDecision(True, "B", angle, float(r["B_fast"]["score"]), {"B_fast": r["B_fast"]["detail"], "phase": "fine"})
                if bestB_fast is None or candB.score > bestB_fast.score:
                    bestB_fast = candB

    # fine で見つからない場合:
    # - 救済処置は行わない（ユーザー要望）
    if bestA is None and bestB_fast is None:
        return FormDecision(
            False,
            None,
            None,
            0.0,
            {
                "reason": "no_detection",
                "coarse": coarse_results,
                "fine_angles": fine,
                "note": "formB_disabled_by_coarse" if not has_qr_in_coarse else "formB_enabled_but_no_detection",
            },
        )

    # Unknown 判定: スコアが低すぎる / A-Bが近すぎる
    # （どちらかが None の場合は -inf として扱う）
    a_score = float(bestA.score) if bestA is not None else float("-inf")
    b_score = float(bestB_fast.score) if bestB_fast is not None else float("-inf")

    # しきい値: 最大スコアが一定未満なら Unknown
    top_score = max(a_score, b_score)
    if float(unknown_score_threshold) > 0 and top_score < float(unknown_score_threshold):
        return FormDecision(False, None, None, float(top_score), {"reason": "below_threshold", "a_score": a_score, "b_score": b_score})

    # マージン: A/B の差が小さすぎたら Unknown
    if float(unknown_margin) > 0 and bestA is not None and bestB_fast is not None:
        if abs(a_score - b_score) < float(unknown_margin):
            return FormDecision(False, None, None, float(top_score), {"reason": "ambiguous", "a_score": a_score, "b_score": b_score})

    # A があり、B_fast より良ければ A を優先
    if bestA is not None and (bestB_fast is None or bestA.score >= bestB_fast.score):
        return bestA

    # ここに来る時点で bestB_fast は存在する想定
    if bestB_fast is None:
        return FormDecision(False, None, None, float(top_score), {"reason": "no_detection", "note": "unexpected_b_fast_none"})

    # B を robust 検出で絞り込み（bestB_fast とその近傍だけ）
    step = float(angles[1] - angles[0]) if len(angles) >= 2 else 10.0
    candidates = [bestB_fast.angle_deg]
    if bestB_fast.angle_deg is not None:
        candidates += [bestB_fast.angle_deg - step, bestB_fast.angle_deg + step]

    bestB: Optional[FormDecision] = None
    for a in candidates:
        if a is None:
            continue
        # 0..360 に正規化
        aa = float(a) % 360.0
        rotated = rotate_image_bound(rectified_bgr, aa)
        rotated, _ = enforce_landscape(rotated)
        if rotated.shape[0] > rotated.shape[1]:
            continue
        okB, scoreB, detB = score_formB(rotated)
        if not okB:
            continue
        cand = FormDecision(True, "B", float(aa), float(scoreB), {"B": detB, "B_fast": bestB_fast.detail.get("B_fast") if bestB_fast.detail else {}})
        if bestB is None or cand.score > bestB.score:
            bestB = cand

    if bestB is not None:
        return bestB

    # 最後のフォールバック: A があるなら A、なければ失敗
    if bestA is not None:
        return bestA
    return FormDecision(False, None, None, 0.0, {"reason": "no_decision_final_fallback"})


"""（template-topn / グローバル特徴による事前絞り込み）

v5 ではユーザー要望により「フォーム確定後は全テンプレを XFeat で照合」します。
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

    def match_with_cached_ref(
        self,
        ref: CachedRef,
        tgt_bgr: np.ndarray,
    ) -> tuple[Any, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """(XFeatHomographyResult相当, H_full, mkpts0, mkpts1) を返す。"""

        # マッチング安定性/速度のためターゲットをリサイズ（後でHを元スケールへ戻す）
        tgt_small, s_tgt = resize_keep_aspect(tgt_bgr, self.match_max_side)
        out1 = self.xfeat.detectAndCompute(tgt_small, top_k=self.top_k)[0]
        out1.update({"image_size": (tgt_small.shape[1], tgt_small.shape[0])})

        matches = self.xfeat.match_lighterglue(ref.out0, out1)
        if isinstance(matches, (list, tuple)) and len(matches) >= 2:
            mkpts0, mkpts1 = matches[0], matches[1]
        elif isinstance(matches, dict) and "mkpts0" in matches and "mkpts1" in matches:
            mkpts0, mkpts1 = matches["mkpts0"], matches["mkpts1"]
        else:
            return (
                # 形だけ合わせた最小オブジェクト（test_recovery_paper の dataclass は重いのでここでは避ける）
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

        # フル解像度に戻す: H_full = inv(S_tgt) * H_small * S_ref
        S_ref = scale_matrix(float(ref.s_ref))
        S_tgt = scale_matrix(float(s_tgt))
        H_full = np.linalg.inv(S_tgt) @ H_small @ S_ref

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


def select_top_templates(
    target_desc: np.ndarray,
    templates: list[CachedRef],
    top_n: int,
) -> list[CachedRef]:
    # v5 では prefilter を使わないため、互換用に「そのまま返す」だけにする。
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
    truth = _case_truth(src_form, src_path) if src_path else _case_truth(src_form, Path(""))
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
        "elapsed_time_stage_5_xfeat_matching_seconds": f"{times.match_s:.6f}",
        "elapsed_time_stage_6_warp_seconds": f"{times.warp_s:.6f}",

        # ---- 実行メタ情報（フルパスなし） ----
        "run_id": str(item.get("run_id") or ""),
        "run_output_root_directory_name": _filename_only(item.get("run_output_root_directory")),
        "run_elapsed_time_total_seconds": str(item.get("run_elapsed_time_total_seconds") or ""),

        # ---- 出力ファイル名（フルパスなし） ----
        "output_degraded_image_filename": _filename_only(item.get("output_degraded_image_path")),
        "output_doc_overlay_image_filename": _filename_only(item.get("output_doc_overlay_image_path")),
        "output_rectified_image_filename": _filename_only(item.get("output_rectified_image_path")),
        "output_rotated_decision_visualization_image_filename": _filename_only(item.get("output_rotated_decision_visualization_image_path")),
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
        # v5 では template-topn は廃止（常に全テンプレ照合）
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

    # 回転スキャン（ユーザー要件: 0..350 を10度刻み）
    p.add_argument(
        "--rotation-step",
        type=float,
        default=float(PIPELINE_DEFAULTS["rotation_scan"]["step_deg"]),
        help="フォーム判定の回転スキャン刻み（度）",
    )
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
            " 例: 0.03 は紙の長辺の 3% をマージンにする。"
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
        f"  --rotation-step       0..350度を何度刻みで回して判定するか（例: 10） [default: {defaults.rotation_step}]",
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
        "  (注) v5 ではテンプレ候補絞り込み（template-topn）は廃止し、常に全テンプレ照合します。",
        "",
        "【ログ】",
        f"  --log-level           ログレベル (DEBUG/INFO/WARNING/ERROR) [default: {defaults.log_level}]",
        f"  --console-log-level   コンソールログレベル (DEBUG/INFO/WARNING/ERROR) [default: {defaults.console_log_level}]",
        "",
        "【出力】",
        f"  --out                 出力ディレクトリ（run_... が作成される） [default: {defaults.out}]",
        "",
        "最小コマンド例（おすすめデフォルト使用）:",
        r"  C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v5.py --limit 1",
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
    t5 = str(row.get("elapsed_time_stage_5_xfeat_matching_seconds") or "")
    t6 = str(row.get("elapsed_time_stage_6_warp_seconds") or "")

    # Ground truth が無い場合（例: C）は、正誤カラムは空欄にする
    truth_part = f"gt_form={gt_form} pred_form={pred_form}"
    if gt_form:
        truth_part += f" form_ok={form_ok} template_ok={template_ok}"

    msg = (
        f"[CASE] id={case_id} ok={ok} ok_warp={ok_warp} stage={stage} "
        f"unknown_reason={unknown_reason} {truth_part} "
        f"best_template={best_tpl_name} inliers={inliers} inlier_ratio={inlier_ratio} "
        f"time_total_s={t_total} (1_degrade={t1},2_doc={t2},3_rectify={t3},4_decide={t4},5_match={t5},6_warp={t6}) "
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
    print(f"  rotation-step       : {args.rotation_step} deg")
    print(f"  rotation-max-workers: {args.rotation_max_workers}")
    if float(getattr(args, "polygon_margin_px", 0.0)) > 0:
        print(f"  polygon-margin      : {args.polygon_margin_px} px (fixed)")
    else:
        print(
            f"  polygon-margin      : ratio={args.polygon_margin_ratio} (min={args.polygon_margin_min_px}px, max={args.polygon_margin_max_px}px)"
        )
    print(f"  marker-preproc      : {args.marker_preproc}")
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
    match_s: float = 0.0
    warp_s: float = 0.0


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
    src_form: str,
    src_path: Path,
    src_bgr: np.ndarray,
    k: int,
    angles: list[float],
    out_dirs: dict[str, Path],
) -> tuple[dict[str, Any], StageTimes]:
    """1枚の入力画像から生成した「1枚の改悪画像（1バリアント）」を処理する。"""

    case_t0 = time.perf_counter()
    case_id = f"{src_form}_{src_path.stem}_deg{k:02d}"
    item: dict[str, Any] = {
        "source_form": src_form,
        "source_path": str(src_path),
        "case": case_id,
        # 注意:
        #   ユーザー要望に合わせて ok の意味を変更する：
        #     ok      = 期待動作として成功したか（C は form_unknown が成功）
        #     ok_warp = warp まで到達したか（aligned 出力が生成されたか）
        "ok": False,
        "ok_warp": False,
        "stage": "start",
        "degraded_variant_index": int(k),
    }
    times = StageTimes()

    # 入力画像の解像度
    try:
        h0, w0 = src_bgr.shape[:2]
        item["source_w"] = int(w0)
        item["source_h"] = int(h0)
    except Exception:
        pass

    # ケースごとに安定した乱数系列（同じ入力なら毎回同じ改悪になる）
    stable = zlib.crc32(f"{src_form}/{src_path.name}".encode("utf-8")) & 0xFFFFFFFF
    case_seed = (int(args.seed) * 1_000_000) + int(stable) * 100 + int(k)
    rng = random.Random(case_seed)

    # 1) degrade
    t0 = time.perf_counter()
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
    times.degrade_s = time.perf_counter() - t0
    out_degraded = out_dirs["degraded"] / f"{case_id}.jpg"
    cv2.imwrite(str(out_degraded), degraded_bgr)
    item["output_degraded_image_path"] = str(out_degraded)
    try:
        hd, wd = degraded_bgr.shape[:2]
        item["degraded_w"] = int(wd)
        item["degraded_h"] = int(hd)
    except Exception:
        pass
    item["stage"] = "degraded"
    item["degrade"] = degrade_meta
    item["H_src_to_degraded"] = H_src_to_deg.astype(float).tolist()

    # 2) DocAligner
    t0 = time.perf_counter()
    poly = detect_polygon_docaligner(model, cb, degraded_bgr)
    times.docaligner_s = time.perf_counter() - t0
    if poly is None:
        item["stage"] = "docaligner_failed"
        item["case_total_s"] = float(time.perf_counter() - case_t0)
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
    cv2.imwrite(str(out_doc), overlay)
    item["output_doc_overlay_image_path"] = str(out_doc)

    # 3) Rectify
    t0 = time.perf_counter()
    rectified, H_deg_to_rect = polygon_to_rectified(
        degraded_bgr,
        poly_exp,
        out_max_side=int(args.docaligner_max_side),
    )
    rectified, _ = enforce_landscape(rectified)
    times.rectify_s = time.perf_counter() - t0
    out_rect = out_dirs["rect"] / f"{case_id}_rect.jpg"
    cv2.imwrite(str(out_rect), rectified)
    item["output_rectified_image_path"] = str(out_rect)
    try:
        hr, wr = rectified.shape[:2]
        item["rectified_w"] = int(wr)
        item["rectified_h"] = int(hr)
    except Exception:
        pass
    item["stage"] = "rectified"
    item["H_degraded_to_rectified"] = H_deg_to_rect.astype(float).tolist()

    # 4) decide form by rotations
    t0 = time.perf_counter()
    decision = decide_form_by_rotations(
        rectified,
        angles=angles,
        max_workers=int(args.rotation_max_workers),
        marker_preproc=str(args.marker_preproc),
        unknown_score_threshold=float(args.unknown_score_threshold),
        unknown_margin=float(args.unknown_margin),
    )
    times.decide_s = time.perf_counter() - t0
    item["form_decision"] = asdict(decision)
    # すぐ参照する項目（ログ/集計向け）
    item["predicted_form"] = str(decision.form or "")
    item["predicted_angle_deg"] = "" if decision.angle_deg is None else float(decision.angle_deg)

    if not decision.ok or decision.form not in ("A", "B") or decision.angle_deg is None:
        item["stage"] = "form_unknown"
        # 期待動作:
        # - A/B: form_unknown になってはいけない
        # - C  : form_unknown になるべき（紙は検出できたが A/B ではない）
        item["ok"] = bool(src_form == "C")
        item["ok_warp"] = False
        item["case_total_s"] = float(time.perf_counter() - case_t0)
        return item, times
    item["stage"] = "form_found"

    # Form correctness for A/B (C は ground truth 未定義のため空扱い)
    if src_form in ("A", "B"):
        item["is_predicted_form_correct"] = bool(decision.form == src_form)
    else:
        item["is_predicted_form_correct"] = None

    chosen = rotate_image_bound(rectified, float(decision.angle_deg))
    chosen, _ = enforce_landscape(chosen)
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
            # 可視化でも WeChat ベース検出を優先
            wechat = getattr(score_formB, "_wechat", None)
            if wechat is not None:
                qrs = detect_qr_codes_wechat_multiscale(chosen, wechat, mode="robust")
            if not qrs:
                qrs = detect_qr_codes_robust(chosen)
        rot_vis = draw_formB_qr_overlay(chosen, qrs)
    out_rot = out_dirs["rot"] / f"{case_id}_rot.jpg"
    cv2.imwrite(str(out_rot), rot_vis)
    item["output_rotated_decision_visualization_image_path"] = str(out_rot)

    # 5) XFeat matching
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

    for ref in candidates:
        tp = Path(ref.template_path)
        if cached_matcher is not None:
            # キャッシュ経路: テンプレ画像の再読込は不要（特徴は事前計算済み）
            res, H_tpl_to_img, mk0, mk1 = cached_matcher.match_with_cached_ref(ref, chosen)
        else:
            tpl_bgr = cv2.imread(str(tp))
            if tpl_bgr is None:
                continue
            res, H_tpl_to_img, mk0, mk1 = matcher.match_and_estimate_h(tpl_bgr, chosen)

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
                elif float(cand.get("inlier_ratio", 0.0)) == float(best.get("inlier_ratio", 0.0)):
                    # reprojection error が小さい方を優先（取れない場合は無視）
                    try:
                        r0 = best.get("reproj_rms", None)
                        r1 = cand.get("reproj_rms", None)
                        if r0 is None and r1 is not None:
                            best = cand
                        elif (r0 is not None) and (r1 is not None) and float(r1) < float(r0):
                            best = cand
                    except Exception:
                        pass

    times.match_s = time.perf_counter() - t0
    item["best_match"] = best
    item["template_match_candidates"] = template_candidate_results
    if best is None or not best.get("ok"):
        item["stage"] = "xfeat_failed"
        item["ok"] = False
        item["ok_warp"] = False
        item["case_total_s"] = float(time.perf_counter() - case_t0)
        return item, times

    tpl_path = Path(str(best["template"]))
    tpl_bgr = cv2.imread(str(tpl_path))
    if tpl_bgr is None:
        item["stage"] = "template_read_failed"
        item["ok"] = False
        item["ok_warp"] = False
        item["case_total_s"] = float(time.perf_counter() - case_t0)
        return item, times

    # Template correctness for A/B only
    if src_form in ("A", "B"):
        try:
            item["is_predicted_best_template_correct"] = bool(Path(str(best.get("template", ""))).name == Path(str(src_path)).name)
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

    # (7) 逆ホモグラフィ（逆行列）安定性
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
        item["case_total_s"] = float(time.perf_counter() - case_t0)
        return item, times

    warped = cv2.warpPerspective(chosen, H_img_to_tpl, (tpl_bgr.shape[1], tpl_bgr.shape[0]))
    out_aligned = out_dirs["aligned"] / f"{case_id}_aligned.jpg"
    cv2.imwrite(str(out_aligned), warped)
    item["output_aligned_image_path"] = str(out_aligned)
    try:
        ha, wa = warped.shape[:2]
        item["aligned_w"] = int(wa)
        item["aligned_h"] = int(ha)
    except Exception:
        pass
    times.warp_s = time.perf_counter() - t0

    # デバッグ用のマッチ可視化（可能な範囲で）
    # 追加: CSV 向けに XFeat の詳細診断も収集
    try:
        res2, H2, mk0, mk1 = matcher.match_and_estimate_h(tpl_bgr, chosen)
        if getattr(res2, "ok", False):
            item["xfeat_best"] = {
                "ref_kpts": int(getattr(res2, "ref_kpts", 0)),
                "tgt_kpts": int(getattr(res2, "tgt_kpts", 0)),
                "matches": int(getattr(res2, "matches", 0)),
                "inliers": int(getattr(res2, "inliers", 0)),
                "inlier_ratio": float(getattr(res2, "inlier_ratio", 0.0)),
                "reproj_rms": getattr(res2, "reproj_rms", None),
            }

        if getattr(res2, "ok", False) and mk0 is not None and mk1 is not None:
            dbg = draw_inlier_matches(tpl_bgr, chosen, mk0, mk1, args.match_max_side)
            out_dbg = out_dirs["debug_matches"] / f"{case_id}_matches.jpg"
            cv2.imwrite(str(out_dbg), dbg)
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
    item["case_total_s"] = float(time.perf_counter() - case_t0)
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
    logger.info("paper_pipeline_v5")
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
        "debug_matches": mkdir(out_root / "5_debug_matches"),
        "aligned": mkdir(out_root / "6_aligned"),
    }

    # 重いモデルをロード
    logger.info("[INFO] Loading DocAligner...")
    model, cb = load_docaligner_model(args.docaligner_model, args.docaligner_type)
    logger.info("[OK] DocAligner loaded")

    logger.info("[INFO] Loading XFeat...")
    matcher = XFeatMatcher(top_k=args.top_k, device=device, match_max_side=args.match_max_side)
    logger.info("[OK] XFeat loaded")

    # WeChat QR detector（利用可能なら）を初期化
    wechat = init_wechat_qr_detector(str(getattr(args, "wechat_model_dir", "")), logger=logger)
    # 引数経由でスレッドに流すと取り回しが悪いので、score_formB に属性としてぶら下げる
    setattr(score_formB, "_wechat", wechat)

    # (4) テンプレ特徴キャッシュ
    cached_matcher: Optional[CachedXFeatMatcher] = None
    try:
        cached_matcher = CachedXFeatMatcher(matcher)
        logger.info("[OK] CachedXFeatMatcher enabled")
    except Exception as e:
        logger.warning("[WARN] CachedXFeatMatcher disabled: %s", e)
        cached_matcher = None

    # フォーム判定用の角度リストを準備
    step = float(args.rotation_step)
    angles = [float(a) for a in np.arange(0.0, 360.0, step) if a < 360.0 - 1e-6]
    angles = [a for a in angles if a <= 350.0 + 1e-6]  # enforce 0..350
    if not angles:
        logger.error("rotation angles list is empty")
        return 1

    src_forms = [s.strip() for s in args.src_forms.split(",") if s.strip()]
    src_forms = [s for s in src_forms if s in ("A", "B", "C")]
    if not src_forms:
        logger.error("src-forms must contain at least one of A,B,C")
        return 1

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
    t_all0 = time.perf_counter()

    # ステージ別の件数/時間（集計用）
    stage_counts: dict[str, int] = {}
    stage_times: dict[str, float] = {
        "degrade_s": 0.0,
        "docaligner_s": 0.0,
        "rectify_s": 0.0,
        "decide_s": 0.0,
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

        logger.info("Processing sources from form %s: %d images", sf, len(sources))
        for sp in sources:
            src_bgr = cv2.imread(str(sp))
            if src_bgr is None:
                logger.warning("failed to read: %s", sp)
                continue

            for k in range(int(args.degrade_n)):
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
                        src_form=sf,
                        src_path=sp,
                        src_bgr=src_bgr,
                        k=k,
                        angles=angles,
                        out_dirs=out_dirs,
                    )
                except Exception as e:
                    item = {
                        "source_form": sf,
                        "source_path": str(sp),
                        "case": f"{sf}_{sp.stem}_deg{k:02d}",
                        "ok": False,
                        "ok_warp": False,
                        "stage": "exception",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
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
                    "match_s": float(st.match_s),
                    "warp_s": float(st.warp_s),
                }

                summary.append(item)

                # build per-case rich csv row + log line (ALWAYS)
                try:
                    row = build_csv_row(args=args, item=item, times=st)
                except Exception as e:
                    # CSV行の構築に失敗しても、処理全体は止めない
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
                stage_times["match_s"] += float(st.match_s)
                stage_times["warp_s"] += float(st.warp_s)

    # サマリ保存（JSON）
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    dt = time.perf_counter() - t_all0

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

