#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper_pipeline_v4.py

C:/Users/takumi/develop/miniconda3/python.exe APA/paper_pipeline_v4.py --degrade-n 3 --template-topn 0

目的
----
既存の検証コード（DocAligner / フォームA・B判定 / XFeat Homography）をベースに、
静止画像の一括処理パイプラインとして統合・運用しやすくする。

特に以下を重視：

- 解像度差に強い処理（polygon margin の比率化）
- 大量処理時に原因追跡しやすいログ/サマリ（logging + stage集計 + 所要時間）
- 検出率向上（マーカー/QR の前処理オプション）
- 高速化（テンプレ特徴キャッシュ + グローバル特徴で候補絞り込み）
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
   - 実処理は高速化のため Coarse-to-Fine（0/90/180/270 で粗探索→近傍のみ探索）
   - フォームA: 3点マーク（TL/TR/BL）が検出できる（`--marker-preproc` で前処理オプション）
   - フォームB: QRコードが検出できる
     - まず高速（軽量）検出で角度候補を絞り、最後に robust 検出で確定
     - `--wechat-model-dir` にモデルがあり、opencv-contrib が入っていれば WeChat QR エンジンを優先（小さいQRに強い）
   - 判定不能/曖昧なら `stage=form_unknown`（Unknown）で終了
5) XFeat matching によるテンプレ照合
   - テンプレは `APA/image/A` または `APA/image/B`（`1.jpg`〜`6.jpg`）
   - グローバル特徴で上位 `--template-topn` 枚へ絞り込み→局所特徴で精密推定（`--template-topn 0` で全探索）
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


# ------------------------------------------------------------
# WeChat QRCode Engine (cv2.wechat_qrcode_WeChatQRCode)
# ------------------------------------------------------------


class WeChatQRDetector:
    """WeChat QR code detector wrapper.

    Why:
      OpenCV's default QRCodeDetector sometimes fails on tiny/low-res QR codes.
      WeChat engine includes a CNN detector + super-resolution model.

    Note:
      Requires opencv-contrib build and 4 model files.
      We keep this wrapper small and cache the heavy detector instance.
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

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Detect and decode QR codes.

        Returns:
          List of dicts: [{data, points, engine}]
        """

        if image_bgr is None:
            return []
        res, points = self.detector.detectAndDecode(image_bgr)

        out: list[dict[str, Any]] = []
        if res is None or points is None:
            return out

        # OpenCV returns tuple/list of strings and a list/np array of points.
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
    """Initialize global WeChat QR detector (heavy) once.

    Returns None if unavailable.
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


# --- reuse implementations from previous work ---
# NOTE: このスクリプトは `python APA/paper_pipeline_v4.py ...` の形で実行される想定。
# その場合 sys.path[0] は `.../APA` になるため、同ディレクトリのモジュールは
# `from test_recovery_paper import ...` の形で import する（`import APA.xxx` は失敗しやすい）。
from test_recovery_paper import (
    XFeatMatcher,
    detect_formA_marker_boxes as _detect_formA_marker_boxes_base,
    draw_inlier_matches,
    ensure_portable_git_on_path,
    now_run_id,
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
# logging
# ------------------------------------------------------------


def setup_logging(
    out_root: Optional[Path],
    level: str = "INFO",
    console_level: Optional[str] = None,
) -> logging.Logger:
    """Configure logging.

    - console: INFO by default (or console_level)
    - file: same as level, saved to out_root/run.log
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
            # Keep running even if file handler fails
            pass

    return logger


# ------------------------------------------------------------
# DocAligner helpers (adapted from test_docaligner_camera_v3.py)
# ------------------------------------------------------------


def patch_capybara_exports() -> None:
    """Expose expected symbols on capybara namespace package (Windows workaround)."""

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
    """Order 4 points to TL/TR/BR/BL."""

    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def expand_polygon(polygon_xy: np.ndarray, margin_px: float, img_w: int, img_h: int) -> np.ndarray:
    """Expand polygon outward by margin_px (best-effort)."""

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
    """Compute margin in px from polygon size ratio.

    We use the polygon's estimated paper size (max(edge lengths)) as the reference.
    This makes behaviour more stable across different image resolutions.
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
    """Warp polygon region to a fronto-parallel rectified image.

    Returns:
        rectified_bgr, H_src_to_rect
    """

    poly = order_quad_tl_tr_br_bl(polygon_xy)

    # Estimate output size based on polygon edges
    w_top = np.linalg.norm(poly[1] - poly[0])
    w_bottom = np.linalg.norm(poly[2] - poly[3])
    h_left = np.linalg.norm(poly[3] - poly[0])
    h_right = np.linalg.norm(poly[2] - poly[1])
    out_w = int(round(max(w_top, w_bottom)))
    out_h = int(round(max(h_left, h_right)))
    out_w = max(320, out_w)
    out_h = max(320, out_h)

    # Limit size for speed
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
    """Rotate without cropping by expanding canvas (similar to imutils.rotate_bound)."""

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


def enforce_portrait(image_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Ensure long side is vertical (portrait). Returns (image, rotated_flag)."""

    h, w = image_bgr.shape[:2]
    if w <= h:
        return image_bgr, False
    # rotate 90deg clockwise
    return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE), True


def enforce_landscape(image_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Ensure long side is horizontal (landscape). Returns (image, rotated_flag)."""

    h, w = image_bgr.shape[:2]
    if w >= h:
        return image_bgr, False
    # rotate 90deg clockwise
    return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE), True


def _thickness_params(image_bgr: np.ndarray) -> tuple[int, float, int]:
    """Return (thickness, font_scale, font_thickness) based on image size."""

    h, w = image_bgr.shape[:2]
    scale = min(w, h) / 1000.0
    thickness = max(6, int(scale * 10))
    font_scale = max(0.8, scale * 1.2)
    font_thickness = max(2, int(scale * 4))
    return thickness, font_scale, font_thickness


def _get_japanese_font(size_px: int) -> ImageFont.FreeTypeFont:
    """Get a Japanese-capable font for Pillow drawing.

    OpenCV's cv2.putText cannot render Japanese, so we use Pillow.
    """

    # NOTE:
    # 以前は Windows フォントパスをハードコードしていたが、
    # Linux/Mac/Docker では存在しないため、可能な限り OS 非依存で解決する。

    # 1) user override
    font_path = os.environ.get("APA_FONT_PATH")
    if font_path:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size=int(size_px))
        except Exception:
            pass

    # 2) common OS fonts
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

    # 3) matplotlib font_manager (best effort)
    try:
        import matplotlib.font_manager as fm

        # try a couple of known JP-capable family names; if not installed, findfont falls back.
        for fam in ["Meiryo", "MS Gothic", "Noto Sans CJK JP", "Noto Sans CJK", "IPAPGothic", "DejaVu Sans"]:
            try:
                p = fm.findfont(fm.FontProperties(family=fam), fallback_to_default=True)
                if p and os.path.exists(p):
                    return ImageFont.truetype(p, size=int(size_px))
            except Exception:
                continue
    except Exception:
        pass

    # 4) fallback: default (may not support Japanese, but keep running)
    return ImageFont.load_default()


def draw_text_pil(
    image_bgr: np.ndarray,
    xy: tuple[int, int],
    text: str,
    color_bgr: tuple[int, int, int],
    font_size: int,
    outline: bool = True,
) -> np.ndarray:
    """Draw text using Pillow so that Japanese doesn't become '???'."""

    # BGR -> RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)

    font = _get_japanese_font(font_size)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))

    x, y = int(xy[0]), int(xy[1])
    if outline:
        # black outline for readability
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
    """ASCII-only fallback when no Japanese-capable font is available."""

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
    """Get marker center from bbox."""

    x, y, w, h = marker.get("bbox", [0, 0, 0, 0])
    return float(x) + float(w) * 0.5, float(y) + float(h) * 0.5


def draw_formA_markers_overlay(image_bgr: np.ndarray, markers: list[dict[str, Any]]) -> np.ndarray:
    """Draw A markers with red boxes and JP corner labels."""

    out = image_bgr.copy()
    thickness, font_scale, font_thickness = _thickness_params(out)
    font_px = max(18, int(font_scale * 28))
    jp = {"top_left": "左上", "top_right": "右上", "bottom_left": "左下"}
    for m in markers:
        x, y, w, h = m.get("bbox", [0, 0, 0, 0])
        corner = str(m.get("corner", ""))
        label = f"{corner}({jp.get(corner, corner)})"
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), thickness)
        # Prefer JP label if possible, otherwise fallback to ASCII.
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
    """Draw B QR with blue box and '右上' label."""

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

    # choose QR top-right point to place label
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
    cv2.polylines(result, [poly], True, (0, 255, 0), 6)
    labels = ["TL", "TR", "BR", "BL"]
    for i, pt in enumerate(poly):
        cv2.circle(result, tuple(pt), 10, (0, 0, 255), -1)
        cv2.putText(result, labels[i], (int(pt[0] + 10), int(pt[1] + 5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return result


def detect_polygon_docaligner(model: Any, cb: Any, image_bgr: np.ndarray, pad_px: int = 100) -> Optional[np.ndarray]:
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
# marker detection wrapper (optional extra preprocessing)
# ------------------------------------------------------------


def _preprocess_variants_for_markers(image_bgr: np.ndarray, mode: str) -> list[tuple[str, np.ndarray]]:
    """Generate preprocessed variants to improve marker detection robustness."""

    if mode == "none":
        return [("bgr", image_bgr)]

    variants: list[tuple[str, np.ndarray]] = [("bgr", image_bgr)]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

    if mode in ("basic", "morph"):
        # illumination-robust contrast
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            g2 = clahe.apply(gray)
            variants.append(("clahe", cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass
        # adaptive threshold (contour-based)
        try:
            bw = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                51,
                5,
            )
            variants.append(("adaptive", cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass

    if mode == "morph":
        try:
            # Morphological closing/opening to reduce noise and connect marker blobs
            k = max(3, int(round(min(image_bgr.shape[:2]) * 0.004)))
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 5)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            variants.append(("adaptive_morph", cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass

    return variants


def detect_formA_marker_boxes(image_bgr: np.ndarray, preproc_mode: str = "none") -> list[dict[str, Any]]:
    """Try marker detection with optional preprocessing variants."""

    best: list[dict[str, Any]] = []
    best_score = -1.0
    for name, var in _preprocess_variants_for_markers(image_bgr, preproc_mode):
        markers = _detect_formA_marker_boxes_base(var)
        # prefer complete detection
        ok = len(markers) == 3
        score = float(sum(m.get("score", 0.0) for m in markers))
        if ok:
            score += 10.0
        # prefer later preprocessing only slightly
        if name != "bgr":
            score += 0.05
        if score > best_score:
            best_score = score
            best = markers
    return best


# ------------------------------------------------------------
# QR detection (robust)
# ------------------------------------------------------------


def _try_decode_qr(qr: cv2.QRCodeDetector, img: np.ndarray) -> Optional[tuple[str, np.ndarray]]:
    """Return (data, points) if detected."""

    try:
        # Multi first (some builds behave better)
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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g2 = clahe.apply(gray)
        candidates.append(("clahe", cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)))
    except Exception:
        pass

    # Otsu binarize
    try:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(("otsu", cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)))
    except Exception:
        pass

    # Adaptive + Morphology (uneven lighting / blur)
    try:
        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            5,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        bw2 = cv2.morphologyEx(bw2, cv2.MORPH_OPEN, kernel)
        candidates.append(("adaptive_morph", cv2.cvtColor(bw2, cv2.COLOR_GRAY2BGR)))
    except Exception:
        pass

    # scales: include upscales (QR may become too small after rectification/rotation)
    base_scales = [1.0, 0.75, 0.5, 0.25]
    up_scales = [1.5, 2.0]
    scales = base_scales + (up_scales if max(h0, w0) < 1800 else [])

    for prep_name, img in candidates:
        h, w = img.shape[:2]
        for s in scales:
            if abs(s - 1.0) < 1e-9:
                test = img
            else:
                new_w = int(round(w * s))
                new_h = int(round(h * s))
                if new_w < 120 or new_h < 120:
                    continue
                if new_w > 6000 or new_h > 6000:
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
    """WeChat engine based detection (best for tiny/low-res QR).

    Returns empty list if detector is not available.
    """

    if wechat is None:
        return []
    try:
        return wechat.detect(image_bgr)
    except Exception:
        return []


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

    # Minimum set (fast). Prefer downscales for stability.
    scales = [1.0, 0.5, 0.25, 0.75]
    if max(h, w) < 1400:
        scales = scales + [1.5]

    for prep, src in candidates:
        for s in scales:
            if abs(s - 1.0) < 1e-9:
                test = src
            else:
                new_w = int(round(w * s))
                new_h = int(round(h * s))
                if new_w < 120 or new_h < 120:
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
# Form decision
# ------------------------------------------------------------


@dataclass
class FormDecision:
    ok: bool
    form: Optional[str]
    angle_deg: Optional[float]
    score: float
    detail: dict[str, Any]


@dataclass
class MarkerGeometryConfig:
    """Constraints to reduce false-positive Form A detections (e.g. Form C misread as A)."""

    # Marker bbox areas should be similar (avoid cases where 1 big blob + 2 tiny noise boxes)
    max_marker_area_ratio: float = 3.0  # max(area)/min(area)

    # Marker size relative to page (rectified image)
    min_marker_area_page_ratio: float = 5e-5
    max_marker_area_page_ratio: float = 5e-3

    # Triangle shape constraint (TL-TR vs TL-BL)
    # Expect dist(TL,TR) / dist(TL,BL) ~= (page_w / page_h)
    max_dist_ratio_relative_error: float = 0.35


def validate_formA_marker_geometry(
    image_bgr: np.ndarray,
    markers: list[dict[str, Any]],
    cfg: MarkerGeometryConfig,
) -> tuple[bool, dict[str, Any]]:
    """Apply additional geometric/scale constraints for Form A markers.

    Returns:
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

    # Triangle distance ratio: require TL/TR/BL centers exist
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

    # Extra constraints to suppress C->A false positives
    cfg = geom_cfg or MarkerGeometryConfig()
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
        # dist in [0..sqrt(2)] -> normalize to [0..1]
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
    qrs = detect_qr_codes_wechat(image_bgr, getattr(score_formB, "_wechat", None))
    if not qrs:
        qrs = detect_qr_codes_robust(image_bgr)
    if not qrs:
        return False, 0.0, {"qrs": []}

    # Prefer "QR at top-right" (in the current rotated image),
    # but do NOT hard-fail if it's slightly off. We instead score by proximity.
    h, w = image_bgr.shape[:2]
    pts = np.asarray(qrs[0]["points"], dtype=np.float32).reshape(-1, 2)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())

    # Score: base 1.0 + relative QR size (encourage clearer detections)
    area = float(cv2.contourArea(pts.astype(np.float32)))
    rel = area / float(max(1, w * h))

    # Proximity to top-right (0..1): higher is better
    # - x: closer to right => larger
    # - y: closer to top   => larger
    x_score = cx / float(max(1, w))
    y_score = 1.0 - (cy / float(max(1, h)))
    pos_score = 0.6 * x_score + 0.4 * y_score

    return True, 1.0 + rel * 10.0 + pos_score, {
        "qrs": qrs,
        "qr_center": [cx, cy],
        "qr_rel_area": rel,
        "qr_pos_score": pos_score,
        "qr_engine": str(qrs[0].get("engine", "opencv")) if qrs else "",
    }


def score_formB_fast(image_bgr: np.ndarray) -> tuple[bool, float, dict[str, Any]]:
    """回転スキャン時の高速判定用（QRがある角度候補を絞る）。"""

    qrs = detect_qr_codes_fast(image_bgr)
    if not qrs:
        return False, 0.0, {"qrs": []}

    h, w = image_bgr.shape[:2]
    pts = np.asarray(qrs[0]["points"], dtype=np.float32).reshape(-1, 2)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())

    # fast score: bias towards top-right but keep it mild
    x_score = cx / float(max(1, w))
    y_score = 1.0 - (cy / float(max(1, h)))
    pos_score = 0.6 * x_score + 0.4 * y_score
    return True, 1.0 + pos_score, {"qrs": qrs, "qr_center": [cx, cy], "qr_pos_score": pos_score}


def decide_form_by_rotations(
    rectified_bgr: np.ndarray,
    angles: list[float],
    max_workers: int = 8,
    marker_preproc: str = "none",
    unknown_score_threshold: float = 0.0,
    unknown_margin: float = 0.0,
) -> FormDecision:
    """Coarse-to-fine rotation scan; return best valid decision.

    NOTE:
    - まず 0/90/180/270 の 4 回で大まかな向き（縦横/上下逆）を推定し、
      その近傍だけ細かく探索する（計算量削減）。
    - QR の robust 検出は重いため、候補角度のみ robust で再検証する。
    - A/B どちらもスコアが低い（閾値未満） or 近すぎる場合は Unknown 扱い。
    """

    def _eval(angle: float) -> dict[str, Any]:
        rotated = rotate_image_bound(rectified_bgr, angle)
        # Enforce landscape after rotation (ユーザー要望: 横長に統一)
        rotated, _ = enforce_landscape(rotated)
        h, w = rotated.shape[:2]
        if h > w:
            return {"angle": float(angle), "skip": True}

        okA, scoreA, detA = score_formA(rotated, marker_preproc=marker_preproc)
        okBf, scoreBf, detBf = score_formB_fast(rotated)

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

    # ----------------------------------
    # Coarse pass: 0/90/180/270 only
    # ----------------------------------

    coarse = [0.0, 90.0, 180.0, 270.0]
    coarse_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(coarse))) as ex:
        futures = [ex.submit(_eval, a) for a in coarse]
        for fut in as_completed(futures):
            r = fut.result()
            if not r or r.get("skip"):
                continue
            coarse_results.append(r)

    if not coarse_results:
        return FormDecision(False, None, None, 0.0, {"reason": "coarse_all_skipped"})

    # pick top-2 coarse angles by max(A_score, B_fast_score)
    coarse_sorted = sorted(
        coarse_results,
        key=lambda rr: max(float(rr["A"]["score"]), float(rr["B_fast"]["score"])),
        reverse=True,
    )
    coarse_top = coarse_sorted[:2]
    base_angles = [float(r["angle"]) for r in coarse_top]

    # ----------------------------------
    # Fine pass: only near base angles (union)
    # ----------------------------------

    window = 50.0  # a bit wider to avoid missing due to detection sensitivity

    # Align fine angles to the user-provided angle list (0..350 step N),
    # rather than generating new angles like 355.
    fine_set: set[float] = set()
    for ba in base_angles:
        for a in angles:
            if _circular_dist_deg(a, ba) <= window:
                fine_set.add(float(a))
    fine = sorted(fine_set)
    if not fine:
        # fallback to full list
        fine = list(angles)

    bestA: Optional[FormDecision] = None
    bestB_fast: Optional[FormDecision] = None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_eval, a) for a in fine]
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

    # If none found (fine pass), fallback to full scan (original behaviour)
    if bestA is None and bestB_fast is None:
        # Full scan is still just 36 angles by default; this prevents regression.
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_eval, a) for a in angles]
            for fut in as_completed(futures):
                r = fut.result()
                if not r or r.get("skip"):
                    continue
                angle = float(r["angle"])
                if r["A"]["ok"]:
                    candA = FormDecision(True, "A", angle, float(r["A"]["score"]), {"A": r["A"]["detail"], "phase": "fallback_full"})
                    if bestA is None or candA.score > bestA.score:
                        bestA = candA
                if r["B_fast"]["ok"]:
                    candB = FormDecision(True, "B", angle, float(r["B_fast"]["score"]), {"B_fast": r["B_fast"]["detail"], "phase": "fallback_full"})
                    if bestB_fast is None or candB.score > bestB_fast.score:
                        bestB_fast = candB

        if bestA is None and bestB_fast is None:
            # FAST で全く検出できない場合でも、robust 側だと拾えるケースがある。
            # まずは 0/90/180/270 のみ robust rescue を実施して、
            # 見つかれば B として確定する。
            rescue_angles = [0.0, 90.0, 180.0, 270.0]
            bestB_rescue: Optional[FormDecision] = None
            for aa in rescue_angles:
                rotated = rotate_image_bound(rectified_bgr, aa)
                rotated, _ = enforce_landscape(rotated)
                if rotated.shape[0] > rotated.shape[1]:
                    continue
                okB, scoreB, detB = score_formB(rotated)
                if not okB:
                    continue
                cand = FormDecision(True, "B", float(aa), float(scoreB), {"B": detB, "rescue": True, "phase": "no_detection_rescue"})
                if bestB_rescue is None or cand.score > bestB_rescue.score:
                    bestB_rescue = cand
            if bestB_rescue is not None:
                return bestB_rescue

            return FormDecision(
                False,
                None,
                None,
                0.0,
                {
                    "reason": "no_detection",
                    "coarse": coarse_results,
                    "fine_angles": fine,
                },
            )

    # Unknown decision: low score or ambiguous
    # (If either is None, treat score as -inf for margin comparison)
    a_score = float(bestA.score) if bestA is not None else float("-inf")
    b_score = float(bestB_fast.score) if bestB_fast is not None else float("-inf")

    # Threshold: if the top score is below threshold, declare Unknown.
    top_score = max(a_score, b_score)
    if float(unknown_score_threshold) > 0 and top_score < float(unknown_score_threshold):
        return FormDecision(False, None, None, float(top_score), {"reason": "below_threshold", "a_score": a_score, "b_score": b_score})

    # Margin: if too close, declare Unknown.
    if float(unknown_margin) > 0 and bestA is not None and bestB_fast is not None:
        if abs(a_score - b_score) < float(unknown_margin):
            return FormDecision(False, None, None, float(top_score), {"reason": "ambiguous", "a_score": a_score, "b_score": b_score})

    # If A found and it looks reliable, prefer A.
    if bestA is not None and (bestB_fast is None or bestA.score >= bestB_fast.score):
        return bestA

    if bestB_fast is None:
        # FAST で QR が全く見つからない場合でも、
        # 透視補正/回転で QR が小さくなり FAST が落ちるケースがある。
        # ここでは最低限の角度（90度刻み）だけ ROBUST を試して救済する。
        rescue_angles = [0.0, 90.0, 180.0, 270.0]
        bestB_rescue: Optional[FormDecision] = None
        for aa in rescue_angles:
            rotated = rotate_image_bound(rectified_bgr, aa)
            rotated, _ = enforce_landscape(rotated)
            if rotated.shape[0] > rotated.shape[1]:
                continue
            okB, scoreB, detB = score_formB(rotated)
            if not okB:
                continue
            cand = FormDecision(True, "B", float(aa), float(scoreB), {"B": detB, "rescue": True})
            if bestB_rescue is None or cand.score > bestB_rescue.score:
                bestB_rescue = cand
        if bestB_rescue is not None:
            return bestB_rescue

        return FormDecision(False, None, None, 0.0, {})

    # Refine B with robust detection only for a few angles.
    # Start from bestB_fast, then try its neighbours (±step).
    step = float(angles[1] - angles[0]) if len(angles) >= 2 else 10.0
    candidates = [bestB_fast.angle_deg]
    if bestB_fast.angle_deg is not None:
        candidates += [bestB_fast.angle_deg - step, bestB_fast.angle_deg + step]

    bestB: Optional[FormDecision] = None
    for a in candidates:
        if a is None:
            continue
        # normalize to 0..360
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

    # fallback: if A exists, return it, else fail
    if bestA is not None:
        return bestA
    return FormDecision(False, None, None, 0.0, {})


# ------------------------------------------------------------
# Template caching + global prefilter
# ------------------------------------------------------------


def compute_global_descriptor(image_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    """Compute a cheap global descriptor for template pre-filtering.

    Current design:
      - resize to fixed max side
      - grayscale histogram (64 bins)
      - HSV histogram (H:24, S:16)

    Returns: 1D float32 vector normalized to unit length.
    """

    img, _ = resize_keep_aspect(image_bgr, max_side=int(size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).reshape(-1)
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).reshape(-1)
    g_hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).reshape(-1)
    v = np.concatenate([h_hist, s_hist, g_hist]).astype(np.float32)
    n = float(np.linalg.norm(v))
    if n > 1e-6:
        v /= n
    return v


@dataclass
class CachedRef:
    template_path: str
    ref_small: np.ndarray
    s_ref: float
    out0: dict[str, Any]
    global_desc: np.ndarray


class CachedXFeatMatcher:
    """XFeat matcher with cached template features."""

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
        g = compute_global_descriptor(template_bgr)
        return CachedRef(template_path=str(template_path), ref_small=ref_small, s_ref=float(s_ref), out0=out0, global_desc=g)

    def match_with_cached_ref(
        self,
        ref: CachedRef,
        tgt_bgr: np.ndarray,
    ) -> tuple[Any, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (XFeatHomographyResult-like, H_full, mkpts0, mkpts1)."""

        # Resize target for matching stability/speed (then scale H back)
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
                # minimal shape-compatible object (reuse dataclass from test_recovery_paper is heavy to import here)
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
            3.5,
            maxIters=1_000,
            confidence=0.999,
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

        # scale H back to full resolution: H_full = inv(S_tgt) * H_small * S_ref
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
    if top_n <= 0 or top_n >= len(templates):
        return templates
    dists = []
    for t in templates:
        d = float(np.linalg.norm(target_desc - t.global_desc))
        dists.append((d, t))
    dists.sort(key=lambda x: x[0])
    return [t for _, t in dists[:top_n]]


# ------------------------------------------------------------
# Homography inversion safety
# ------------------------------------------------------------


def safe_invert_homography(
    H: np.ndarray,
    inliers: int,
    inlier_ratio: float,
    min_inliers: int,
    min_inlier_ratio: float,
    max_cond: float,
) -> tuple[bool, Optional[np.ndarray], str, float, float]:
    """Safely invert homography.

    - reject if inliers too small
    - reject if inlier ratio too small
    - reject if matrix is near singular (det close to 0 or cond too large)
    """

    if int(inliers) < int(min_inliers):
        return False, None, f"inliers<{min_inliers} ({inliers})", float("nan"), float("nan")
    if float(inlier_ratio) < float(min_inlier_ratio):
        return False, None, f"inlier_ratio<{min_inlier_ratio:.3f} ({inlier_ratio:.3f})", float("nan"), float("nan")

    H = np.asarray(H, dtype=np.float64)
    det = float(np.linalg.det(H))
    if not math.isfinite(det) or abs(det) < 1e-12:
        return False, None, f"det too small ({det:.3e})", float("nan"), float(det)
    try:
        cond = float(np.linalg.cond(H))
        if not math.isfinite(cond) or (max_cond > 0 and cond > float(max_cond)):
            return False, None, f"cond too large ({cond:.3e})", float(cond), float(det)
    except Exception:
        # cond computation can fail; still try inversion with exception handling
        cond = float("nan")

    try:
        H_inv = np.linalg.inv(H)
        return True, H_inv, "ok", float(cond), float(det)
    except Exception as e:
        return False, None, f"inv failed: {e}", float(cond), float(det)


# ------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------


def _bool_to_str(v: Any) -> str:
    if v is None:
        return ""
    return "TRUE" if bool(v) else "FALSE"


def _to_json_cell(v: Any) -> str:
    """Serialize a complex python object for a single CSV cell."""

    if v is None:
        return ""
    try:
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(v)


def _path_or_empty(p: Any) -> str:
    if not p:
        return ""
    return str(p)


def _filename_only(p: Any) -> str:
    """Return only the filename part (no directory)."""

    if not p:
        return ""
    try:
        return Path(str(p)).name
    except Exception:
        return str(p)


def _filenames_only_list(v: Any) -> list[str]:
    """Convert list of paths -> list of filenames."""

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


def compute_reproj_rms(H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    """Compute RMS reprojection error (px)."""

    try:
        src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 1, 2)
        dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(src, np.asarray(H, dtype=np.float64))
        err = np.linalg.norm(proj - dst, axis=2).reshape(-1)
        return float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")
    except Exception:
        return float("nan")


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
    """Ground-truth definition for this pipeline.

    - If src_form is A/B, the expected decided form is same.
    - Expected template is the same file stem (1..6) within that form.
    - If src_form is C (or others), ground truth is unknown.
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
    """Build a rich, analysis-friendly CSV row.

    Column names are intentionally verbose so that anyone can understand them.
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

    # NOTE: CSV は「フルパス禁止」の要望に従い、原則 filename のみ出力する。
    src_filename = _filename_only(item.get("source_path"))

    expected_behavior_label = ""
    if src_form == "C":
        expected_behavior_label = "C_should_be_rejected_as_form_unknown"
    elif src_form in ("A", "B"):
        expected_behavior_label = "A_or_B_should_be_correct_form_and_template_and_warp"
    else:
        expected_behavior_label = "unknown_source_form"

    row: dict[str, Any] = {
        # ---- identity (short, human-friendly) ----
        "case_id": str(item.get("case") or ""),
        "source_form_folder_name(A_or_B_or_C)": src_form,
        "source_image_filename": src_filename,
        "source_image_filename_stem": str(src_path.stem) if src_path else "",
        "degraded_variant_index": str(item.get("degraded_variant_index") or ""),

        # ---- ground truth (A/B only) ----
        "ground_truth_source_form(A_or_B)": gt_form,
        "ground_truth_source_template_filename(if_A_or_B)": gt_template_filename,
        "ground_truth_source_template_number(if_A_or_B)": gt_template_number,

        # ---- predictions ----
        "predicted_decided_form(A_or_B_or_empty)": predicted_form,
        "predicted_decided_rotation_angle_deg": "" if predicted_angle is None else str(predicted_angle),
        "predicted_best_template_filename": best_template_filename,
        "predicted_best_template_number": best_template_number,

        # ---- correctness (A/B only) ----
        "is_predicted_form_correct": _bool_to_str(is_form_correct) if gt_form else "",
        "is_predicted_best_template_correct": _bool_to_str(is_template_correct) if gt_form else "",

        # ---- pipeline status ----
        "pipeline_final_ok(warp_done)": _bool_to_str(item.get("ok_warp")),
        "pipeline_final_ok(expected_behavior)": _bool_to_str(item.get("ok")),
        "pipeline_stop_stage": str(item.get("stage") or ""),
        "pipeline_expected_behavior_label": expected_behavior_label,
        "pipeline_predicted_form_raw(A_or_B_or_empty)": str(item.get("predicted_form") or ""),

        # ---- timings ----
        "elapsed_time_total_one_case_seconds": f"{float(item.get('case_total_s', 0.0)):.6f}",
        "elapsed_time_stage_1_degrade_seconds": f"{times.degrade_s:.6f}",
        "elapsed_time_stage_2_docaligner_seconds": f"{times.docaligner_s:.6f}",
        "elapsed_time_stage_3_rectify_seconds": f"{times.rectify_s:.6f}",
        "elapsed_time_stage_4_form_decision_seconds": f"{times.decide_s:.6f}",
        "elapsed_time_stage_5_xfeat_matching_seconds": f"{times.match_s:.6f}",
        "elapsed_time_stage_6_warp_seconds": f"{times.warp_s:.6f}",

        # ---- run metadata (no full paths) ----
        "run_id": str(item.get("run_id") or ""),
        "run_output_root_directory_name": _filename_only(item.get("run_output_root_directory")),
        "run_elapsed_time_total_seconds": str(item.get("run_elapsed_time_total_seconds") or ""),

        # ---- output filenames (no full paths) ----
        "output_degraded_image_filename": _filename_only(item.get("output_degraded_image_path")),
        "output_doc_overlay_image_filename": _filename_only(item.get("output_doc_overlay_image_path")),
        "output_rectified_image_filename": _filename_only(item.get("output_rectified_image_path")),
        "output_rotated_decision_visualization_image_filename": _filename_only(item.get("output_rotated_decision_visualization_image_path")),
        "output_debug_matches_image_filename": _filename_only(item.get("output_debug_matches_image_path")),
        "output_aligned_image_filename": _filename_only(item.get("output_aligned_image_path")),

        # ---- images: resolutions ----
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

        # ---- degradation parameters (rich) ----
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

        # ---- DocAligner ----
        "docaligner_polygon_xy_json": _to_json_cell(item.get("polygon")),
        "docaligner_polygon_margin_mode(ratio_or_fixed_px)": str(poly_margin.get("mode") or ""),
        "docaligner_polygon_margin_computed_px": str(poly_margin.get("computed_px") or poly_margin.get("value") or ""),
        "docaligner_polygon_margin_details_json": _to_json_cell(poly_margin),

        # ---- form decision debug ----
        "form_decision_score": str(dec.get("score") or ""),
        "form_decision_detail_json": _to_json_cell(dec.get("detail")),

        # ---- XFeat best match ----
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

        # ---- homography stability ----
        "homography_inversion_ok": _bool_to_str(inv.get("ok")),
        "homography_inversion_reject_reason": str(inv.get("reason") or ""),
        "homography_matrix_condition_number": str(inv.get("cond") or ""),
        "homography_matrix_determinant": str(inv.get("det") or ""),

        # ---- run configuration (selected, for quick filtering) ----
        "run_config_rotation_step_deg": str(getattr(args, "rotation_step", "")),
        "run_config_template_topn": str(getattr(args, "template_topn", "")),
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

    # exception info (if any)
    if item.get("stage") == "exception":
        row["exception_error_message"] = str(item.get("error") or "")
        row["exception_traceback"] = str(item.get("traceback") or "")
    else:
        row["exception_error_message"] = ""
        row["exception_traceback"] = ""

    return row


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(form: str) -> list[Path]:
    base = Path(__file__).resolve().parent / "image" / form
    paths: list[Path] = []
    for i in range(1, 7):
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

    p.add_argument("--src-forms", type=str, default="A,B,C", help="Input forms to degrade from: comma separated (A,B,C)")
    p.add_argument("--degrade-n", type=int, default=1, help="Number of degraded variants per source image")
    p.add_argument("--degrade-w", type=int, default=2400)
    p.add_argument("--degrade-h", type=int, default=1800)
    p.add_argument("--max-rot", type=float, default=180.0, help="Degradation rotation. >=180 enables 0-360 uniform rotation.")
    p.add_argument("--min-abs-rot", type=float, default=0.0)
    p.add_argument("--rotation-mode", choices=["uniform", "snap"], default="uniform")
    p.add_argument("--snap-step-deg", type=float, default=90.0)
    p.add_argument("--perspective", type=float, default=0.08)
    p.add_argument("--min-visible-area-ratio", type=float, default=0.25)
    p.add_argument("--max-attempts", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # WeChat QR models
    p.add_argument(
        "--wechat-model-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "models" / "wechat_qrcode"),
        help="WeChat QRCode Engine のモデルディレクトリ（detect/sr の prototxt/caffemodel を配置）",
    )

    # 回転スキャン（ユーザー要件: 0..350 を10度刻み）
    p.add_argument("--rotation-step", type=float, default=10.0, help="フォーム判定の回転スキャン刻み（度）")
    p.add_argument("--rotation-max-workers", type=int, default=8, help="回転スキャンの並列数（スレッド）")

    p.add_argument("--docaligner-model", choices=["lcnet050", "lcnet100", "fastvit_t8", "fastvit_sa24"], default="fastvit_sa24")
    p.add_argument("--docaligner-type", choices=["point", "heatmap"], default="heatmap")
    # 透視補正後の紙画像が小さすぎると QR が潰れて検出しづらいので、デフォルトは少し大きめ。
    p.add_argument("--docaligner-max-side", type=int, default=2400, help="Max side length for rectified paper")
    # (1) polygon margin: ratio-based to be robust across resolutions
    p.add_argument(
        "--polygon-margin-ratio",
        type=float,
        default=0.03,
        help=(
            "DocAligner polygon を外側に広げるマージン（紙サイズに対する比率）。"
            " 例: 0.03 は紙の長辺の 3% をマージンにする。"
        ),
    )
    p.add_argument(
        "--polygon-margin-min-px",
        type=float,
        default=10.0,
        help="ratio-based マージンの下限(px)",
    )
    p.add_argument(
        "--polygon-margin-max-px",
        type=float,
        default=200.0,
        help="ratio-based マージンの上限(px)（0以下で無制限）",
    )
    p.add_argument(
        "--polygon-margin-px",
        type=float,
        default=0.0,
        help="互換用: 固定pxマージン（>0 の場合 ratio を上書き）",
    )

    # (2) logging
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    p.add_argument("--console-log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    # (3) extra preprocessing
    p.add_argument(
        "--marker-preproc",
        choices=["none", "basic", "morph"],
        default="basic",
        help="フォームAマーカー検出の前処理（照明ムラ対策）",
    )

    # (4) template caching / prefilter
    p.add_argument(
        "--template-topn",
        type=int,
        default=3,
        help="グローバル特徴で候補テンプレを絞り込む上位N（0で全探索）",
    )

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--top-k", type=int, default=1024, help="XFeatの特徴点数（大きいほど高精度だが遅い）")
    p.add_argument("--match-max-side", type=int, default=1200, help="XFeat用にリサイズする最大辺(px)（大きいほど高精度だが遅い）")

    # (6) Unknown threshold
    p.add_argument(
        "--unknown-score-threshold",
        type=float,
        default=1.2,
        help="フォーム判定スコアがこの値未満なら Unknown 扱い",
    )
    p.add_argument(
        "--unknown-margin",
        type=float,
        default=0.15,
        help="A/B スコア差がこの値未満なら Unknown 扱い（曖昧）",
    )

    # (7) homography stability
    p.add_argument("--min-inliers-for-warp", type=int, default=10, help="warp を許可する最小 inlier 数")
    p.add_argument("--min-inlier-ratio-for-warp", type=float, default=0.15, help="warp を許可する最小 inlier_ratio")
    p.add_argument("--max-h-cond", type=float, default=1e6, help="Homography 行列の条件数上限（大きいと不安定）")

    p.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "output_pipeline"))
    p.add_argument("--limit", type=int, default=0, help="Debug: limit number of source images per form (0=all)")

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
        f"  --template-topn       グローバル特徴でテンプレ候補を絞る上位N（0で全探索） [default: {defaults.template_topn}]",
        "",
        "【ログ】",
        f"  --log-level           ログレベル (DEBUG/INFO/WARNING/ERROR) [default: {defaults.log_level}]",
        f"  --console-log-level   コンソールログレベル (DEBUG/INFO/WARNING/ERROR) [default: {defaults.console_log_level}]",
        "",
        "【出力】",
        f"  --out                 出力ディレクトリ（run_... が作成される） [default: {defaults.out}]",
        "",
        "最小コマンド例（おすすめデフォルト使用）:",
        r"  C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v4.py --limit 1",
        "",
    ]
    print("\n".join(lines))


def log_case_summary(logger: logging.Logger, row: dict[str, Any]) -> None:
    """Always print one-line case summary for readability."""

    case_id = str(row.get("case_id") or "")
    # user-facing ok = expected-behavior success
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

    t_total = str(row.get("elapsed_time_total_one_case_seconds") or "")
    t1 = str(row.get("elapsed_time_stage_1_degrade_seconds") or "")
    t2 = str(row.get("elapsed_time_stage_2_docaligner_seconds") or "")
    t3 = str(row.get("elapsed_time_stage_3_rectify_seconds") or "")
    t4 = str(row.get("elapsed_time_stage_4_form_decision_seconds") or "")
    t5 = str(row.get("elapsed_time_stage_5_xfeat_matching_seconds") or "")
    t6 = str(row.get("elapsed_time_stage_6_warp_seconds") or "")

    # If ground truth is unavailable (e.g., C), keep correctness columns blank.
    truth_part = f"gt_form={gt_form} pred_form={pred_form}"
    if gt_form:
        truth_part += f" form_ok={form_ok} template_ok={template_ok}"

    msg = (
        f"[CASE] id={case_id} ok={ok} ok_warp={ok_warp} stage={stage} {truth_part} "
        f"best_template={best_tpl_name} inliers={inliers} inlier_ratio={inlier_ratio} "
        f"time_total_s={t_total} (1_degrade={t1},2_doc={t2},3_rectify={t3},4_decide={t4},5_match={t5},6_warp={t6}) "
        f"src={src}"
    )

    if ok == "TRUE":
        logger.info(msg)
    else:
        # failure is important for later analysis
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
    """Print dataset-level statistics at the end of the log.

    Focus:
      - expected-behavior success (ユーザー要望の主KPI)
      - A/B form+template accuracy
      - C rejection success rate (should be form_unknown)
      - false positive analysis: C predicted as A/B
      - stage time averages (mean/median)
    """

    total = len(summary)
    if total == 0:
        logger.info("[STATS] no cases")
        return

    ok_warp = sum(1 for s in summary if bool(s.get("ok_warp")))
    ok_expected = sum(1 for s in summary if bool(s.get("ok")))

    # Per source form buckets
    by_src: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": [], "other": []}
    for s in summary:
        sf = str(s.get("source_form") or "")
        if sf in by_src:
            by_src[sf].append(s)
        else:
            by_src["other"].append(s)

    # A/B accuracy (form correct, template correct)
    def _count_true(items: list[dict[str, Any]], key: str) -> int:
        return sum(1 for it in items if bool(it.get(key)))

    a_items = by_src["A"]
    b_items = by_src["B"]
    c_items = by_src["C"]

    a_form_ok = _count_true(a_items, "is_predicted_form_correct")
    b_form_ok = _count_true(b_items, "is_predicted_form_correct")
    a_tpl_ok = _count_true(a_items, "is_predicted_best_template_correct")
    b_tpl_ok = _count_true(b_items, "is_predicted_best_template_correct")

    # C should be rejected as form_unknown
    c_reject_ok = sum(1 for it in c_items if str(it.get("stage")) == "form_unknown")
    c_fp_as_A = sum(1 for it in c_items if str(it.get("predicted_form") or "") == "A")
    c_fp_as_B = sum(1 for it in c_items if str(it.get("predicted_form") or "") == "B")

    # Stage timing per-case (mean/median)
    t_total_cases = [float(s.get("case_total_s", 0.0)) for s in summary if s.get("case_total_s") is not None]
    t1 = [float(s.get("stage_times", {}).get("degrade_s", 0.0)) for s in summary if isinstance(s.get("stage_times"), dict)]
    # If stage_times are not embedded, fallback to overall sums / total.

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

    # Stage time aggregates
    logger.info("[STATS] stage time totals (s) (same as SUMMARY)")
    for k, v in stage_times.items():
        logger.info("  %-12s : %.2f", k, float(v))

    # Means from aggregate totals
    logger.info("[STATS] stage time mean per case (s)")
    for k, v in stage_times.items():
        logger.info("  %-12s : %.3f", k, float(v) / float(total))

    # Total time mean/median
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
    print(f"  template-topn       : {args.template_topn}")
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
# main pipeline split
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
    """Process one degraded variant of one source image."""

    case_t0 = time.perf_counter()
    case_id = f"{src_form}_{src_path.stem}_deg{k:02d}"
    item: dict[str, Any] = {
        "source_form": src_form,
        "source_path": str(src_path),
        "case": case_id,
        # NOTE:
        #   ユーザー要望に合わせて ok の意味を変更する：
        #     ok      = 期待動作として成功したか（C は form_unknown が成功）
        #     ok_warp = warp まで到達したか（aligned 出力が生成されたか）
        "ok": False,
        "ok_warp": False,
        "stage": "start",
        "degraded_variant_index": int(k),
    }
    times = StageTimes()

    # source resolution
    try:
        h0, w0 = src_bgr.shape[:2]
        item["source_w"] = int(w0)
        item["source_h"] = int(h0)
    except Exception:
        pass

    # stable RNG per case
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

    # (1) polygon margin: ratio-based by default
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
    # quick access (for logging/statistics)
    item["predicted_form"] = str(decision.form or "")
    item["predicted_angle_deg"] = "" if decision.angle_deg is None else float(decision.angle_deg)

    if not decision.ok or decision.form not in ("A", "B") or decision.angle_deg is None:
        item["stage"] = "form_unknown"
        # Expected behaviour:
        # - A/B: should NOT become form_unknown
        # - C  : should become form_unknown (paper detected but not A/B)
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

    # visualize decision evidence
    if decision.form == "A":
        markers = ((decision.detail or {}).get("A") or {}).get("markers") or []
        rot_vis = draw_formA_markers_overlay(chosen, markers)
    else:
        qrs = ((decision.detail or {}).get("B") or {}).get("qrs")
        if not qrs:
            qrs = detect_qr_codes_robust(chosen)
        rot_vis = draw_formB_qr_overlay(chosen, qrs)
    out_rot = out_dirs["rot"] / f"{case_id}_rot.jpg"
    cv2.imwrite(str(out_rot), rot_vis)
    item["output_rotated_decision_visualization_image_path"] = str(out_rot)

    # 5) XFeat matching (template caching + prefilter)
    t0 = time.perf_counter()
    templates = templates_A if decision.form == "A" else templates_B
    best: Optional[dict[str, Any]] = None

    # prefilter templates by global descriptor
    target_desc = compute_global_descriptor(chosen)
    candidates = select_top_templates(target_desc, templates, top_n=int(args.template_topn))
    item["template_prefilter"] = {
        "topn": int(args.template_topn),
        "candidates": [c.template_path for c in candidates],
        "total": len(templates),
    }

    template_candidate_results: list[dict[str, Any]] = []

    for ref in candidates:
        tp = Path(ref.template_path)
        tpl_bgr = cv2.imread(str(tp))
        if tpl_bgr is None:
            continue

        if cached_matcher is not None:
            res, H_tpl_to_img, mk0, mk1 = cached_matcher.match_with_cached_ref(ref, chosen)
        else:
            res, H_tpl_to_img, mk0, mk1 = matcher.match_and_estimate_h(tpl_bgr, chosen)

        ok = bool(getattr(res, "ok", False)) and H_tpl_to_img is not None
        cand = {
            "template": str(tp),
            "ok": ok,
            "inliers": int(getattr(res, "inliers", 0)),
            "matches": int(getattr(res, "matches", 0)),
            "inlier_ratio": float(getattr(res, "inlier_ratio", 0.0)),
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

    # (7) inverse homography stability
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

    # debug matches (best effort)
    # debug matches (best effort) + capture richer XFeat diagnostics for CSV
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
    # Expected-behaviour success:
    # - A/B: form correct AND template correct AND warp done
    # - C  : reaching "done" is actually a false-positive (should have been rejected)
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

    # Output root (create early so we can place log file)
    run_id = now_run_id()
    out_root = mkdir(Path(args.out) / f"run_{run_id}")
    logger = setup_logging(out_root, level=str(args.log_level), console_level=str(args.console_log_level))

    logger.info("=" * 70)
    logger.info("paper_pipeline_v4")
    logger.info("=" * 70)
    logger.info("OpenCV: %s", cv2.__version__)
    logger.info("torch : %s", torch.__version__)
    logger.info("src-forms: %s", args.src_forms)
    print_config(args)

    # Setup device for XFeat
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")
    ensure_portable_git_on_path()

    # Output directories (numbered)
    out_dirs = {
        "degraded": mkdir(out_root / "1_degraded"),
        "doc": mkdir(out_root / "2_doc"),
        "rect": mkdir(out_root / "3_rectified"),
        "rot": mkdir(out_root / "4_rectified_rot"),
        "debug_matches": mkdir(out_root / "5_debug_matches"),
        "aligned": mkdir(out_root / "6_aligned"),
    }

    # Load heavy models
    logger.info("[INFO] Loading DocAligner...")
    model, cb = load_docaligner_model(args.docaligner_model, args.docaligner_type)
    logger.info("[OK] DocAligner loaded")

    logger.info("[INFO] Loading XFeat...")
    matcher = XFeatMatcher(top_k=args.top_k, device=device, match_max_side=args.match_max_side)
    logger.info("[OK] XFeat loaded")

    # Initialize WeChat QR detector (optional)
    wechat = init_wechat_qr_detector(str(getattr(args, "wechat_model_dir", "")), logger=logger)
    # Bind to score_formB via function attribute to avoid threading/arg plumbing
    setattr(score_formB, "_wechat", wechat)

    # (4) template cache
    cached_matcher: Optional[CachedXFeatMatcher] = None
    try:
        cached_matcher = CachedXFeatMatcher(matcher)
        logger.info("[OK] CachedXFeatMatcher enabled")
    except Exception as e:
        logger.warning("[WARN] CachedXFeatMatcher disabled: %s", e)
        cached_matcher = None

    # Prepare angles for form detection
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

    # Templates for final alignment (A/B only)
    template_paths_A = list_images("A")
    template_paths_B = list_images("B")
    if not template_paths_A or not template_paths_B:
        logger.error("templates not found. Expected APA/image/A and APA/image/B")
        return 1

    # warm-up template cache
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
        # still need global desc for prefilter: build minimal cache
        for pth in template_paths_A:
            img = cv2.imread(str(pth))
            if img is None:
                continue
            templates_A.append(CachedRef(str(pth), img, 1.0, {}, compute_global_descriptor(img)))
        for pth in template_paths_B:
            img = cv2.imread(str(pth))
            if img is None:
                continue
            templates_B.append(CachedRef(str(pth), img, 1.0, {}, compute_global_descriptor(img)))
        logger.info("[OK] template global-desc cache built: A=%d B=%d", len(templates_A), len(templates_B))

    summary: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    t_all0 = time.perf_counter()

    # stage aggregates
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
                    # keep running even if a row build fails
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

    # Save summary
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    dt = time.perf_counter() - t_all0

    # Fill run elapsed (same value for all rows so filtering becomes easy)
    for r in csv_rows:
        r["run_elapsed_time_total_seconds"] = f"{dt:.6f}"
        r.setdefault("run_id", str(run_id))
        # NOTE: ユーザー要望により CSV にフルパスは出さない。
        # run 出力は run_id / run_output_root_directory_name で特定できる。

    # Write rich summary.csv
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

    # (2) stage summary
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
