#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""APA_back.py

APA.py から唯一 import される「重い実装」側モジュール。

目的
----
- `APA/apa_input` の画像を順に処理し、`9_demo` 相当の出力画像（左=入力+根拠、右=aligned）だけを生成する。
- 精度（アルゴリズム/ハイパーパラメータ）は `paper_pipeline_v18.py` の挙動から変更しない方針で、
  必要な実装をこのファイルへ集約する。

注意
----
- WeChat QR（`cv2.wechat_qrcode_WeChatQRCode`）には opencv-contrib とモデルファイルが必要。
- XFeat は torch.hub からロードする（環境により git が必要）。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import queue
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

try:
    from turbojpeg import TJPF_BGR, TurboJPEG

    _TURBOJPEG_IMPORT_OK = True
except Exception:
    TJPF_BGR = None  # type: ignore
    TurboJPEG = None  # type: ignore
    _TURBOJPEG_IMPORT_OK = False

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# デフォルト設定（paper_pipeline_v18.py と同値を維持）
# ============================================================


PIPELINE_DEFAULTS: dict[str, Any] = {
    "wechat": {
        "model_dir": str(Path(__file__).resolve().parent / "models" / "wechat_qrcode"),
    },
    "uvdoc": {
        "ckpt_path": str(Path(__file__).resolve().parent / "third_party" / "UVDoc" / "model" / "best_model.pkl"),
    },
    "background_division": {
        "enable": True,
        "sigma_ratio": 0.03,
        "sigma_min": 15.0,
        "sigma_max": 120.0,
        "bg_min": 6.0,
    },
    "xfeat": {
        "device_default": "cpu",
        "top_k": 3072,
        "match_max_side_px": 1400,
    },
    "rotation_scan": {
        "max_workers": 12,
        "scan_angles_2_deg": [0.0, 180.0],
    },
    "docaligner": {
        "model": "fastvit_sa24",
        "type": "heatmap",
        "rectified_max_side_px": 3200,
        "pad_px": 320,
        "pad_px_auto_ratio": 0.10,
        "pad_px_auto_min": 120,
        "pad_px_auto_max": 800,
        "multi": {
            "enable": True,
            "extra_models": ["fastvit_t8", "lcnet100"],
            "extra_types": ["heatmap", "point"],
            "pad_px_candidates": [240, 400, 650, 900],
            "input_scales": [0.75, 0.6, 1.15],
            "max_infer_runs": 8,
            "max_polygon_candidates": 4,
            "eval_max_candidates": 2,
            "eval_max_margins": 2,
            "eval_rotation_workers": 1,
        },
        "polygon_margin": {
            "ratio": 0.18,
            "min_px": 10.0,
            "max_px": 800.0,
            "fixed_px": 0.0,
        },
        "rectify_padding": {
            "enable": True,
            "pad_px": 800,
            "border_value": [0, 0, 0],
        },
        "advanced_fallback": {
            "enable": True,
            "trigger_on_form_unknown_no_detection": True,
        },
    },
    "marker": {
        "preproc_mode": "morph",
        "clahe": {"clipLimit": 3.0, "tileGridSize": [8, 8]},
        "adaptive_threshold": {"block_size": 61, "C": 3},
        "morph": {"kernel_ratio": 0.006, "kernel_min": 5},
    },
    "formA": {
        "geometry": {
            "max_marker_area_ratio": 3.0,
            "min_marker_area_page_ratio": 5e-5,
            "max_marker_area_page_ratio": 5e-3,
            "max_dist_ratio_relative_error": 0.35,
            "surround_pad_ratio": 2.0,
            "surround_pad_px_min": 8,
            "surround_pad_px_max": 120,
            "surround_min_mean_gray": 175.0,
            "surround_max_ink_ratio": 0.05,
            "surround_adaptive_block_size": 41,
            "surround_adaptive_C": 9,
        }
    },
    "qr": {
        "min_test_side_px": 120,
        "wechat": {
            "fast": {
                "variants": ["bgr", "gray", "clahe"],
                "scales": [0.75, 1.0, 1.25],
            },
            "robust": {
                "variants": ["bgr", "gray", "clahe", "adaptive_threshold", "adaptive_morph"],
                "scales": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            },
            "up_scale_enable_max_side_px": 1200,
            "max_test_side_px": 6500,
            "adaptive_morph_kernel": [5, 5],
        },
        "clahe": {"clipLimit": 3.0, "tileGridSize": [8, 8]},
        "adaptive_threshold": {"block_size": 61, "C": 3},
    },
    "homography": {
        "find": {
            "ransac_reproj_threshold_px": 4.0,
            "max_iters": 5000,
            "confidence": 0.999,
        },
        "invert": {
            "det_abs_min": 1e-12,
        },
    },
    "unknown": {
        "score_threshold": 1.2,
        "margin": 0.15,
    },
    "warp": {
        "min_inliers": 60,
        "min_inlier_ratio": 0.06,
        "max_h_cond": 1e6,
    },
    "save_images": {
        "jpeg_quality": 95,
    },
    "visual": {
        "polygon_line_thickness": 4,
        "polygon_point_radius": 10,
        "polygon_label_font_scale": 1.0,
        "polygon_label_thickness": 2,
    },
}


# ============================================================
# 共通ユーティリティ
# ============================================================


def ensure_portable_git_on_path() -> None:
    portable_git_bin = r"C:\Users\takumi\develop\git\bin"
    if os.path.exists(portable_git_bin):
        os.environ["PATH"] = portable_git_bin + os.pathsep + os.environ.get("PATH", "")


def now_run_id() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(*, log_dir: Path, level: str = "INFO", console_level: str = "INFO") -> logging.Logger:
    mkdir(log_dir)

    logger = logging.getLogger("APA")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_dir / "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(fh)

    return logger


_TURBOJPEG: Optional[Any] = None


def _get_turbojpeg() -> Optional[Any]:
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


def write_image(path: Path, image_bgr: np.ndarray, *, jpeg_quality: int = 95) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    ext = str(path.suffix).lower()
    if ext in (".jpg", ".jpeg"):
        tj = _get_turbojpeg()
        if tj is not None and TJPF_BGR is not None:
            try:
                buf = tj.encode(image_bgr, quality=int(jpeg_quality), pixel_format=TJPF_BGR)
                with open(path, "wb") as f:
                    f.write(buf)
                return True
            except Exception:
                pass
        try:
            return bool(cv2.imwrite(str(path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]))
        except Exception:
            return False

    try:
        return bool(cv2.imwrite(str(path), image_bgr))
    except Exception:
        return False


def resize_keep_aspect(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= int(max_side):
        return img, 1.0
    s = float(max_side) / float(m)
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, float(s)


def scale_matrix(s: float) -> np.ndarray:
    return np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


# ============================================================
# DocAligner（paper_pipeline_v18.py の必要部分を移植）
# ============================================================


def patch_capybara_exports() -> None:
    try:
        import capybara as cb
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing 'capybara' module (expected: capybara-docsaid).") from e

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


def load_docaligner_model(model_name: str, model_type: str) -> tuple[Any, Any]:
    patch_capybara_exports()
    import capybara as cb
    from docaligner import DocAligner, ModelType

    mtype = ModelType.heatmap if model_type == "heatmap" else ModelType.point
    model = DocAligner(model_type=mtype, model_cfg=model_name)
    return model, cb


def order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def _poly_signed_area_xy(poly_xy: np.ndarray) -> float:
    p = np.asarray(poly_xy, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    s = 0.0
    for i in range(len(p)):
        x1, y1 = float(p[i][0]), float(p[i][1])
        x2, y2 = float(p[(i + 1) % len(p)][0]), float(p[(i + 1) % len(p)][1])
        s += x1 * y2 - x2 * y1
    return 0.5 * float(s)


def _intersect_lines(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    return np.array([px, py], dtype=np.float32)


def _offset_quad_by_normals(poly_xy: np.ndarray, margin_px: float) -> Optional[np.ndarray]:
    poly = order_quad_tl_tr_br_bl(poly_xy).astype(np.float32)
    if margin_px <= 0:
        return poly
    area = _poly_signed_area_xy(poly)
    is_ccw = area > 0

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(4):
        p0 = poly[i]
        p1 = poly[(i + 1) % 4]
        v = (p1 - p0).astype(np.float32)
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        dx, dy = float(v[0] / n), float(v[1] / n)
        if is_ccw:
            nx, ny = dy, -dx
        else:
            nx, ny = -dy, dx
        off = np.array([nx, ny], dtype=np.float32) * float(margin_px)
        lines.append((p0 + off, p1 + off))

    out_pts: list[np.ndarray] = []
    for i in range(4):
        a1, a2 = lines[i - 1]
        b1, b2 = lines[i]
        inter = _intersect_lines(a1, a2, b1, b2)
        if inter is None:
            return None
        out_pts.append(inter)
    out = np.stack(out_pts, axis=0).astype(np.float32)
    return order_quad_tl_tr_br_bl(out)


def expand_polygon(polygon_xy: np.ndarray, margin_px: float, img_w: int, img_h: int) -> np.ndarray:
    poly = np.asarray(polygon_xy, dtype=np.float32).reshape(4, 2)
    if margin_px <= 0:
        return order_quad_tl_tr_br_bl(poly)

    out = _offset_quad_by_normals(poly, float(margin_px))
    if out is None:
        center = poly.mean(axis=0)
        pts: list[np.ndarray] = []
        for pt in poly:
            v = pt - center
            n = float(np.linalg.norm(v))
            pts.append(pt if n < 1e-6 else pt + (v / n) * float(margin_px))
        out = order_quad_tl_tr_br_bl(np.asarray(pts, dtype=np.float32))

    out[:, 0] = np.clip(out[:, 0], 0, max(0, img_w - 1))
    out[:, 1] = np.clip(out[:, 1], 0, max(0, img_h - 1))
    return out


def polygon_margin_px_from_ratio(polygon_xy: np.ndarray, ratio: float, min_px: float, max_px: float) -> float:
    poly = order_quad_tl_tr_br_bl(polygon_xy)
    w_top = float(np.linalg.norm(poly[1] - poly[0]))
    w_bottom = float(np.linalg.norm(poly[2] - poly[3]))
    h_left = float(np.linalg.norm(poly[3] - poly[0]))
    h_right = float(np.linalg.norm(poly[2] - poly[1]))
    ref = max(w_top, w_bottom, h_left, h_right)
    px = float(ref) * float(ratio)
    px = max(float(min_px), px)
    if float(max_px) > 0:
        px = min(float(max_px), px)
    return float(px)


def polygon_to_rectified(image_bgr: np.ndarray, polygon_xy: np.ndarray, out_max_side: int) -> tuple[np.ndarray, np.ndarray]:
    poly = order_quad_tl_tr_br_bl(polygon_xy)
    w_top = np.linalg.norm(poly[1] - poly[0])
    w_bottom = np.linalg.norm(poly[2] - poly[3])
    h_left = np.linalg.norm(poly[3] - poly[0])
    h_right = np.linalg.norm(poly[2] - poly[1])
    out_w = int(round(max(w_top, w_bottom)))
    out_h = int(round(max(h_left, h_right)))
    out_w = max(320, out_w)
    out_h = max(320, out_h)
    m = max(out_w, out_h)
    if m > int(out_max_side):
        s = float(out_max_side) / float(m)
        out_w = int(round(out_w * s))
        out_h = int(round(out_h * s))

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(poly.astype(np.float32), dst)
    rectified = cv2.warpPerspective(image_bgr, H, (out_w, out_h))
    return rectified, H


def enforce_landscape(image_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    h, w = image_bgr.shape[:2]
    if w >= h:
        return image_bgr, False
    return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE), True


def _landscape_rotation_matrix_if_applied(*, w: int, h: int, rotated_90cw: bool) -> np.ndarray:
    if not rotated_90cw:
        return np.eye(3, dtype=np.float64)
    return np.array([[0.0, -1.0, float(h - 1)], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotate_image_bound(image_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
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


def rotate_image_bound_with_matrix(image_bgr: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    a = float(angle_deg) % 360.0
    h, w = image_bgr.shape[:2]
    if abs(a - 0.0) < 1e-6:
        return image_bgr, np.eye(3, dtype=np.float64)
    if abs(a - 180.0) < 1e-6:
        M = np.array([[-1.0, 0.0, float(w - 1)], [0.0, -1.0, float(h - 1)], [0.0, 0.0, 1.0]], dtype=np.float64)
        return cv2.rotate(image_bgr, cv2.ROTATE_180), M

    center = (w / 2.0, h / 2.0)
    M2 = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M2[0, 0])
    sin = abs(M2[0, 1])
    new_w = int(round((h * sin) + (w * cos)))
    new_h = int(round((h * cos) + (w * sin)))
    M2[0, 2] += (new_w / 2.0) - center[0]
    M2[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(image_bgr, M2, (new_w, new_h))
    M3 = np.array(
        [[float(M2[0, 0]), float(M2[0, 1]), float(M2[0, 2])], [float(M2[1, 0]), float(M2[1, 1]), float(M2[1, 2])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rotated, M3


def _clamp_poly_to_image(poly_xy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    poly = order_quad_tl_tr_br_bl(np.asarray(poly_xy, dtype=np.float32).reshape(4, 2))
    poly[:, 0] = np.clip(poly[:, 0], 0, max(0, int(img_w) - 1))
    poly[:, 1] = np.clip(poly[:, 1], 0, max(0, int(img_h) - 1))
    return poly


def _quad_quality(poly_xy: np.ndarray, img_w: int, img_h: int) -> dict[str, Any]:
    poly = order_quad_tl_tr_br_bl(np.asarray(poly_xy, dtype=np.float32).reshape(4, 2))
    area = float(abs(cv2.contourArea(poly.reshape(-1, 1, 2).astype(np.float32))))
    img_area = float(max(1, img_w * img_h))
    area_ratio = float(area / img_area)
    edges = [
        float(np.linalg.norm(poly[1] - poly[0])),
        float(np.linalg.norm(poly[2] - poly[1])),
        float(np.linalg.norm(poly[3] - poly[2])),
        float(np.linalg.norm(poly[0] - poly[3])),
    ]
    e_min = float(min(edges)) if edges else 0.0
    e_max = float(max(edges)) if edges else 0.0
    edge_ratio = float(e_max / max(1e-6, e_min))
    return {"area": area, "area_ratio": area_ratio, "edge_min": e_min, "edge_max": e_max, "edge_ratio": edge_ratio}


def _is_valid_quad(poly_xy: np.ndarray, img_w: int, img_h: int) -> tuple[bool, dict[str, Any]]:
    q = _quad_quality(poly_xy, img_w, img_h)
    min_side = float(min(img_w, img_h))
    ok = True
    reasons: list[str] = []
    if q["area_ratio"] < 0.01:
        ok = False
        reasons.append("area_ratio_too_small")
    if q["edge_min"] < (min_side * 0.025):
        ok = False
        reasons.append("edge_min_too_small")
    if q["edge_ratio"] > 18.0:
        ok = False
        reasons.append("edge_ratio_too_large")
    q["ok"] = ok
    q["reasons"] = reasons
    return ok, q


def _unique_quad_key(poly_xy: np.ndarray, *, decimals: int = 1) -> tuple[tuple[float, float], ...]:
    poly = order_quad_tl_tr_br_bl(np.asarray(poly_xy, dtype=np.float32).reshape(4, 2))
    return tuple((round(float(x), decimals), round(float(y), decimals)) for x, y in poly.tolist())


def _compute_pad_px_auto(image_bgr: np.ndarray) -> int:
    h, w = image_bgr.shape[:2]
    cfg = PIPELINE_DEFAULTS.get("docaligner") or {}
    ratio = float(cfg.get("pad_px_auto_ratio", 0.08) or 0.08)
    pmin = int(cfg.get("pad_px_auto_min", 120) or 120)
    pmax = int(cfg.get("pad_px_auto_max", 800) or 800)
    pad = int(round(float(min(h, w)) * ratio))
    pad = max(pmin, pad)
    pad = min(pmax, pad)
    return int(pad)


# ------------------------------------------------------------
# v18準拠: DocAligner polygon 正規化（重複点/退化quadの修復）
#
# NOTE:
#   ここは精度に直結するため、paper_pipeline_v18.py の実装思想に合わせて
#   「必ず4点（重複なし）」を得ることを目的に、画像エッジを用いた修復も行う。
# ------------------------------------------------------------


def _min_pairwise_distance_xy(pts_xy: np.ndarray) -> float:
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0.0
    dmin = float("inf")
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            dmin = min(dmin, d)
    return float(dmin) if math.isfinite(dmin) else 0.0


def _default_duplicate_threshold_px(*, pts_xy: np.ndarray, image_bgr: Optional[np.ndarray] = None) -> float:
    """重複点（ほぼ同一点）とみなす距離しきい値(px)を決める。"""

    try:
        if image_bgr is not None:
            h, w = image_bgr.shape[:2]
            s = float(min(h, w))
        else:
            pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
            xs = pts[:, 0]
            ys = pts[:, 1]
            s = float(max(1.0, min(float(xs.max() - xs.min()), float(ys.max() - ys.min()))))
        return float(max(6.0, min(80.0, s * 0.012)))
    except Exception:
        return 12.0


def _is_degenerate_quad_by_geometry(
    quad_xy: np.ndarray,
    *,
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
    dup_thresh_px: float,
) -> tuple[bool, dict[str, Any]]:
    quad = order_quad_tl_tr_br_bl(np.asarray(quad_xy, dtype=np.float32).reshape(4, 2))
    area = float(abs(cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.float32))))
    edges = [
        float(np.linalg.norm(quad[1] - quad[0])),
        float(np.linalg.norm(quad[2] - quad[1])),
        float(np.linalg.norm(quad[3] - quad[2])),
        float(np.linalg.norm(quad[0] - quad[3])),
    ]
    e_min = float(min(edges)) if edges else 0.0
    min_pair = _min_pairwise_distance_xy(quad)

    reasons: list[str] = []
    if min_pair < float(dup_thresh_px):
        reasons.append("duplicate_points")
    if area <= 1.0:
        reasons.append("area_too_small")
    if e_min <= 1.0:
        reasons.append("edge_min_too_small")
    if img_w is not None and img_h is not None:
        min_side = float(min(img_w, img_h))
        if e_min < (min_side * 0.01):
            reasons.append("edge_min_too_small_relative")

    return (len(reasons) > 0), {"area": area, "edge_min": e_min, "min_pair_dist": min_pair, "reasons": reasons}


def _score_quad_by_edge_support(*, quad_xy: np.ndarray, edge_u8: np.ndarray) -> float:
    quad = order_quad_tl_tr_br_bl(np.asarray(quad_xy, dtype=np.float32).reshape(4, 2))
    h, w = edge_u8.shape[:2]
    edge = edge_u8
    if edge.ndim == 3:
        edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)

    r = 3
    hits = 0
    total = 0
    for i in range(4):
        p0 = quad[i]
        p1 = quad[(i + 1) % 4]
        seg_len = float(np.linalg.norm(p1 - p0))
        n = int(max(30, min(180, seg_len / 8.0)))
        for t in np.linspace(0.0, 1.0, n, dtype=np.float32):
            x = int(round(float(p0[0] * (1.0 - t) + p1[0] * t)))
            y = int(round(float(p0[1] * (1.0 - t) + p1[1] * t)))
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            total += 1
            x0 = max(0, x - r)
            x1 = min(w, x + r + 1)
            y0 = max(0, y - r)
            y1 = min(h, y + r + 1)
            if int(np.max(edge[y0:y1, x0:x1])) > 0:
                hits += 1
    if total <= 0:
        return 0.0
    return float(hits) / float(total)


def _refine_quad_corners_by_shi_tomasi(*, image_gray: np.ndarray, quad_xy: np.ndarray) -> np.ndarray:
    quad = order_quad_tl_tr_br_bl(np.asarray(quad_xy, dtype=np.float32).reshape(4, 2))
    h, w = image_gray.shape[:2]
    out = quad.copy()
    win = int(max(25, min(140, min(h, w) * 0.06)))

    for i in range(4):
        cx, cy = float(quad[i][0]), float(quad[i][1])
        x0 = int(max(0, round(cx - win)))
        y0 = int(max(0, round(cy - win)))
        x1 = int(min(w, round(cx + win)))
        y1 = int(min(h, round(cy + win)))
        roi = image_gray[y0:y1, x0:x1]
        if roi.size < 40:
            continue
        try:
            pts = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=50,
                qualityLevel=0.01,
                minDistance=max(5, int(win * 0.08)),
                blockSize=7,
            )
        except Exception:
            pts = None
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            continue
        pts[:, 0] += float(x0)
        pts[:, 1] += float(y0)
        d = np.linalg.norm(pts - np.array([[cx, cy]], dtype=np.float32), axis=1)
        j = int(np.argmin(d))
        if float(d[j]) <= float(win) * 0.75:
            out[i] = pts[j]

    return order_quad_tl_tr_br_bl(out)


def _recover_quad_from_edges(image_bgr: np.ndarray) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """画像のエッジ/輪郭から4隅を再推定する（DocAligner出力が退化した時の修復）。"""

    meta: dict[str, Any] = {"ok": False, "method": "", "detail": {}}
    if image_bgr is None:
        meta["detail"]["reason"] = "image_is_none"
        return None, meta
    h, w = image_bgr.shape[:2]
    if h < 80 or w < 80:
        meta["detail"]["reason"] = "too_small"
        return None, meta

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0.0)

    e = cv2.Canny(gray, 40, 140)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    best_quad: Optional[np.ndarray] = None
    best_score = float("-inf")

    try:
        contours, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        contours = []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40] if contours else []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(h * w) * 0.02:
            continue
        hull = cv2.convexHull(cnt)
        peri = float(cv2.arcLength(hull, True))
        if peri <= 1:
            continue
        for eps_ratio in [0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
            pts = approx.reshape(-1, 2).astype(np.float32)
            if pts.shape[0] != 4 or (not cv2.isContourConvex(approx)):
                continue
            quad = order_quad_tl_tr_br_bl(pts)
            dup_thr = _default_duplicate_threshold_px(pts_xy=quad, image_bgr=image_bgr)
            deg, _deg_meta = _is_degenerate_quad_by_geometry(quad, img_w=int(w), img_h=int(h), dup_thresh_px=float(dup_thr))
            if deg:
                continue
            ok, q = _is_valid_quad(quad, img_w=int(w), img_h=int(h))
            if not ok:
                continue
            support = _score_quad_by_edge_support(quad_xy=quad, edge_u8=e)
            score = float(q.get("area_ratio", 0.0)) * 10.0 + float(support) * 8.0
            if score > best_score:
                best_score = score
                best_quad = quad
                meta["detail"]["contour"] = {"eps_ratio": float(eps_ratio), "score": float(score), "support": float(support), "q": q}

    if best_quad is not None:
        try:
            best_quad = _refine_quad_corners_by_shi_tomasi(image_gray=gray, quad_xy=best_quad)
        except Exception:
            pass
        meta.update({"ok": True, "method": "contour_approx"})
        return best_quad, meta

    meta["detail"]["reason"] = "edge_recover_failed"
    return None, meta


def normalize_polygon_to_quad_with_meta(
    poly_xy: np.ndarray,
    *,
    image_bgr: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """polygon を「必ず4点（重複なし）」へ正規化する。"""

    meta: dict[str, Any] = {"ok": False, "method": "", "issue": "", "detail": {}}
    if poly_xy is None:
        meta["issue"] = "poly_is_none"
        return None, meta
    pts = np.asarray(poly_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        meta["issue"] = "pts_lt_3"
        return None, meta

    dup_thr = _default_duplicate_threshold_px(pts_xy=pts, image_bgr=image_bgr)

    if pts.shape[0] == 4:
        try:
            quad0 = order_quad_tl_tr_br_bl(pts)
            deg, deg_meta = _is_degenerate_quad_by_geometry(
                quad0,
                img_w=int(image_bgr.shape[1]) if image_bgr is not None else None,
                img_h=int(image_bgr.shape[0]) if image_bgr is not None else None,
                dup_thresh_px=float(dup_thr),
            )
            meta["detail"]["direct_quad_check"] = deg_meta
            if not deg:
                meta.update({"ok": True, "method": "direct_order"})
                return quad0, meta
            meta["issue"] = "duplicate_or_degenerate_quad"
        except Exception as e:
            meta["issue"] = f"direct_order_failed:{e}"

        if image_bgr is not None:
            quad_r, rec_meta = _recover_quad_from_edges(image_bgr)
            meta["detail"]["edge_recover"] = rec_meta
            if quad_r is not None:
                meta.update({"ok": True, "method": f"repair:{rec_meta.get('method', '')}"})
                return quad_r, meta

        try:
            hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(np.float32)
            if hull.shape[0] >= 3:
                rect = cv2.minAreaRect(hull.reshape(-1, 1, 2))
                box = cv2.boxPoints(rect).astype(np.float32)
                quad = order_quad_tl_tr_br_bl(box)
                meta.update({"ok": True, "method": "fallback_minAreaRect_from_hull"})
                return quad, meta
        except Exception as e:
            meta["detail"]["minAreaRect_error"] = str(e)

        return None, meta

    if image_bgr is not None:
        quad_r, rec_meta = _recover_quad_from_edges(image_bgr)
        meta["detail"]["edge_recover"] = rec_meta
        if quad_r is not None:
            meta.update({"ok": True, "method": f"repair:{rec_meta.get('method', '')}"})
            return quad_r, meta

    try:
        hull = cv2.convexHull(pts.astype(np.float32))
        hull2 = hull.reshape(-1, 2).astype(np.float32)
        if hull2.shape[0] < 3:
            meta["issue"] = "hull_lt_3"
            return None, meta
        rect = cv2.minAreaRect(hull2.reshape(-1, 1, 2))
        box = cv2.boxPoints(rect).astype(np.float32)
        if box.shape != (4, 2):
            meta["issue"] = "minAreaRect_box_not_4"
            return None, meta
        quad = order_quad_tl_tr_br_bl(box)
        meta.update({"ok": True, "method": "minAreaRect"})
        return quad, meta
    except Exception as e:
        meta["issue"] = f"minAreaRect_failed:{e}"
        return None, meta


def detect_polygon_fallback_opencv(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """DocAligner が全滅した場合のフォールバック（OpenCV 輪郭ベース）。"""

    if image_bgr is None:
        return None

    try:
        h, w = image_bgr.shape[:2]
        if h < 64 or w < 64:
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0.0)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:10]:
            area = float(cv2.contourArea(cnt))
            if area < float(h * w) * 0.05:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            pts = approx.reshape(-1, 2).astype(np.float32)
            if pts.shape[0] == 4 and cv2.isContourConvex(approx):
                quad = order_quad_tl_tr_br_bl(pts)
                return quad
            # 4点が取れない場合は minAreaRect で四角形化
            hull = cv2.convexHull(cnt.reshape(-1, 2).astype(np.float32))
            box = cv2.boxPoints(cv2.minAreaRect(hull.reshape(-1, 1, 2))).astype(np.float32)
            if box.shape == (4, 2):
                return order_quad_tl_tr_br_bl(box)
    except Exception:
        return None

    return None


def detect_polygon_fallback_advanced(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """DocAligner が 2点/3点等で失敗した場合の高精度フォールバック。"""

    if image_bgr is None:
        return None
    h, w = image_bgr.shape[:2]
    if h < 80 or w < 80:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0.0)

    variants: list[tuple[str, np.ndarray]] = [("gray", gray_blur)]
    try:
        clahe_cfg = (PIPELINE_DEFAULTS.get("marker") or {}).get("clahe") or {"clipLimit": 3.0, "tileGridSize": [8, 8]}
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_cfg.get("clipLimit", 3.0)),
            tileGridSize=tuple(int(x) for x in clahe_cfg.get("tileGridSize", [8, 8])),
        )
        variants.append(("clahe", clahe.apply(gray)))
    except Exception:
        pass
    try:
        at_cfg = (PIPELINE_DEFAULTS.get("qr") or {}).get("adaptive_threshold") or {"block_size": 61, "C": 3}
        blk = int(at_cfg.get("block_size", 61))
        if blk % 2 == 0:
            blk += 1
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, int(at_cfg.get("C", 3)))
        variants.append(("adaptive_inv", bw))
    except Exception:
        pass

    best_quad: Optional[np.ndarray] = None
    best_score = float("-inf")

    def _try_contours(src_u8: np.ndarray) -> None:
        nonlocal best_quad, best_score
        if src_u8.ndim == 2:
            e = cv2.Canny(src_u8, 40, 140)
        else:
            e = cv2.Canny(cv2.cvtColor(src_u8, cv2.COLOR_BGR2GRAY), 40, 140)
        e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        contours, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < float(h * w) * 0.03:
                continue
            hull = cv2.convexHull(cnt)
            peri = float(cv2.arcLength(hull, True))
            if peri <= 1:
                continue

            quad: Optional[np.ndarray] = None
            for eps_ratio in [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
                pts = approx.reshape(-1, 2).astype(np.float32)
                if pts.shape[0] == 4 and cv2.isContourConvex(approx):
                    quad = order_quad_tl_tr_br_bl(pts)
                    break
            if quad is None:
                box = cv2.boxPoints(cv2.minAreaRect(hull.reshape(-1, 1, 2))).astype(np.float32)
                if box.shape == (4, 2):
                    quad = order_quad_tl_tr_br_bl(box)
            if quad is None:
                continue
            quad = _clamp_poly_to_image(quad, img_w=int(w), img_h=int(h))
            ok, q = _is_valid_quad(quad, img_w=int(w), img_h=int(h))
            if not ok:
                continue
            support = _score_quad_by_edge_support(quad_xy=quad, edge_u8=e)
            score = float(q.get("area_ratio", 0.0)) * 10.0 + float(support) * 6.0 - float(q.get("edge_ratio", 0.0)) * 0.01
            if score > best_score:
                best_score = score
                best_quad = quad

    for _name, src in variants:
        _try_contours(src)

    if best_quad is None:
        return None
    try:
        return _refine_quad_corners_by_shi_tomasi(image_gray=gray, quad_xy=best_quad)
    except Exception:
        return best_quad


def _run_docaligner_once_with_meta(
    *,
    model: Any,
    cb: Any,
    image_bgr: np.ndarray,
    pad_px: int,
    input_scale: float = 1.0,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """DocAligner を1回実行し、polygon(4x2) と正規化メタを返す。"""

    if image_bgr is None:
        return None, {"ok": False, "issue": "image_is_none"}

    s = float(input_scale)
    img = image_bgr
    if abs(s - 1.0) > 1e-9:
        h, w = img.shape[:2]
        new_w = max(8, int(round(w * s)))
        new_h = max(8, int(round(h * s)))
        interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    padded = cb.pad(img, int(pad_px))
    poly = model(img=padded, do_center_crop=False)
    if poly is None:
        return None, {"ok": False, "issue": "model_returned_none"}

    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        return None, {"ok": False, "issue": "poly_lt_3"}

    poly, norm_meta = normalize_polygon_to_quad_with_meta(poly, image_bgr=padded)
    if poly is None:
        if not isinstance(norm_meta, dict):
            norm_meta = {"ok": False, "issue": "normalize_failed"}
        return None, norm_meta

    poly = poly - float(pad_px)
    if abs(s - 1.0) > 1e-9:
        poly = poly / float(s)

    if isinstance(norm_meta, dict):
        norm_meta = dict(norm_meta)
        norm_meta.setdefault("note", "normalized_on_padded_image_before_unpad")

    return poly.astype(np.float32), (norm_meta if isinstance(norm_meta, dict) else {"ok": False})


def _run_docaligner_once(*, model: Any, cb: Any, image_bgr: np.ndarray, pad_px: int, input_scale: float = 1.0) -> Optional[np.ndarray]:
    poly, _norm_meta = _run_docaligner_once_with_meta(
        model=model,
        cb=cb,
        image_bgr=image_bgr,
        pad_px=int(pad_px),
        input_scale=float(input_scale),
    )
    return poly


def detect_polygon_docaligner(
    model: Any,
    cb: Any,
    image_bgr: np.ndarray,
    *,
    pad_px: Optional[int] = None,
    input_scale: float = 1.0,
) -> Optional[np.ndarray]:
    """DocAligner を1回実行して polygon(4x2) を返す（v18準拠の pad auto + strict check）。"""

    if image_bgr is None:
        return None

    if pad_px is None:
        cfg_doc = PIPELINE_DEFAULTS.get("docaligner") or {}
        pad_base = int(cfg_doc.get("pad_px") or 200)
        pad_auto = _compute_pad_px_auto(image_bgr)
        pad_px = int(max(pad_base, pad_auto))

    poly = _run_docaligner_once(model=model, cb=cb, image_bgr=image_bgr, pad_px=int(pad_px), input_scale=float(input_scale))
    if poly is None:
        return None

    ok, _q = _is_valid_quad(poly, img_w=int(image_bgr.shape[1]), img_h=int(image_bgr.shape[0]))
    if not ok:
        return None
    return poly


def _iter_docaligner_settings(*, args: argparse.Namespace, image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """DocAligner候補設定（model/type/pad/scale）を優先度順に列挙する（v18準拠）。"""

    cfg_doc = PIPELINE_DEFAULTS.get("docaligner") or {}
    cfg_multi = (cfg_doc.get("multi") or {}) if isinstance(cfg_doc, dict) else {}

    model0 = str(getattr(args, "docaligner_model", cfg_doc.get("model") or "fastvit_sa24"))
    type0 = str(getattr(args, "docaligner_type", cfg_doc.get("type") or "heatmap"))

    models = [model0] + [str(x) for x in (cfg_multi.get("extra_models") or [])]
    types = [type0] + [str(x) for x in (cfg_multi.get("extra_types") or [])]

    pad_base = int(cfg_doc.get("pad_px") or 200)
    pad_auto = _compute_pad_px_auto(image_bgr)
    pad0 = int(max(pad_base, pad_auto))
    pad_candidates = [pad0] + [int(x) for x in (cfg_multi.get("pad_px_candidates") or [])] + [pad_base, pad_auto]
    pad_candidates = [int(x) for x in pad_candidates if int(x) >= 0]

    seen_pad: set[int] = set()
    pad_list: list[int] = []
    for p in pad_candidates:
        if p in seen_pad:
            continue
        seen_pad.add(int(p))
        pad_list.append(int(p))

    scales_raw = [float(x) for x in (cfg_multi.get("input_scales") or [])]
    scales = [1.0] + [float(s) for s in scales_raw if abs(float(s) - 1.0) > 1e-9]

    def _uniq_str(xs: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for x in xs:
            x2 = str(x)
            if x2 in seen:
                continue
            seen.add(x2)
            out.append(x2)
        return out

    models_u = _uniq_str(models)
    types_u = _uniq_str(types)

    model_pri = {"lcnet050": 0, "lcnet100": 1, "fastvit_t8": 2, "fastvit_sa24": 3}
    rest_models = [m for m in _uniq_str(models_u) if str(m) != str(model0)]
    rest_models = sorted(rest_models, key=lambda m: model_pri.get(str(m), 99))
    models_u = [str(model0)] + rest_models

    rest_types = [t for t in _uniq_str(types_u) if str(t) != str(type0)]
    rest_types = [t for t in ["heatmap", "point"] if t in rest_types]
    types_u = [str(type0)] + rest_types

    def _uniq_settings(xs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, int, float]] = set()
        out: list[dict[str, Any]] = []
        for d in xs:
            key = (str(d["model"]), str(d["type"]), int(d["pad_px"]), float(d["input_scale"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out

    settings: list[dict[str, Any]] = []
    for m in models_u:
        for t in types_u:
            for p in pad_list:
                settings.append({"model": m, "type": t, "pad_px": int(p), "input_scale": 1.0})
            for s in scales:
                if abs(float(s) - 1.0) < 1e-9:
                    continue
                settings.append({"model": m, "type": t, "pad_px": int(pad0), "input_scale": float(s)})

    return _uniq_settings(settings)


def _margin_px_candidates_for_eval(*, args: argparse.Namespace, poly_xy: np.ndarray) -> list[float]:
    fixed = float(getattr(args, "polygon_margin_px", 0.0) or 0.0)
    if fixed > 0:
        return [float(fixed)]

    ratios = [
        float(getattr(args, "polygon_margin_ratio", 0.0) or 0.0),
        0.0,
        0.06,
    ]
    seen: set[float] = set()
    ratios2: list[float] = []
    for r in ratios:
        r2 = float(r)
        if r2 in seen:
            continue
        seen.add(r2)
        ratios2.append(r2)

    out: list[float] = []
    for r in ratios2:
        if r <= 0:
            out.append(0.0)
        else:
            out.append(
                polygon_margin_px_from_ratio(
                    poly_xy,
                    ratio=float(r),
                    min_px=float(getattr(args, "polygon_margin_min_px", 0.0) or 0.0),
                    max_px=float(getattr(args, "polygon_margin_max_px", 0.0) or 0.0),
                )
            )

    seen2: set[int] = set()
    out2: list[float] = []
    for m in out:
        k = int(round(float(m)))
        if k in seen2:
            continue
        seen2.add(k)
        out2.append(float(m))
    return out2


def _quick_eval_form_scores_for_candidate(
    rectified_landscape_bgr: np.ndarray,
    *,
    wechat: "WeChatQRDetectorPool",
    rotation_max_workers: int,
    marker_preproc: str,
    unknown_score_threshold: float,
    unknown_margin: float,
    formA_geom_cfg: Optional["MarkerGeometryConfig"] = None,
) -> tuple["FormDecision", float]:
    decision = decide_form_by_rotations(
        rectified_landscape_bgr,
        wechat=wechat,
        max_workers=int(rotation_max_workers),
        marker_preproc=str(marker_preproc),
        unknown_score_threshold=float(unknown_score_threshold),
        unknown_margin=float(unknown_margin),
        formA_geom_cfg=formA_geom_cfg,
    )
    return decision, float(decision.score)


def detect_polygon_docaligner_multi(
    *,
    logger: logging.Logger,
    args: argparse.Namespace,
    degraded_bgr: np.ndarray,
    wechat: "WeChatQRDetectorPool",
    formA_geom_cfg: Optional["MarkerGeometryConfig"] = None,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """DocAligner を複数条件で実行し、最良polygonを返す（v18準拠・精度優先）。"""

    if degraded_bgr is None:
        return None, {"ok": False, "reason": "image_is_none"}

    cfg_doc = PIPELINE_DEFAULTS.get("docaligner") or {}
    cfg_multi = (cfg_doc.get("multi") or {}) if isinstance(cfg_doc, dict) else {}
    enable_multi = bool(cfg_multi.get("enable", True))

    h, w = degraded_bgr.shape[:2]

    def _get_model_cached(model_name: str, model_type: str) -> tuple[Any, Any]:
        cache = getattr(detect_polygon_docaligner_multi, "_model_cache", None)
        if cache is None:
            cache = {}
            setattr(detect_polygon_docaligner_multi, "_model_cache", cache)
        key = (str(model_name), str(model_type))
        if key in cache:
            return cache[key]
        m, cb2 = load_docaligner_model(str(model_name), str(model_type))
        cache[key] = (m, cb2)
        return m, cb2

    if not enable_multi:
        m0, cb0 = _get_model_cached(str(getattr(args, "docaligner_model", cfg_doc.get("model") or "fastvit_sa24")), str(getattr(args, "docaligner_type", cfg_doc.get("type") or "heatmap")))
        poly = detect_polygon_docaligner(m0, cb0, degraded_bgr)
        return poly, {"ok": poly is not None, "mode": "single"}

    settings = _iter_docaligner_settings(args=args, image_bgr=degraded_bgr)
    max_infer_runs = int(cfg_multi.get("max_infer_runs") or 8)
    max_poly_candidates = int(cfg_multi.get("max_polygon_candidates") or 4)

    candidates: list[dict[str, Any]] = []
    all_polys: list[dict[str, Any]] = []
    seen_keys: set[tuple[tuple[float, float], ...]] = set()

    infer_runs = 0
    for st in settings:
        if infer_runs >= max_infer_runs:
            break
        model_name = str(st["model"])
        model_type = str(st["type"])
        pad_px = int(st["pad_px"])
        input_scale = float(st["input_scale"])

        try:
            model, cb = _get_model_cached(model_name, model_type)
        except Exception as e:
            logger.debug("[DocAligner] model load failed: %s", e)
            continue

        infer_runs += 1
        try:
            poly, norm_meta = _run_docaligner_once_with_meta(
                model=model,
                cb=cb,
                image_bgr=degraded_bgr,
                pad_px=int(pad_px),
                input_scale=float(input_scale),
            )
        except Exception as e:
            logger.debug("[DocAligner] infer failed: %s", e)
            continue

        if poly is None:
            continue

        poly = _clamp_poly_to_image(poly, img_w=int(w), img_h=int(h))
        ok, q = _is_valid_quad(poly, img_w=int(w), img_h=int(h))
        all_polys.append(
            {
                "poly": order_quad_tl_tr_br_bl(poly).astype(np.float32),
                "quality": q,
                "setting": {"model": model_name, "type": model_type, "pad_px": int(pad_px), "input_scale": float(input_scale)},
                "norm": norm_meta if isinstance(norm_meta, dict) else {"ok": False, "issue": "norm_meta_not_dict"},
                "strict_ok": bool(ok),
            }
        )
        if not ok:
            continue

        k = _unique_quad_key(poly, decimals=1)
        if k in seen_keys:
            continue
        seen_keys.add(k)

        candidates.append(
            {
                "poly": order_quad_tl_tr_br_bl(poly).astype(np.float32),
                "quality": q,
                "setting": {"model": model_name, "type": model_type, "pad_px": int(pad_px), "input_scale": float(input_scale)},
                "norm": norm_meta if isinstance(norm_meta, dict) else {"ok": False, "issue": "norm_meta_not_dict"},
            }
        )
        if len(candidates) >= max_poly_candidates:
            break

    if not candidates:
        # strict で全滅した場合の救済（v18思想）
        if all_polys:
            def _relaxed_score(d: dict[str, Any]) -> float:
                q = (d.get("quality") or {})
                area = float(q.get("area_ratio") or 0.0)
                edge_min = float(q.get("edge_min") or 0.0)
                return area - (1.0 if edge_min <= 1.0 else 0.0)

            sorted_all = sorted(all_polys, key=_relaxed_score, reverse=True)
            for d in sorted_all[: max(2, int(max_poly_candidates))]:
                candidates.append({"poly": d["poly"], "quality": d["quality"], "setting": d["setting"], "relaxed": True})

        fb = detect_polygon_fallback_opencv(degraded_bgr)
        if fb is not None:
            ok_fb, q_fb = _is_valid_quad(fb, img_w=int(w), img_h=int(h))
            if ok_fb:
                candidates.append({"poly": order_quad_tl_tr_br_bl(fb).astype(np.float32), "quality": q_fb, "setting": {"model": "opencv_fallback", "type": "contour"}, "relaxed": True})

        adv = detect_polygon_fallback_advanced(degraded_bgr)
        if adv is not None:
            ok_adv, q_adv = _is_valid_quad(adv, img_w=int(w), img_h=int(h))
            if ok_adv:
                candidates.append({"poly": order_quad_tl_tr_br_bl(np.asarray(adv, dtype=np.float32)).astype(np.float32), "quality": q_adv, "setting": {"model": "opencv_fallback", "type": "advanced"}, "relaxed": True})

        if not candidates:
            return None, {"ok": False, "reason": "no_valid_polygon", "infer_runs": infer_runs, "all_polys": all_polys}

    best: Optional[dict[str, Any]] = None
    evals: list[dict[str, Any]] = []

    edge_u8: Optional[np.ndarray] = None

    def _lazy_edge_u8() -> np.ndarray:
        nonlocal edge_u8
        if edge_u8 is None:
            g = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2GRAY)
            e = cv2.Canny(g, 40, 140)
            e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
            edge_u8 = e
        return edge_u8

    try:
        eval_max_candidates = int(cfg_multi.get("eval_max_candidates") or 0)
    except Exception:
        eval_max_candidates = 0
    if eval_max_candidates <= 0:
        eval_max_candidates = int(max(1, min(2, len(candidates))))

    def _cand_quality_score(d: dict[str, Any]) -> float:
        q = d.get("quality") or {}
        area = float(q.get("area_ratio") or 0.0)
        edge_ratio = float(q.get("edge_ratio") or 1e9)
        edge_min = float(q.get("edge_min") or 0.0)
        return area * 10.0 - (edge_ratio * 0.15) + (edge_min * 1e-4)

    candidates_for_eval = sorted(candidates, key=_cand_quality_score, reverse=True)[: int(eval_max_candidates)]

    try:
        eval_rotation_workers = int(cfg_multi.get("eval_rotation_workers") or 1)
    except Exception:
        eval_rotation_workers = 1
    eval_rotation_workers = int(max(1, min(4, eval_rotation_workers)))

    try:
        eval_max_margins = int(cfg_multi.get("eval_max_margins") or 2)
    except Exception:
        eval_max_margins = 2
    eval_max_margins = int(max(1, min(4, eval_max_margins)))

    for c in candidates_for_eval:
        poly = np.asarray(c["poly"], dtype=np.float32)
        margin_list = _margin_px_candidates_for_eval(args=args, poly_xy=poly)[: int(eval_max_margins)]
        for margin_px in margin_list:
            try:
                rect, _H, poly_exp, rect_meta = rectify_with_margin_and_optional_padding(
                    degraded_bgr,
                    polygon_xy=poly,
                    margin_px=float(margin_px),
                    out_max_side=int(getattr(args, "docaligner_max_side")),
                )
                rect, _ = enforce_landscape(rect)
                decision, score = _quick_eval_form_scores_for_candidate(
                    rect,
                    wechat=wechat,
                    rotation_max_workers=int(eval_rotation_workers),
                    marker_preproc=str(getattr(args, "marker_preproc")),
                    unknown_score_threshold=float(getattr(args, "unknown_score_threshold")),
                    unknown_margin=float(getattr(args, "unknown_margin")),
                    formA_geom_cfg=formA_geom_cfg,
                )
                rec = {
                    "setting": c.get("setting"),
                    "quality": c.get("quality"),
                    "margin_px": float(margin_px),
                    "decision": asdict(decision),
                    "score": float(score),
                    "rectify": rect_meta,
                }
                evals.append(rec)

                if best is None:
                    best = {"poly": poly, "margin_px": float(margin_px), **rec}
                else:
                    cur_score = float(score)
                    best_score = float(best.get("score") or 0.0)
                    if cur_score > best_score + 1e-6:
                        best = {"poly": poly, "margin_px": float(margin_px), **rec}
                    elif abs(cur_score - best_score) <= 1e-6:
                        try:
                            cur_area = float(((c.get("quality") or {}).get("area_ratio") or 0.0))
                            best_area = float(((best.get("quality") or {}).get("area_ratio") or 0.0))
                            edge = _lazy_edge_u8()
                            cur_support = _score_quad_by_edge_support(quad_xy=poly_exp, edge_u8=edge)
                            best_support = float(best.get("edge_support") or 0.0)
                            if cur_support > best_support + 0.02:
                                best = {"poly": poly, "margin_px": float(margin_px), **rec, "edge_support": float(cur_support)}
                            elif abs(cur_support - best_support) <= 0.02:
                                if cur_area > best_area:
                                    best = {"poly": poly, "margin_px": float(margin_px), **rec, "edge_support": float(cur_support)}
                        except Exception:
                            pass
            except Exception as e:
                logger.debug("[DocAligner] candidate eval failed: %s", e)
                continue

    if best is None:
        best_q = None
        for c in candidates:
            q = c.get("quality") or {}
            if best_q is None or float(q.get("area_ratio") or 0.0) > float(best_q.get("quality", {}).get("area_ratio") or 0.0):
                best_q = c
        poly_best = np.asarray(best_q["poly"], dtype=np.float32) if best_q is not None else None
        return poly_best, {
            "ok": poly_best is not None,
            "reason": "fallback_by_quality",
            "infer_runs": infer_runs,
            "candidates": candidates,
            "evals": evals,
            "all_polys": all_polys,
        }

    return np.asarray(best["poly"], dtype=np.float32), {
        "ok": True,
        "reason": "multi_best_by_form_score",
        "infer_runs": infer_runs,
        "best": best,
        "candidates": candidates,
        "all_polys": all_polys,
        "evals": evals,
    }


def pick_best_docaligner_candidate_by_form_score(
    *,
    logger: logging.Logger,
    image_bgr: np.ndarray,
    poly_fallback: np.ndarray,
    doc_meta: dict[str, Any],
    wechat: "WeChatQRDetectorPool",
    marker_preproc: str,
    unknown_score_threshold: float,
    unknown_margin: float,
    formA_geom_cfg: Optional[MarkerGeometryConfig],
    docaligner_max_side: int,
    polygon_margin_ratio: float,
    polygon_margin_min_px: float,
    polygon_margin_max_px: float,
    polygon_margin_fixed_px: float,
) -> tuple[np.ndarray, Optional[float], dict[str, Any]]:
    """DocAligner multi の候補（polygon）を、rectify→フォーム判定スコアで再評価して最良を選ぶ。

    目的:
      - 面積最大の polygon が「紙を少し欠いた」quad になり、マーカー/QR が画面外で no_detection になるケースを減らす。
      - paper_pipeline_v18 の候補選択思想（rectify→decideで最良候補を採用）に揃える。

    注意:
      - 精度に関わるハイパーパラメータは変更しない。
      - ここでの評価は「候補選択のため」であり、後段の本処理（rectify/decide）は同じ関数・同じ引数で行う。
    """

    meta: dict[str, Any] = {
        "ok": False,
        "reason": "not_evaluated",
        "picked_margin_px": None,
        "picked_score": None,
        "picked_form": None,
        "evals": [],
    }

    if image_bgr is None:
        meta["reason"] = "image_is_none"
        return np.asarray(poly_fallback, dtype=np.float32), None, meta

    # 候補 polygon を取り出す（なければ fallback のみ）
    cand_list: list[dict[str, Any]] = []
    try:
        for c in list((doc_meta or {}).get("candidates") or []):
            if not isinstance(c, dict) or (c.get("poly") is None):
                continue
            poly = np.asarray(c.get("poly"), dtype=np.float32).reshape(4, 2)
            cand_list.append({"poly": order_quad_tl_tr_br_bl(poly), "quality": c.get("quality") or {}, "setting": c.get("setting") or {}})
    except Exception:
        cand_list = []

    if not cand_list:
        cand_list = [{"poly": order_quad_tl_tr_br_bl(np.asarray(poly_fallback, dtype=np.float32).reshape(4, 2)), "quality": {}, "setting": {"fallback": True}}]

    # 評価数を絞る（paper_pipeline_v18 相当）
    cfg_multi = ((PIPELINE_DEFAULTS.get("docaligner") or {}).get("multi") or {})
    try:
        eval_max_candidates = int(cfg_multi.get("eval_max_candidates") or 2)
    except Exception:
        eval_max_candidates = 2
    eval_max_candidates = max(1, min(6, eval_max_candidates))

    def _area_ratio(d: dict[str, Any]) -> float:
        try:
            return float((d.get("quality") or {}).get("area_ratio") or 0.0)
        except Exception:
            return 0.0

    cand_list = sorted(cand_list, key=_area_ratio, reverse=True)[:eval_max_candidates]

    # margin 候補（固定指定があればそれだけ）
    def _margin_candidates(poly_xy: np.ndarray) -> list[float]:
        if float(polygon_margin_fixed_px) > 0.0:
            return [float(polygon_margin_fixed_px)]

        ratios = [float(polygon_margin_ratio), 0.0, 0.06]
        pxs: list[float] = []
        for r in ratios:
            if r <= 0:
                pxs.append(0.0)
            else:
                pxs.append(
                    polygon_margin_px_from_ratio(
                        poly_xy,
                        ratio=float(r),
                        min_px=float(polygon_margin_min_px),
                        max_px=float(polygon_margin_max_px),
                    )
                )
        # uniq by rounded px
        seen: set[int] = set()
        out: list[float] = []
        for m in pxs:
            k = int(round(float(m)))
            if k in seen:
                continue
            seen.add(k)
            out.append(float(m))
        return out

    try:
        eval_max_margins = int(cfg_multi.get("eval_max_margins") or 2)
    except Exception:
        eval_max_margins = 2
    eval_max_margins = max(1, min(6, eval_max_margins))

    best: Optional[dict[str, Any]] = None

    for c in cand_list:
        poly = np.asarray(c["poly"], dtype=np.float32).reshape(4, 2)
        margins = _margin_candidates(poly)[:eval_max_margins]
        for margin_px in margins:
            try:
                rect, _H, _polyexp, _rmeta = rectify_with_margin_and_optional_padding(
                    image_bgr,
                    polygon_xy=poly,
                    margin_px=float(margin_px),
                    out_max_side=int(docaligner_max_side),
                )
                rect, _ = enforce_landscape(rect)
                decision = decide_form_by_rotations(
                    rect,
                    wechat=wechat,
                    max_workers=1,
                    marker_preproc=str(marker_preproc),
                    unknown_score_threshold=float(unknown_score_threshold),
                    unknown_margin=float(unknown_margin),
                    formA_geom_cfg=formA_geom_cfg,
                )
                ok_form = bool(decision.ok) and (str(decision.form) in ("A", "B")) and (decision.angle_deg is not None)
                score = float(decision.score)
                reason = ""
                if not ok_form:
                    try:
                        reason, _ = extract_form_unknown_reason(asdict(decision))
                    except Exception:
                        reason = "unknown"

                rec = {
                    "ok_form": bool(ok_form),
                    "score": float(score),
                    "reason": str(reason),
                    "form": str(decision.form or ""),
                    "angle": None if decision.angle_deg is None else float(decision.angle_deg),
                    "margin_px": float(margin_px),
                    "area_ratio": float(_area_ratio(c)),
                }
                meta["evals"].append(rec)

                key = (1 if ok_form else 0, float(score), float(_area_ratio(c)))
                if best is None:
                    best = {"poly": poly, "key": key, "rec": rec}
                else:
                    if key > tuple(best.get("key") or (0, 0.0, 0.0)):
                        best = {"poly": poly, "key": key, "rec": rec}
            except Exception as e:
                logger.debug("[DocAligner] candidate eval failed: %s", e)
                continue

    if best is None:
        meta["reason"] = "no_eval_succeeded"
        return np.asarray(poly_fallback, dtype=np.float32), None, meta

    meta["ok"] = True
    meta["reason"] = "picked"
    meta["picked_margin_px"] = float(best["rec"]["margin_px"])
    meta["picked_score"] = float(best["rec"]["score"])
    meta["picked_form"] = str(best["rec"]["form"])
    return order_quad_tl_tr_br_bl(np.asarray(best["poly"], dtype=np.float32)), float(best["rec"]["margin_px"]), meta


def _get_rectify_padding_cfg() -> dict[str, Any]:
    cfg_doc = PIPELINE_DEFAULTS.get("docaligner") or {}
    cfg = (cfg_doc.get("rectify_padding") or {}) if isinstance(cfg_doc, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def _apply_rectify_padding(image_bgr: np.ndarray, *, required_margin_px: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cfg = _get_rectify_padding_cfg()
    if not bool(cfg.get("enable", False)):
        return image_bgr, np.eye(3, dtype=np.float64), {"applied": False, "reason": "disabled"}

    pad_cfg = int(cfg.get("pad_px", 0) or 0)
    pad_need = int(max(0.0, float(required_margin_px)))
    pad_px = int(max(pad_cfg, pad_need + 12))
    pad_px = int(max(0, min(2000, pad_px)))
    if pad_px <= 0:
        return image_bgr, np.eye(3, dtype=np.float64), {"applied": False, "reason": "pad_px<=0"}

    border_value = cfg.get("border_value", [0, 0, 0])
    try:
        bgr = tuple(int(x) for x in border_value)
        if len(bgr) != 3:
            bgr = (0, 0, 0)
    except Exception:
        bgr = (0, 0, 0)

    padded = cv2.copyMakeBorder(image_bgr, pad_px, pad_px, pad_px, pad_px, borderType=cv2.BORDER_CONSTANT, value=bgr)
    T = np.array([[1.0, 0.0, float(pad_px)], [0.0, 1.0, float(pad_px)], [0.0, 0.0, 1.0]], dtype=np.float64)
    return padded, T, {"applied": True, "pad_px": int(pad_px), "border_value": list(bgr)}


def rectify_with_margin_and_optional_padding(
    image_bgr: np.ndarray,
    *,
    polygon_xy: np.ndarray,
    margin_px: float,
    out_max_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    h, w = image_bgr.shape[:2]
    poly = order_quad_tl_tr_br_bl(np.asarray(polygon_xy, dtype=np.float32).reshape(4, 2))
    poly_exp_overlay = expand_polygon(poly, float(margin_px), img_w=int(w), img_h=int(h))

    padded, T_deg_to_pad, pad_meta = _apply_rectify_padding(image_bgr, required_margin_px=float(margin_px))
    Hp, Wp = padded.shape[:2]
    poly_pad = (poly + np.array([[float(T_deg_to_pad[0, 2]), float(T_deg_to_pad[1, 2])]], dtype=np.float32)).astype(np.float32)
    poly_exp_pad = expand_polygon(poly_pad, float(margin_px), img_w=int(Wp), img_h=int(Hp))
    rectified, H_pad_to_rect = polygon_to_rectified(padded, poly_exp_pad, out_max_side=int(out_max_side))
    H_deg_to_rect = np.asarray(H_pad_to_rect, dtype=np.float64) @ np.asarray(T_deg_to_pad, dtype=np.float64)
    meta = {"rectify_padding": pad_meta, "poly_exp_overlay": poly_exp_overlay.astype(float).tolist(), "poly_exp_padded": poly_exp_pad.astype(float).tolist()}
    return rectified, H_deg_to_rect, poly_exp_overlay, meta


# ============================================================
# WeChat QR
# ============================================================


class WeChatQRDetectorPool:
    def __init__(self, model_dir: str, pool_size: int):
        self.model_dir = str(model_dir)
        self.pool_size = int(pool_size)
        if self.pool_size <= 0:
            raise ValueError("pool_size must be >= 1")

        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=self.pool_size)
        for _ in range(self.pool_size):
            self._q.put(self._init_detector(self.model_dir))

    @staticmethod
    def _init_detector(model_dir: str) -> Any:
        if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
            raise RuntimeError("cv2.wechat_qrcode_WeChatQRCode is not available")

        detect_proto = os.path.join(model_dir, "detect.prototxt")
        detect_caffe = os.path.join(model_dir, "detect.caffemodel")
        sr_proto = os.path.join(model_dir, "sr.prototxt")
        sr_caffe = os.path.join(model_dir, "sr.caffemodel")
        if not all(map(os.path.exists, [detect_proto, detect_caffe, sr_proto, sr_caffe])):
            raise FileNotFoundError("WeChat QR model files not found")
        return cv2.wechat_qrcode_WeChatQRCode(detect_proto, detect_caffe, sr_proto, sr_caffe)

    @staticmethod
    def _decode_from_detector(detector: Any, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        res, points = detector.detectAndDecode(image_bgr)
        out: list[dict[str, Any]] = []
        if res is None or points is None:
            return out
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
        if image_bgr is None:
            return []
        det = self._q.get()
        try:
            return self._decode_from_detector(det, image_bgr)
        finally:
            self._q.put(det)


def _preprocess_variants_for_qr(image_bgr: np.ndarray, variant_names: list[str]) -> list[tuple[str, np.ndarray]]:
    if image_bgr is None:
        return []
    names = [str(x) for x in (variant_names or [])] or ["bgr"]
    out: list[tuple[str, np.ndarray]] = []
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
            g = _get_gray()
            if g is not None:
                try:
                    clahe_cfg = PIPELINE_DEFAULTS.get("qr", {}).get("clahe", {})
                    clahe = cv2.createCLAHE(
                        clipLimit=float(clahe_cfg.get("clipLimit", 2.0)),
                        tileGridSize=tuple(int(x) for x in clahe_cfg.get("tileGridSize", [8, 8])),
                    )
                    g2 = clahe.apply(g)
                    out.append(("clahe", cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)))
                except Exception:
                    pass
        elif name == "adaptive_threshold":
            g = _get_gray()
            if g is not None:
                try:
                    at_cfg = PIPELINE_DEFAULTS.get("qr", {}).get("adaptive_threshold", {})
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
                    pass
        elif name == "adaptive_morph":
            g = _get_gray()
            if g is not None:
                try:
                    at_cfg = PIPELINE_DEFAULTS.get("qr", {}).get("adaptive_threshold", {})
                    block_size = int(at_cfg.get("block_size", 51))
                    c_val = int(at_cfg.get("C", 5))
                    if block_size < 3:
                        block_size = 3
                    if block_size % 2 == 0:
                        block_size += 1
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
                    pass
        else:
            continue

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
    wechat: WeChatQRDetectorPool,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    cfg_all = PIPELINE_DEFAULTS.get("qr", {}).get("wechat", {})
    cfg = cfg_all.get(str(mode), {}) if isinstance(cfg_all, dict) else {}
    variant_names = list(cfg.get("variants") or ["bgr"])
    scales = [float(s) for s in (cfg.get("scales") or [1.0])]

    h0, w0 = image_bgr.shape[:2]
    up_enable_max = int(cfg_all.get("up_scale_enable_max_side_px", 0) or 0)
    if up_enable_max > 0 and max(h0, w0) >= up_enable_max:
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

            qrs = wechat.detect(test)
            if not qrs:
                continue

            if abs(s - 1.0) > 1e-9:
                for q in qrs:
                    try:
                        pts = np.asarray(q.get("points"), dtype=np.float32).reshape(-1, 2)
                        pts = pts / float(s)
                        q["points"] = pts.tolist()
                    except Exception:
                        continue

            for q in qrs:
                q.setdefault("engine", "wechat")
                q["prep"] = prep_name
                q["scale"] = float(s)

            if str(mode) == "fast":
                return qrs

            try:
                score, det = score_best_qr_candidate(image_bgr, qrs)
                if float(score) >= 50.0 or bool(det.get("qr_is_in_top_right_quadrant")):
                    return det.get("qrs") or qrs
            except Exception:
                score = 0.0

            if float(score) > best_score:
                best_score = float(score)
                best = qrs

    return best


def score_best_qr_candidate(image_bgr: np.ndarray, qrs: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    h, w = image_bgr.shape[:2]
    best = None
    best_score = float("-inf")
    best_detail: dict[str, Any] = {}

    for q in (qrs or []):
        try:
            pts = np.asarray(q.get("points"), dtype=np.float32).reshape(-1, 2)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            area = float(abs(cv2.contourArea(pts.astype(np.float32))))
            rel = area / float(max(1, w * h))

            nx = cx / float(max(1, w))
            ny = cy / float(max(1, h))

            is_in_top_right_quadrant = (nx > 0.5) and (ny < 0.5)
            is_in_bottom_left_quadrant = (nx < 0.5) and (ny > 0.5)

            if is_in_top_right_quadrant:
                quadrant_bonus = 100.0
            elif is_in_bottom_left_quadrant:
                quadrant_bonus = -100.0
            else:
                quadrant_bonus = 0.0

            dist_from_top_right = math.sqrt((nx - 1.0) ** 2 + (ny - 0.0) ** 2)
            pos_score = max(0.0, 1.0 - (dist_from_top_right / 1.41421356))
            final_pos_score = pos_score**2

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

    reordered = [best] + [q for q in (qrs or []) if q is not best]
    detail = {
        "qrs": reordered,
        **best_detail,
        "qr_engine": str(best.get("engine", "wechat")),
        "qr_prep": str(best.get("prep", "")),
        "qr_scale": best.get("scale", None),
    }
    return float(best_score), detail


# ============================================================
# フォームAマーカー検出 / 幾何制約
# ============================================================


def detect_formA_marker_boxes_base(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = image_bgr.shape[:2]

    marker_cfg = PIPELINE_DEFAULTS.get("marker", {}) if isinstance(PIPELINE_DEFAULTS.get("marker", {}), dict) else {}
    corner_margin_ratio = float(marker_cfg.get("corner_margin_ratio", 0.12) or 0.12)
    corner_margin_ratio = float(max(0.05, min(0.30, corner_margin_ratio)))

    corner_margin_x = int(w * corner_margin_ratio)
    corner_margin_y = int(h * corner_margin_ratio)
    corners = {
        "top_left": (0, 0, corner_margin_x, corner_margin_y),
        "top_right": (w - corner_margin_x, 0, w, corner_margin_y),
        "bottom_left": (0, h - corner_margin_y, corner_margin_x, h),
        "bottom_right": (w - corner_margin_x, h - corner_margin_y, w, h),
    }

    min_size_ratio = float(marker_cfg.get("marker_min_size_ratio", 0.008) or 0.008)
    max_size_ratio = float(marker_cfg.get("marker_max_size_ratio", 0.07) or 0.07)
    min_size = min(w, h) * float(min_size_ratio)
    max_size = min(w, h) * float(max_size_ratio)
    min_area = min_size**2
    max_area = max_size**2

    bin_list: list[tuple[str, np.ndarray]] = []
    for th in (50, 80, 120):
        _, b = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)
        bin_list.append((f"th_{th}", b))
    _, b_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_list.append(("otsu", b_otsu))

    found: dict[str, dict[str, Any]] = {}
    kernel = np.ones((3, 3), np.uint8)

    for method, binary in bin_list:
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, ww, hh = cv2.boundingRect(contour)
            area_rect = ww * hh
            area_contour = float(cv2.contourArea(contour))
            if not (min_area < area_contour < max_area):
                continue
            ar = float(ww) / float(hh) if hh else 0.0
            if not (0.4 < ar < 2.5):
                continue

            cx, cy = x + ww // 2, y + hh // 2
            corner_name = None
            for name, (x1, y1, x2, y2) in corners.items():
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    corner_name = name
                    break
            if corner_name not in ("top_left", "top_right", "bottom_left"):
                continue

            fill_ratio = area_contour / float(area_rect) if area_rect else 0.0
            if fill_ratio <= 0.45:
                continue

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_val = float(cv2.mean(gray, mask=mask)[0])
            if mean_val >= 180:
                continue

            eps = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, eps, True)

            corner_xy = {
                "top_left": (0.0, 0.0),
                "top_right": (float(w - 1), 0.0),
                "bottom_left": (0.0, float(h - 1)),
            }.get(corner_name, (0.0, 0.0))
            dist = float(np.hypot(float(cx) - float(corner_xy[0]), float(cy) - float(corner_xy[1])))
            max_dist = float(np.hypot(float(corner_margin_x), float(corner_margin_y)))
            pos_score = max(0.0, 1.0 - (dist / max(1e-6, max_dist)))

            aspect_score = 1.0 - abs(ar - 1.0) * 0.5
            intensity_score = (180.0 - mean_val) / 180.0
            score = aspect_score * 0.22 + fill_ratio * 0.33 + intensity_score * 0.35 + pos_score * 0.10

            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype(np.float32)
                pts = order_quad_tl_tr_br_bl(pts)
            else:
                pts = np.array([[x, y], [x + ww - 1, y], [x + ww - 1, y + hh - 1], [x, y + hh - 1]], dtype=np.float32)

            info = {
                "corner": corner_name,
                "bbox": [int(x), int(y), int(ww), int(hh)],
                "points": pts.tolist(),
                "score": float(score),
                "pos_score": float(pos_score),
                "method": method,
            }
            if corner_name not in found or score > float(found[corner_name]["score"]):
                found[corner_name] = info

    return [found[k] for k in ("top_left", "top_right", "bottom_left") if k in found]


def _preprocess_variants_for_markers(image_bgr: np.ndarray, mode: str) -> list[tuple[str, np.ndarray]]:
    if mode == "none":
        return [("bgr", image_bgr)]

    variants: list[tuple[str, np.ndarray]] = [("bgr", image_bgr)]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

    if mode in ("basic", "morph"):
        try:
            clahe_cfg = PIPELINE_DEFAULTS["marker"]["clahe"]
            clahe = cv2.createCLAHE(clipLimit=float(clahe_cfg["clipLimit"]), tileGridSize=tuple(int(x) for x in clahe_cfg["tileGridSize"]))
            g2 = clahe.apply(gray)
            variants.append(("clahe", cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass
        try:
            at = PIPELINE_DEFAULTS["marker"]["adaptive_threshold"]
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, int(at["block_size"]), int(at["C"]))
            variants.append(("adaptive", cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass

    if mode == "morph":
        try:
            morph_cfg = PIPELINE_DEFAULTS["marker"]["morph"]
            k = max(int(morph_cfg["kernel_min"]), int(round(min(image_bgr.shape[:2]) * float(morph_cfg["kernel_ratio"]))))
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            at = PIPELINE_DEFAULTS["marker"]["adaptive_threshold"]
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, int(at["block_size"]), int(at["C"]))
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            variants.append(("adaptive_morph", cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass

    return variants


def detect_formA_marker_boxes(image_bgr: np.ndarray, *, preproc_mode: str) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    best_score = -1.0
    for name, var in _preprocess_variants_for_markers(image_bgr, preproc_mode):
        markers = detect_formA_marker_boxes_base(var)
        ok = len(markers) == 3
        score = float(sum(m.get("score", 0.0) for m in markers))
        if ok:
            score += 10.0
        if name != "bgr":
            score += 0.05
        if score > best_score:
            best_score = score
            best = markers
    return best


@dataclass
class MarkerGeometryConfig:
    max_marker_area_ratio: float = 3.0
    min_marker_area_page_ratio: float = 5e-5
    max_marker_area_page_ratio: float = 5e-3
    max_dist_ratio_relative_error: float = 0.35
    surround_pad_ratio: float = 2.0
    surround_pad_px_min: int = 8
    surround_pad_px_max: int = 120
    surround_min_mean_gray: float = 190.0
    surround_max_ink_ratio: float = 0.05
    surround_adaptive_block_size: int = 41
    surround_adaptive_C: int = 9


def make_formA_geom_cfg_for_apa_input() -> MarkerGeometryConfig:
    """apa_input（実写）向けのフォームA幾何制約（recall寄り）。

    paper_pipeline_v18.py の target/hard_target と同じ考え方で、
    机模様・影・枠線の写り込みによる取りこぼし（no_detection）を減らす。

    NOTE:
      精度/ハイパーパラメータを勝手に変更しないため、
      ここでの値は paper_pipeline_v18 の target 用と同値にする。
    """

    cfg_dict = (PIPELINE_DEFAULTS.get("formA") or {}).get("geometry") or {}
    allowed = set(getattr(MarkerGeometryConfig, "__dataclass_fields__", {}).keys())
    base = MarkerGeometryConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})

    # v18 target/hard_target と同じ緩和
    return MarkerGeometryConfig(
        **{
            **asdict(base),
            "surround_min_mean_gray": 150.0,
            "surround_max_ink_ratio": 0.08,
            "min_marker_area_page_ratio": 3.5e-5,
        }
    )


def _marker_center_xy(marker: dict[str, Any]) -> tuple[float, float]:
    x, y, w, h = marker.get("bbox", [0, 0, 0, 0])
    return float(x) + float(w) * 0.5, float(y) + float(h) * 0.5


def validate_formA_marker_geometry(image_bgr: np.ndarray, markers: list[dict[str, Any]], cfg: MarkerGeometryConfig) -> tuple[bool, dict[str, Any]]:
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
        corner_to_center[str(m.get("corner", ""))] = _marker_center_xy(m)

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

    def _check_surrounding_blankness(*, gray_img: np.ndarray, bbox_xywh: tuple[float, float, float, float]) -> tuple[bool, dict[str, Any]]:
        x, y, bw, bh = bbox_xywh
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
        bx0 = int(max(0, math.floor(x - x0)))
        by0 = int(max(0, math.floor(y - y0)))
        bx1 = int(min(roi.shape[1], math.ceil(x - x0 + bw)))
        by1 = int(min(roi.shape[0], math.ceil(y - y0 + bh)))
        mask = np.ones_like(roi, dtype=np.uint8)
        if bx1 > bx0 and by1 > by0:
            mask[by0:by1, bx0:bx1] = 0
        ring_area = int(mask.sum())
        if ring_area <= 0:
            return False, {"ok": False, "reason": "ring_area_zero", "roi": [x0, y0, x1, y1]}
        mean_gray = float((roi.astype(np.float32) * mask.astype(np.float32)).sum() / float(ring_area))
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
        }

    try:
        gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        surround_details: dict[str, Any] = {}
        for m in markers:
            corner = str(m.get("corner", ""))
            x, y, bw, bh = m.get("bbox", [0, 0, 0, 0])
            ok_blank, sdet = _check_surrounding_blankness(gray_img=gray_img, bbox_xywh=(x, y, bw, bh))
            surround_details[corner or "unknown"] = sdet
            if not ok_blank:
                detail["reasons"].append(f"marker_surrounding_not_blank:{corner or 'unknown'}")
        detail["surrounding_blankness"] = surround_details
    except Exception as e:
        detail["reasons"].append(f"surrounding_blankness_check_failed:{e}")

    ok = len(detail["reasons"]) == 0
    detail["ok"] = ok
    return ok, detail


def score_formA(
    image_bgr: np.ndarray,
    *,
    marker_preproc: str,
    geom_cfg: Optional[MarkerGeometryConfig] = None,
) -> tuple[bool, float, dict[str, Any]]:
    markers = detect_formA_marker_boxes(image_bgr, preproc_mode=str(marker_preproc))
    ok = len(markers) == 3
    if not ok:
        return False, 0.0, {"markers": markers}

    if geom_cfg is not None:
        cfg = geom_cfg
    else:
        cfg_dict = (PIPELINE_DEFAULTS.get("formA") or {}).get("geometry") or {}
        allowed = set(getattr(MarkerGeometryConfig, "__dataclass_fields__", {}).keys())
        cfg = MarkerGeometryConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})

    geom_ok, geom_detail = validate_formA_marker_geometry(image_bgr, markers, cfg)
    if not geom_ok:
        return False, 0.0, {"markers": markers, "geometry": geom_detail}

    base_score = float(sum(m.get("score", 0.0) for m in markers))

    h, w = image_bgr.shape[:2]
    expected = {"top_left": (0.0, 0.0), "top_right": (1.0, 0.0), "bottom_left": (0.0, 1.0)}
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
        closeness = max(0.0, 1.0 - (dist / 1.41421356))
        per_corner[corner] = float(closeness)
        pos_scores.append(float(closeness))
    pos_score = float(np.mean(pos_scores)) if pos_scores else 0.0
    score = base_score + pos_score * 2.0

    return True, float(score), {
        "markers": markers,
        "geometry": geom_detail,
        "pos_score": pos_score,
        "pos_score_per_corner": per_corner,
        "base_score": base_score,
        "marker_preproc": marker_preproc,
    }


def score_formB_fast(image_bgr: np.ndarray, *, wechat: WeChatQRDetectorPool) -> tuple[bool, float, dict[str, Any]]:
    qrs = detect_qr_codes_wechat_multiscale(image_bgr, wechat, mode="fast")
    if not qrs:
        return False, 0.0, {"qrs": [], "reason": "wechat_no_qr", "phase": "fast"}
    best_score, detail = score_best_qr_candidate(image_bgr, qrs)
    score = 1.0 + float(best_score)
    detail["phase"] = "fast"
    return True, float(score), detail


def score_formB(image_bgr: np.ndarray, *, wechat: WeChatQRDetectorPool) -> tuple[bool, float, dict[str, Any]]:
    qrs = detect_qr_codes_wechat_multiscale(image_bgr, wechat, mode="robust")
    if not qrs:
        return False, 0.0, {"qrs": [], "reason": "wechat_no_qr"}
    best_score, detail = score_best_qr_candidate(image_bgr, qrs)
    score = 1.0 + float(best_score)
    return True, float(score), detail


@dataclass
class FormDecision:
    ok: bool
    form: Optional[str]
    angle_deg: Optional[float]
    score: float
    detail: dict[str, Any]


def extract_form_unknown_reason(decision: Any) -> tuple[str, dict[str, Any]]:
    if decision is None:
        return "no_decision", {}

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
    return reason, diag


def decide_form_by_rotations(
    rectified_bgr: np.ndarray,
    *,
    wechat: WeChatQRDetectorPool,
    max_workers: int,
    marker_preproc: str,
    unknown_score_threshold: float,
    unknown_margin: float,
    formA_geom_cfg: Optional[MarkerGeometryConfig] = None,
) -> FormDecision:
    scan_angles = [float(a) for a in (PIPELINE_DEFAULTS.get("rotation_scan") or {}).get("scan_angles_2_deg", [])] or [0.0, 180.0]
    thr = float(unknown_score_threshold or 0.0)
    scan_results: list[dict[str, Any]] = []
    rejected_by_threshold: dict[str, Any] = {}
    rotated_cache: dict[float, np.ndarray] = {}

    def _get_rot(angle: float) -> np.ndarray:
        a = float(angle)
        if a not in rotated_cache:
            rotated_cache[a] = rotate_image_bound(rectified_bgr, a)
        return rotated_cache[a]

    valid_angles: list[float] = []
    for a in scan_angles:
        rot = _get_rot(a)
        h, w = rot.shape[:2]
        if h > w:
            scan_results.append({"angle": float(a), "skip": True, "reason": "portrait"})
            continue
        valid_angles.append(float(a))
    if not valid_angles:
        return FormDecision(False, None, None, 0.0, {"reason": "no_valid_angle", "scan_angles": scan_angles})

    def _run_parallel(func, angles: list[float]) -> list[dict[str, Any]]:
        if int(max_workers) <= 1 or len(angles) <= 1:
            return [func(a) for a in angles]
        with ThreadPoolExecutor(max_workers=min(int(max_workers), len(angles))) as ex:
            futures = [ex.submit(func, a) for a in angles]
            return [f.result() for f in as_completed(futures)]

    def _merge_scan(angle: float, key: str, value: Any) -> None:
        for sr in scan_results:
            if abs(float(sr.get("angle", -999.0)) - float(angle)) < 1e-6:
                sr[key] = value
                return
        scan_results.append({"angle": float(angle), "skip": False, key: value})

    # A
    def _eval_A(angle: float) -> dict[str, Any]:
        rot = _get_rot(angle)
        okA, scoreA, detA = score_formA(rot, marker_preproc=str(marker_preproc), geom_cfg=formA_geom_cfg)
        return {"angle": float(angle), "skip": False, "A": {"ok": bool(okA), "score": float(scoreA), "detail": detA}}

    bestA: Optional[FormDecision] = None
    for r in _run_parallel(_eval_A, valid_angles):
        scan_results.append(r)
        if (r.get("A") or {}).get("ok"):
            cand = FormDecision(True, "A", float(r["angle"]), float(r["A"]["score"]), {"A": r["A"]["detail"], "phase": "formA_found"})
            if bestA is None or cand.score > bestA.score:
                bestA = cand
    if bestA is not None:
        if thr > 0 and float(bestA.score) < thr:
            rejected_by_threshold["A"] = {"score": float(bestA.score), "phase": "formA_found"}
        else:
            return bestA

    # A_morph
    if str(marker_preproc) != "morph":

        def _eval_A_morph(angle: float) -> dict[str, Any]:
            rot = _get_rot(angle)
            okA, scoreA, detA = score_formA(rot, marker_preproc="morph", geom_cfg=formA_geom_cfg)
            return {"angle": float(angle), "skip": False, "A_morph": {"ok": bool(okA), "score": float(scoreA), "detail": detA}}

        bestA_m: Optional[FormDecision] = None
        for r in _run_parallel(_eval_A_morph, valid_angles):
            _merge_scan(float(r["angle"]), "A_morph", r.get("A_morph"))
            if (r.get("A_morph") or {}).get("ok"):
                cand = FormDecision(True, "A", float(r["angle"]), float(r["A_morph"]["score"]), {"A": r["A_morph"]["detail"], "phase": "formA_found_fallback_morph"})
                if bestA_m is None or cand.score > bestA_m.score:
                    bestA_m = cand
        if bestA_m is not None:
            if thr > 0 and float(bestA_m.score) < thr:
                rejected_by_threshold["A_morph"] = {"score": float(bestA_m.score), "phase": "formA_found_fallback_morph"}
            else:
                return bestA_m

    # B_fast
    def _eval_B_fast(angle: float) -> dict[str, Any]:
        rot = _get_rot(angle)
        okB, scoreB, detB = score_formB_fast(rot, wechat=wechat)
        return {"angle": float(angle), "skip": False, "B_fast": {"ok": bool(okB), "score": float(scoreB), "detail": detB}}

    bestB_fast: Optional[FormDecision] = None
    for r in _run_parallel(_eval_B_fast, valid_angles):
        _merge_scan(float(r["angle"]), "B_fast", r.get("B_fast"))
        if (r.get("B_fast") or {}).get("ok"):
            cand = FormDecision(True, "B", float(r["angle"]), float(r["B_fast"]["score"]), {"B_fast": r["B_fast"]["detail"], "phase": "formB_fast_found"})
            if bestB_fast is None or cand.score > bestB_fast.score:
                bestB_fast = cand
    if bestB_fast is not None:
        if thr > 0 and float(bestB_fast.score) < thr:
            rejected_by_threshold["B_fast"] = {"score": float(bestB_fast.score), "phase": "formB_fast_found"}
        else:
            return bestB_fast

    # B_robust
    def _eval_B(angle: float) -> dict[str, Any]:
        rot = _get_rot(angle)
        okB, scoreB, detB = score_formB(rot, wechat=wechat)
        return {"angle": float(angle), "skip": False, "B_robust": {"ok": bool(okB), "score": float(scoreB), "detail": detB}}

    bestB: Optional[FormDecision] = None
    for r in _run_parallel(_eval_B, valid_angles):
        _merge_scan(float(r["angle"]), "B_robust", r.get("B_robust"))
        if (r.get("B_robust") or {}).get("ok"):
            cand = FormDecision(True, "B", float(r["angle"]), float(r["B_robust"]["score"]), {"B": r["B_robust"]["detail"], "phase": "formB_robust_fallback"})
            if bestB is None or cand.score > bestB.score:
                bestB = cand
    if bestB is not None:
        if thr > 0 and float(bestB.score) < thr:
            rejected_by_threshold["B_robust"] = {"score": float(bestB.score), "phase": "formB_robust_fallback"}
        else:
            return bestB

    if rejected_by_threshold:
        best_rejected = max((float(v.get("score", 0.0)) for v in rejected_by_threshold.values()), default=0.0)
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
            },
        )

    return FormDecision(False, None, None, 0.0, {"reason": "no_detection", "scan": scan_results, "scan_angles": scan_angles})


# ============================================================
# UVDoc（必要実装を内製化：UVDocnet + bilinear_unwarping）
# ============================================================


IMG_SIZE = [488, 712]


def bilinear_unwarping(warped_img: torch.Tensor, point_positions: torch.Tensor, img_size: tuple[int, int]) -> torch.Tensor:
    import torch.nn.functional as F

    upsampled_grid = F.interpolate(point_positions, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True)
    grid = upsampled_grid.transpose(1, 2).transpose(2, 3)

    # paper_pipeline_v18 と同じ補正（端欠け抑制）
    try:
        gx = grid[..., 0]
        gy = grid[..., 1]
        eps = 1e-6

        gx_min = gx.amin(dim=(1, 2), keepdim=True)
        gx_max = gx.amax(dim=(1, 2), keepdim=True)
        gy_min = gy.amin(dim=(1, 2), keepdim=True)
        gy_max = gy.amax(dim=(1, 2), keepdim=True)

        gx_span = (gx_max - gx_min).clamp_min(eps)
        gy_span = (gy_max - gy_min).clamp_min(eps)

        gx_center = (gx_max + gx_min) * 0.5
        gy_center = (gy_max + gy_min) * 0.5

        gx = (gx - gx_center) * (2.0 / gx_span)
        gy = (gy - gy_center) * (2.0 / gy_span)
        grid = torch.stack([gx, gy], dim=-1)
        grid = grid.clamp(-1.0, 1.0)
    except Exception:
        pass

    return F.grid_sample(warped_img, grid, align_corners=True, padding_mode="border")


def conv3x3(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> torch.nn.Conv2d:
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)


def dilated_conv_bn_act(in_channels: int, out_channels: int, act_fn: torch.nn.Module, BatchNorm: Any, dilation: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            bias=False,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        ),
        BatchNorm(out_channels),
        act_fn,
    )


def dilated_conv(in_channels: int, out_channels: int, kernel_size: int, dilation: int, stride: int = 1) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
        )
    )


class ResidualBlockWithDilation(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        BatchNorm: Any,
        kernel_size: int,
        stride: int = 1,
        downsample: Optional[torch.nn.Module] = None,
        is_activation: bool = True,
        is_top: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.downsample = downsample
        self.is_activation = is_activation
        self.is_top = is_top
        if self.stride != 1 or self.is_top:
            self.conv1 = conv3x3(in_channels, out_channels, kernel_size, self.stride)
            self.conv2 = conv3x3(out_channels, out_channels, kernel_size)
        else:
            self.conv1 = dilated_conv(in_channels, out_channels, kernel_size, dilation=3)
            self.conv2 = dilated_conv(out_channels, out_channels, kernel_size, dilation=3)
        self.bn1 = BatchNorm(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn2 = BatchNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2 += residual
        out = self.relu(out2)
        return out


class ResnetStraight(torch.nn.Module):
    def __init__(
        self,
        num_filter: int,
        map_num: list[int],
        BatchNorm: Any,
        block_nums: list[int] = [3, 4, 6, 3],
        block: Any = ResidualBlockWithDilation,
        kernel_size: int = 5,
        stride: list[int] = [1, 1, 2, 2],
    ):
        super().__init__()
        self.in_channels = num_filter * map_num[0]
        self.stride = stride
        self.relu = torch.nn.ReLU(inplace=True)
        self.block_nums = block_nums
        self.kernel_size = kernel_size

        self.layer1 = self.blocklayer(block, num_filter * map_num[0], self.block_nums[0], BatchNorm, kernel_size=self.kernel_size, stride=self.stride[0])
        self.layer2 = self.blocklayer(block, num_filter * map_num[1], self.block_nums[1], BatchNorm, kernel_size=self.kernel_size, stride=self.stride[1])
        self.layer3 = self.blocklayer(block, num_filter * map_num[2], self.block_nums[2], BatchNorm, kernel_size=self.kernel_size, stride=self.stride[2])

    def blocklayer(self, block: Any, out_channels: int, block_nums: int, BatchNorm: Any, kernel_size: int, stride: int = 1) -> torch.nn.Sequential:
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = torch.nn.Sequential(conv3x3(self.in_channels, out_channels, kernel_size=kernel_size, stride=stride), BatchNorm(out_channels))

        layers: list[torch.nn.Module] = []
        layers.append(block(self.in_channels, out_channels, BatchNorm, kernel_size, stride, downsample, is_top=True))
        self.in_channels = out_channels
        for _ in range(1, block_nums):
            layers.append(block(out_channels, out_channels, BatchNorm, kernel_size, is_activation=True, is_top=False))
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return out3


class UVDocnet(torch.nn.Module):
    def __init__(self, num_filter: int, kernel_size: int = 5):
        super().__init__()
        self.num_filter = num_filter
        self.in_channels = 3
        self.kernel_size = kernel_size
        self.stride = [1, 2, 2, 2]

        BatchNorm = torch.nn.BatchNorm2d
        act_fn = torch.nn.ReLU(inplace=True)
        map_num = [1, 2, 4, 8, 16]

        self.resnet_head = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.num_filter * map_num[0], bias=False, kernel_size=self.kernel_size, stride=2, padding=self.kernel_size // 2),
            BatchNorm(self.num_filter * map_num[0]),
            act_fn,
            torch.nn.Conv2d(self.num_filter * map_num[0], self.num_filter * map_num[0], bias=False, kernel_size=self.kernel_size, stride=2, padding=self.kernel_size // 2),
            BatchNorm(self.num_filter * map_num[0]),
            act_fn,
        )

        self.resnet_down = ResnetStraight(
            self.num_filter,
            map_num,
            BatchNorm,
            block_nums=[3, 4, 6, 3],
            block=ResidualBlockWithDilation,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

        map_num_i = 2
        self.bridge_1 = torch.nn.Sequential(dilated_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn, BatchNorm, dilation=1))
        self.bridge_2 = torch.nn.Sequential(dilated_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn, BatchNorm, dilation=2))
        self.bridge_3 = torch.nn.Sequential(dilated_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn, BatchNorm, dilation=5))
        self.bridge_4 = torch.nn.Sequential(*[dilated_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn, BatchNorm, dilation=d) for d in [8, 3, 2]])
        self.bridge_5 = torch.nn.Sequential(*[dilated_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn, BatchNorm, dilation=d) for d in [12, 7, 4]])
        self.bridge_6 = torch.nn.Sequential(*[dilated_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn, BatchNorm, dilation=d) for d in [18, 12, 6]])

        self.bridge_concat = torch.nn.Sequential(
            torch.nn.Conv2d(self.num_filter * map_num[map_num_i] * 6, self.num_filter * map_num[2], bias=False, kernel_size=1, stride=1, padding=0),
            BatchNorm(self.num_filter * map_num[2]),
            act_fn,
        )

        self.out_point_positions2D = torch.nn.Sequential(
            torch.nn.Conv2d(self.num_filter * map_num[2], self.num_filter * map_num[0], bias=False, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2, padding_mode="reflect"),
            BatchNorm(self.num_filter * map_num[0]),
            torch.nn.PReLU(),
            torch.nn.Conv2d(self.num_filter * map_num[0], 2, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2, padding_mode="reflect"),
        )

        self.out_point_positions3D = torch.nn.Sequential(
            torch.nn.Conv2d(self.num_filter * map_num[2], self.num_filter * map_num[0], bias=False, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2, padding_mode="reflect"),
            BatchNorm(self.num_filter * map_num[0]),
            torch.nn.PReLU(),
            torch.nn.Conv2d(self.num_filter * map_num[0], 3, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2, padding_mode="reflect"),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, torch.nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                torch.nn.init.xavier_normal_(m.weight, gain=0.2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        resnet_head = self.resnet_head(x)
        resnet_down = self.resnet_down(resnet_head)
        bridge_1 = self.bridge_1(resnet_down)
        bridge_2 = self.bridge_2(resnet_down)
        bridge_3 = self.bridge_3(resnet_down)
        bridge_4 = self.bridge_4(resnet_down)
        bridge_5 = self.bridge_5(resnet_down)
        bridge_6 = self.bridge_6(resnet_down)
        bridge_concat = torch.cat([bridge_1, bridge_2, bridge_3, bridge_4, bridge_5, bridge_6], dim=1)
        bridge = self.bridge_concat(bridge_concat)
        out_point_positions2D = self.out_point_positions2D(bridge)
        out_point_positions3D = self.out_point_positions3D(bridge)
        return out_point_positions2D, out_point_positions3D


class UVDocUnwrapper:
    def __init__(self, *, ckpt_path: Path, device: str):
        self.device = torch.device(device)
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"UVDoc checkpoint not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=self.device)
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise RuntimeError("Unexpected UVDoc checkpoint format")
        model = UVDocnet(num_filter=32, kernel_size=5)
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)
        model.eval()
        self.model = model
        self.img_size = tuple(int(x) for x in IMG_SIZE)

    @torch.no_grad()
    def unwarp_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = cv2.resize(img_rgb, self.img_size).transpose(2, 0, 1)
        inp_t = torch.from_numpy(inp).unsqueeze(0).to(self.device)
        point_positions2D, _ = self.model(inp_t)

        out_w = int(img_rgb.shape[1])
        out_h = int(img_rgb.shape[0])
        warped_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        unwarped = bilinear_unwarping(warped_img=warped_t, point_positions=torch.unsqueeze(point_positions2D[0], dim=0), img_size=(out_w, out_h))
        out_rgb = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


# ============================================================
# 背景除算法
# ============================================================


def apply_background_division(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = dict((PIPELINE_DEFAULTS.get("background_division") or {}))
    if not bool(cfg.get("enable", True)):
        return image_bgr, {"applied": False, "reason": "disabled"}
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
        bg = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma, sigmaY=sigma)
        bg = np.maximum(bg.astype(np.float32), bg_min)
        L_corr = cv2.divide(L.astype(np.float32), bg, scale=255.0)
        L_corr = np.clip(L_corr, 0, 255).astype(np.uint8)
        lab2 = cv2.merge([L_corr, A, B])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out, {"applied": True, "sigma": float(sigma), "bg_min": float(bg_min)}
    except Exception as e:
        return image_bgr, {"applied": False, "reason": f"exception:{e}"}


# ============================================================
# XFeat matching（paper_pipeline_v18.py 相当）
# ============================================================


def compute_reproj_rms(H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    src = src_pts.reshape(-1, 1, 2).astype(np.float32)
    dst = dst_pts.reshape(-1, 1, 2).astype(np.float32)
    proj = cv2.perspectiveTransform(src, np.asarray(H, dtype=np.float64))
    err = np.linalg.norm(proj - dst, axis=2).reshape(-1)
    return float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")


def refine_homography_least_squares(H_init: np.ndarray, mkpts0: np.ndarray, mkpts1: np.ndarray, inlier_mask: np.ndarray) -> tuple[np.ndarray, Optional[float]]:
    H0 = np.asarray(H_init, dtype=np.float64)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.size != len(mkpts0) or mask.size != len(mkpts1):
        return H0, None
    if int(mask.sum()) < 4:
        return H0, None
    p0 = np.asarray(mkpts0, dtype=np.float32)[mask]
    p1 = np.asarray(mkpts1, dtype=np.float32)[mask]
    H_ls, _ = cv2.findHomography(p0, p1, 0)
    if H_ls is None:
        return H0, None
    try:
        rms = compute_reproj_rms(np.asarray(H_ls, dtype=np.float64), p0, p1)
    except Exception:
        rms = None
    return np.asarray(H_ls, dtype=np.float64), rms


@dataclass
class CachedRef:
    template_path: str
    s_ref: float
    out0: dict[str, Any]
    template_bgr: Optional[np.ndarray] = None


class XFeatMatcher:
    def __init__(self, *, top_k: int, device: str, match_max_side: int):
        ensure_portable_git_on_path()
        self.device = str(device)
        self.top_k = int(top_k)
        self.match_max_side = int(match_max_side)
        self.xfeat = (
            torch.hub.load(
                "verlab/accelerated_features",
                "XFeat",
                pretrained=True,
                top_k=self.top_k,
            )
            .to(self.device)
            .eval()
        )


class CachedXFeatMatcher:
    def __init__(self, base: XFeatMatcher):
        self.base = base
        self.xfeat = base.xfeat
        self.top_k = int(base.top_k)
        self.match_max_side = int(base.match_max_side)

    def prepare_ref(self, template_bgr: np.ndarray, template_path: str) -> CachedRef:
        ref_small, s_ref = resize_keep_aspect(template_bgr, self.match_max_side)
        out0 = self.xfeat.detectAndCompute(ref_small, top_k=self.top_k)[0]
        out0.update({"image_size": (ref_small.shape[1], ref_small.shape[0])})
        return CachedRef(template_path=str(template_path), s_ref=float(s_ref), out0=out0, template_bgr=template_bgr)

    def prepare_target(self, tgt_bgr: np.ndarray) -> tuple[dict[str, Any], float, np.ndarray]:
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
    ) -> tuple[bool, dict[str, Any]]:
        matches = self.xfeat.match_lighterglue(ref.out0, out1)
        if isinstance(matches, (list, tuple)) and len(matches) >= 2:
            mkpts0, mkpts1 = matches[0], matches[1]
        elif isinstance(matches, dict) and "mkpts0" in matches and "mkpts1" in matches:
            mkpts0, mkpts1 = matches["mkpts0"], matches["mkpts1"]
        else:
            return False, {"ok": False, "inliers": 0, "matches": 0, "inlier_ratio": 0.0, "H_ref_to_tgt": None}

        mkpts0 = np.asarray(mkpts0, dtype=np.float32)
        mkpts1 = np.asarray(mkpts1, dtype=np.float32)
        if len(mkpts0) < 4:
            return False, {"ok": False, "inliers": 0, "matches": int(len(mkpts0)), "inlier_ratio": 0.0, "H_ref_to_tgt": None}

        H_small, mask = cv2.findHomography(
            mkpts0,
            mkpts1,
            cv2.USAC_MAGSAC,
            float(PIPELINE_DEFAULTS["homography"]["find"]["ransac_reproj_threshold_px"]),
            maxIters=int(PIPELINE_DEFAULTS["homography"]["find"]["max_iters"]),
            confidence=float(PIPELINE_DEFAULTS["homography"]["find"]["confidence"]),
        )
        if H_small is None or mask is None:
            return False, {"ok": False, "inliers": 0, "matches": int(len(mkpts0)), "inlier_ratio": 0.0, "H_ref_to_tgt": None}

        mask = mask.reshape(-1).astype(bool)
        inliers = int(mask.sum())
        matches_n = int(len(mask))
        inlier_ratio = float(inliers) / float(matches_n) if matches_n else 0.0

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
        return True, {
            "ok": True,
            "inliers": inliers,
            "matches": matches_n,
            "inlier_ratio": float(inlier_ratio),
            "reproj_rms": reproj,
            "H_ref_to_tgt": H_full.astype(float).tolist(),
        }


def safe_invert_homography(
    H: np.ndarray,
    *,
    inliers: int,
    inlier_ratio: float,
    min_inliers: int,
    min_inlier_ratio: float,
    max_cond: float,
) -> tuple[bool, Optional[np.ndarray], str, float, float]:
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
        if not math.isfinite(cond) or (float(max_cond) > 0 and cond > float(max_cond)):
            return False, None, f"cond too large ({cond:.3e})", float(cond), float(det)
    except Exception:
        cond = float("nan")

    try:
        return True, np.linalg.inv(H), "ok", float(cond), float(det)
    except Exception as e:
        return False, None, f"inv failed: {e}", float(cond), float(det)


# ============================================================
# デモ画像（9_demo 相当）生成
# ============================================================


def _thickness_params(image_bgr: np.ndarray) -> tuple[int, float, int]:
    h, w = image_bgr.shape[:2]
    scale = min(w, h) / 1000.0
    thickness = max(6, int(scale * 10))
    font_scale = max(0.8, scale * 1.2)
    font_thickness = max(2, int(scale * 4))
    return thickness, font_scale, font_thickness


def _get_japanese_font(size_px: int) -> ImageFont.FreeTypeFont:
    font_path = os.environ.get("APA_FONT_PATH")
    if font_path:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size=int(size_px))
        except Exception:
            pass

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
        candidates += ["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/Helvetica.ttc"]
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
    return ImageFont.load_default()


def draw_text_pil(image_bgr: np.ndarray, xy: tuple[int, int], text: str, color_bgr: tuple[int, int, int], font_size: int) -> np.ndarray:
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    font = _get_japanese_font(font_size)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    x, y = int(xy[0]), int(xy[1])
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=color_rgb)
    out_rgb = np.array(pil)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def _hstack_with_padding(left_bgr: np.ndarray, right_bgr: np.ndarray, *, pad_color_bgr: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    lh, lw = left_bgr.shape[:2]
    rh, rw = right_bgr.shape[:2]
    out_h = max(lh, rh)

    def _pad(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h == out_h:
            return img
        pad = out_h - h
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=pad_color_bgr)

    return np.hstack([_pad(left_bgr), _pad(right_bgr)])


def _draw_polygon_outline(image_bgr: np.ndarray, poly_xy: np.ndarray, color_bgr: tuple[int, int, int], thickness: int) -> np.ndarray:
    out = image_bgr.copy()
    pts = np.asarray(poly_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, color_bgr, int(thickness))
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


def _perspective_transform_points(points_xy: np.ndarray, H_3x3: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    H = np.asarray(H_3x3, dtype=np.float64)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def _generate_demo9_image(
    *,
    input_bgr: np.ndarray,
    polygon_xy: np.ndarray,
    polygon_margin_px: float,
    H_input_to_rectified_landscape: np.ndarray,
    rectified_landscape_size_wh: tuple[int, int],
    decided_form: str,
    decided_angle_deg: float,
    decision_markers: Optional[list[dict[str, Any]]],
    decision_qrs: Optional[list[dict[str, Any]]],
    aligned_bgr: Optional[np.ndarray],
    stage_label: str,
) -> np.ndarray:
    poly_exp = expand_polygon(
        np.asarray(polygon_xy, dtype=np.float32),
        margin_px=float(polygon_margin_px),
        img_w=int(input_bgr.shape[1]),
        img_h=int(input_bgr.shape[0]),
    )
    left = draw_polygon_overlay(input_bgr, poly_exp)

    # 進捗（ステージ）を左上へ
    try:
        left = draw_text_pil(left, (10, 10), f"stage={stage_label}", (0, 0, 255), font_size=28)
    except Exception:
        pass

    rect_w, rect_h = int(rectified_landscape_size_wh[0]), int(rectified_landscape_size_wh[1])
    dummy_rect = np.zeros((max(1, rect_h), max(1, rect_w), 3), dtype=np.uint8)
    _dummy_rot, M_rect_to_chosen = rotate_image_bound_with_matrix(dummy_rect, float(decided_angle_deg))

    H_in_to_rect = np.asarray(H_input_to_rectified_landscape, dtype=np.float64)
    H_in_to_chosen = np.asarray(M_rect_to_chosen, dtype=np.float64) @ H_in_to_rect
    try:
        H_chosen_to_in = np.linalg.inv(H_in_to_chosen)
    except Exception:
        H_chosen_to_in = None

    if H_chosen_to_in is not None:
        if decided_form == "A" and decision_markers:
            for m in decision_markers:
                try:
                    x, y, bw, bh = m.get("bbox", [0, 0, 0, 0])
                    box = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float32)
                    box_in = _perspective_transform_points(box, H_chosen_to_in)
                    left = _draw_polygon_outline(left, box_in, (0, 0, 255), thickness=6)
                except Exception:
                    continue
        if decided_form == "B" and decision_qrs:
            try:
                pts = np.asarray(decision_qrs[0].get("points"), dtype=np.float32).reshape(-1, 2)
                pts_in = _perspective_transform_points(pts, H_chosen_to_in)
                left = _draw_polygon_outline(left, pts_in, (255, 0, 0), thickness=6)
            except Exception:
                pass

    if aligned_bgr is None:
        # 失敗時は右を黒背景+ラベルにする
        right = np.zeros((left.shape[0], left.shape[0], 3), dtype=np.uint8)
        try:
            right = draw_text_pil(right, (10, 10), "NO_OUTPUT", (0, 0, 255), font_size=36)
        except Exception:
            pass
    else:
        right = aligned_bgr
    return _hstack_with_padding(left, right)


# ============================================================
# 実行（APA.py から呼ばれる）
# ============================================================


@dataclass
class StageTimes:
    docaligner_s: float = 0.0
    rectify_s: float = 0.0
    decide_s: float = 0.0
    uvdoc_s: float = 0.0
    bgdiv_s: float = 0.0
    match_s: float = 0.0
    warp_s: float = 0.0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 入出力
    p.add_argument("--input-dir", type=str, default=str(Path(__file__).resolve().parent / "apa_input"))
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "apa_output"))
    p.add_argument("--log-dir", type=str, default=str(Path(__file__).resolve().parent / "apa_log"))
    p.add_argument("--template-dir", type=str, default=str(Path(__file__).resolve().parent / "apa_template"))
    p.add_argument("--limit", type=int, default=0, help="先頭N枚だけ処理（0=全て）")

    # ログ
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    p.add_argument("--console-log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    # WeChat
    p.add_argument("--wechat-model-dir", type=str, default=str(PIPELINE_DEFAULTS["wechat"]["model_dir"]))

    # DocAligner
    p.add_argument("--docaligner-model", choices=["lcnet050", "lcnet100", "fastvit_t8", "fastvit_sa24"], default=str(PIPELINE_DEFAULTS["docaligner"]["model"]))
    p.add_argument("--docaligner-type", choices=["point", "heatmap"], default=str(PIPELINE_DEFAULTS["docaligner"]["type"]))
    p.add_argument("--docaligner-max-side", type=int, default=int(PIPELINE_DEFAULTS["docaligner"]["rectified_max_side_px"]))
    p.add_argument("--polygon-margin-ratio", type=float, default=float(PIPELINE_DEFAULTS["docaligner"]["polygon_margin"]["ratio"]))
    p.add_argument("--polygon-margin-min-px", type=float, default=float(PIPELINE_DEFAULTS["docaligner"]["polygon_margin"]["min_px"]))
    p.add_argument("--polygon-margin-max-px", type=float, default=float(PIPELINE_DEFAULTS["docaligner"]["polygon_margin"]["max_px"]))
    p.add_argument("--polygon-margin-px", type=float, default=float(PIPELINE_DEFAULTS["docaligner"]["polygon_margin"]["fixed_px"]))

    # フォーム判定
    p.add_argument("--rotation-max-workers", type=int, default=int(PIPELINE_DEFAULTS["rotation_scan"]["max_workers"]))
    p.add_argument("--marker-preproc", choices=["none", "basic", "morph"], default=str(PIPELINE_DEFAULTS["marker"]["preproc_mode"]))
    p.add_argument("--unknown-score-threshold", type=float, default=float(PIPELINE_DEFAULTS["unknown"]["score_threshold"]))
    p.add_argument("--unknown-margin", type=float, default=float(PIPELINE_DEFAULTS["unknown"]["margin"]))

    # XFeat
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=str(PIPELINE_DEFAULTS["xfeat"]["device_default"]))
    p.add_argument("--top-k", type=int, default=int(PIPELINE_DEFAULTS["xfeat"]["top_k"]))
    p.add_argument("--match-max-side", type=int, default=int(PIPELINE_DEFAULTS["xfeat"]["match_max_side_px"]))

    # warp許可
    p.add_argument("--min-inliers-for-warp", type=int, default=int(PIPELINE_DEFAULTS["warp"]["min_inliers"]))
    p.add_argument("--min-inlier-ratio-for-warp", type=float, default=float(PIPELINE_DEFAULTS["warp"]["min_inlier_ratio"]))
    p.add_argument("--max-h-cond", type=float, default=float(PIPELINE_DEFAULTS["warp"]["max_h_cond"]))

    return p


def _list_input_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _list_templates(template_dir: Path) -> tuple[list[Path], list[Path]]:
    dirA = template_dir / "A"
    dirB = template_dir / "B"
    A = sorted([p for p in dirA.glob("*.jpg") if p.is_file()]) if dirA.exists() else []
    B = sorted([p for p in dirB.glob("*.jpg") if p.is_file()]) if dirB.exists() else []
    return A, B


def run_apa_pipeline(args: argparse.Namespace) -> tuple[Path, Path]:
    """APA.py から呼ぶ実行関数。

    戻り値:
      (out_run_dir, log_run_dir)
    """

    run_id = f"run_{now_run_id()}"
    out_root = Path(str(args.output_dir)) / run_id
    log_root = Path(str(args.log_dir)) / run_id

    logger = setup_logging(log_dir=log_root, level=str(args.log_level), console_level=str(args.console_log_level))

    logger.info("=" * 70)
    logger.info("APA pipeline (apa_input -> demo9 -> apa_output)")
    logger.info("=" * 70)
    logger.info("OpenCV: %s", cv2.__version__)
    logger.info("torch : %s", torch.__version__)
    logger.info("input-dir   : %s", str(args.input_dir))
    logger.info("template-dir: %s", str(args.template_dir))
    logger.info("output-dir  : %s", str(out_root))
    logger.info("log-dir     : %s", str(log_root))

    mkdir(out_root)

    input_paths = _list_input_images(Path(str(args.input_dir)))
    if int(getattr(args, "limit", 0) or 0) > 0:
        input_paths = input_paths[: int(args.limit)]
    if not input_paths:
        raise FileNotFoundError(f"No input images found in: {args.input_dir}")

    template_A, template_B = _list_templates(Path(str(args.template_dir)))
    if not template_A or not template_B:
        raise FileNotFoundError("Templates not found. Expected: apa_template/A/*.jpg and apa_template/B/*.jpg")

    # device
    device = "cuda" if str(args.device) == "auto" and torch.cuda.is_available() else (str(args.device) if str(args.device) != "auto" else "cpu")
    ensure_portable_git_on_path()

    # Load models
    logger.info("[INFO] Loading DocAligner...")
    model, cb = load_docaligner_model(str(args.docaligner_model), str(args.docaligner_type))
    logger.info("[OK] DocAligner loaded")

    logger.info("[INFO] Loading XFeat...")
    matcher = XFeatMatcher(top_k=int(args.top_k), device=str(device), match_max_side=int(args.match_max_side))
    cached_matcher = CachedXFeatMatcher(matcher)
    logger.info("[OK] XFeat loaded")

    logger.info("[INFO] Loading templates (and caching features)...")
    templates_A: list[CachedRef] = []
    templates_B: list[CachedRef] = []
    for pth in template_A:
        img = cv2.imread(str(pth))
        if img is None:
            continue
        templates_A.append(cached_matcher.prepare_ref(img, str(pth)))
    for pth in template_B:
        img = cv2.imread(str(pth))
        if img is None:
            continue
        templates_B.append(cached_matcher.prepare_ref(img, str(pth)))
    logger.info("[OK] template cache built: A=%d B=%d", len(templates_A), len(templates_B))

    logger.info("[INFO] Initializing WeChat QR detector pool...")
    wechat = WeChatQRDetectorPool(model_dir=str(args.wechat_model_dir), pool_size=int(args.rotation_max_workers))
    logger.info("[OK] WeChat QR ready")

    logger.info("[INFO] Loading UVDoc...")
    uvdoc = UVDocUnwrapper(ckpt_path=Path(str(PIPELINE_DEFAULTS["uvdoc"]["ckpt_path"])), device=str(device))
    logger.info("[OK] UVDoc loaded")

    # apa_input（実写）向けのフォームA幾何制約（recall寄り）を1回だけ構築
    formA_geom_cfg_for_apa = make_formA_geom_cfg_for_apa_input()

    # CSV summary
    rows: list[dict[str, Any]] = []
    t_run0 = time.perf_counter()
    jpeg_quality = int((PIPELINE_DEFAULTS.get("save_images") or {}).get("jpeg_quality") or 95)

    for idx, in_path in enumerate(input_paths, start=1):
        case_id = f"{idx:04d}_{in_path.stem}"
        t0 = time.perf_counter()
        times = StageTimes()

        item: dict[str, Any] = {
            "case_id": case_id,
            "input_filename": in_path.name,
            "stage": "start",
            "pred_form": "",
            "pred_angle": "",
            "best_template": "",
            "output_demo9": "",
            "error": "",
        }

        img_in = cv2.imread(str(in_path))
        if img_in is None:
            item["stage"] = "read_failed"
            item["error"] = "cv2.imread returned None"
            rows.append({**item, "elapsed_total_s": f"{time.perf_counter()-t0:.6f}"})
            logger.warning("[CASE] %s stage=%s file=%s", case_id, item["stage"], in_path.name)
            continue

        # 2) docaligner
        td0 = time.perf_counter()
        poly, doc_meta = detect_polygon_docaligner_multi(
            logger=logger,
            args=args,
            degraded_bgr=img_in,
            wechat=wechat,
            formA_geom_cfg=formA_geom_cfg_for_apa,
        )
        times.docaligner_s = time.perf_counter() - td0
        if poly is None:
            item["stage"] = "docaligner_failed"
            item["error"] = "no polygon"
            demo9 = _generate_demo9_image(
                input_bgr=img_in,
                polygon_xy=np.array([[0, 0], [img_in.shape[1]-1, 0], [img_in.shape[1]-1, img_in.shape[0]-1], [0, img_in.shape[0]-1]], dtype=np.float32),
                polygon_margin_px=0.0,
                H_input_to_rectified_landscape=np.eye(3, dtype=np.float64),
                rectified_landscape_size_wh=(img_in.shape[1], img_in.shape[0]),
                decided_form="",
                decided_angle_deg=0.0,
                decision_markers=None,
                decision_qrs=None,
                aligned_bgr=None,
                stage_label=item["stage"],
            )
            out_path = out_root / f"{case_id}_demo9.jpg"
            write_image(out_path, demo9, jpeg_quality=jpeg_quality)
            item["output_demo9"] = str(out_path.name)
            rows.append({**item, **asdict(times), "elapsed_total_s": f"{time.perf_counter()-t0:.6f}"})
            logger.warning("[CASE] %s stage=%s file=%s", case_id, item["stage"], in_path.name)
            continue

        # margin
        if float(getattr(args, "polygon_margin_px", 0.0) or 0.0) > 0:
            margin_px = float(args.polygon_margin_px)
        else:
            picked_margin_px = None
            try:
                if isinstance(doc_meta, dict) and isinstance(doc_meta.get("best"), dict):
                    picked_margin_px = float(doc_meta["best"].get("margin_px"))
            except Exception:
                picked_margin_px = None

            if picked_margin_px is not None and math.isfinite(float(picked_margin_px)):
                margin_px = float(picked_margin_px)
            else:
                margin_px = polygon_margin_px_from_ratio(
                    poly,
                    ratio=float(args.polygon_margin_ratio),
                    min_px=float(args.polygon_margin_min_px),
                    max_px=float(args.polygon_margin_max_px),
                )

        # 3) rectify
        tr0 = time.perf_counter()
        rectified, H_in_to_rect, poly_exp, rect_meta = rectify_with_margin_and_optional_padding(
            img_in,
            polygon_xy=poly,
            margin_px=float(margin_px),
            out_max_side=int(args.docaligner_max_side),
        )
        rect0_h, rect0_w = rectified.shape[:2]
        rectified, rot90 = enforce_landscape(rectified)
        M_rect_to_land = _landscape_rotation_matrix_if_applied(w=int(rect0_w), h=int(rect0_h), rotated_90cw=bool(rot90))
        H_in_to_rect_land = np.asarray(M_rect_to_land, dtype=np.float64) @ np.asarray(H_in_to_rect, dtype=np.float64)
        times.rectify_s = time.perf_counter() - tr0

        # 4) decide
        tdec0 = time.perf_counter()
        decision = decide_form_by_rotations(
            rectified,
            wechat=wechat,
            max_workers=int(args.rotation_max_workers),
            marker_preproc=str(args.marker_preproc),
            unknown_score_threshold=float(args.unknown_score_threshold),
            unknown_margin=float(args.unknown_margin),
            formA_geom_cfg=formA_geom_cfg_for_apa,
        )
        times.decide_s = time.perf_counter() - tdec0
        item["pred_form"] = str(decision.form or "")
        item["pred_angle"] = "" if decision.angle_deg is None else str(float(decision.angle_deg))

        if (not decision.ok) or (decision.form not in ("A", "B")) or (decision.angle_deg is None):
            item["stage"] = "form_unknown"
            reason, _ = extract_form_unknown_reason(asdict(decision))
            item["error"] = str(reason)
            demo9 = _generate_demo9_image(
                input_bgr=img_in,
                polygon_xy=poly,
                polygon_margin_px=float(margin_px),
                H_input_to_rectified_landscape=H_in_to_rect_land,
                rectified_landscape_size_wh=(int(rectified.shape[1]), int(rectified.shape[0])),
                decided_form=str(decision.form or ""),
                decided_angle_deg=float(decision.angle_deg or 0.0),
                decision_markers=None,
                decision_qrs=None,
                aligned_bgr=None,
                stage_label=item["stage"],
            )
            out_path = out_root / f"{case_id}_demo9.jpg"
            write_image(out_path, demo9, jpeg_quality=jpeg_quality)
            item["output_demo9"] = str(out_path.name)
            rows.append({**item, **asdict(times), "elapsed_total_s": f"{time.perf_counter()-t0:.6f}"})
            elapsed_total = time.perf_counter() - t0
            logger.warning(
                "[CASE] %s stage=%s file=%s pred_form=%s reason=%s total=%.3fs (doc=%.3f rectify=%.3f decide=%.3f)",
                case_id,
                item["stage"],
                in_path.name,
                item["pred_form"],
                item.get("error"),
                float(elapsed_total),
                times.docaligner_s,
                times.rectify_s,
                times.decide_s,
            )
            continue

        # chosen
        chosen = rotate_image_bound(rectified, float(decision.angle_deg))

        # 5) uvdoc
        tuv0 = time.perf_counter()
        try:
            chosen_uv = uvdoc.unwarp_bgr(chosen)
        except Exception as e:
            item["stage"] = "uvdoc_failed"
            item["error"] = str(e)
            chosen_uv = chosen
        times.uvdoc_s = time.perf_counter() - tuv0

        # 6) bgdiv
        tbg0 = time.perf_counter()
        bgdiv_bgr, bgdiv_meta = apply_background_division(chosen_uv)
        times.bgdiv_s = time.perf_counter() - tbg0

        # 7) match all templates (A/B)
        tmatch0 = time.perf_counter()
        templates = templates_A if str(decision.form) == "A" else templates_B
        out1, _s_tgt, invS_tgt = cached_matcher.prepare_target(bgdiv_bgr)

        best: Optional[dict[str, Any]] = None
        for ref in templates:
            ok, res = cached_matcher.match_with_cached_ref_and_prepared_target(ref, out1=out1, invS_tgt=invS_tgt)
            if not ok:
                continue
            if best is None:
                best = {"template": ref.template_path, **res}
            else:
                if int(res.get("inliers", 0)) > int(best.get("inliers", 0)):
                    best = {"template": ref.template_path, **res}
                elif int(res.get("inliers", 0)) == int(best.get("inliers", 0)):
                    if float(res.get("inlier_ratio", 0.0)) > float(best.get("inlier_ratio", 0.0)):
                        best = {"template": ref.template_path, **res}
        times.match_s = time.perf_counter() - tmatch0

        if best is None or not bool(best.get("ok")):
            item["stage"] = "xfeat_failed"
            item["error"] = "no best match"
            demo9 = _generate_demo9_image(
                input_bgr=img_in,
                polygon_xy=poly,
                polygon_margin_px=float(margin_px),
                H_input_to_rectified_landscape=H_in_to_rect_land,
                rectified_landscape_size_wh=(int(rectified.shape[1]), int(rectified.shape[0])),
                decided_form=str(decision.form),
                decided_angle_deg=float(decision.angle_deg),
                decision_markers=((decision.detail or {}).get("A") or {}).get("markers") if str(decision.form) == "A" else None,
                decision_qrs=((decision.detail or {}).get("B") or {}).get("qrs") if str(decision.form) == "B" else None,
                aligned_bgr=None,
                stage_label=item["stage"],
            )
            out_path = out_root / f"{case_id}_demo9.jpg"
            write_image(out_path, demo9, jpeg_quality=jpeg_quality)
            item["output_demo9"] = str(out_path.name)
            rows.append({**item, **asdict(times), "elapsed_total_s": f"{time.perf_counter()-t0:.6f}"})
            logger.warning("[CASE] %s stage=%s file=%s", case_id, item["stage"], in_path.name)
            continue

        item["best_template"] = Path(str(best.get("template") or "")).name

        # 8) warp to template
        tw0 = time.perf_counter()
        H_tpl_to_img = np.asarray(best.get("H_ref_to_tgt"), dtype=np.float64)
        ok_inv, H_img_to_tpl, inv_reason, h_cond, h_det = safe_invert_homography(
            H_tpl_to_img,
            inliers=int(best.get("inliers", 0)),
            inlier_ratio=float(best.get("inlier_ratio", 0.0)),
            min_inliers=int(args.min_inliers_for_warp),
            min_inlier_ratio=float(args.min_inlier_ratio_for_warp),
            max_cond=float(args.max_h_cond),
        )
        if not ok_inv or H_img_to_tpl is None:
            item["stage"] = "homography_unstable"
            item["error"] = str(inv_reason)
            aligned = None
        else:
            tpl_path = Path(str(best.get("template") or ""))
            tpl_bgr = cv2.imread(str(tpl_path))
            if tpl_bgr is None:
                item["stage"] = "template_read_failed"
                item["error"] = "cv2.imread template returned None"
                aligned = None
            else:
                aligned = cv2.warpPerspective(bgdiv_bgr, H_img_to_tpl, (tpl_bgr.shape[1], tpl_bgr.shape[0]))
                item["stage"] = "done"
        times.warp_s = time.perf_counter() - tw0

        # demo9（出力はこれだけ）
        decision_markers = ((decision.detail or {}).get("A") or {}).get("markers") if str(decision.form) == "A" else None
        decision_qrs = None
        if str(decision.form) == "B":
            dB = (decision.detail or {}).get("B")
            dBf = (decision.detail or {}).get("B_fast")
            decision_qrs = (dB or {}).get("qrs") or (dBf or {}).get("qrs")

        demo9 = _generate_demo9_image(
            input_bgr=img_in,
            polygon_xy=poly,
            polygon_margin_px=float(margin_px),
            H_input_to_rectified_landscape=H_in_to_rect_land,
            rectified_landscape_size_wh=(int(rectified.shape[1]), int(rectified.shape[0])),
            decided_form=str(decision.form),
            decided_angle_deg=float(decision.angle_deg),
            decision_markers=decision_markers,
            decision_qrs=decision_qrs,
            aligned_bgr=aligned,
            stage_label=item["stage"],
        )
        out_path = out_root / f"{case_id}_demo9.jpg"
        write_image(out_path, demo9, jpeg_quality=jpeg_quality)
        item["output_demo9"] = str(out_path.name)

        elapsed_total = time.perf_counter() - t0
        rows.append({**item, **asdict(times), "elapsed_total_s": f"{elapsed_total:.6f}"})
        logger.info(
            "[CASE] %s stage=%s file=%s pred_form=%s best=%s total=%.3fs (doc=%.3f rectify=%.3f decide=%.3f uvdoc=%.3f bgdiv=%.3f match=%.3f warp=%.3f)",
            case_id,
            item["stage"],
            in_path.name,
            item["pred_form"],
            item["best_template"],
            float(elapsed_total),
            times.docaligner_s,
            times.rectify_s,
            times.decide_s,
            times.uvdoc_s,
            times.bgdiv_s,
            times.match_s,
            times.warp_s,
        )

    elapsed_run = time.perf_counter() - t_run0
    logger.info("[DONE] total_images=%d elapsed=%.3fs", len(input_paths), float(elapsed_run))

    # summary.csv
    csv_path = log_root / "summary.csv"
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("[DONE] summary.csv: %s", str(csv_path))

    return out_root, log_root
