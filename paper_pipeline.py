#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper_pipeline.py

目的
----
ユーザー指示のパイプラインを、既存の検証コード（DocAligner / フォームA・B判定 / XFeat Homography）を
ベースにして「静止画像一括処理」として実装する。

パイプライン概要
----------------
入力: `APA/image/{A,B,C}/*.jpg`（全画像）

1) 改悪生成（`APA/test_recovery_paper.py` の実装を流用）
2) 改悪画像に DocAligner を適用し、紙の四角枠（polygon）を得る
   - 得られない場合、その画像の処理は終了
3) polygon を用いて紙領域を透視補正（perspective warp）
4) 透視補正後の紙画像を 0〜350度（10度刻み）で回転させ、フォーム判定を並列に実行
   - フォームA: 3点マーク（TL/TR/BL）が検出できる
   - フォームB: 右上にQRコードが検出できる
   - 見つからなければ、その画像の処理は終了
5) 確定フォームに応じて、全テンプレ（`APA/image/A` or `APA/image/B`）と XFeat matching を実行
   - Homography を推定し、対応点一致度（inliers）最大のテンプレを採用
6) 採用した Homography の逆で、改悪画像（透視補正＋回転確定済み）をテンプレ座標にワープして保存

出力
----
`APA/output_pipeline/run_YYYYmmdd_HHMMSS/` 配下に（処理順が分かるように番号付き）：
 - 1_degraded/             : 改悪画像
 - 2_doc/                  : DocAligner polygon 可視化
 - 3_rectified/            : 透視補正した紙画像
 - 4_rectified_rot/        : フォーム確定に使った回転後画像
 - 5_debug_matches/        : best template のマッチ可視化
 - 6_aligned/              : best template にワープした結果
 - summary.json / summary.csv

注意
----
 - torch.hub 経由の XFeat 読み込みで git が必要になることがあるため、
   portable git を PATH に追加する処理を `test_recovery_paper` から流用する。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont


# --- reuse implementations from previous work ---
# NOTE: このスクリプトは `python APA/paper_pipeline.py ...` の形で実行される想定。
# その場合 sys.path[0] は `.../APA` になるため、同ディレクトリのモジュールは
# `from test_recovery_paper import ...` の形で import する（`import APA.xxx` は失敗しやすい）。
from test_recovery_paper import (
    XFeatMatcher,
    detect_formA_marker_boxes,
    draw_inlier_matches,
    ensure_portable_git_on_path,
    now_run_id,
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

    # Windows default Japanese font
    candidates = [
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\meiryob.ttc",
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size=int(size_px))
        except Exception:
            pass
    # Fallback: default (may not support Japanese, but keep running)
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
        out = draw_text_pil(
            out,
            (int(x), max(5, int(y) - font_px - 4)),
            label,
            color_bgr=(0, 0, 255),
            font_size=font_px,
            outline=True,
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
    out = draw_text_pil(
        out,
        (int(tr[0] + 10), max(5, int(tr[1] - font_px - 4))),
        "右上",
        color_bgr=(255, 0, 0),
        font_size=font_px,
        outline=True,
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
    - 複数の前処理（gray/CLAHE/Otsu）
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


def score_formA(image_bgr: np.ndarray) -> tuple[bool, float, dict[str, Any]]:
    """フォームA判定。

    3点マーカー検出（TL/TR/BL）ができることに加えて、
    それぞれが「本来の位置（左上/右上/左下）」に近いほどスコア加点する。
    """

    markers = detect_formA_marker_boxes(image_bgr)
    ok = len(markers) == 3
    if not ok:
        return False, 0.0, {"markers": markers}

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

    return True, float(score), {"markers": markers, "pos_score": pos_score, "pos_score_per_corner": per_corner, "base_score": base_score}


def score_formB(image_bgr: np.ndarray) -> tuple[bool, float, dict[str, Any]]:
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

    return True, 1.0 + rel * 10.0 + pos_score, {"qrs": qrs, "qr_center": [cx, cy], "qr_rel_area": rel, "qr_pos_score": pos_score}


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
) -> FormDecision:
    """Try all angles in parallel; return best valid decision.

    NOTE:
    - QR の robust 検出は重く、全角度×並列で回すと極端に遅くなる/環境により不安定。
      そのため、ここでは
        1) 全角度を並列で FAST 判定（QRの角度候補を絞る）
        2) FASTで最良だった角度（+近傍）だけ ROBUST で再検証
      の二段階にする。
    """

    def _eval(angle: float) -> dict[str, Any]:
        rotated = rotate_image_bound(rectified_bgr, angle)
        # Enforce landscape after rotation (ユーザー要望: 横長に統一)
        rotated, _ = enforce_landscape(rotated)
        h, w = rotated.shape[:2]
        if h > w:
            return {"angle": float(angle), "skip": True}

        okA, scoreA, detA = score_formA(rotated)
        okBf, scoreBf, detBf = score_formB_fast(rotated)

        return {
            "angle": float(angle),
            "skip": False,
            "A": {"ok": bool(okA), "score": float(scoreA), "detail": detA},
            "B_fast": {"ok": bool(okBf), "score": float(scoreBf), "detail": detBf},
        }

    bestA: Optional[FormDecision] = None
    bestB_fast: Optional[FormDecision] = None
    b_fast_angles: list[float] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_eval, a) for a in angles]
        for fut in as_completed(futures):
            r = fut.result()
            if not r or r.get("skip"):
                continue

            angle = float(r["angle"])

            # Track best A
            if r["A"]["ok"]:
                candA = FormDecision(True, "A", angle, float(r["A"]["score"]), {"A": r["A"]["detail"]})
                if bestA is None or candA.score > bestA.score:
                    bestA = candA

            # Track best B (fast)
            if r["B_fast"]["ok"]:
                b_fast_angles.append(angle)
                candB = FormDecision(True, "B", angle, float(r["B_fast"]["score"]), {"B_fast": r["B_fast"]["detail"]})
                if bestB_fast is None or candB.score > bestB_fast.score:
                    bestB_fast = candB

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

    # 回転スキャン（ユーザー要件: 0..350 を10度刻み）
    p.add_argument("--rotation-step", type=float, default=10.0, help="フォーム判定の回転スキャン刻み（度）")
    p.add_argument("--rotation-max-workers", type=int, default=8, help="回転スキャンの並列数（スレッド）")

    p.add_argument("--docaligner-model", choices=["lcnet050", "lcnet100", "fastvit_t8", "fastvit_sa24"], default="fastvit_sa24")
    p.add_argument("--docaligner-type", choices=["point", "heatmap"], default="heatmap")
    # 透視補正後の紙画像が小さすぎると QR が潰れて検出しづらいので、デフォルトは少し大きめ。
    p.add_argument("--docaligner-max-side", type=int, default=2400, help="Max side length for rectified paper")
    p.add_argument(
        "--polygon-margin",
        type=float,
        default=80.0,
        help="DocAligner polygon を外側に広げるマージン(px)。端のQR/マーカー欠け対策。",
    )

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--top-k", type=int, default=1024, help="XFeatの特徴点数（大きいほど高精度だが遅い）")
    p.add_argument("--match-max-side", type=int, default=1200, help="XFeat用にリサイズする最大辺(px)（大きいほど高精度だが遅い）")

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
        f"  --docaligner-model    使用モデル（精度/速度のトレードオフ） [default: {defaults.docaligner_model}]",
        f"  --docaligner-max-side 透視補正後の紙画像の最大辺(px) [default: {defaults.docaligner_max_side}]",
        f"  --polygon-margin      polygon を外側に広げる(px)。端のQR/マーカー欠け対策 [default: {defaults.polygon_margin}]",
        "",
        "【フォーム判定】",
        f"  --rotation-step       0..350度を何度刻みで回して判定するか（例: 10） [default: {defaults.rotation_step}]",
        f"  --rotation-max-workers 回転スキャンの並列数（スレッド） [default: {defaults.rotation_max_workers}]",
        "",
        "【XFeat（位置合わせ）】",
        f"  --device              XFeatの実行デバイス（cpu/cuda/auto） [default: {defaults.device}]",
        f"  --top-k               特徴点数（大きいほど高精度だが遅い） [default: {defaults.top_k}]",
        f"  --match-max-side      マッチング前にリサイズする最大辺(px)（大きいほど高精度だが遅い） [default: {defaults.match_max_side}]",
        "",
        "【出力】",
        f"  --out                 出力ディレクトリ（run_... が作成される） [default: {defaults.out}]",
        "",
        "最小コマンド例（おすすめデフォルト使用）:",
        r"  C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline.py --limit 1",
        "",
    ]
    print("\n".join(lines))


def print_config(args: argparse.Namespace) -> None:
    """起動時に主要設定を一覧表示する（引数が多い問題への対策）。"""

    print("[CONFIG]")
    print(f"  src-forms          : {args.src_forms}")
    print(f"  limit              : {args.limit}")
    print(f"  degrade-n           : {args.degrade_n}")
    print(f"  rotation-step       : {args.rotation_step} deg")
    print(f"  rotation-max-workers: {args.rotation_max_workers}")
    print(f"  polygon-margin      : {args.polygon_margin} px")
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


def main(argv=None) -> int:
    args = parse_args(argv)

    if getattr(args, "explain", False):
        print_explain()
        return 0

    print("=" * 70)
    print("paper_pipeline")
    print("=" * 70)
    print(f"OpenCV: {cv2.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"src-forms: {args.src_forms}")
    print_config(args)

    # Setup device for XFeat
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")
    ensure_portable_git_on_path()

    # Output directories
    out_root = mkdir(Path(args.out) / f"run_{now_run_id()}")
    # 出力は「処理順が分かる」ように番号付きフォルダ名にする
    degraded_dir = mkdir(out_root / "1_degraded")
    doc_dir = mkdir(out_root / "2_doc")
    rect_dir = mkdir(out_root / "3_rectified")
    rot_dir = mkdir(out_root / "4_rectified_rot")
    debug_matches_dir = mkdir(out_root / "5_debug_matches")
    aligned_dir = mkdir(out_root / "6_aligned")

    # Load heavy models
    print("[INFO] Loading DocAligner...")
    model, cb = load_docaligner_model(args.docaligner_model, args.docaligner_type)
    print("[OK] DocAligner loaded")

    print("[INFO] Loading XFeat...")
    matcher = XFeatMatcher(top_k=args.top_k, device=device, match_max_side=args.match_max_side)
    print("[OK] XFeat loaded")

    # Prepare angles for form detection
    step = float(args.rotation_step)
    angles = [float(a) for a in np.arange(0.0, 360.0, step) if a < 360.0 - 1e-6]
    angles = [a for a in angles if a <= 350.0 + 1e-6]  # enforce 0..350
    if not angles:
        print("[ERROR] rotation angles list is empty")
        return 1

    src_forms = [s.strip() for s in args.src_forms.split(",") if s.strip()]
    src_forms = [s for s in src_forms if s in ("A", "B", "C")]
    if not src_forms:
        print("[ERROR] src-forms must contain at least one of A,B,C")
        return 1

    # Templates for final alignment (A/B only)
    templates_A = list_images("A")
    templates_B = list_images("B")
    if not templates_A or not templates_B:
        print("[ERROR] templates not found. Expected APA/image/A and APA/image/B")
        return 1

    summary: list[dict[str, Any]] = []
    t_all0 = time.time()

    for sf in src_forms:
        sources = list_images(sf)
        if args.limit and args.limit > 0:
            sources = sources[: int(args.limit)]

        if not sources:
            print(f"[WARN] no sources: APA/image/{sf}")
            continue

        print(f"\n[INFO] Processing sources from form {sf}: {len(sources)} images")
        for sp in sources:
            src_bgr = cv2.imread(str(sp))
            if src_bgr is None:
                print(f"[WARN] failed to read: {sp}")
                continue

            for k in range(int(args.degrade_n)):
                case_id = f"{sf}_{sp.stem}_deg{k:02d}"
                item: dict[str, Any] = {
                    "source_form": sf,
                    "source_path": str(sp),
                    "case": case_id,
                    "ok": False,
                    "stage": "start",
                }

                # Make per-case RNG so that results are reproducible and independent of processing order.
                # NOTE: Python の hash() は実行ごとに値が変わり得るため、
                # 安定なハッシュ（crc32）を使って seed を作る。
                stable = zlib.crc32(f"{sf}/{sp.name}".encode("utf-8")) & 0xFFFFFFFF
                case_seed = (int(args.seed) * 1_000_000) + int(stable) * 100 + int(k)
                rng = random.Random(case_seed)

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
                cv2.imwrite(str(degraded_dir / f"{case_id}.jpg"), degraded_bgr)
                item["stage"] = "degraded"
                item["degrade"] = degrade_meta
                item["H_src_to_degraded"] = H_src_to_deg.astype(float).tolist()

                # DocAligner polygon
                poly = detect_polygon_docaligner(model, cb, degraded_bgr)
                if poly is None:
                    item["stage"] = "docaligner_failed"
                    summary.append(item)
                    continue

                item["stage"] = "docaligner_ok"
                item["polygon"] = poly.astype(float).tolist()
                # Expand polygon slightly to avoid clipping markers/QR near the paper edge
                poly_exp = expand_polygon(
                    poly,
                    margin_px=float(args.polygon_margin),
                    img_w=int(degraded_bgr.shape[1]),
                    img_h=int(degraded_bgr.shape[0]),
                )

                overlay = draw_polygon_overlay(degraded_bgr, poly_exp)
                cv2.imwrite(str(doc_dir / f"{case_id}_doc.jpg"), overlay)

                # Rectify (perspective)
                rectified, H_deg_to_rect = polygon_to_rectified(
                    degraded_bgr,
                    poly_exp,
                    out_max_side=int(args.docaligner_max_side),
                )
                # ユーザー要望: 透視補正後は横長（landscape）に統一
                rectified, _ = enforce_landscape(rectified)
                cv2.imwrite(str(rect_dir / f"{case_id}_rect.jpg"), rectified)
                item["stage"] = "rectified"
                item["H_degraded_to_rectified"] = H_deg_to_rect.astype(float).tolist()

                # Decide form by rotations
                decision = decide_form_by_rotations(
                    rectified,
                    angles=angles,
                    max_workers=int(args.rotation_max_workers),
                )
                item["form_decision"] = asdict(decision)
                if not decision.ok or decision.form not in ("A", "B") or decision.angle_deg is None:
                    item["stage"] = "form_not_found"
                    summary.append(item)
                    continue

                item["stage"] = "form_found"
                # Create the chosen rotated image used for matching
                chosen = rotate_image_bound(rectified, float(decision.angle_deg))
                chosen, _ = enforce_landscape(chosen)

                # 4_rectified_rot は「フォーム判定に使った回転後画像」なので、
                # 判定に使った特徴（A=3点マーカー / B=QR）を可視化して保存する。
                if decision.form == "A":
                    markers = ((decision.detail or {}).get("A") or {}).get("markers") or []
                    rot_vis = draw_formA_markers_overlay(chosen, markers)
                else:
                    # B の場合、ROBUSTで確定した情報があればそれを優先。
                    qrs = ((decision.detail or {}).get("B") or {}).get("qrs")
                    if not qrs:
                        qrs = detect_qr_codes_robust(chosen)
                    rot_vis = draw_formB_qr_overlay(chosen, qrs)

                cv2.imwrite(str(rot_dir / f"{case_id}_rot.jpg"), rot_vis)

                # Match against all templates of the decided form
                templates = templates_A if decision.form == "A" else templates_B
                best: Optional[dict[str, Any]] = None
                for tp in templates:
                    tpl_bgr = cv2.imread(str(tp))
                    if tpl_bgr is None:
                        continue
                    res, H_tpl_to_img, mk0, mk1 = matcher.match_and_estimate_h(tpl_bgr, chosen)
                    if not res.ok or H_tpl_to_img is None or mk0 is None or mk1 is None:
                        cand = {
                            "template": str(tp),
                            "ok": False,
                            "inliers": int(getattr(res, "inliers", 0)),
                            "matches": int(getattr(res, "matches", 0)),
                            "inlier_ratio": float(getattr(res, "inlier_ratio", 0.0)),
                        }
                    else:
                        cand = {
                            "template": str(tp),
                            "ok": True,
                            "H_template_to_image": H_tpl_to_img.astype(float).tolist(),
                            **asdict(res),
                        }
                    if best is None:
                        best = cand
                    else:
                        # choose by inliers (対応点の一致数として採用)
                        if int(cand.get("inliers", 0)) > int(best.get("inliers", 0)):
                            best = cand
                        elif int(cand.get("inliers", 0)) == int(best.get("inliers", 0)):
                            if float(cand.get("inlier_ratio", 0.0)) > float(best.get("inlier_ratio", 0.0)):
                                best = cand

                item["best_match"] = best
                if best is None or not best.get("ok"):
                    item["stage"] = "xfeat_failed"
                    summary.append(item)
                    continue

                # Warp chosen image to template plane
                tpl_path = Path(str(best["template"]))
                tpl_bgr = cv2.imread(str(tpl_path))
                if tpl_bgr is None:
                    item["stage"] = "template_read_failed"
                    summary.append(item)
                    continue

                H_tpl_to_img = np.asarray(best["H_ref_to_tgt"], dtype=np.float64)
                H_img_to_tpl = np.linalg.inv(H_tpl_to_img)
                warped = cv2.warpPerspective(chosen, H_img_to_tpl, (tpl_bgr.shape[1], tpl_bgr.shape[0]))
                cv2.imwrite(str(aligned_dir / f"{case_id}_aligned.jpg"), warped)

                # debug matches (visualization uses internal resized coords)
                try:
                    # recompute mkpts for visualization
                    res2, _, mk0, mk1 = matcher.match_and_estimate_h(tpl_bgr, chosen)
                    if res2.ok and mk0 is not None and mk1 is not None:
                        dbg = draw_inlier_matches(tpl_bgr, chosen, mk0, mk1, args.match_max_side)
                        cv2.imwrite(str(debug_matches_dir / f"{case_id}_matches.jpg"), dbg)
                except Exception:
                    pass

                item["stage"] = "done"
                item["ok"] = True
                summary.append(item)
                print(f"  [OK] {case_id}: form={decision.form} angle={decision.angle_deg} best={tpl_path.name} inliers={best.get('inliers')}")

    # Save summary
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = out_root / "summary.csv"
    header = [
        "case",
        "source_form",
        "source_path",
        "ok",
        "stage",
        "decided_form",
        "decided_angle",
        "best_template",
        "best_inliers",
        "best_inlier_ratio",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for it in summary:
            dec = it.get("form_decision", {}) or {}
            best = it.get("best_match", {}) or {}
            row = [
                str(it.get("case", "")),
                str(it.get("source_form", "")),
                str(it.get("source_path", "")),
                str(it.get("ok", "")),
                str(it.get("stage", "")),
                str(dec.get("form", "")),
                str(dec.get("angle_deg", "")),
                str(best.get("template", "")),
                str(best.get("inliers", "")),
                str(best.get("inlier_ratio", "")),
            ]
            f.write(",".join(row) + "\n")

    dt = time.time() - t_all0
    print(f"\n[DONE] outputs: {out_root}")
    print(f"[DONE] elapsed: {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
