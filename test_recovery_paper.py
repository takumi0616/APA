#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_recovery_paper.py

目的
----
フォームA/Bのテンプレ（正解）画像を基準にして、

- テンプレから自動生成した改悪画像（回転/射影/背景合成など）

を XFeat matching で位置合わせ（Homography 推定）できるかを評価する。

重要な方針（ユーザー指示）
--------------------------
- bad/ は無視（改悪はテンプレから生成）
- 品質改善（2値化/影補正など）は実装しない
- XFeat matching による Homography 性能を見たい

使い方（例）
------------
静止画像（テンプレ→改悪生成→評価）:

  python APA/test_recovery_paper.py --mode images --form A
  python APA/test_recovery_paper.py --mode images --form B

"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# XFeat deps
import torch


# ------------------------------------------------------------
# Windows environment: keep stdout UTF-8 friendly
# ------------------------------------------------------------
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def ensure_portable_git_on_path() -> None:
    """torch.hub may try to call git; this PC uses portable Git not on PATH."""

    portable_git_bin = r"C:\Users\takumi\develop\git\bin"
    if os.path.exists(portable_git_bin):
        os.environ["PATH"] = portable_git_bin + os.pathsep + os.environ.get("PATH", "")


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


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


# ------------------------------------------------------------
# Synthetic degradation generation
# ------------------------------------------------------------


def random_background(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Generate a simple random background (noise + gradients)."""

    bg = np.zeros((h, w, 3), dtype=np.uint8)

    # base color
    base = np.array([rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)], dtype=np.uint8)
    bg[:, :] = base

    # gradient overlay
    gx = np.linspace(0, 1, w, dtype=np.float32)
    gy = np.linspace(0, 1, h, dtype=np.float32)
    g = (np.outer(gy, gx) * 255.0).astype(np.float32)
    g3 = np.stack([g, g, g], axis=-1)
    bg = to_uint8(0.6 * bg.astype(np.float32) + 0.4 * g3)

    # noise (milder)
    n = np.zeros((h, w, 3), dtype=np.float32)
    n[:, :, 0] = np.random.normal(0, 8, size=(h, w))
    n[:, :, 1] = np.random.normal(0, 8, size=(h, w))
    n[:, :, 2] = np.random.normal(0, 8, size=(h, w))
    bg = to_uint8(bg.astype(np.float32) + n)

    # random lines (milder)
    for _ in range(rng.randint(3, 10)):
        x1, y1 = rng.randint(0, w - 1), rng.randint(0, h - 1)
        x2, y2 = rng.randint(0, w - 1), rng.randint(0, h - 1)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        cv2.line(bg, (x1, y1), (x2, y2), color, rng.randint(1, 2), lineType=cv2.LINE_AA)

    return bg


def warp_template_to_random_view(
    template_bgr: np.ndarray,
    out_size: tuple[int, int],
    rng: random.Random,
    max_rotation_deg: float = 12.0,
    min_abs_rotation_deg: float = 0.0,
    rotation_mode: str = "uniform",
    snap_step_deg: float = 90.0,
    perspective_jitter: float = 0.08,
    min_visible_area_ratio: float = 0.25,
    max_attempts: int = 50,
    save_steps_dir: Optional[Path] = None,
    step_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create a degraded image by warping template into a random quadrilateral on a random background.

    Returns:
        degraded_bgr, H_ref_to_deg
    """

    h, w = template_bgr.shape[:2]
    out_w, out_h = out_size

    # IMPORTANT:
    # 以前は、回転後に頂点を単純クリップしていたため、
    #   - 紙が画面外に逃げる
    #   - 紙が極端に小さくなる
    #   - マスクがほぼ0（紙が写っていない）
    # のケースが出ていた。
    #
    # ここでは「紙が必ず画面内に十分写る」まで再生成する。

    margin = int(min(out_w, out_h) * 0.08)
    base_w_min = int(out_w * 0.70)
    base_w_max = int(out_w * 0.92)
    min_visible_area_px = int(out_w * out_h * float(min_visible_area_ratio))

    dst_quad = None
    base_w = 0
    base_h = 0
    angle = 0.0
    for _attempt in range(max_attempts):
        # destination quad center
        cx = rng.randint(margin, out_w - margin)
        cy = rng.randint(margin, out_h - margin)

        # Make the paper occupy larger area so that it keeps enough resolution.
        base_w = rng.randint(base_w_min, base_w_max)
        base_h = int(base_w * (h / w))
        base_h = max(120, min(base_h, int(out_h * 0.85)))

        # start from axis-aligned rectangle
        x1, y1 = cx - base_w // 2, cy - base_h // 2
        x2, y2 = cx + base_w // 2, cy + base_h // 2
        rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        # rotation
        # rotation_mode:
        #   - uniform: 0〜360一様（max_rotation_deg>=180時） or [-max_rot, +max_rot]
        #   - snap:   snap_step_deg刻み（デフォルト90度）にスナップして「上下逆/横向き」を確実に作る
        if max_rotation_deg >= 180:
            if rotation_mode == "snap":
                step = float(max(1.0, snap_step_deg))
                # 0, step, 2*step, ... の候補からランダム（0度近傍は min_abs_rotation_deg で除外）
                candidates = [i * step for i in range(int(round(360.0 / step)))]
                rng.shuffle(candidates)
                angle = 0.0
                for cand in candidates:
                    dist0 = min(cand % 360.0, 360.0 - (cand % 360.0))
                    if dist0 >= float(min_abs_rotation_deg):
                        angle = float(cand % 360.0)
                        break
            else:
                for _ in range(100):
                    angle = rng.uniform(0.0, 360.0)
                    dist0 = min(angle, 360.0 - angle)
                    if dist0 >= float(min_abs_rotation_deg):
                        break
                else:
                    angle = rng.uniform(0.0, 360.0)
        else:
            angle = rng.uniform(-max_rotation_deg, max_rotation_deg)

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rect_rot = cv2.transform(rect.reshape(-1, 1, 2), M).reshape(4, 2)

        # perspective jitter
        jitter = perspective_jitter * min(base_w, base_h)
        rect_rot += np.array(
            [[rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)] for _ in range(4)],
            dtype=np.float32,
        )

        # --- Keep the quad inside the output by scaling (not clipping) ---
        # Clipping can collapse the quad and effectively remove the paper.
        # Instead, if some corners go out of bounds, scale the quad towards the center.
        inset = 2.0
        dx = rect_rot[:, 0] - cx
        dy = rect_rot[:, 1] - cy
        max_dx = float(np.max(np.abs(dx))) if len(dx) else 0.0
        max_dy = float(np.max(np.abs(dy))) if len(dy) else 0.0
        allow_x = float(min(cx, (out_w - 1) - cx)) - inset
        allow_y = float(min(cy, (out_h - 1) - cy)) - inset
        if allow_x <= 1 or allow_y <= 1:
            continue

        sx = allow_x / max_dx if max_dx > 1e-6 else 1.0
        sy = allow_y / max_dy if max_dy > 1e-6 else 1.0
        s = float(min(1.0, sx, sy))
        if s < 1.0:
            rect_rot = np.stack([cx + dx * s, cy + dy * s], axis=1).astype(np.float32)

        # Accept only if all corners are inside
        if (
            (rect_rot[:, 0].min() >= 0)
            and (rect_rot[:, 1].min() >= 0)
            and (rect_rot[:, 0].max() <= out_w - 1)
            and (rect_rot[:, 1].max() <= out_h - 1)
        ):
            # IMPORTANT:
            # ここでは「並べ替え」をしない。
            # rect は [TL, TR, BR, BL] の順番で生成しており、
            # 回転・射影後もこの“対応関係”を保ったまま Homography を作るべき。
            # order_quad_* で画面上のTL/TR/BR/BLに並べ替えると、
            # 角の対応が崩れて意図した回転（例：QRが下側）が壊れる。
            cand = rect_rot.astype(np.float32)

            # quick visible-area check by warping a ones mask
            tmp_mask = cv2.warpPerspective(
                np.ones((h, w), dtype=np.uint8) * 255,
                cv2.getPerspectiveTransform(
                    np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32),
                    cand,
                ),
                (out_w, out_h),
            )
            if int(cv2.countNonZero(tmp_mask)) >= min_visible_area_px:
                dst_quad = cand
                break

    if dst_quad is None:
        # 最後の手段: クリップして必ず返す（ただしログ的には不自然になる）
        rect_rot[:, 0] = np.clip(rect_rot[:, 0], 0, out_w - 1)
        rect_rot[:, 1] = np.clip(rect_rot[:, 1], 0, out_h - 1)
        dst_quad = order_quad_tl_tr_br_bl(rect_rot)

    src_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_quad, dst_quad)

    bg = random_background(out_h, out_w, rng)
    if save_steps_dir is not None:
        mkdir(save_steps_dir)
        cv2.imwrite(str(save_steps_dir / f"{step_prefix}01_background.jpg"), bg)

    warped = cv2.warpPerspective(template_bgr, H, (out_w, out_h))
    if save_steps_dir is not None:
        cv2.imwrite(str(save_steps_dir / f"{step_prefix}02_warped_template.jpg"), warped)

    # mask for blending
    mask = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, H, (out_w, out_h))
    if save_steps_dir is not None:
        cv2.imwrite(str(save_steps_dir / f"{step_prefix}03_mask.jpg"), mask)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    degraded = np.where(mask3 > 0, warped, bg)
    if save_steps_dir is not None:
        cv2.imwrite(str(save_steps_dir / f"{step_prefix}04_composited.jpg"), degraded)

    # add *mild* blur/noise to simulate capturing conditions
    if rng.random() < 0.35:
        k = 3
        degraded = cv2.GaussianBlur(degraded, (k, k), rng.uniform(0.4, 0.8))
    if rng.random() < 0.35:
        degraded = to_uint8(degraded.astype(np.float32) + np.random.normal(0, 4, size=degraded.shape))

    if save_steps_dir is not None:
        cv2.imwrite(str(save_steps_dir / f"{step_prefix}05_final_degraded.jpg"), degraded)

    meta = {
        "angle_deg": float(angle),
        "rotation_mode": str(rotation_mode),
        "snap_step_deg": float(snap_step_deg),
        "base_w": int(base_w),
        "base_h": int(base_h),
        "out_w": int(out_w),
        "out_h": int(out_h),
        "perspective_jitter": float(perspective_jitter),
        "min_visible_area_ratio": float(min_visible_area_ratio),
        "max_attempts": int(max_attempts),
    }
    return degraded, H, meta


# ------------------------------------------------------------
# Form feature detection (for debug/metadata only)
# ------------------------------------------------------------


def detect_qr_codes_multiscale(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """フォームB想定: OpenCV QRCodeDetector のマルチスケール検出（test_capture_formB.py を踏襲）。"""

    qr_results: list[dict[str, Any]] = []
    qr_detector = cv2.QRCodeDetector()
    height, width = image_bgr.shape[:2]

    scales = [0.5, 0.25, 1.0, 0.125, 0.75]
    for scale in scales:
        if scale == 1.0:
            test_image = image_bgr
        else:
            new_width = int(width * scale)
            new_height = int(height * scale)
            if new_width < 100 or new_height < 100:
                continue
            test_image = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

        try:
            data, points, _ = qr_detector.detectAndDecode(test_image)
            if data and points is not None:
                pts = points[0]
                if scale != 1.0:
                    pts = pts / scale
                pts = pts.astype(np.float32)
                qr_results.append(
                    {
                        "data": data,
                        "points": pts.tolist(),
                        "scale": scale,
                    }
                )
                break
        except Exception:
            pass

    return qr_results


def detect_formA_marker_boxes(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """フォームA想定: 3点マークを検出し、各マーカーの角点（矩形4隅 or approx）を返す。

    注: ここでの角点は、まずは安定性優先で bbox の 4隅を採用する。
        （輪郭近似で4点が取れた場合はそれを利用）

    元実装: APA/test_capture_formA.py
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = image_bgr.shape[:2]

    # corner regions (15%)
    corner_margin_x = int(w * 0.15)
    corner_margin_y = int(h * 0.15)
    corners = {
        "top_left": (0, 0, corner_margin_x, corner_margin_y),
        "top_right": (w - corner_margin_x, 0, w, corner_margin_y),
        "bottom_left": (0, h - corner_margin_y, corner_margin_x, h),
        "bottom_right": (w - corner_margin_x, h - corner_margin_y, w, h),
    }

    # size range
    min_size = min(w, h) * 0.005
    max_size = min(w, h) * 0.08
    min_area = min_size**2
    max_area = max_size**2

    # binarization trials
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
            if fill_ratio <= 0.4:
                continue

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_val = float(cv2.mean(gray, mask=mask)[0])
            if mean_val >= 180:
                continue

            eps = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, eps, True)

            # score
            aspect_score = 1.0 - abs(ar - 1.0) * 0.5
            intensity_score = (180.0 - mean_val) / 180.0
            score = aspect_score * 0.25 + fill_ratio * 0.35 + intensity_score * 0.4

            # corners
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype(np.float32)
                pts = order_quad_tl_tr_br_bl(pts)
            else:
                pts = np.array(
                    [[x, y], [x + ww - 1, y], [x + ww - 1, y + hh - 1], [x, y + hh - 1]],
                    dtype=np.float32,
                )

            info = {
                "corner": corner_name,
                "bbox": [int(x), int(y), int(ww), int(hh)],
                "points": pts.tolist(),
                "score": float(score),
                "method": method,
            }

            if corner_name not in found or score > float(found[corner_name]["score"]):
                found[corner_name] = info

    # keep TL/TR/BL
    return [found[k] for k in ("top_left", "top_right", "bottom_left") if k in found]


# ------------------------------------------------------------
# XFeat matching / homography
# ------------------------------------------------------------


@dataclass
class XFeatHomographyResult:
    ok: bool
    ref_kpts: int
    tgt_kpts: int
    matches: int
    inliers: int
    inlier_ratio: float
    reproj_rms: Optional[float]
    H_ref_to_tgt: Optional[list[list[float]]]


def resize_keep_aspect(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Resize so that max(H,W) == max_side (if larger). Returns (resized, scale)."""

    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0
    s = float(max_side) / float(m)
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, s


def scale_matrix(s: float) -> np.ndarray:
    return np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)


def compute_reproj_rms(H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    """Compute RMS reprojection error on matched points."""

    src = src_pts.reshape(-1, 1, 2).astype(np.float32)
    dst = dst_pts.reshape(-1, 1, 2).astype(np.float32)
    proj = cv2.perspectiveTransform(src, H)
    err = np.linalg.norm(proj - dst, axis=2).reshape(-1)
    return float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")


def refine_homography_least_squares(
    H_init: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inlier_mask: np.ndarray,
) -> tuple[np.ndarray, Optional[float]]:
    """Refine homography using inliers only.

    Why:
      cv2.findHomography(..., USAC_MAGSAC, ...) is robust, but the returned H can
      be slightly suboptimal for warp quality. A common approach is:
        1) robust estimation -> inlier mask
        2) least-squares re-fit on inliers

    Returns:
      (H_refined, reproj_rms_on_inliers)
    """

    H0 = np.asarray(H_init, dtype=np.float64)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.size != len(mkpts0) or mask.size != len(mkpts1):
        return H0, None

    if int(mask.sum()) < 4:
        return H0, None

    p0 = np.asarray(mkpts0, dtype=np.float32)[mask]
    p1 = np.asarray(mkpts1, dtype=np.float32)[mask]

    # method=0 => a simple (non-robust) least squares fit.
    H_ls, _ = cv2.findHomography(p0, p1, 0)
    if H_ls is None:
        return H0, None

    try:
        rms = compute_reproj_rms(np.asarray(H_ls, dtype=np.float64), p0, p1)
    except Exception:
        rms = None

    return np.asarray(H_ls, dtype=np.float64), rms


class XFeatMatcher:
    def __init__(self, top_k: int = 4096, device: str = "cpu", match_max_side: int = 1200):
        ensure_portable_git_on_path()
        self.device = device
        self.top_k = top_k
        self.match_max_side = match_max_side
        self.xfeat = torch.hub.load(
            "verlab/accelerated_features",
            "XFeat",
            pretrained=True,
            top_k=top_k,
        ).to(device)
        self.xfeat.eval()

    def match_and_estimate_h(self, ref_bgr: np.ndarray, tgt_bgr: np.ndarray) -> tuple[XFeatHomographyResult, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Return result + (H, mkpts0, mkpts1)"""

        t0 = time.time()

        # Resize for matching stability/speed (then scale H back)
        ref_small, s_ref = resize_keep_aspect(ref_bgr, self.match_max_side)
        tgt_small, s_tgt = resize_keep_aspect(tgt_bgr, self.match_max_side)

        out0 = self.xfeat.detectAndCompute(ref_small, top_k=self.top_k)[0]
        out1 = self.xfeat.detectAndCompute(tgt_small, top_k=self.top_k)[0]
        # (Upstream requires image_size for some matchers; keep it)
        out0.update({"image_size": (ref_small.shape[1], ref_small.shape[0])})
        out1.update({"image_size": (tgt_small.shape[1], tgt_small.shape[0])})

        matches = self.xfeat.match_lighterglue(out0, out1)
        if isinstance(matches, (list, tuple)) and len(matches) >= 2:
            mkpts0, mkpts1 = matches[0], matches[1]
        elif isinstance(matches, dict) and "mkpts0" in matches and "mkpts1" in matches:
            mkpts0, mkpts1 = matches["mkpts0"], matches["mkpts1"]
        else:
            return (
                XFeatHomographyResult(False, 0, 0, 0, 0, 0.0, None, None),
                None,
                None,
                None,
            )

        mkpts0 = np.asarray(mkpts0, dtype=np.float32)
        mkpts1 = np.asarray(mkpts1, dtype=np.float32)

        # These points are in "small" coordinate system.
        ref_kpts = int(len(out0.get("keypoints", [])) or 0)
        tgt_kpts = int(len(out1.get("keypoints", [])) or 0)

        if len(mkpts0) < 4:
            return (
                XFeatHomographyResult(False, ref_kpts, tgt_kpts, int(len(mkpts0)), 0, 0.0, None, None),
                None,
                mkpts0,
                mkpts1,
            )

        H, mask = cv2.findHomography(
            mkpts0,
            mkpts1,
            cv2.USAC_MAGSAC,
            3.5,
            maxIters=1_000,
            confidence=0.999,
        )

        if H is None or mask is None:
            return (
                XFeatHomographyResult(False, ref_kpts, tgt_kpts, int(len(mkpts0)), 0, 0.0, None, None),
                None,
                mkpts0,
                mkpts1,
            )

        mask = mask.reshape(-1).astype(bool)
        inliers = int(mask.sum())
        matches_n = int(len(mask))
        inlier_ratio = float(inliers) / float(matches_n) if matches_n else 0.0

        # Refine H by least-squares on inliers for better warp quality.
        reproj = None
        if inliers >= 4:
            try:
                H_refined, rms = refine_homography_least_squares(H, mkpts0, mkpts1, mask)
                if H_refined is not None:
                    H = H_refined
                reproj = rms if rms is not None else compute_reproj_rms(H, mkpts0[mask], mkpts1[mask])
            except Exception:
                reproj = compute_reproj_rms(H, mkpts0[mask], mkpts1[mask])

        _ = time.time() - t0

        # scale H back to full resolution: H_full = inv(S_tgt) * H_small * S_ref
        S_ref = scale_matrix(s_ref)
        S_tgt = scale_matrix(s_tgt)
        H_full = np.linalg.inv(S_tgt) @ H @ S_ref

        return (
            XFeatHomographyResult(
                ok=True,
                ref_kpts=ref_kpts,
                tgt_kpts=tgt_kpts,
                matches=matches_n,
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                reproj_rms=reproj,
                H_ref_to_tgt=H_full.astype(float).tolist(),
            ),
            H_full,
            mkpts0,
            mkpts1,
        )


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------


def draw_inlier_matches(
    ref_bgr: np.ndarray,
    tgt_bgr: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    match_max_side: int,
) -> np.ndarray:
    """Draw projected ref corners on tgt + inlier matches."""

    # NOTE:
    # mkpts0/mkpts1 are in the "matching" coordinate system (resized images).
    # For visualization to look correct, we must draw on the SAME resized images.
    # This prevents the left/right size mismatch and “not matching” appearance.

    ref_vis, _ = resize_keep_aspect(ref_bgr, match_max_side)
    tgt_vis, _ = resize_keep_aspect(tgt_bgr, match_max_side)

    Hm, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 3.5)
    if Hm is None or mask is None:
        return tgt_vis
    mask = mask.reshape(-1).astype(bool)

    h, w = ref_vis.shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, Hm)

    tgt2 = tgt_vis.copy()
    for i in range(len(warped)):
        p1 = tuple(warped[i - 1][0].astype(int))
        p2 = tuple(warped[i][0].astype(int))
        cv2.line(tgt2, p1, p2, (0, 255, 0), 4)

    k0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in mkpts0]
    k1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in mkpts1]
    matches = [cv2.DMatch(i, i, 0) for i, m in enumerate(mask) if m]
    canvas = cv2.drawMatches(ref_vis, k0, tgt2, k1, matches, None, matchColor=(0, 255, 0), flags=2)
    return canvas


# ------------------------------------------------------------
# Image mode
# ------------------------------------------------------------


def load_templates(form: str) -> list[Path]:
    base = Path(__file__).resolve().parent / "image" / form
    paths = []
    for i in range(1, 7):
        p = base / f"{i}.jpg"
        if p.exists():
            paths.append(p)
    return paths


def run_images_mode(args: argparse.Namespace) -> int:
    out_root = mkdir(Path(args.out) / f"run_{now_run_id()}_{args.form}_images")
    debug_dir = mkdir(out_root / "debug")
    warped_dir = mkdir(out_root / "warped")
    degraded_dir = mkdir(out_root / "degraded")
    steps_root = mkdir(out_root / "degrade_steps") if args.save_degrade_steps else None

    # reproducibility
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    templates = load_templates(args.form)
    if not templates:
        print(f"[ERROR] templates not found: APA/image/{args.form}/*.jpg")
        return 1

    if args.only_template is not None:
        templates = [p for p in templates if p.stem == str(args.only_template)]
        if not templates:
            print(f"[ERROR] only_template={args.only_template} not found.")
            return 1

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")
    matcher = XFeatMatcher(top_k=args.top_k, device=device, match_max_side=args.match_max_side)

    summary: list[dict[str, Any]] = []

    for tp in templates:
        template_bgr = cv2.imread(str(tp))
        if template_bgr is None:
            print(f"[WARN] failed to read: {tp}")
            continue

        template_features: dict[str, Any] = {}
        if args.detect_features:
            if args.form == "A":
                template_features["markers"] = detect_formA_marker_boxes(template_bgr)
            if args.form == "B":
                template_features["qrs"] = detect_qr_codes_multiscale(template_bgr)

        # generate N degraded variants
        for k in range(args.degrade_n):
            name = f"{tp.stem}_deg{k:02d}"
            step_dir = (steps_root / name) if steps_root is not None else None
            degraded_bgr, H_gt, degrade_meta = warp_template_to_random_view(
                template_bgr,
                out_size=(args.degrade_w, args.degrade_h),
                rng=rng,
                max_rotation_deg=args.max_rot,
                min_abs_rotation_deg=args.min_abs_rot,
                rotation_mode=args.rotation_mode,
                snap_step_deg=args.snap_step_deg,
                perspective_jitter=args.perspective,
                min_visible_area_ratio=args.min_visible_area_ratio,
                max_attempts=args.max_attempts,
                save_steps_dir=step_dir,
                step_prefix="",
            )
            cv2.imwrite(str(degraded_dir / f"{name}.jpg"), degraded_bgr)

            # Safety check: ensure the paper is actually visible in the degraded image.
            # (This should already be satisfied by the generator constraints, but keep
            # a cheap post-check so that we can catch unexpected failures early.)
            # Use edge density as a lightweight proxy.
            try:
                gray_chk = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2GRAY)
                edges_chk = cv2.Canny(gray_chk, 50, 150)
                edge_ratio = float((edges_chk > 0).mean())
                if edge_ratio < 0.001:
                    print(f"[WARN] degraded may not contain enough paper pixels: {name} edge_ratio={edge_ratio:.6f}")
            except Exception:
                pass

            res, H, mk0, mk1 = matcher.match_and_estimate_h(template_bgr, degraded_bgr)
            item: dict[str, Any] = {
                "template": str(tp),
                "case": name,
                "ok": res.ok,
                "degrade": degrade_meta,
                "H_gt_ref_to_tgt": H_gt.astype(float).tolist(),
                **asdict(res),
            }

            if args.detect_features:
                item["template_features"] = template_features
                if args.form == "A":
                    item["degraded_features"] = {"markers": detect_formA_marker_boxes(degraded_bgr)}
                if args.form == "B":
                    item["degraded_features"] = {"qrs": detect_qr_codes_multiscale(degraded_bgr)}
            summary.append(item)

            if not res.ok or H is None or mk0 is None or mk1 is None:
                continue

            # warp degraded back to template plane (inverse H)
            H_inv = np.linalg.inv(H)
            warped = cv2.warpPerspective(degraded_bgr, H_inv, (template_bgr.shape[1], template_bgr.shape[0]))
            cv2.imwrite(str(warped_dir / f"{name}_warped.jpg"), warped)

            # debug matches image
            try:
                dbg = draw_inlier_matches(
                    template_bgr,
                    degraded_bgr,
                    mk0,
                    mk1,
                    args.match_max_side,
                )
                cv2.imwrite(str(debug_dir / f"{name}_matches.jpg"), dbg)
            except Exception:
                pass

        print(f"[OK] processed template: {tp.name}")

    # save summary
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # simple csv
    csv_path = out_root / "summary.csv"
    header = [
        "template",
        "case",
        "ok",
        "ref_kpts",
        "tgt_kpts",
        "matches",
        "inliers",
        "inlier_ratio",
        "reproj_rms",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for it in summary:
            row = [
                str(it.get("template", "")),
                str(it.get("case", "")),
                str(it.get("ok", "")),
                str(it.get("ref_kpts", "")),
                str(it.get("tgt_kpts", "")),
                str(it.get("matches", "")),
                str(it.get("inliers", "")),
                str(it.get("inlier_ratio", "")),
                str(it.get("reproj_rms", "")),
            ]
            f.write(",".join(row) + "\n")

    print(f"\n[DONE] outputs: {out_root}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["images"], required=True)
    p.add_argument("--form", choices=["A", "B"], required=True)
    p.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "output_recovery"))

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--top-k", type=int, default=2048)
    p.add_argument(
        "--match-max-side",
        type=int,
        default=1200,
        help="XFeat matching 用の最大辺（大きいほど精度↑/速度↓）",
    )
    p.add_argument(
        "--detect-features",
        action="store_true",
        help="フォームA(マーカー)/B(QR)の検出も行い、summary.jsonに入れる（Homographyには未使用）",
    )

    # images mode options
    p.add_argument("--degrade-n", type=int, default=5)
    # 改悪画像が小さすぎると現実の撮影条件と乖離するため、デフォルトを少し大きめにする
    p.add_argument("--degrade-w", type=int, default=2400)
    p.add_argument("--degrade-h", type=int, default=1800)
    # Make default degradation milder (user feedback)
    p.add_argument("--max-rot", type=float, default=12.0)
    p.add_argument(
        "--min-abs-rot",
        type=float,
        default=0.0,
        help=(
            "フル回転モード（--max-rot>=180）時に、0度に近い回転を避けたい場合の下限（度）。"
            " 例: --max-rot 180 --min-abs-rot 120 とすると、ほぼ上下逆/横向きが必ず混ざる。"
        ),
    )
    p.add_argument("--perspective", type=float, default=0.08)

    p.add_argument(
        "--rotation-mode",
        choices=["uniform", "snap"],
        default="uniform",
        help=(
            "回転角の生成方法。uniform=連続一様乱数、snap=角度を一定刻みにスナップ（上下逆/横向きを確実に作る）。"
        ),
    )
    p.add_argument(
        "--snap-step-deg",
        type=float,
        default=90.0,
        help="rotation-mode=snap のときの刻み角度（度）。例: 90 -> 0/90/180/270",
    )
    p.add_argument(
        "--min-visible-area-ratio",
        type=float,
        default=0.25,
        help="改悪画像内で紙（テンプレ）が占める最小面積比（0〜1）。紙が写っていないケースを防ぐ。",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=50,
        help="紙が十分写る改悪を生成するための最大リトライ回数。",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--save-degrade-steps",
        action="store_true",
        help="改悪生成の途中経過（背景/ワープ/マスク/合成/最終）を out_root/degrade_steps/<case>/ に保存する",
    )

    p.add_argument(
        "--only-template",
        type=int,
        default=None,
        help="デバッグ用：テンプレ番号（1〜6）を指定した場合、その1枚だけ処理する",
    )

    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    print("=" * 70)
    print("test_recovery_paper")
    print("=" * 70)
    print(f"OpenCV: {cv2.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"mode: {args.mode} / form: {args.form}")

    if args.mode == "images":
        return run_images_mode(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
