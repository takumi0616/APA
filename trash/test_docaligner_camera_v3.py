#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DocAligner Real-time Document Detection v3
==========================================

v2 からの改善点（カメラ周りの汎用化）
------------------------------------
python test_recovery_paper.py --mode images --form A --device cpu --only-template 1 --degrade-n 1 --degrade-w 1200 --degrade-h 900 --match-max-side 900

C:\Users\takumi\develop\miniconda3\python.exe C:\Users\takumi\develop\APA\test_recovery_paper.py --mode images --form A --device cpu

`test_camera.py` と同様に、カメラが複数ある環境（例: Mac内蔵 + iPhone連係カメラ + USBカメラ / Windows + 複数Webカメラ）でも
「どのカメラを使うか」を起動時に明示できるようにしました。

追加した CLI オプション:
  - --index N     : カメラ index を指定して起動（指定した場合、他 index の探索をしない）
  - --list        : 利用可能なカメラ index を列挙して終了
  - --identify    : 各 index を順番にプレビュー表示して “どれがどれか” を目視で確認
  - --max-index N : list/identify/自動検出の探索上限

Controls (runtime):
- 'q' or ESC: 終了
- 's': 画像保存
- 'p': 透視変換プレビュー
- 'm': モデル切り替え（heatmap/point）
- '1': lcnet050 (Point, 最軽量)
- '2': lcnet100 (Heatmap, バランス)
- '3': fastvit_t8 (Heatmap, 軽量)
- '4': fastvit_sa24 (Heatmap, 最高精度)
- 't': 平滑化 ON/OFF
- '+'/'-': ボックスマージン調整

Created: January 6, 2026
Updated: January 8, 2026 (v3)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import deque
from datetime import datetime
from typing import Optional

import cv2
import numpy as np


# Fix encoding for Windows (best-effort)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Suppress warnings
warnings.filterwarnings("ignore")


def patch_capybara_exports() -> None:
    """Patch Docsaid's `capybara` namespace package to expose expected symbols.

    `capybara-docsaid` installs a *namespace package* named `capybara`.
    In this environment, `capybara` has no `__init__.py`, so attributes like
    `capybara.get_curdir` are not exported at the top-level.

    However, `docaligner-docsaid` expects `import capybara as cb` and then uses
    top-level attributes (cb.get_curdir, cb.pad, cb.ONNXEngine, ...).

    This function imports the real implementations from submodules and attaches
    them to the already-imported `capybara` module object *before* importing
    `docaligner`.
    """

    import capybara as cb  # namespace package

    # path/utils
    from capybara.utils.custom_path import Path, get_curdir
    from capybara.utils.utils import download_from_google

    # mixins/enums
    from capybara.mixins import EnumCheckMixin
    from capybara.onnxengine.enum import Backend

    # ONNX engine
    from capybara.onnxengine.engine import ONNXEngine

    # vision helpers
    from capybara.vision.functionals import centercrop, imbinarize, pad
    from capybara.vision.geometric import imresize
    from capybara.vision.improc import is_numpy_img

    # structures
    from capybara.structures.polygons import Polygons

    # Attach to module object (only if missing)
    for name, obj in {
        "Path": Path,
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


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DocAligner Real-time Document Detection (v3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--index", type=int, default=None, help="Force camera index to use")
    p.add_argument(
        "--list",
        action="store_true",
        help="List available camera indices and exit (no DocAligner load)",
    )
    p.add_argument(
        "--identify",
        action="store_true",
        help=(
            "Interactively identify which physical camera corresponds to each index. "
            "Press keys: 'n' next, 'q' or ESC quit."
        ),
    )
    p.add_argument(
        "--max-index",
        type=int,
        default=4,
        help="Max camera index to probe when auto-detecting/listing/identifying",
    )
    return p.parse_args(argv)


def check_camera_availability(max_index: int = 4) -> list[int]:
    available: list[int] = []
    print("=" * 60)
    print("Camera Detection Test")
    print("=" * 60)
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        try:
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"  [OK] Camera {i}: Detected ({width}x{height}, {fps:.1f}fps)")
                    available.append(i)
        finally:
            cap.release()
    if not available:
        print("  [NG] No available cameras found")
    print()
    return available


def identify_cameras(max_index: int = 4) -> int:
    print("=" * 60)
    print("Camera Identify Mode")
    print("=" * 60)
    print("Keys: 'n' next, 'q' or ESC quit")
    print()

    window_name = "DocAligner v3 - Camera Identify"
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        print(f"[INFO] Showing camera index={idx}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"[WARN] Failed to read frame from index={idx}")
                    break
                cv2.putText(
                    frame,
                    f"index={idx}  (press n:next, q:quit)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    break
                if key == ord("q") or key == 27:
                    cv2.destroyAllWindows()
                    return 0
        finally:
            cap.release()

    cv2.destroyAllWindows()
    return 0


class PolygonSmoother:
    """時間的平滑化クラス（v2 の実装を踏襲）"""

    def __init__(self, buffer_size=3, outlier_threshold=100):
        self.buffer = deque(maxlen=buffer_size)
        self.outlier_threshold = outlier_threshold
        self.last_valid_polygon = None
        self.no_detect_count = 0
        self.max_no_detect = 10

    def update(self, polygon, use_filter=True):
        if polygon is None or len(polygon) < 4:
            self.no_detect_count += 1
            if self.no_detect_count > self.max_no_detect:
                self.last_valid_polygon = None
                self.buffer.clear()
            elif self.last_valid_polygon is not None:
                return self.last_valid_polygon
            return None

        self.no_detect_count = 0

        if not use_filter:
            self.last_valid_polygon = polygon
            return polygon

        if self.last_valid_polygon is not None:
            diff = np.abs(polygon - self.last_valid_polygon).max()
            if diff > self.outlier_threshold:
                self.buffer.clear()

        self.buffer.append(polygon.copy())
        self.last_valid_polygon = polygon

        if len(self.buffer) < 2:
            return polygon

        stacked = np.stack(list(self.buffer))
        smoothed = np.median(stacked, axis=0)
        return smoothed

    def reset(self):
        self.buffer.clear()
        self.last_valid_polygon = None
        self.no_detect_count = 0


def set_camera_resolution(cap, width, height, fps):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # NOTE: OpenCV property name is CAP_PROP_FPS (not CAP_PROP_FRAME_FPS)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    return actual_width, actual_height, actual_fps


def save_frame(frame, save_dir="docaligner_captures_v3"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, save_dir)
    os.makedirs(full_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"doc_{timestamp}.jpg"
    filepath = os.path.join(full_path, filename)

    cv2.imwrite(filepath, frame)
    return filepath


def expand_polygon(polygon, margin=20):
    if polygon is None or len(polygon) < 4:
        return polygon

    center = polygon.mean(axis=0)
    expanded = []
    for pt in polygon:
        direction = pt - center
        length = np.linalg.norm(direction)
        if length > 0:
            unit_direction = direction / length
            new_pt = pt + unit_direction * margin
        else:
            new_pt = pt
        expanded.append(new_pt)
    return np.array(expanded)


def draw_polygon(frame, polygon, color=(0, 255, 0), thickness=3, expand_margin=0):
    if polygon is None or len(polygon) < 4:
        return frame

    if expand_margin > 0:
        polygon = expand_polygon(polygon, expand_margin)

    result = frame.copy()
    pts = polygon.astype(np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.2, result, 0.8, 0, result)

    cv2.polylines(result, [pts], True, color, thickness)

    for i, pt in enumerate(pts):
        cv2.circle(result, tuple(pt), 8, (0, 0, 255), -1)
        cv2.circle(result, tuple(pt), 10, (255, 255, 255), 2)
        labels = ["TL", "TR", "BR", "BL"]
        cv2.putText(
            result,
            labels[i],
            (pt[0] + 15, pt[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return result


def draw_info_panel(frame, fps, doc_detected, model_name, smoothing_enabled, camera_index: int):
    result = frame.copy()

    cv2.rectangle(result, (5, 5), (420, 150), (0, 0, 0), -1)
    cv2.rectangle(result, (5, 5), (420, 150), (0, 255, 0), 1)

    cv2.putText(
        result,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
    )

    status_color = (0, 255, 0) if doc_detected else (0, 0, 255)
    status_text = "Document: DETECTED" if doc_detected else "Document: NOT FOUND"
    cv2.putText(
        result,
        status_text,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        1,
    )

    cv2.putText(
        result,
        f"Camera index: {camera_index}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    cv2.putText(
        result,
        f"Model: {model_name}",
        (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    smooth_text = "Smoothing: ON" if smoothing_enabled else "Smoothing: OFF"
    smooth_color = (0, 255, 0) if smoothing_enabled else (100, 100, 100)
    cv2.putText(
        result,
        smooth_text,
        (10, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        smooth_color,
        1,
    )

    cv2.putText(
        result,
        "Keys: q=Exit s=Save m=Model 1-4=Select",
        (10, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (150, 150, 150),
        1,
    )

    return result


def get_perspective_transform(frame, polygon, output_size=(800, 600)):
    if polygon is None or len(polygon) < 4:
        return None

    dst = np.array(
        [
            [0, 0],
            [output_size[0] - 1, 0],
            [output_size[0] - 1, output_size[1] - 1],
            [0, output_size[1] - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(polygon.astype(np.float32), dst)
    warped = cv2.warpPerspective(frame, M, output_size)
    return warped


def build_model_configs(ModelType):
    return {
        "1": {"type": ModelType.point, "cfg": "lcnet050", "name": "lcnet050 (Point, 最軽量)"},
        "2": {"type": ModelType.heatmap, "cfg": "lcnet100", "name": "lcnet100 (Heatmap, バランス)"},
        "3": {"type": ModelType.heatmap, "cfg": "fastvit_t8", "name": "fastvit_t8 (Heatmap, 軽量)"},
        "4": {"type": ModelType.heatmap, "cfg": "fastvit_sa24", "name": "fastvit_sa24 (Heatmap, 最高精度)"},
    }


def load_model(model_key: str, DocAligner, model_configs):
    config = model_configs.get(model_key, model_configs["4"])
    print(f"Loading model: {config['name']}...")
    model = DocAligner(model_type=config["type"], model_cfg=config["cfg"])
    print(f"[OK] Model loaded: {config['name']}")
    return model, config["name"]


def main(argv=None) -> int:
    args = parse_args(argv)

    # camera-only utilities (no heavy deps)
    if args.list:
        cams = check_camera_availability(max_index=args.max_index)
        print("Available camera indices:", cams)
        print("If you don't know which one is built-in, run: --identify")
        return 0 if cams else 1
    if args.identify:
        return identify_cameras(max_index=args.max_index)

    print()
    print("=" * 60)
    print("DocAligner Real-time Document Detection v3")
    print("Camera selection: --index / --list / --identify")
    print("=" * 60)
    print()
    print(f"OpenCV Version: {cv2.__version__}")
    print()

    # Decide camera index
    if args.index is not None:
        camera_index = args.index
    else:
        cams = check_camera_availability(max_index=args.max_index)
        if not cams:
            print("Error: No available cameras.")
            return 1
        camera_index = cams[0]

    # Open camera (IMPORTANT: do NOT probe other indices here)
    print(f"Opening camera index={camera_index} ...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera index={camera_index}")
        print("Hint: run with --list or --identify to find a working index")
        return 1

    width, height, fps = set_camera_resolution(cap, 1280, 720, 30)
    print(f"[OK] Camera opened: {width}x{height} @ {fps:.1f}fps")
    print()

    # Heavy deps are imported only after we know we really need to run DocAligner.
    # Also patch capybara exports before importing docaligner.
    try:
        patch_capybara_exports()
        import capybara as cb
        from docaligner import DocAligner, ModelType
    except Exception as e:
        cap.release()
        print("[ERROR] Failed to import DocAligner dependencies.")
        print("        Please ensure `docaligner` and `capybara` are installed.")
        print("        Error:", repr(e))
        return 1

    model_configs = build_model_configs(ModelType)

    print("利用可能なモデル:")
    for key, config in model_configs.items():
        print(f"  [{key}] {config['name']}")
    print()

    current_model_key = "4"
    model, model_name = load_model(current_model_key, DocAligner, model_configs)
    print()

    print("=" * 60)
    print("操作方法:")
    print("  'q' or ESC: 終了")
    print("  's': 画像保存")
    print("  'p': 透視変換プレビュー")
    print("  't': 平滑化のON/OFF切り替え")
    print("  '1'-'4': モデル切り替え")
    print("  '+'/'-': ボックスマージン調整")
    print("=" * 60)
    print()

    show_perspective = False
    smoothing_enabled = True
    smoother = PolygonSmoother(buffer_size=5, outlier_threshold=50)
    box_margin = 30

    prev_time = cv2.getTickCount()
    fps_display = 0.0
    frame_count = 0

    main_window = f"DocAligner v3 - Document Detection (cam {camera_index})"
    perspective_window = "Perspective Correction"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to get frame")
                break

            current_time = cv2.getTickCount()
            time_diff = (current_time - prev_time) / cv2.getTickFrequency()
            if time_diff >= 1.0:
                fps_display = frame_count / time_diff
                frame_count = 0
                prev_time = current_time
            frame_count += 1

            padded_frame = cb.pad(frame, 100)
            polygon = model(img=padded_frame, do_center_crop=False)
            if polygon is not None:
                polygon = polygon - 100

            if smoothing_enabled:
                polygon = smoother.update(polygon)

            doc_detected = polygon is not None and len(polygon) >= 4

            result = frame.copy()
            if doc_detected:
                result = draw_polygon(result, polygon, expand_margin=box_margin)

            result = draw_info_panel(
                result,
                fps_display,
                doc_detected,
                model_name,
                smoothing_enabled,
                camera_index=camera_index,
            )

            cv2.putText(
                result,
                f"Margin: {box_margin}px (+/-)",
                (width - 220, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow(main_window, result)

            if show_perspective and doc_detected:
                warped = get_perspective_transform(frame, polygon)
                if warped is not None:
                    cv2.imshow(perspective_window, warped)
            else:
                try:
                    cv2.destroyWindow(perspective_window)
                except Exception:
                    pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n終了します...")
                break
            elif key == ord("s"):
                filepath = save_frame(result)
                print(f"保存しました: {filepath}")
                if doc_detected:
                    warped = get_perspective_transform(frame, polygon)
                    if warped is not None:
                        warped_path = filepath.replace(".jpg", "_corrected.jpg")
                        cv2.imwrite(warped_path, warped)
                        print(f"補正画像を保存: {warped_path}")
            elif key == ord("p"):
                show_perspective = not show_perspective
                print(f"透視変換プレビュー: {'ON' if show_perspective else 'OFF'}")
            elif key == ord("t"):
                smoothing_enabled = not smoothing_enabled
                if not smoothing_enabled:
                    smoother.reset()
                print(f"平滑化: {'ON' if smoothing_enabled else 'OFF'}")
            elif key == ord("+") or key == ord("="):
                box_margin = min(100, box_margin + 10)
                print(f"マージン: {box_margin}px")
            elif key == ord("-") or key == ord("_"):
                box_margin = max(0, box_margin - 10)
                print(f"マージン: {box_margin}px")
            else:
                ch = chr(key) if 0 <= key < 256 else ""
                if ch in model_configs:
                    new_key = ch
                    if new_key != current_model_key:
                        current_model_key = new_key
                        model, model_name = load_model(
                            current_model_key,
                            DocAligner,
                            model_configs,
                        )
                        smoother.reset()

    except KeyboardInterrupt:
        print("\n\nCtrl+Cで終了...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("DocAligner Document Detection v3 Complete")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
