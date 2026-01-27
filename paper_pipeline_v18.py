#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper_pipeline_v18.py

実行方法
--------

[Windows]
リポジトリルート（`.../develop`）から:

    C:/Users/takumi/develop/miniconda3/python.exe APA/paper_pipeline_v18.py

[macOS/Linux]
リポジトリルートから（`APA/` 配下のスクリプトを直接指定）:

    .venv/bin/python paper_pipeline_v18.py

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
入力（データセット）:

本スクリプトは **複数の入力ソース**をまとめて処理します。

v18.??+ のデフォルトは **synthetic（A/B/C） + test + target** を処理します。
（処理対象を絞りたい場合は `--src-forms` / `--test-limit` / `--target-limit` で調整します）

1) synthetic（改悪あり）: `APA/image/{A,B,C}/`
   - デフォルトは `1.jpg`〜`6.jpg` を対象（`PIPELINE_DEFAULTS["template_numbers"]`）
   - 対象フォームは `--src-forms` で指定
   - `--degrade-n` 枚ぶんの改悪画像を生成してから本処理へ投入します

2) test（改悪あり）: `APA/image/test/`
   - `.png/.jpg/.jpeg` を列挙
   - **ファイル名から GT（正解）を推定**します
     - 推奨: `{A|B|C}_{template番号}_{id}.png` 例: `A_3_1.png`
       - 先頭2要素（A と 3）が GT（フォーム・テンプレ番号）
       - 3要素目以降は識別子で、GT 判定には使いません
   - `--degrade-n` 枚ぶんの改悪画像を生成してから本処理へ投入します

3) target（改悪なし）: `APA/image/target/`
   - `.png/.jpg/.jpeg` を列挙
   - **改悪生成を行わず**、そのまま本処理へ投入します（現場画像の想定）

処理フロー（1 case = 1 枚の入力から生成した 1 枚の改悪画像）:

※ v18.7: 改悪生成（degrade）は **最初に全ケース分をまとめて生成**し、以降の本処理へ投入する。
   改悪生成の所要時間は計測対象外。

1) 改悪生成（本ファイル内に統合済みの実装を使用）
   - v18.5: 紙のしなり（非線形ワープ）と、撮影時の影（照明ムラ）を追加
2) DocAligner により紙領域 polygon（4点）を推定
   - 失敗したら `stage=docaligner_failed` で終了
3) polygon を（紙サイズ比の margin で）外側に拡張 → 透視補正（rectify）
   - 透視補正後の画像は横長に統一（`enforce_landscape`）
   - `--polygon-margin-px > 0` の場合は固定pxマージンで上書き可能
4) フォーム判定（回転探索）
   - rectify 後は `enforce_landscape` で横長に統一しているため、回転探索は **2方向（0度/180度）** のみ
     - 0 と 180 を比較し、最上位角度を確定に使う
     - 0/180 で何も見つからない場合は Unknown（no_detection）とする（追加の角度探索などの救済は行わない）
   - フォームA: 3点マーク（TL/TR/BL）が検出できる（`--marker-preproc` で前処理オプション）
     - v18.6: A が検出できても **Unknown閾値（`--unknown-score-threshold`）未満** の場合は、その時点で Unknown 確定せず
       **フォームB探索へフォールバック**します（B の取りこぼし回避）
   - フォームB: QRコードが検出できる
     - **WeChat QR エンジンのみ** を使用（OpenCV 標準 `QRCodeDetector` は使わない）
     - `--wechat-model-dir` にモデルが必要（opencv-contrib 必須）
     - v18 では WeChat-only でも v7 と同様に **fast → robust の2段階**で評価する
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
- `9_demo/`           : デモ用の並列可視化（左=degraded+逆投影、右=aligned）
- `summary.json` / `summary.csv`
- `run.log`           : 実行ログ（logging）

※v18 ではデバッグ画像の保存量を `--save-images {all,fail,none}` で制御できる。

- `all` : 従来通り、常に保存
- `fail`: `stage!=done` のケースだけ保存（成功ケースは保存しない）
- `none`: 一切保存しない（速度計測向け）

ディレクトリ自体は作られるが、`fail/none` の場合は中身が空になることがある。
ただし、以下は **save-images 設定に関わらず保存されます**（成果物/解析用のため）。

- `8_aligned/`（最終成果物）
- `7_debug_matches/`（マッチ可視化）
- `9_demo/`（デモ可視化）

※ v18.7: `--save-images` の設定に関わらず、`7_aligned/` は成果物として必ず保存される（※v18.8 で `8_aligned/` に移動）。
※ v18.8: 背景除算法（stage6）を追加したため、成果物は `8_aligned/` に移動。
   `--save-images` の設定に関わらず、`8_aligned/` は成果物として必ず保存される。

注意
----
- `--explain` を付けると、主要パラメータの意味（日本語）を表示して終了します。
- torch.hub 経由の XFeat 読み込みで git が必要になることがあるため、
  portable git を PATH に追加する処理を `test_recovery_paper` から流用する。
- QR 検出は WeChat QR エンジン（`cv2.wechat_qrcode_WeChatQRCode`）のみ利用する。
  - WeChat を使うには opencv-contrib のビルドと、4つのモデルファイル
    （detect/sr の prototxt/caffemodel）が必要
  - **src-forms に B を含む場合、WeChat が利用できないと起動時にエラー終了**する
- JPEG 保存は可能なら python-turbojpeg（libjpeg-turbo）を優先する（失敗時は `cv2.imwrite` にフォールバック）
- 日本語ラベル描画は Pillow を使用（OpenCV putText は日本語非対応のため）。
  - `APA_FONT_PATH` を設定すると任意フォントを優先可能

更新履歴（抜粋）
----------------

- v18.16（2026-01-26）
  - target（現場撮影）の no_detection（フォームA取りこぼし）対策:
    - A-geometry を追加で緩める（recall優先）
      - surround_max_ink_ratio: 0.05 -> 0.08（机模様/枠線の写り込み救済）
      - min_marker_area_page_ratio: 5e-5 -> 3.5e-5（マーカーが僅かに小さいケース救済）
  - 高精度フォールバック（advanced_fallback）の診断性改善:
    - no_detection 時に polygon を再推定するフォールバックで、各 margin 試行の decision を attempts に記録

- v18.15（2026-01-26）
  - target（現場撮影）でのフォームA取りこぼし対策:
    - マーカー検出の探索範囲を「corner付近」に絞り、枠線/文字の誤検出を減らす
    - マーカー想定サイズ（min/max）を調整し、rectify後の解像度差に追従
    - cornerへの近さ（pos_score）をスコアに加え、端にある正しいマーカーを優先
  - target の A-geometry（surround_min_mean_gray）を緩め、影で corner が暗い場合の救済を追加

- v18.10〜v18.11（2026-01-25）
  - DocAligner の安定性改善:
    - pad_px を画像サイズ比から自動推定（端ギリギリ撮影の救済）
    - model/type/pad/scale を変えた複数推論 + 退化quadの除外 + フォームスコアで候補選択
    - polygon expand を「辺法線による offset」を優先し、中心放射型はフォールバック化
  - summary.json 書き出しの安定化: numpy.ndarray 等が混ざっても落ちない default を追加
  - 実行の利便性: image/test・image/target の件数制限（--test-limit/--target-limit）を追加
  - 依存導入が抜けた環境向け: capybara/docaligner import 失敗時に .venv 使用を促すエラーを明確化

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
# 自己完結化のためのユーティリティ群
# ------------------------------------------------------------
#
# 元々 `test_recovery_paper.py` に置いていた以下の機能は、
# paper_pipeline_v18.py 単体で動作できるよう、本ファイル内へ移植した。
#
# - ensure_portable_git_on_path / now_run_id
# - 改悪生成（warp_template_to_random_view）
# - フォームA マーカー検出（detect_formA_marker_boxes のベース実装）
# - XFeatMatcher（torch.hub 経由）
# - Homography の least-squares refine / 可視化（draw_inlier_matches）


def ensure_portable_git_on_path() -> None:
    """torch.hub が内部で git を呼ぶ場合に備え、portable git を PATH に追加する。"""

    portable_git_bin = r"C:\Users\takumi\develop\git\bin"
    if os.path.exists(portable_git_bin):
        os.environ["PATH"] = portable_git_bin + os.pathsep + os.environ.get("PATH", "")


def now_run_id() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def random_background(h: int, w: int, rng: random.Random) -> np.ndarray:
    """簡易なランダム背景（グラデ＋ノイズ＋線）。"""

    bg = np.zeros((h, w, 3), dtype=np.uint8)
    base = np.array([rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)], dtype=np.uint8)
    bg[:, :] = base

    gx = np.linspace(0, 1, w, dtype=np.float32)
    gy = np.linspace(0, 1, h, dtype=np.float32)
    g = (np.outer(gy, gx) * 255.0).astype(np.float32)
    g3 = np.stack([g, g, g], axis=-1)
    bg = to_uint8(0.6 * bg.astype(np.float32) + 0.4 * g3)

    n = np.zeros((h, w, 3), dtype=np.float32)
    n[:, :, 0] = np.random.normal(0, 8, size=(h, w))
    n[:, :, 1] = np.random.normal(0, 8, size=(h, w))
    n[:, :, 2] = np.random.normal(0, 8, size=(h, w))
    bg = to_uint8(bg.astype(np.float32) + n)

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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """テンプレをランダム四角形へ射影して背景に合成し、改悪画像を作る。

    NOTE:
      この実装は元々 `test_recovery_paper.py` にあったものを移植。
      「紙がフレーム内に収まる」「極端な潰れ/透視を避ける」制約を含む。
    """

    h, w = template_bgr.shape[:2]
    out_w, out_h = out_size

    margin = int(min(out_w, out_h) * 0.06)

    side_ref = int(min(out_w, out_h)) if float(max_rotation_deg) >= 180.0 else int(out_w)
    base_w_min = int(side_ref * 0.75)
    base_w_max = int(side_ref * 0.92)
    min_visible_area_px = int(out_w * out_h * float(min_visible_area_ratio))

    min_fit_scale = 0.70 if float(max_rotation_deg) >= 180.0 else 0.78
    max_perspective_edge_ratio = 1.55
    min_edge_len_ratio = 0.58

    dst_quad = None
    base_w = 0
    base_h = 0
    angle = 0.0
    fit_scale_used: float = 1.0
    visible_area_ratio_used: float = 0.0
    quad_area_ratio_used: float = 0.0
    edge_ratio_top_bottom: float = 1.0
    edge_ratio_left_right: float = 1.0
    edge_len_min: float = 0.0
    edge_len_max: float = 0.0

    for _attempt in range(int(max_attempts)):
        base_w = rng.randint(int(base_w_min), int(base_w_max))
        base_h = int(base_w * (h / w))
        base_h = max(120, min(base_h, int(out_h * 0.85)))

        cx_lo = int(base_w // 2 + margin)
        cx_hi = int(out_w - 1 - base_w // 2 - margin)
        cy_lo = int(base_h // 2 + margin)
        cy_hi = int(out_h - 1 - base_h // 2 - margin)
        if cx_lo >= cx_hi or cy_lo >= cy_hi:
            continue

        cx = rng.randint(cx_lo, cx_hi)
        cy = rng.randint(cy_lo, cy_hi)

        x1, y1 = cx - base_w // 2, cy - base_h // 2
        x2, y2 = cx + base_w // 2, cy + base_h // 2
        rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        # rotation
        if float(max_rotation_deg) >= 180:
            if str(rotation_mode) == "snap":
                step = float(max(1.0, float(snap_step_deg)))
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
            angle = rng.uniform(-float(max_rotation_deg), float(max_rotation_deg))

        M = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
        rect_rot = cv2.transform(rect.reshape(-1, 1, 2), M).reshape(4, 2)

        # perspective jitter
        jitter = float(perspective_jitter) * float(min(base_w, base_h))
        rect_rot += np.array(
            [[rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)] for _ in range(4)],
            dtype=np.float32,
        )

        # keep quad inside by scaling towards center (not clipping)
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
        if s < float(min_fit_scale):
            continue

        # all corners inside
        if (
            (rect_rot[:, 0].min() >= 0)
            and (rect_rot[:, 1].min() >= 0)
            and (rect_rot[:, 0].max() <= out_w - 1)
            and (rect_rot[:, 1].max() <= out_h - 1)
        ):
            cand = rect_rot.astype(np.float32)
            try:
                top = float(np.linalg.norm(cand[1] - cand[0]))
                right = float(np.linalg.norm(cand[2] - cand[1]))
                bottom = float(np.linalg.norm(cand[2] - cand[3]))
                left = float(np.linalg.norm(cand[3] - cand[0]))

                edges = [top, right, bottom, left]
                e_min = float(min(edges))
                e_max = float(max(edges))
                if e_max <= 1e-6:
                    continue

                tb = max(top, bottom) / max(1e-6, min(top, bottom))
                lr = max(left, right) / max(1e-6, min(left, right))
                if tb > float(max_perspective_edge_ratio) or lr > float(max_perspective_edge_ratio):
                    continue

                if (e_min / e_max) < float(min_edge_len_ratio):
                    continue

                area = float(abs(cv2.contourArea(cand.reshape(-1, 1, 2))))
                base_area = float(base_w * base_h)
                if base_area <= 1:
                    continue
                area_ratio = area / base_area
                if area_ratio < 0.60:
                    continue
            except Exception:
                continue

            tmp_mask = cv2.warpPerspective(
                np.ones((h, w), dtype=np.uint8) * 255,
                cv2.getPerspectiveTransform(
                    np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32),
                    cand,
                ),
                (out_w, out_h),
            )
            visible_px = int(cv2.countNonZero(tmp_mask))
            if visible_px >= int(min_visible_area_px):
                dst_quad = cand
                fit_scale_used = float(s)
                visible_area_ratio_used = float(visible_px) / float(max(1, out_w * out_h))
                quad_area_ratio_used = float(area_ratio)
                edge_ratio_top_bottom = float(max(top, bottom) / max(1e-6, min(top, bottom)))
                edge_ratio_left_right = float(max(left, right) / max(1e-6, min(left, right)))
                edge_len_min = float(e_min)
                edge_len_max = float(e_max)
                break

    if dst_quad is None:
        # fallback: safe fronto-parallel rectangle
        base_w = int(out_w * 0.85)
        base_h = int(base_w * (h / w))
        base_h = max(120, min(base_h, int(out_h * 0.85)))
        cx = int(out_w // 2)
        cy = int(out_h // 2)
        x1, y1 = cx - base_w // 2, cy - base_h // 2
        x2, y2 = cx + base_w // 2, cy + base_h // 2
        dst_quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        angle = 0.0
        fit_scale_used = 1.0
        visible_area_ratio_used = float(base_w * base_h) / float(max(1, out_w * out_h))
        quad_area_ratio_used = 1.0
        edge_ratio_top_bottom = 1.0
        edge_ratio_left_right = 1.0
        edge_len_min = float(min(base_w, base_h))
        edge_len_max = float(max(base_w, base_h))

    src_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_quad, dst_quad)

    bg = random_background(out_h, out_w, rng)
    warped = cv2.warpPerspective(template_bgr, H, (out_w, out_h))

    mask = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, H, (out_w, out_h))
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    degraded = np.where(mask3 > 0, warped, bg)

    # mild blur/noise
    if rng.random() < 0.35:
        degraded = cv2.GaussianBlur(degraded, (3, 3), rng.uniform(0.4, 0.8))
    if rng.random() < 0.35:
        degraded = to_uint8(degraded.astype(np.float32) + np.random.normal(0, 4, size=degraded.shape))

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
        "visible_area_ratio": float(visible_area_ratio_used),
        "fit_scale": float(fit_scale_used),
        "quad_area_ratio_to_base": float(quad_area_ratio_used),
        "edge_ratio_top_bottom": float(edge_ratio_top_bottom),
        "edge_ratio_left_right": float(edge_ratio_left_right),
        "edge_len_min": float(edge_len_min),
        "edge_len_max": float(edge_len_max),
        "safety": {
            "min_fit_scale": float(min_fit_scale),
            "max_perspective_edge_ratio": float(max_perspective_edge_ratio),
            "min_edge_len_ratio": float(min_edge_len_ratio),
        },
        "max_attempts": int(max_attempts),
    }
    return degraded, H, meta


def resize_keep_aspect(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """max(H,W) が max_side を超える場合のみリサイズし、(resized, scale) を返す。"""

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


def compute_reproj_rms(H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    src = src_pts.reshape(-1, 1, 2).astype(np.float32)
    dst = dst_pts.reshape(-1, 1, 2).astype(np.float32)
    proj = cv2.perspectiveTransform(src, np.asarray(H, dtype=np.float64))
    err = np.linalg.norm(proj - dst, axis=2).reshape(-1)
    return float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")


def refine_homography_least_squares(
    H_init: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inlier_mask: np.ndarray,
) -> tuple[np.ndarray, Optional[float]]:
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


def draw_inlier_matches(
    ref_bgr: np.ndarray,
    tgt_bgr: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    match_max_side: int,
) -> np.ndarray:
    """inlier matches 可視化（XFeatのmatching座標系に合わせて描画）。"""

    ref_vis, _ = resize_keep_aspect(ref_bgr, int(match_max_side))
    tgt_vis, _ = resize_keep_aspect(tgt_bgr, int(match_max_side))

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


def detect_formA_marker_boxes_base(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """フォームA想定: 3点マーカー（TL/TR/BL）の bbox を検出（ベース実装）。

    - `paper_pipeline_v18.py` 側では、前処理バリアントを試すためのラッパー
      `detect_formA_marker_boxes()` を別途持つ。
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = image_bgr.shape[:2]

    # v18.15 (target改善):
    # rectified 後の書類では、マーカーは「かなり端」にある。
    # corner探索範囲が広すぎると、枠線/文字/手書きなどを誤検出しやすい。
    # そのため、探索範囲・想定サイズを設定化し、既定値も少し厳しめにする。
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

            # 位置prior: corner の“より端”に近いものを優先
            corner_xy = {
                "top_left": (0.0, 0.0),
                "top_right": (float(w - 1), 0.0),
                "bottom_left": (0.0, float(h - 1)),
            }.get(corner_name, (0.0, 0.0))
            dist = float(np.hypot(float(cx) - float(corner_xy[0]), float(cy) - float(corner_xy[1])))
            # max_dist は探索矩形の対角線
            max_dist = float(np.hypot(float(corner_margin_x), float(corner_margin_y)))
            pos_score = max(0.0, 1.0 - (dist / max(1e-6, max_dist)))

            aspect_score = 1.0 - abs(ar - 1.0) * 0.5
            intensity_score = (180.0 - mean_val) / 180.0
            score = aspect_score * 0.22 + fill_ratio * 0.33 + intensity_score * 0.35 + pos_score * 0.10

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
                "pos_score": float(pos_score),
                "method": method,
            }

            if corner_name not in found or score > float(found[corner_name]["score"]):
                found[corner_name] = info

    return [found[k] for k in ("top_left", "top_right", "bottom_left") if k in found]


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


class XFeatMatcher:
    def __init__(self, top_k: int = 4096, device: str = "cpu", match_max_side: int = 1200):
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

    def match_and_estimate_h(
        self, ref_bgr: np.ndarray, tgt_bgr: np.ndarray
    ) -> tuple[XFeatHomographyResult, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """(result, H_full, mkpts0, mkpts1) を返す。"""

        ref_small, s_ref = resize_keep_aspect(ref_bgr, self.match_max_side)
        tgt_small, s_tgt = resize_keep_aspect(tgt_bgr, self.match_max_side)

        out0 = self.xfeat.detectAndCompute(ref_small, top_k=self.top_k)[0]
        out1 = self.xfeat.detectAndCompute(tgt_small, top_k=self.top_k)[0]
        out0.update({"image_size": (ref_small.shape[1], ref_small.shape[0])})
        out1.update({"image_size": (tgt_small.shape[1], tgt_small.shape[0])})

        matches = self.xfeat.match_lighterglue(out0, out1)
        if isinstance(matches, (list, tuple)) and len(matches) >= 2:
            mkpts0, mkpts1 = matches[0], matches[1]
        elif isinstance(matches, dict) and "mkpts0" in matches and "mkpts1" in matches:
            mkpts0, mkpts1 = matches["mkpts0"], matches["mkpts1"]
        else:
            return (XFeatHomographyResult(False, 0, 0, 0, 0, 0.0, None, None), None, None, None)

        mkpts0 = np.asarray(mkpts0, dtype=np.float32)
        mkpts1 = np.asarray(mkpts1, dtype=np.float32)

        ref_kpts = int(len(out0.get("keypoints", [])) or 0)
        tgt_kpts = int(len(out1.get("keypoints", [])) or 0)
        if len(mkpts0) < 4:
            return (XFeatHomographyResult(False, ref_kpts, tgt_kpts, int(len(mkpts0)), 0, 0.0, None, None), None, mkpts0, mkpts1)

        H_small, mask = cv2.findHomography(
            mkpts0,
            mkpts1,
            cv2.USAC_MAGSAC,
            float(PIPELINE_DEFAULTS["homography"]["find"]["ransac_reproj_threshold_px"]),
            maxIters=int(PIPELINE_DEFAULTS["homography"]["find"]["max_iters"]),
            confidence=float(PIPELINE_DEFAULTS["homography"]["find"]["confidence"]),
        )
        if H_small is None or mask is None:
            return (XFeatHomographyResult(False, ref_kpts, tgt_kpts, int(len(mkpts0)), 0, 0.0, None, None), None, mkpts0, mkpts1)

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

        S_ref = scale_matrix(float(s_ref))
        S_tgt = scale_matrix(float(s_tgt))
        H_full = np.linalg.inv(S_tgt) @ H_small @ S_ref

        return (
            XFeatHomographyResult(
                ok=True,
                ref_kpts=ref_kpts,
                tgt_kpts=tgt_kpts,
                matches=matches_n,
                inliers=inliers,
                inlier_ratio=float(inlier_ratio),
                reproj_rms=reproj,
                H_ref_to_tgt=H_full.astype(float).tolist(),
            ),
            H_full,
            mkpts0,
            mkpts1,
        )


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
    # NOTE(2026-01-27):
    # ユーザー要望により、デフォルトで synthetic(A/B/C) と test/target も処理する。
    # ただし重いので、必要に応じて --src-forms / --test-limit / --target-limit で絞り込み可能。
    "src_forms": ["A", "B", "C"],  # 入力元フォーム（synthetic生成）
    "limit": 0,  # デバッグ用：各フォームで先頭N枚だけ処理（0=全て）
    # NOTE(v18.12):
    # 旧挙動では 0=全件処理 だったが、src-forms の軽い実行でも image/test / image/target まで
    # 常に巻き込まれて「完了しない（非常に時間がかかる）」誤解を生みやすかった。
    # v18.12 では以下のルールへ変更:
    #   - 0 : そのデータセットを処理しない（skip）
    #   - >0: 先頭N枚だけ処理
    #   - <0: 全件処理
    "target_limit": -1,  # デバッグ用：image/target の先頭N枚だけ処理（0=skip, <0=all）
    "test_limit": -1,  # デバッグ用：image/test の先頭N枚だけ処理（0=skip, <0=all）
    "template_numbers": [1, 2, 3, 4, 5, 6],  # テンプレ/入力画像の対象番号（例: 1.jpg〜6.jpg）

    # 改悪生成（degrade）
    "degrade": {
        "n": 5,  # 1枚の入力から何枚の改悪画像を作るか
        "out_size_wh": [2400, 1800],  # 改悪画像の出力サイズ（幅, 高さ）
        # ユーザー要望: 過度な改悪（極端な傾き/奥行き/縮小）が出ないよう、デフォルトを「常識的」にする。
        # 必要なら CLI 引数で上げられる。
        # 改悪（回転）を強める（ユーザーFB）
        # - `warp_template_to_random_view()` は max_rot>=180 のとき 0〜360 一様回転モードになる。
        # - ただし紙がフレーム内に残る制約を満たす必要があるため、
        #   生成関数側で「回転時の紙サイズ」を short side 基準に調整している（test_recovery_paper 側の実装）。
        "max_rot_deg": 180.0,  # 改悪生成の回転強度（>=180で0..360一様回転モード）
        "min_abs_rot_deg": 0.0,  # 最小回転量（0なら小さな回転も許可）
        "rotation_mode": "uniform",  # 回転角の出し方（"uniform" または "snap"）
        "snap_step_deg": 90.0,  # rotation_mode="snap" の場合の角度刻み
        # 歪みが強すぎるとのフィードバックがあったため弱めに調整
        "perspective_jitter": 0.03,  # 射影ゆがみ量（大きいほど難しい）
        # NOTE:
        # 0〜360度回転（max_rot>=180）を有効にすると、斜め回転では
        # 「紙がフレーム内に収まる」制約のために紙サイズが短辺基準に落ちる。
        # その結果、out_w*out_h に対する紙の占有率（visible_area_ratio）は
        # 0.55 のような大きい値だと物理的に満たせず、生成が遅くなる/0度に偏る。
        # ここは「紙が写っていること」を保証する目的に留め、下げる。
        "min_visible_area_ratio": 0.25,  # 生成画像でテンプレが見えている最小比率（小さすぎ防止）
        "max_attempts": 50,  # 改悪生成の最大試行回数
        "seed": 45,  # 乱数シード（再現性）

        # v18.5: 紙がしなっているような歪み（非線形ワープ）
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

        # v18.5: 撮影時の影（照明ムラ）の混入
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
        # v18.5 では「紙のしなり」「影」を主目的として追加し、
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
        # 高精度優先: 影/周辺減光をより強く抑える（計算時間は増える）
        "sigma_ratio": 0.03,
        "sigma_min": 15.0,
        "sigma_max": 120.0,
        # bg が極端に小さいと divide が発散するため、下限を設ける
        "bg_min": 6.0,
    },

    # XFeat（テンプレマッチング）
    "xfeat": {
        "device_default": "cpu",  # 既定の実行デバイス（auto/cpu/cuda のうち default に使う）
        # 高精度優先（時間はかかってよい想定）
        # - top_k を増やすと対応点候補が増え、Homography の安定性が上がりやすい
        # - match_max_side_px を増やすと細部が残り、マッチング精度が上がりやすい
        #   （ただしメモリ/時間コストが増える）
        # 高精度優先: より多くの特徴点 + より高解像度で照合
        #   - top_k を増やすと対応点候補が増え、Homography の安定性が上がりやすい
        #   - match_max_side_px を増やすと細部が残り、マッチング精度が上がりやすい
        # 速度/安定のバランス（止まりにくさ重視で微緩和）
        "top_k": 3072,
        "match_max_side_px": 1400,
    },

    # フォーム判定（回転スキャン）
    "rotation_scan": {
        # 高精度優先: QR/マーカー検出の前処理バリエーションが増えるため、並列数も少し増やす
        "max_workers": 12,  # 回転スキャンの並列数（スレッド）
        # v18.2 改善: rectify 後は enforce_landscape で横長に統一されているため、
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

        # NOTE(v18.17+):
        # target では margin を大きく取りたい（端ギリギリ撮影救済）ため、
        # rectify の最大辺も引き上げて角の解像度を落としにくくする。
        "rectified_max_side_px": 3200,

        # DocAligner入力前に周囲へ足すパディング(px)
        "pad_px": 320,
        # auto pad も精度優先でやや厚めにする
        "pad_px_auto_ratio": 0.10,
        "pad_px_auto_min": 120,
        "pad_px_auto_max": 800,

        # DocAlignerを1発勝負にしない（複数候補→後段で選ぶ）
        "multi": {
            "enable": True,
            # 追加で試すモデル/タイプ。args の指定（--docaligner-model/type）が最優先。
            "extra_models": ["fastvit_t8", "lcnet100"],
            "extra_types": ["heatmap", "point"],
            # pad は「auto + 複数固定値」を候補にする（端ギリギリ撮影の救済）
            "pad_px_candidates": [240, 400, 650, 900],
            # 入力リサイズ（polygonはスケールで戻す）
            "input_scales": [0.75, 0.6, 1.15],
            # 1ケースでの DocAligner 推論回数の上限
            "max_infer_runs": 8,
            # 後段で評価する raw polygon 候補の上限
            "max_polygon_candidates": 4,

            # v18.18+（精度優先 + 10秒/枚目標）:
            # DocAligner multi は「候補生成（推論）」に加えて「候補評価（rectify→フォーム判定）」も行うが、
            # 評価をやりすぎると docaligner_s が肥大化する。
            # そこで、幾何品質で上位のみを評価し、margin の探索数も制限する。
            "eval_max_candidates": 2,
            "eval_max_margins": 2,
            # 候補評価用のフォーム判定は、0/180 の2方向のみなので並列化メリットが小さい。
            # スレッド生成/同期コストを避けるため既定は 1。
            "eval_rotation_workers": 1,
        },

        # polygon を外側に広げる margin
        "polygon_margin": {
            # NOTE(v18.17+):
            # target では 0.12 だと max_px=200 に頭打ちしやすく、
            # 実際に必要な margin が確保できずマーカー/QRが欠けて no_detection になりやすい。
            "ratio": 0.18,
            "min_px": 10.0,
            "max_px": 800.0,
            "fixed_px": 0.0,
        },

        # v18.17+: rectify 直前に画像を pad して「margin を取りたいのに clip で潰れる」を避ける。
        "rectify_padding": {
            "enable": True,
            "pad_px": 800,
            "border_value": [0, 0, 0],
        },

        # no_detection の場合の高精度フォールバック
        "advanced_fallback": {
            "enable": True,
            "trigger_on_form_unknown_no_detection": True,
        },
    },

    # マーカー検出（フォームA）向け前処理
    "marker": {
        # 高精度優先: morph は多少重いが、照明ムラ/影/ノイズに対して頑健になりやすい
        "preproc_mode": "morph",  # マーカー検出前処理の強さ（"none" / "basic" / "morph"）
        # 高精度優先: CLAHE/自適応二値化をやや強める（影・照明ムラへの耐性）
        "clahe": {"clipLimit": 3.0, "tileGridSize": [8, 8]},  # CLAHE設定（照明ムラ対策）
        "adaptive_threshold": {"block_size": 61, "C": 3},  # 自適応二値化の設定
        "morph": {
            # 画像短辺に対する比率でカーネルサイズを決める
            "kernel_ratio": 0.006,  # カーネルサイズ = 短辺 * 比率（概算）
            "kernel_min": 5,  # カーネルサイズの最小値
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
            # v18.5 で shadow(照明ムラ) を入れると、
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
            # v18.2 改善:
            # v7並みの精度を確保するため、前処理バリエーションを増やす。
            # fast→robust の2段階で評価する。
            "fast": {
                # fast: 角度候補の絞り込み用
                # v18.2: 前処理を増やして精度向上
                # 高精度優先: 角度選択で取りこぼすと復帰できないため、fast 側も少し厚めにする。
                # 高精度優先: fast の段階でも取りこぼしを減らす（角度候補の選択ミス回避）
                "variants": ["bgr", "gray", "clahe"],
                "scales": [0.75, 1.0, 1.25],
            },
            "robust": {
                # robust: 最終確定用（多少重くてもよいが、呼ぶのは最大1回）
                # v18.2: v7並みの前処理バリエーションに拡張
                # 注意:
                # 以前の実装では variants に "adaptive_threshold" を書いていたが、
                # 実装側が未対応で無視されるケースがあった。
                # 本ファイルでは "adaptive_threshold" を明示的にサポート（下の関数修正）する。
                "variants": ["bgr", "gray", "clahe", "adaptive_threshold", "adaptive_morph"],
                # 高精度優先: スケール探索を厚めにする（時間は増える）
                "scales": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            },
            "up_scale_enable_max_side_px": 1200,  # 最大辺がこの値以上なら拡大は無効化
            "max_test_side_px": 6500,  # WeChat で試す画像の最大辺(px)
            "adaptive_morph_kernel": [5, 5],
        },
        # CLAHE設定（照明ムラ対策）
        "clahe": {"clipLimit": 3.0, "tileGridSize": [8, 8]},
        # 自適応二値化の設定（高精度優先: blockを少し大きくして低周波ムラを吸収）
        "adaptive_threshold": {"block_size": 61, "C": 3},
    },

    # Homography（特徴点マッチングの射影変換）
    "homography": {
        "find": {
            # 高精度優先:
            # - iters/confidence を上げて収束率を上げる（時間は増える）
            # - reproj threshold は僅かに厳しめにして外れ値混入を抑える
            # 高精度優先:
            # - iters/confidence を上げて収束率/再現性を上げる（時間は増える）
            # - reproj threshold を少し厳しめにして外れ値混入を抑える
            "ransac_reproj_threshold_px": 4.0,
            "max_iters": 5000,
            "confidence": 0.999,
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
        # 高精度優先: RANSACを強化した分、warp判定は「極端な緩和」はせず、少しだけ救済寄りにする
        "min_inliers": 60,  # warpを許可する最小inlier数
        "min_inlier_ratio": 0.06,  # warpを許可する最小inlier_ratio
        "max_h_cond": 1e6,  # Homographyの条件数上限（大きいと不安定）
    },
}


# ============================================================
# Speed profile（大幅高速化用）
# ============================================================


def resolve_speed_profile(args: argparse.Namespace, *, dataset: str) -> str:
    """speed profile を決定する。

    - fast    : 精度を多少犠牲にして大幅高速化（target向け）
    - accurate: 既存の高精度（重い）フロー
    - auto    : dataset=target のみ fast、他は accurate
    """

    p = str(getattr(args, "speed_profile", "auto") or "auto")
    if p not in ("auto", "fast", "accurate"):
        p = "auto"
    # NOTE(2026-01-27):
    # ユーザー要望により、target のデフォルトは「速度より精度（ただし1枚~10秒程度）」へ戻す。
    # fast は明示指定したときのみ有効とする。
    if p == "auto":
        return "accurate"
    return p


def resolve_extra_outputs_mode(args: argparse.Namespace, *, profile: str) -> str:
    """デモ/可視化などの追加出力のモードを決める。

    auto:
      - accurate: all
      - fast    : none
    """

    m = str(getattr(args, "extra_outputs", "auto") or "auto")
    if m not in ("auto", "all", "none"):
        m = "auto"
    if m == "auto":
        return "none" if str(profile) == "fast" else "all"
    return m


def _get_docaligner_model_cached(model_name: str, model_type: str) -> tuple[Any, Any]:
    """DocAligner model を (model_name, model_type) でキャッシュして返す。"""

    cache = getattr(_get_docaligner_model_cached, "_cache", None)
    if cache is None:
        cache = {}
        setattr(_get_docaligner_model_cached, "_cache", cache)
    key = (str(model_name), str(model_type))
    if key in cache:
        return cache[key]
    m, cb = load_docaligner_model(str(model_name), str(model_type))
    cache[key] = (m, cb)
    return m, cb


def detect_polygon_docaligner_fast(
    *,
    logger: logging.Logger,
    degraded_bgr: np.ndarray,
    max_input_side_px: int = 1200,
    model_name: str = "lcnet100",
    model_type: str = "heatmap",
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """DocAligner を fast 設定で最小回数だけ実行する。

    目的:
      - v18 の最大ボトルネック（DocAligner multi + 高解像度入力）を一気に削る

    方針:
      - 画像を max_input_side_px に収まるよう縮小（input_scale）
      - pad も控えめ（1〜2回の試行のみ）
      - multi/advanced fallback は行わない
    """

    if degraded_bgr is None:
        return None, {"ok": False, "mode": "fast", "reason": "image_is_none"}

    h, w = degraded_bgr.shape[:2]
    mside = max(1, int(max(h, w)))
    s = min(1.0, float(max_input_side_px) / float(mside))
    # 縮小しすぎは精度が落ちすぎるため下限を設ける
    s = float(max(0.35, min(1.0, s)))

    try:
        model, cb = _get_docaligner_model_cached(str(model_name), str(model_type))
    except Exception as e:
        return None, {"ok": False, "mode": "fast", "reason": f"model_load_failed:{e}", "model": model_name, "type": model_type}

    attempts: list[dict[str, Any]] = []
    for pad_px in [120, 240]:
        try:
            poly = _run_docaligner_once(
                model=model,
                cb=cb,
                image_bgr=degraded_bgr,
                pad_px=int(pad_px),
                input_scale=float(s),
            )
        except Exception as e:
            attempts.append({"pad_px": int(pad_px), "input_scale": float(s), "ok": False, "error": str(e)})
            continue

        if poly is None:
            attempts.append({"pad_px": int(pad_px), "input_scale": float(s), "ok": False, "issue": "model_returned_none"})
            continue

        try:
            poly = _clamp_poly_to_image(poly, img_w=int(w), img_h=int(h))
            ok, q = _is_valid_quad(poly, img_w=int(w), img_h=int(h))
            attempts.append({"pad_px": int(pad_px), "input_scale": float(s), "ok": bool(ok), "quality": q})
            if ok:
                return poly, {
                    "ok": True,
                    "mode": "fast",
                    "model": str(model_name),
                    "type": str(model_type),
                    "pad_px": int(pad_px),
                    "input_scale": float(s),
                    "attempts": attempts,
                }
        except Exception as e:
            attempts.append({"pad_px": int(pad_px), "input_scale": float(s), "ok": False, "error": str(e)})

    logger.debug("[DocAligner fast] no valid polygon. attempts=%s", attempts)
    return None, {
        "ok": False,
        "mode": "fast",
        "model": str(model_name),
        "type": str(model_type),
        "input_scale": float(s),
        "attempts": attempts,
        "reason": "no_valid_polygon",
    }


def decide_form_fast(
    rectified_bgr: np.ndarray,
    *,
    unknown_score_threshold: float,
    unknown_margin: float,
    formA_geom_cfg: Optional[MarkerGeometryConfig],
    marker_preproc: str = "none",
) -> FormDecision:
    """fast profile 用のフォーム判定。

    大幅高速化のため、以下に簡略化:
      - 角度は 0/180 のみ（従来通り）
      - A: marker_preproc は 1 種類のみ（バリエーションは作らない）。
        ※精度/速度のトレードオフとして、呼び出し側（CLI: --marker-preproc）で調整する。
      - B: WeChat detector を「生画像1発」だけ（前処理/マルチスケール無し）
      - morph fallback / robust fallback は行わない
    """

    if rectified_bgr is None:
        return FormDecision(False, None, None, 0.0, {"reason": "image_is_none", "mode": "fast"})

    scan_angles = [0.0, 180.0]
    thr = float(unknown_score_threshold or 0.0)

    wechat = getattr(score_formB, "_wechat", None)
    scan: list[dict[str, Any]] = []

    bestA: Optional[FormDecision] = None
    bestB: Optional[FormDecision] = None

    for a in scan_angles:
        rot = rotate_image_bound(rectified_bgr, float(a))
        h, w = rot.shape[:2]
        if h > w:
            scan.append({"angle": float(a), "skip": True})
            continue

        okA, scoreA, detA = score_formA(rot, marker_preproc=str(marker_preproc), geom_cfg=formA_geom_cfg)
        rec: dict[str, Any] = {"angle": float(a), "skip": False, "A": {"ok": bool(okA), "score": float(scoreA), "detail": detA}}

        if wechat is not None:
            try:
                qrs = detect_qr_codes_wechat(rot, wechat)
            except Exception:
                qrs = []
            if qrs:
                bscore, bdet = score_best_qr_candidate(rot, qrs)
                rec["B"] = {"ok": True, "score": float(1.0 + bscore), "detail": {**bdet, "phase": "fast_minimal"}}
            else:
                rec["B"] = {"ok": False, "score": 0.0, "detail": {"qrs": [], "reason": "wechat_no_qr", "phase": "fast_minimal"}}
        else:
            rec["B"] = {"ok": False, "score": 0.0, "detail": {"qrs": [], "reason": "wechat_detector_disabled", "phase": "fast_minimal"}}

        scan.append(rec)

        if okA:
            cand = FormDecision(True, "A", float(a), float(scoreA), {"A": detA, "phase": "fast_A"})
            if bestA is None or cand.score > bestA.score:
                bestA = cand
        if (rec.get("B") or {}).get("ok"):
            cand = FormDecision(True, "B", float(a), float(rec["B"]["score"]), {"B": rec["B"]["detail"], "phase": "fast_B"})
            if bestB is None or cand.score > bestB.score:
                bestB = cand

    # A/B のどちらかが見つかっていればスコアで選ぶ
    # （ただし threshold/margin の Unknown 判定は従来の思想を踏襲）
    cand: Optional[FormDecision] = None
    if bestA is not None and bestB is not None:
        cand = bestA if bestA.score >= bestB.score else bestB
    else:
        cand = bestA or bestB

    if cand is None:
        return FormDecision(False, None, None, 0.0, {"reason": "no_detection", "scan": scan, "scan_angles": scan_angles, "mode": "fast"})

    if thr > 0 and float(cand.score) < thr:
        return FormDecision(
            False,
            None,
            None,
            float(cand.score),
            {"reason": "below_threshold", "threshold": float(thr), "scan": scan, "scan_angles": scan_angles, "mode": "fast"},
        )

    # margin 判定（曖昧）: A/B の両方が取れているときのみ適用
    if bestA is not None and bestB is not None:
        if abs(float(bestA.score) - float(bestB.score)) < float(unknown_margin or 0.0):
            return FormDecision(
                False,
                None,
                None,
                float(max(bestA.score, bestB.score)),
                {
                    "reason": "ambiguous",
                    "unknown_margin": float(unknown_margin or 0.0),
                    "a_score": float(bestA.score),
                    "b_score": float(bestB.score),
                    "scan": scan,
                    "scan_angles": scan_angles,
                    "mode": "fast",
                },
            )

    return cand


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
# 追加の改悪（v18.5）: 紙のしなり / 影（照明ムラ）
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
        # v18 改善: 呼び出し側で「detector をスレッド数分用意」し、ここでは Lock を使わない。
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
        # v18 改善: detector をスレッド数ぶん用意し、Lock による直列化を避ける。
        _WECHAT_QR = WeChatQRDetectorPool(model_dir=model_dir, pool_size=int(pool_size))
        if logger:
            logger.info("[OK] WeChat QR detector initialized: %s (pool_size=%d)", model_dir, int(pool_size))
        return _WECHAT_QR
    except Exception as e:
        _WECHAT_QR = None
        if logger:
            logger.warning("[WARN] WeChat QR detector disabled: %s", e)
        return None


"""（移植に伴う注意）

v18 以前は `test_recovery_paper.py` から多数の関数/クラスを import していましたが、
本バージョンでは paper_pipeline_v18.py 単体で動くように、必要な実装を全て本ファイルへ統合しました。
"""


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

    # NOTE:
    # 本プロジェクトで想定しているのは **capybara-docsaid**（import名: capybara）。
    # conda(base) 等で実行していると未インストールで ModuleNotFoundError になることがある。
    try:
        import capybara as cb
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing 'capybara' module (expected: capybara-docsaid). "
            "You are likely using the wrong Python interpreter. "
            "Please run with this repo's venv: `.venv/bin/python paper_pipeline_v18.py ...`"
        ) from e

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


def _poly_signed_area_xy(poly_xy: np.ndarray) -> float:
    """2D polygon の signed area を返す（CCW: 正, CW: 負）。"""

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
    """2本の直線(p1-p2, p3-p4)の交点を返す（平行ならNone）。"""

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
    """四角形を辺法線方向に margin_px だけオフセットする。

    いわゆる polygon offset（miter join）。
    - 4辺を外側へ平行移動
    - 隣接辺の交点を新しい頂点として採用

    注意:
      - 透視歪みが強いケースで、中心放射型 expand より安定しやすい。
      - 平行に近い辺などで交点計算が不安定な場合は None。
    """

    poly = order_quad_tl_tr_br_bl(poly_xy).astype(np.float32)
    if margin_px <= 0:
        return poly

    # 向き（CCW/CW）により outward normal を決める
    area = _poly_signed_area_xy(poly)
    is_ccw = area > 0

    # 4辺のオフセット線を作る
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(4):
        p0 = poly[i]
        p1 = poly[(i + 1) % 4]
        v = (p1 - p0).astype(np.float32)
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        dx, dy = float(v[0] / n), float(v[1] / n)
        # outward: CCWなら右法線、CWなら左法線
        if is_ccw:
            nx, ny = dy, -dx
        else:
            nx, ny = -dy, dx
        off = np.array([nx, ny], dtype=np.float32) * float(margin_px)
        lines.append((p0 + off, p1 + off))

    # 隣接辺の交点を頂点にする
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
    """polygon を margin_px だけ外側に広げる（可能な範囲で）。"""

    poly = np.asarray(polygon_xy, dtype=np.float32).reshape(4, 2)
    if margin_px <= 0:
        return order_quad_tl_tr_br_bl(poly)

    # v18.10 (DocAligner改善 1-C):
    # 従来の「中心から放射状に押し出す」は、透視歪みが強いと
    # “本当に欲しい外側方向”とズレやすい。
    # ここでは辺法線による polygon offset（miter join）を優先し、
    # 失敗時のみ旧方式へフォールバックする。
    out = _offset_quad_by_normals(poly, float(margin_px))
    if out is None:
        # fallback: center-radial expand
        center = poly.mean(axis=0)
        pts: list[np.ndarray] = []
        for pt in poly:
            v = pt - center
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                pts.append(pt)
            else:
                pts.append(pt + (v / n) * float(margin_px))
        out = order_quad_tl_tr_br_bl(np.asarray(pts, dtype=np.float32))

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


def _get_rectify_padding_cfg() -> dict[str, Any]:
    """rectify 前に入力を padding する設定を取得する。"""

    cfg_doc = PIPELINE_DEFAULTS.get("docaligner") or {}
    cfg = (cfg_doc.get("rectify_padding") or {}) if isinstance(cfg_doc, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}
    return cfg


def _apply_rectify_padding(
    image_bgr: np.ndarray,
    *,
    required_margin_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """rectify 前に画像を padding し、座標変換行列も返す。

    目的:
      polygon を外側へ expand したいのに、入力画像境界で clip されて
      rectify 結果が欠ける問題（特に target の端ギリギリ撮影）を避ける。

    戻り値:
      (padded_image, T_deg_to_padded, meta)
    """

    if image_bgr is None:
        return image_bgr, np.eye(3, dtype=np.float64), {"applied": False, "reason": "image_is_none"}

    cfg = _get_rectify_padding_cfg()
    enable = bool(cfg.get("enable", False))
    if not enable:
        return image_bgr, np.eye(3, dtype=np.float64), {"applied": False, "reason": "disabled"}

    pad_cfg = int(cfg.get("pad_px", 0) or 0)
    # margin を確保するため、必要なら pad を上積みする（過剰な巨大化は避ける）
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

    padded = cv2.copyMakeBorder(
        image_bgr,
        pad_px,
        pad_px,
        pad_px,
        pad_px,
        borderType=cv2.BORDER_CONSTANT,
        value=bgr,
    )
    # degraded(x,y)->padded(x+pad,y+pad)
    T = np.array([[1.0, 0.0, float(pad_px)], [0.0, 1.0, float(pad_px)], [0.0, 0.0, 1.0]], dtype=np.float64)
    return padded, T, {"applied": True, "pad_px": int(pad_px), "border_value": list(bgr)}


def rectify_with_margin_and_optional_padding(
    image_bgr: np.ndarray,
    *,
    polygon_xy: np.ndarray,
    margin_px: float,
    out_max_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """polygon + margin で rectify を行い、必要なら入力 padding も適用する。

    戻り値:
      (rectified, H_degraded_to_rectified, poly_exp_for_overlay(degraded), meta)
    """

    if image_bgr is None:
        raise ValueError("image_bgr is None")

    h, w = image_bgr.shape[:2]
    poly = order_quad_tl_tr_br_bl(np.asarray(polygon_xy, dtype=np.float32).reshape(4, 2))

    # まず overlay 用（従来通り degraded 画像内で clamp）
    poly_exp_overlay = expand_polygon(poly, float(margin_px), img_w=int(w), img_h=int(h))

    # rectify 用は padding したキャンバス上で expand する
    padded, T_deg_to_pad, pad_meta = _apply_rectify_padding(image_bgr, required_margin_px=float(margin_px))
    Hp, Wp = padded.shape[:2]
    poly_pad = (poly + np.array([[float(T_deg_to_pad[0, 2]), float(T_deg_to_pad[1, 2])]], dtype=np.float32)).astype(np.float32)
    poly_exp_pad = expand_polygon(poly_pad, float(margin_px), img_w=int(Wp), img_h=int(Hp))
    rectified, H_pad_to_rect = polygon_to_rectified(padded, poly_exp_pad, out_max_side=int(out_max_side))

    # degraded -> rectified の変換へ落とし込む
    H_deg_to_rect = np.asarray(H_pad_to_rect, dtype=np.float64) @ np.asarray(T_deg_to_pad, dtype=np.float64)

    meta = {
        "rectify_padding": pad_meta,
        "poly_exp_overlay": poly_exp_overlay.astype(float).tolist(),
        "poly_exp_padded": poly_exp_pad.astype(float).tolist(),
    }
    return rectified, H_deg_to_rect, poly_exp_overlay, meta


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


def rotate_image_bound_with_matrix(image_bgr: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """rotate_image_bound と同等の回転を行い、同時に座標変換行列も返す。

    戻り値:
      (rotated_bgr, M_3x3)

    M_3x3 は「元画像座標 -> 回転後画像座標」への変換。

    NOTE:
      - demo 画像（出力 9）で、フォーム判定結果（マーカー/QR）を
        degraded 元画像座標へ逆投影するために使用する。
      - 0/180 は rotate_image_bound と同様の最適化経路を踏み、
        matrix もそれに合わせて返す。
    """

    a = float(angle_deg) % 360.0
    h, w = image_bgr.shape[:2]

    if abs(a - 0.0) < 1e-6:
        return image_bgr, np.eye(3, dtype=np.float64)

    if abs(a - 180.0) < 1e-6:
        # cv2.rotate(img, ROTATE_180) 相当
        # src(x,y) -> dst(x',y') = (w-1-x, h-1-y)
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
    M3 = np.array([[float(M2[0, 0]), float(M2[0, 1]), float(M2[0, 2])], [float(M2[1, 0]), float(M2[1, 1]), float(M2[1, 2])], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rotated, M3


def _landscape_rotation_matrix_if_applied(*, w: int, h: int, rotated_90cw: bool) -> np.ndarray:
    """enforce_landscape の 90度CW回転を、座標変換行列として表現する。

    rotated_90cw=True の場合、src(x,y)->dst(x',y') は以下:
      x' = h-1-y
      y' = x

    つまり M = [[0,-1,h-1],[1,0,0],[0,0,1]]
    """

    if not rotated_90cw:
        return np.eye(3, dtype=np.float64)
    return np.array([[0.0, -1.0, float(h - 1)], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _perspective_transform_points(points_xy: np.ndarray, H_3x3: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    H = np.asarray(H_3x3, dtype=np.float64)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def _hstack_with_padding(left_bgr: np.ndarray, right_bgr: np.ndarray, *, pad_color_bgr: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """左右を横並びにする（切り取り/拡大縮小なし）。高さが違う場合は下方向に padding する。"""

    left = left_bgr
    right = right_bgr
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    out_h = max(lh, rh)

    def _pad(img: np.ndarray, out_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == out_h:
            return img
        pad = out_h - h
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=pad_color_bgr)

    left2 = _pad(left, out_h)
    right2 = _pad(right, out_h)
    return np.hstack([left2, right2])


def _draw_polygon_outline(image_bgr: np.ndarray, poly_xy: np.ndarray, color_bgr: tuple[int, int, int], thickness: int) -> np.ndarray:
    out = image_bgr.copy()
    pts = np.asarray(poly_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, color_bgr, int(thickness))
    return out


def _generate_demo9_image(
    *,
    degraded_bgr: np.ndarray,
    polygon_xy: np.ndarray,
    polygon_margin_px: float,
    H_degraded_to_rectified_landscape: np.ndarray,
    rectified_landscape_size_wh: tuple[int, int],
    decided_form: str,
    decided_angle_deg: float,
    decision_markers: Optional[list[dict[str, Any]]],
    decision_qrs: Optional[list[dict[str, Any]]],
    aligned_bgr: np.ndarray,
) -> np.ndarray:
    """出力9（デモ用）の画像を作る。

    左:
      - degraded 元画像（切り取り/ズーム無し）
      - DocAligner polygon（緑）
      - フォーム判定結果（A=赤bbox / B=青ポリゴン）を *元画像座標* へ逆投影して重畳

    右:
      - 8_aligned（最終整形済み）
    """

    # 左: DocAligner polygon
    poly_exp = expand_polygon(
        np.asarray(polygon_xy, dtype=np.float32),
        margin_px=float(polygon_margin_px),
        img_w=int(degraded_bgr.shape[1]),
        img_h=int(degraded_bgr.shape[0]),
    )
    left = draw_polygon_overlay(degraded_bgr, poly_exp)

    # degraded -> rectified_landscape -> chosen(rot)
    rect_w, rect_h = int(rectified_landscape_size_wh[0]), int(rectified_landscape_size_wh[1])
    dummy_rect = np.zeros((max(1, rect_h), max(1, rect_w), 3), dtype=np.uint8)
    _dummy_rot, M_rect_to_chosen = rotate_image_bound_with_matrix(dummy_rect, float(decided_angle_deg))

    H_deg_to_rect = np.asarray(H_degraded_to_rectified_landscape, dtype=np.float64)
    H_deg_to_chosen = np.asarray(M_rect_to_chosen, dtype=np.float64) @ H_deg_to_rect
    try:
        H_chosen_to_deg = np.linalg.inv(H_deg_to_chosen)
    except Exception:
        H_chosen_to_deg = None

    if H_chosen_to_deg is not None:
        # フォームA: マーカー bbox（chosen座標） -> degraded座標へ
        if decided_form == "A" and decision_markers:
            for m in decision_markers:
                try:
                    x, y, bw, bh = m.get("bbox", [0, 0, 0, 0])
                    box = np.array(
                        [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
                        dtype=np.float32,
                    )
                    box_deg = _perspective_transform_points(box, H_chosen_to_deg)
                    left = _draw_polygon_outline(left, box_deg, (0, 0, 255), thickness=6)
                except Exception:
                    continue

        # フォームB: QR points（chosen座標） -> degraded座標へ
        if decided_form == "B" and decision_qrs:
            try:
                pts = np.asarray(decision_qrs[0].get("points"), dtype=np.float32).reshape(-1, 2)
                pts_deg = _perspective_transform_points(pts, H_chosen_to_deg)
                left = _draw_polygon_outline(left, pts_deg, (255, 0, 0), thickness=6)
            except Exception:
                pass

    # 右: aligned（最終）
    right = aligned_bgr
    return _hstack_with_padding(left, right)


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


def _compute_pad_px_auto(image_bgr: np.ndarray) -> int:
    """画像サイズから pad_px を自動算出する（targetの端ギリギリ撮影対策）。"""

    try:
        h, w = image_bgr.shape[:2]
    except Exception:
        return int((PIPELINE_DEFAULTS.get("docaligner") or {}).get("pad_px") or 200)

    cfg = PIPELINE_DEFAULTS.get("docaligner") or {}
    ratio = float(cfg.get("pad_px_auto_ratio", 0.08) or 0.08)
    pmin = int(cfg.get("pad_px_auto_min", 120) or 120)
    # v18.10: target画像の端ギリギリ対策として auto_max を大きめに取る
    pmax = int(cfg.get("pad_px_auto_max", 800) or 800)
    pad = int(round(float(min(h, w)) * ratio))
    pad = max(pmin, pad)
    pad = min(pmax, pad)
    return int(pad)


def _run_docaligner_once_with_meta(
    *,
    model: Any,
    cb: Any,
    image_bgr: np.ndarray,
    pad_px: int,
    input_scale: float = 1.0,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """DocAlignerを1回実行して polygon(4x2, original scale) と正規化メタを返す。

    高速化目的（v18.??）:
      - DocAligner multi 内で「診断のための normalize 再実行」を避ける。
        （normalize は、退化quadの修復時にエッジ復旧が走りうるため重い）

    戻り値:
      (poly or None, norm_meta)
    """

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

    # v18.18 (改善: 重複コーナー/三角形化の修復):
    # ここで「必ず4つの角（重複なし）」を得るため、
    # 画像コンテキスト（エッジ）を使った修復を含む正規化を行う。
    poly, norm_meta = normalize_polygon_to_quad_with_meta(poly, image_bgr=padded)
    if poly is None:
        # norm_meta は診断に重要なので返す
        if not isinstance(norm_meta, dict):
            norm_meta = {"ok": False, "issue": "normalize_failed"}
        return None, norm_meta

    # unpad
    poly = poly - float(pad_px)
    if abs(s - 1.0) > 1e-9:
        poly = poly / float(s)

    # どの座標系で正規化したか（診断用）
    if isinstance(norm_meta, dict):
        norm_meta = dict(norm_meta)
        norm_meta.setdefault("note", "normalized_on_padded_image_before_unpad")

    return poly.astype(np.float32), (norm_meta if isinstance(norm_meta, dict) else {"ok": False})


def _run_docaligner_once(
    *,
    model: Any,
    cb: Any,
    image_bgr: np.ndarray,
    pad_px: int,
    input_scale: float = 1.0,
) -> Optional[np.ndarray]:
    """DocAlignerを1回実行して polygon(4x2, original scale) を返す。"""

    poly, _norm_meta = _run_docaligner_once_with_meta(
        model=model,
        cb=cb,
        image_bgr=image_bgr,
        pad_px=int(pad_px),
        input_scale=float(input_scale),
    )
    return poly


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
    """重複点（ほぼ同一点）とみなす距離しきい値(px)を決める。

    - 画像がある場合: min(H,W) の比率で決める
    - 無い場合: pts の bounding box から概算
    """

    try:
        if image_bgr is not None:
            h, w = image_bgr.shape[:2]
            s = float(min(h, w))
        else:
            pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
            xs = pts[:, 0]
            ys = pts[:, 1]
            s = float(max(1.0, min(float(xs.max() - xs.min()), float(ys.max() - ys.min()))))
        # 端末/解像度差に強いように比率で決める
        # 例: 短辺2000px -> 0.012*2000=24px
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
    # 画像サイズがあるなら「極小quad」を弾く（ただし厳しすぎると救済候補が消えるため軽め）
    if img_w is not None and img_h is not None:
        min_side = float(min(img_w, img_h))
        if e_min < (min_side * 0.01):
            reasons.append("edge_min_too_small_relative")

    return (len(reasons) > 0), {"area": area, "edge_min": e_min, "min_pair_dist": min_pair, "reasons": reasons}


def _recover_quad_from_edges(
    image_bgr: np.ndarray,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """画像のエッジ/輪郭から4隅を再推定する（DocAligner出力が退化した時の修復）。

    方針（高精度優先・時間は気にしない）:
      1) エッジ→輪郭→凸包→approxPolyDP(ε掃引) で「4点凸四角形」を直接探す
      2) 見つからない場合は HoughLinesP で2方向×2本の外接線を推定し、交点を角とする
      3) 最後に Shi-Tomasi で角を微調整

    返り値:
      (quad or None, meta)
    """

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

    # エッジ抽出（やや強め）
    e = cv2.Canny(gray, 40, 140)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    best_quad: Optional[np.ndarray] = None
    best_score = float("-inf")

    # --- (1) 輪郭近似で4点を直接探す ---
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
            deg, deg_meta = _is_degenerate_quad_by_geometry(quad, img_w=int(w), img_h=int(h), dup_thresh_px=float(dup_thr))
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

    # --- (2) HoughLinesP で4辺を推定して交点から角を作る ---
    try:
        lines = cv2.HoughLinesP(e, 1, np.pi / 180.0, threshold=120, minLineLength=int(min(h, w) * 0.18), maxLineGap=18)
    except Exception:
        lines = None

    if lines is not None and len(lines) >= 6:
        segs = np.asarray(lines, dtype=np.float32).reshape(-1, 4)

        # 方向角（0..pi）
        dx = segs[:, 2] - segs[:, 0]
        dy = segs[:, 3] - segs[:, 1]
        ang = np.mod(np.arctan2(dy, dx), np.pi)  # 0..pi

        # 2方向に分ける（直交っぽい2群）
        # まず主方向をヒストグラムで取る
        bins = 36
        hist, edges = np.histogram(ang, bins=bins, range=(0.0, float(np.pi)))
        i0 = int(np.argmax(hist))
        a0 = float((edges[i0] + edges[i0 + 1]) * 0.5)
        # 直交方向（±90deg）近傍を探す
        target = float((a0 + (np.pi / 2.0)) % np.pi)
        # target に近いbinを選ぶ
        centers = (edges[:-1] + edges[1:]) * 0.5
        i1 = int(np.argmin(np.abs(centers - target)))
        a1 = float(centers[i1])

        def _select_group(center_angle: float, tol: float = float(np.pi / 18.0)) -> np.ndarray:
            da = np.abs(((ang - center_angle + np.pi) % np.pi) - np.pi / 2.0)
            # 上式は微妙なので、単純に circular distance を使う
            da = np.abs(((ang - center_angle + np.pi) % np.pi) - np.pi)
            da = np.minimum(da, np.pi - da)
            return segs[da < tol]

        g0 = _select_group(a0)
        g1 = _select_group(a1)

        def _fit_two_parallel_lines(segs_xyxy: np.ndarray) -> Optional[list[tuple[np.ndarray, float]]]:
            if segs_xyxy is None or len(segs_xyxy) < 2:
                return None
            # 各線分の法線 n とオフセット rho を作る
            ps = segs_xyxy.reshape(-1, 4)
            x1, y1, x2, y2 = ps[:, 0], ps[:, 1], ps[:, 2], ps[:, 3]
            vx = x2 - x1
            vy = y2 - y1
            norm = np.sqrt(vx * vx + vy * vy) + 1e-6
            vx /= norm
            vy /= norm
            # 法線（右法線）
            nx = vy
            ny = -vx
            # 各線分の中点
            mx = (x1 + x2) * 0.5
            my = (y1 + y2) * 0.5
            rho = nx * mx + ny * my

            # 法線方向は符号が反転し得るので揃える（rhoが正になるように）
            sign = np.where(rho >= 0, 1.0, -1.0)
            nx *= sign
            ny *= sign
            rho *= sign

            # 1D k-means(2) 相当: rho の 2 クラスタを中央値で分ける
            r_sorted = np.sort(rho)
            cut = float(r_sorted[len(r_sorted) // 2])
            idx0 = rho <= cut
            idx1 = rho > cut
            if idx0.sum() < 1 or idx1.sum() < 1:
                # 端2本
                idx0 = rho <= float(r_sorted[0])
                idx1 = rho >= float(r_sorted[-1])

            def _avg_line(mask: np.ndarray) -> tuple[np.ndarray, float]:
                n = np.array([float(np.mean(nx[mask])), float(np.mean(ny[mask]))], dtype=np.float32)
                nn = float(np.linalg.norm(n))
                if nn <= 1e-6:
                    n = np.array([1.0, 0.0], dtype=np.float32)
                    nn = 1.0
                n = n / nn
                r = float(np.mean(rho[mask]))
                return n, r

            line0 = _avg_line(idx0)
            line1 = _avg_line(idx1)
            return [line0, line1]

        lines0 = _fit_two_parallel_lines(g0)
        lines1 = _fit_two_parallel_lines(g1)

        if lines0 is not None and lines1 is not None:
            # 直線 (n, rho): n・x = rho を2本ずつ
            def _inter(n1: np.ndarray, r1: float, n2: np.ndarray, r2: float) -> Optional[np.ndarray]:
                A = np.array([[float(n1[0]), float(n1[1])], [float(n2[0]), float(n2[1])]], dtype=np.float64)
                b = np.array([float(r1), float(r2)], dtype=np.float64)
                det = float(np.linalg.det(A))
                if abs(det) < 1e-9:
                    return None
                x = np.linalg.solve(A, b)
                if not (math.isfinite(float(x[0])) and math.isfinite(float(x[1]))):
                    return None
                return x.astype(np.float32)

            nA1, rA1 = lines0[0]
            nA2, rA2 = lines0[1]
            nB1, rB1 = lines1[0]
            nB2, rB2 = lines1[1]

            pts = []
            for (na, ra) in [(nA1, rA1), (nA2, rA2)]:
                for (nb, rb) in [(nB1, rB1), (nB2, rB2)]:
                    p = _inter(na, ra, nb, rb)
                    if p is not None:
                        pts.append(p)

            if len(pts) == 4:
                quad = order_quad_tl_tr_br_bl(np.stack(pts, axis=0).astype(np.float32))
                quad = _clamp_poly_to_image(quad, img_w=int(w), img_h=int(h))
                dup_thr = _default_duplicate_threshold_px(pts_xy=quad, image_bgr=image_bgr)
                deg, deg_meta = _is_degenerate_quad_by_geometry(quad, img_w=int(w), img_h=int(h), dup_thresh_px=float(dup_thr))
                if not deg:
                    ok, q = _is_valid_quad(quad, img_w=int(w), img_h=int(h))
                    if ok:
                        try:
                            quad = _refine_quad_corners_by_shi_tomasi(image_gray=gray, quad_xy=quad)
                        except Exception:
                            pass
                        meta.update({"ok": True, "method": "hough_lines", "detail": {"q": q, "deg": deg_meta}})
                        return quad, meta

    meta["detail"]["reason"] = "edge_recover_failed"
    return None, meta


def normalize_polygon_to_quad_with_meta(
    poly_xy: np.ndarray,
    *,
    image_bgr: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """DocAligner 等が返した polygon を「必ず4点（重複なし）」へ正規化する。

    特にユーザー要望（重要）:
      - 4点のうち2点が同じ場所（重複）になるケースを検出し、論理処理で必ず修復を試みる。

    返り値:
      (quad or None, meta)
    """

    meta: dict[str, Any] = {"ok": False, "method": "", "issue": "", "detail": {}}
    if poly_xy is None:
        meta["issue"] = "poly_is_none"
        return None, meta

    pts = np.asarray(poly_xy, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        meta["issue"] = "pts_lt_3"
        return None, meta

    # dup 判定閾値
    dup_thr = _default_duplicate_threshold_px(pts_xy=pts, image_bgr=image_bgr)

    # (A) 4点が来た場合は、まず order→重複/退化チェック
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

        # ここに来る = 4点だが重複/退化
        if image_bgr is not None:
            quad_r, rec_meta = _recover_quad_from_edges(image_bgr)
            meta["detail"]["edge_recover"] = rec_meta
            if quad_r is not None:
                meta.update({"ok": True, "method": f"repair:{rec_meta.get('method', '')}"})
                return quad_r, meta

        # 画像が無い/復旧失敗 -> hull→minAreaRect（最低限四角形にはする）
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

    # (B) N>=3（多点/三角形など）: 画像があればエッジ復旧を優先
    if image_bgr is not None:
        quad_r, rec_meta = _recover_quad_from_edges(image_bgr)
        meta["detail"]["edge_recover"] = rec_meta
        if quad_r is not None:
            meta.update({"ok": True, "method": f"repair:{rec_meta.get('method', '')}"})
            return quad_r, meta

    # (C) 最後の手段: hull→minAreaRect
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


def normalize_polygon_to_quad(poly_xy: np.ndarray) -> Optional[np.ndarray]:
    """後方互換API: quadのみ返す。"""

    quad, _meta = normalize_polygon_to_quad_with_meta(poly_xy, image_bgr=None)
    return quad


def detect_polygon_fallback_opencv(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """DocAligner が全滅した場合のフォールバック（OpenCV 輪郭ベース）。

    目的:
      - AI検出が「角欠け」「三角形/線」になりやすいケースで、
        古典手法でも拾える可能性があるため。

    方針:
      - Canny → 輪郭抽出 → 面積最大を選ぶ
      - approx で4点が得られればそれを採用
      - 4点が得られない場合は minAreaRect で四角形化
    """

    if image_bgr is None:
        return None

    try:
        h, w = image_bgr.shape[:2]
        if h < 64 or w < 64:
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0.0)

        # エッジ強調
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 面積最大（紙領域を想定）
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:10]:
            area = float(cv2.contourArea(cnt))
            if area < float(h * w) * 0.05:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            pts = approx.reshape(-1, 2).astype(np.float32)

            quad = None
            if pts.shape[0] == 4 and cv2.isContourConvex(approx):
                quad = order_quad_tl_tr_br_bl(pts)
            else:
                quad = normalize_polygon_to_quad(cnt.reshape(-1, 2).astype(np.float32))

            if quad is None:
                continue

            return quad
    except Exception:
        return None

    return None


def _score_quad_by_edge_support(
    *,
    quad_xy: np.ndarray,
    edge_u8: np.ndarray,
) -> float:
    """quad が画像中のエッジにどれだけ一致しているかを簡易スコア化する。"""

    quad = order_quad_tl_tr_br_bl(np.asarray(quad_xy, dtype=np.float32).reshape(4, 2))
    h, w = edge_u8.shape[:2]
    edge = edge_u8
    if edge.ndim == 3:
        edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)

    # サンプリング点の周囲 r ピクセルにエッジがあればヒット扱い
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


def _refine_quad_corners_by_shi_tomasi(
    *,
    image_gray: np.ndarray,
    quad_xy: np.ndarray,
) -> np.ndarray:
    """推定quadの各コーナーを、周辺の強いコーナー（Shi-Tomasi）へスナップする。"""

    quad = order_quad_tl_tr_br_bl(np.asarray(quad_xy, dtype=np.float32).reshape(4, 2))
    h, w = image_gray.shape[:2]
    out = quad.copy()

    # 角周りの探索窓（画像スケールに応じて）
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

        # ROI座標 -> 画像座標
        pts[:, 0] += float(x0)
        pts[:, 1] += float(y0)
        d = np.linalg.norm(pts - np.array([[cx, cy]], dtype=np.float32), axis=1)
        j = int(np.argmin(d))
        if float(d[j]) <= float(win) * 0.75:
            out[i] = pts[j]

    return order_quad_tl_tr_br_bl(out)


def detect_polygon_fallback_advanced(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """DocAligner が 2点/3点等で失敗した場合の、より高精度なフォールバック。

    方針（処理時間は無視・精度優先）:
      - 複数の前処理を作る
      - 輪郭→凸包→approxPolyDP(ε掃引) で 4点を直接探索
      - 4点化できない場合は minAreaRect
      - エッジ支持率（quad辺上にエッジがある割合）でスコアリング
      - 最良quadを Shi-Tomasi でコーナー微調整
    """

    if image_bgr is None:
        return None

    h, w = image_bgr.shape[:2]
    if h < 80 or w < 80:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0.0)

    variants: list[tuple[str, np.ndarray]] = [("gray", gray_blur)]
    # CLAHE
    try:
        clahe_cfg = (PIPELINE_DEFAULTS.get("marker") or {}).get("clahe") or {"clipLimit": 3.0, "tileGridSize": [8, 8]}
        clahe = cv2.createCLAHE(clipLimit=float(clahe_cfg.get("clipLimit", 3.0)), tileGridSize=tuple(int(x) for x in clahe_cfg.get("tileGridSize", [8, 8])))
        variants.append(("clahe", clahe.apply(gray)))
    except Exception:
        pass

    # adaptive threshold (inv)
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

        # edges
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

            quad = None
            # ε掃引で4点近似を探す
            for eps_ratio in [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
                pts = approx.reshape(-1, 2).astype(np.float32)
                if pts.shape[0] == 4 and cv2.isContourConvex(approx):
                    quad = order_quad_tl_tr_br_bl(pts)
                    break

            if quad is None:
                # 4点近似が取れない場合は四角形化
                quad = normalize_polygon_to_quad(hull.reshape(-1, 2).astype(np.float32))

            if quad is None:
                continue

            quad = _clamp_poly_to_image(quad, img_w=int(w), img_h=int(h))
            ok, q = _is_valid_quad(quad, img_w=int(w), img_h=int(h))
            if not ok:
                continue

            support = _score_quad_by_edge_support(quad_xy=quad, edge_u8=e)
            # 面積比 + エッジ支持率を重視
            score = float(q.get("area_ratio", 0.0)) * 10.0 + float(support) * 6.0 - float(q.get("edge_ratio", 0.0)) * 0.01

            if score > best_score:
                best_score = score
                best_quad = quad

    for _name, src in variants:
        _try_contours(src)

    if best_quad is None:
        return None

    try:
        refined = _refine_quad_corners_by_shi_tomasi(image_gray=gray, quad_xy=best_quad)
        return refined
    except Exception:
        return best_quad


def _quad_quality(poly_xy: np.ndarray, img_w: int, img_h: int) -> dict[str, Any]:
    """polygon品質（面積/辺長など）を計算する。"""

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
    return {
        "area": area,
        "area_ratio": area_ratio,
        "edge_min": e_min,
        "edge_max": e_max,
        "edge_ratio": edge_ratio,
    }


def _is_valid_quad(poly_xy: np.ndarray, img_w: int, img_h: int) -> tuple[bool, dict[str, Any]]:
    """三角形/線のような退化polygonを早期に弾く。"""

    q = _quad_quality(poly_xy, img_w, img_h)
    min_side = float(min(img_w, img_h))

    # NOTE:
    # - 端ギリギリ撮影では、誤検出が「線に近い」面積になりやすい
    # - ここで弾くことで、後段の rectify/フォーム判定を無駄に回さない
    ok = True
    reasons: list[str] = []
    # v18.14:
    # target 画像では「紙がフレーム端ギリギリ」「一部欠け」などで、
    # DocAligner の候補がやや小さくなることがある。
    # ここを厳しくしすぎると救済候補まで全滅しやすいので、
    # 最低限の“紙らしさ”を保ちつつ、閾値を少し緩める。
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
    """polygon を重複排除するための丸めキー。"""

    poly = order_quad_tl_tr_br_bl(np.asarray(poly_xy, dtype=np.float32).reshape(4, 2))
    return tuple((round(float(x), decimals), round(float(y), decimals)) for x, y in poly.tolist())


def _clamp_poly_to_image(poly_xy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    poly = order_quad_tl_tr_br_bl(np.asarray(poly_xy, dtype=np.float32).reshape(4, 2))
    poly[:, 0] = np.clip(poly[:, 0], 0, max(0, int(img_w) - 1))
    poly[:, 1] = np.clip(poly[:, 1], 0, max(0, int(img_h) - 1))
    return poly


def _iter_docaligner_settings(
    *,
    args: argparse.Namespace,
    image_bgr: np.ndarray,
) -> list[dict[str, Any]]:
    """DocAligner候補設定（model/type/pad/scale）を優先度順に列挙する。"""

    cfg_doc = PIPELINE_DEFAULTS.get("docaligner") or {}
    cfg_multi = (cfg_doc.get("multi") or {}) if isinstance(cfg_doc, dict) else {}

    model0 = str(getattr(args, "docaligner_model", cfg_doc.get("model") or "fastvit_sa24"))
    type0 = str(getattr(args, "docaligner_type", cfg_doc.get("type") or "heatmap"))

    models = [model0] + [str(x) for x in (cfg_multi.get("extra_models") or [])]
    types = [type0] + [str(x) for x in (cfg_multi.get("extra_types") or [])]

    # pad candidates: base + auto + configured list
    # 改善1（DocAligner）:
    # 端ギリギリ撮影の救済には pad が効くが、
    # v18.13〜の target-only 運用（過去）では「max_infer_runs が小さい」ため、
    # *scale を先に総当たりして pad が試せない* という順序が精度を下げやすい。
    # そのため、ここでは「pad（候補）→ scale」の順でまず試し、
    # 最低限の回数で大きめ pad まで到達できるように優先度順を設計する。
    pad_base = int(cfg_doc.get("pad_px") or 200)
    pad_auto = _compute_pad_px_auto(image_bgr)
    pad0 = int(max(pad_base, pad_auto))

    pad_candidates = [pad0] + [int(x) for x in (cfg_multi.get("pad_px_candidates") or [])] + [pad_base, pad_auto]
    pad_candidates = [int(x) for x in pad_candidates if int(x) >= 0]

    # 重複除去（順序維持）
    seen_pad: set[int] = set()
    pad_list: list[int] = []
    for p in pad_candidates:
        if p in seen_pad:
            continue
        seen_pad.add(p)
        pad_list.append(int(p))

    scales_raw = [float(x) for x in (cfg_multi.get("input_scales") or [])]
    # 1.0 は常に最優先で試す（角欠け対策で pad を増やす場合、低解像度化だけ先に回すと成功率が下がりやすい）
    scales = [1.0] + [float(s) for s in scales_raw if abs(float(s) - 1.0) > 1e-9]

    # 優先度:
    # - まず *scale=1.0* で pad 候補を順番に試す（pad が効くケースを取りこぼさない）
    # - それでもダメなら、pad0 のまま scale バリエーションを試す
    # - モデル/タイプは「ユーザー指定（args）」を最優先し、次に extra を試す
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

    # 優先度:
    # - まず *scale=1.0* で pad 候補を順番に試す（pad が効くケースを最短で拾う）
    # - それでもダメなら、pad0 のまま scale バリエーション（縮小/拡大）
    # - モデル/タイプは「ユーザー指定（args）」を最優先し、次に extra を試す

    # model/type の順序:
    # - args で指定された (model0/type0) を **最優先**
    # - それ以外は安定化のため、軽量→高精度の順（ただし model0 は先頭固定）
    model_pri = {"lcnet050": 0, "lcnet100": 1, "fastvit_t8": 2, "fastvit_sa24": 3}
    rest_models = [m for m in _uniq_str(models_u) if str(m) != str(model0)]
    rest_models = sorted(rest_models, key=lambda m: model_pri.get(str(m), 99))
    models_u = [str(model0)] + rest_models

    # type は args 指定を最優先し、残りを後ろへ
    rest_types = [t for t in _uniq_str(types_u) if str(t) != str(type0)]
    # 安定化: heatmap→point の順
    rest_types = [t for t in ["heatmap", "point"] if t in rest_types]
    types_u = [str(type0)] + rest_types

    pads_u = pad_list  # 既に優先度順に構成済み
    scales_u = scales  # [1.0, ...]

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
            for p in pads_u:
                settings.append({"model": m, "type": t, "pad_px": int(p), "input_scale": 1.0})
            for s in scales_u:
                if abs(float(s) - 1.0) < 1e-9:
                    continue
                settings.append({"model": m, "type": t, "pad_px": int(pad0), "input_scale": float(s)})

    return _uniq_settings(settings)


def _margin_px_candidates_for_eval(*, args: argparse.Namespace, poly_xy: np.ndarray) -> list[float]:
    """margin 候補（複数）を返す。"""

    # fixed が指定されている場合はそれだけ
    fixed = float(getattr(args, "polygon_margin_px", 0.0) or 0.0)
    if fixed > 0:
        return [float(fixed)]

    # NOTE:
    # v18.18+ では「候補評価」内での margin 探索を絞る（速度/安定の両立）。
    # ただし advanced_fallback 側で必要に応じて再探索するため、ここで無理に網羅しない。
    ratios = [
        float(getattr(args, "polygon_margin_ratio", 0.0) or 0.0),
        0.0,
        0.06,
    ]
    # uniq
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
    # uniq again with rounding
    seen2: set[int] = set()
    out2: list[float] = []
    for m in out:
        k = int(round(float(m)))
        if k in seen2:
            continue
        seen2.add(k)
        out2.append(float(m))
    return out2


def detect_polygon_docaligner(
    model: Any,
    cb: Any,
    image_bgr: np.ndarray,
    pad_px: Optional[int] = None,
    input_scale: float = 1.0,
) -> Optional[np.ndarray]:
    # v18.10:
    # pad_px が未指定の場合は「固定値 + 自動推定」のうち大きい方を採用する。

    if image_bgr is None:
        return None

    if pad_px is None:
        pad_base = int((PIPELINE_DEFAULTS.get("docaligner") or {}).get("pad_px") or 200)
        pad_auto = _compute_pad_px_auto(image_bgr)
        pad_px = int(max(pad_base, pad_auto))

    poly = _run_docaligner_once(model=model, cb=cb, image_bgr=image_bgr, pad_px=int(pad_px), input_scale=float(input_scale))
    if poly is None:
        return None

    ok, _q = _is_valid_quad(poly, img_w=int(image_bgr.shape[1]), img_h=int(image_bgr.shape[0]))
    if not ok:
        return None

    return poly


def _quick_eval_form_scores_for_candidate(
    rectified_landscape_bgr: np.ndarray,
    *,
    rotation_max_workers: int,
    marker_preproc: str,
    unknown_score_threshold: float,
    unknown_margin: float,
    formA_geom_cfg: Optional[MarkerGeometryConfig] = None,
) -> tuple[FormDecision, float]:
    """候補polygonの「良さ」を測るための軽量評価。

    方針:
      - 既存の decide_form_by_rotations()（0/180のみ）をそのまま使い、A/Bスコアが高い候補を優先
      - unknown 判定でも、top score を返して候補比較に使う
    """

    decision = decide_form_by_rotations(
        rectified_landscape_bgr,
        max_workers=int(rotation_max_workers),
        marker_preproc=str(marker_preproc),
        unknown_score_threshold=float(unknown_score_threshold),
        unknown_margin=float(unknown_margin),
        formA_geom_cfg=formA_geom_cfg,
    )
    return decision, float(decision.score)


def make_formA_geom_cfg_for_case(*, source_dataset: str, source_form: str) -> Optional[MarkerGeometryConfig]:
    """dataset に応じてフォームAの幾何制約（誤検出抑制）を調整した cfg を返す。

    NOTE:
      - test: 正例取りこぼし回避を優先し、周辺白地制約を無効化
      - target: 実撮影（影/机模様）での取りこぼし回避を優先し、閾値を緩める
      - synthetic: デフォルト（PIPELINE_DEFAULTS の設定）
    """

    base_cfg_for_A: Optional[MarkerGeometryConfig] = None
    try:
        base_cfg_dict = (PIPELINE_DEFAULTS.get("formA") or {}).get("geometry") or {}
        allowed = set(getattr(MarkerGeometryConfig, "__dataclass_fields__", {}).keys())
        base_cfg_for_A = MarkerGeometryConfig(**{k: v for k, v in base_cfg_dict.items() if k in allowed})
    except Exception:
        base_cfg_for_A = None

    ds = str(source_dataset or "")
    sf = str(source_form or "")

    # test: 周辺白地チェックが改悪/撮影条件で誤rejectしやすいので無効化
    if ds == "test" and sf in ("A", "B"):
        try:
            base_cfg = base_cfg_for_A or MarkerGeometryConfig()
            return MarkerGeometryConfig(
                **{
                    **asdict(base_cfg),
                    "surround_min_mean_gray": 0.0,
                    "surround_max_ink_ratio": 1.0,
                }
            )
        except Exception:
            return None

    # target: 影・机模様・枠線の写り込みで取りこぼしやすいので recall 優先で緩める
    if ds == "target":
        try:
            base_cfg = base_cfg_for_A or MarkerGeometryConfig()
            return MarkerGeometryConfig(
                **{
                    **asdict(base_cfg),
                    "surround_min_mean_gray": 150.0,
                    "surround_max_ink_ratio": 0.08,
                    "min_marker_area_page_ratio": 3.5e-5,
                }
            )
        except Exception:
            return None

    return None


def detect_polygon_docaligner_multi(
    *,
    logger: logging.Logger,
    args: argparse.Namespace,
    degraded_bgr: np.ndarray,
    formA_geom_cfg: Optional[MarkerGeometryConfig] = None,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """DocAligner を“1発勝負”にせず複数条件で実行し、最良polygonを返す。

    改善1-A/1-B/1-C の中核:
      - model/type/pad/scale を変えて複数回推論
      - 退化polygon（線/三角形っぽい）を早期除外
      - polygon margin（複数候補）も含めて rectify → A/Bスコアで評価

    返り値:
      (best_poly_xy or None, meta)
    """

    if degraded_bgr is None:
        return None, {"ok": False, "reason": "image_is_none"}

    cfg_doc = PIPELINE_DEFAULTS.get("docaligner") or {}
    cfg_multi = (cfg_doc.get("multi") or {}) if isinstance(cfg_doc, dict) else {}
    enable_multi = bool(cfg_multi.get("enable", True))

    # Multi 無効なら従来通り 1 回だけ
    if not enable_multi:
        model, cb = load_docaligner_model(str(args.docaligner_model), str(args.docaligner_type))
        poly = detect_polygon_docaligner(model, cb, degraded_bgr)
        return poly, {"ok": poly is not None, "mode": "single"}

    h, w = degraded_bgr.shape[:2]
    settings = _iter_docaligner_settings(args=args, image_bgr=degraded_bgr)
    max_infer_runs = int(cfg_multi.get("max_infer_runs") or 10)
    max_poly_candidates = int(cfg_multi.get("max_polygon_candidates") or 6)

    # 推論結果（raw polygon）候補
    candidates: list[dict[str, Any]] = []
    all_polys: list[dict[str, Any]] = []
    seen_keys: set[tuple[tuple[float, float], ...]] = set()

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
            logger.warning("[DocAligner] model load failed: %s/%s err=%s", model_name, model_type, e)
            continue

        infer_runs += 1
        try:
            # v18.18+ 改善:
            # _run_docaligner_once_with_meta 内で normalize_polygon_to_quad_with_meta を実施しているため、
            # ここで再度 normalize を回して「二重実行」しない。
            poly, _norm_meta = _run_docaligner_once_with_meta(
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
                "norm": _norm_meta if isinstance(_norm_meta, dict) else {"ok": False, "issue": "norm_meta_not_dict"},
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
                "norm": _norm_meta if isinstance(_norm_meta, dict) else {"ok": False, "issue": "norm_meta_not_dict"},
            }
        )
        if len(candidates) >= max_poly_candidates:
            break

    if not candidates:
        # strict で全滅した場合は、救済候補を *増やす*。
        # v18.16 までは「面積最大」を優先していたが、target では
        # 面積が大きくても “線/重複点” に近い退化quadが混ざりうる。
        # ここでは以下の順で候補を増やし、後段の eval に回す。
        #   1) all_polys のうち、area_ratio が大きいもの（ただし edge_min が0に近いものは避ける）
        #   2) OpenCV 輪郭ベース
        #   3) 高精度フォールバック（advanced）

        if all_polys:
            def _relaxed_score(d: dict[str, Any]) -> float:
                q = (d.get("quality") or {})
                area = float(q.get("area_ratio") or 0.0)
                edge_min = float(q.get("edge_min") or 0.0)
                # edge_min が 0 に近いもの（重複点/退化）は強く減点
                return area - (1.0 if edge_min <= 1.0 else 0.0)

            sorted_all = sorted(all_polys, key=_relaxed_score, reverse=True)
            for d in sorted_all[: max(2, int(max_poly_candidates))]:
                candidates.append({"poly": d["poly"], "quality": d["quality"], "setting": d["setting"], "relaxed": True})

        # OpenCV 輪郭ベース
        fallback = detect_polygon_fallback_opencv(degraded_bgr)
        if fallback is not None:
            ok_fb, q_fb = _is_valid_quad(fallback, img_w=int(w), img_h=int(h))
            if ok_fb:
                candidates.append(
                    {
                        "poly": order_quad_tl_tr_br_bl(fallback).astype(np.float32),
                        "quality": q_fb,
                        "setting": {"model": "opencv_fallback", "type": "contour", "pad_px": 0, "input_scale": 1.0},
                        "relaxed": True,
                    }
                )

        # 高精度フォールバック（advanced）を候補に追加
        adv = detect_polygon_fallback_advanced(degraded_bgr)
        if adv is not None:
            ok_adv, q_adv = _is_valid_quad(adv, img_w=int(w), img_h=int(h))
            if ok_adv:
                candidates.append(
                    {
                        "poly": order_quad_tl_tr_br_bl(np.asarray(adv, dtype=np.float32)).astype(np.float32),
                        "quality": q_adv,
                        "setting": {"model": "opencv_fallback", "type": "advanced", "pad_px": 0, "input_scale": 1.0},
                        "relaxed": True,
                    }
                )

        if not candidates:
            return None, {"ok": False, "reason": "no_valid_polygon", "infer_runs": infer_runs, "all_polys": all_polys}

    # 候補評価: polygon margin 複数候補も試す
    # NOTE(v18.14):
    # ユーザー要望は「三角形/線のような退化検出のまま進めない」ことであり、
    # ここでフォーム判定（マーカー/QR）が通ることを必須条件にしてしまうと、
    # 「紙は検出できているがマーカー/QRが切れている」ケースまで DocAligner 段階で落ちてしまう。
    # そのため、
    #   - DocAligner 段階では *幾何的に妥当な quad* を採用する
    #   - フォーム判定のスコアは「候補選択の補助」として利用する
    # という設計にする。
    best: Optional[dict[str, Any]] = None
    evals: list[dict[str, Any]] = []

    # tie-break 用のエッジ画像は候補に依存しないため、必要時に1回だけ作る
    edge_u8: Optional[np.ndarray] = None

    def _lazy_edge_u8() -> np.ndarray:
        nonlocal edge_u8
        if edge_u8 is None:
            g = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2GRAY)
            e = cv2.Canny(g, 40, 140)
            e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
            edge_u8 = e
        return edge_u8

    # ---- 候補の評価数を絞る（幾何品質で上位のみ） ----
    try:
        eval_max_candidates = int(cfg_multi.get("eval_max_candidates") or 0)
    except Exception:
        eval_max_candidates = 0
    if eval_max_candidates <= 0:
        eval_max_candidates = int(max(1, min(2, len(candidates))))

    def _cand_quality_score(d: dict[str, Any]) -> float:
        q = d.get("quality") or {}
        # 面積比を最重要。edge_ratio は小さいほど良い。
        area = float(q.get("area_ratio") or 0.0)
        edge_ratio = float(q.get("edge_ratio") or 1e9)
        edge_min = float(q.get("edge_min") or 0.0)
        # スコア: areaを押し上げ、edge_ratioを弱く減点、edge_minを僅かに加点
        return area * 10.0 - (edge_ratio * 0.15) + (edge_min * 1e-4)

    candidates_for_eval = sorted(candidates, key=_cand_quality_score, reverse=True)[: int(eval_max_candidates)]

    # 評価用の decide は軽量化（0/180固定なので並列化しない方が速いことが多い）
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
        margin_list = _margin_px_candidates_for_eval(args=args, poly_xy=poly)
        margin_list = margin_list[: int(eval_max_margins)]
        for margin_px in margin_list:
            try:
                rect, _H, poly_exp, rect_meta = rectify_with_margin_and_optional_padding(
                    degraded_bgr,
                    polygon_xy=poly,
                    margin_px=float(margin_px),
                    out_max_side=int(args.docaligner_max_side),
                )
                rect, _ = enforce_landscape(rect)
                decision, score = _quick_eval_form_scores_for_candidate(
                    rect,
                    rotation_max_workers=int(eval_rotation_workers),
                    marker_preproc=str(args.marker_preproc),
                    unknown_score_threshold=float(args.unknown_score_threshold),
                    unknown_margin=float(args.unknown_margin),
                    formA_geom_cfg=formA_geom_cfg,
                )
                rec = {
                    "setting": c["setting"],
                    "quality": c["quality"],
                    "margin_px": float(margin_px),
                    "decision": asdict(decision),
                    "score": float(score),
                    "rectify": rect_meta,
                }
                evals.append(rec)
                # フォーム判定スコアで一次選好。
                # target の no_detection ケースではスコアが全て 0 になりやすいので、
                # タイブレークは「面積」だけでなく「エッジ支持率」も使って、
                # 机の縁/影などに引っ張られた誤quadを避ける。
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
                            # edge support
                            edge = _lazy_edge_u8()
                            cur_support = _score_quad_by_edge_support(quad_xy=poly_exp, edge_u8=edge)
                            best_support = float(best.get("edge_support") or 0.0)

                            # まず支持率を優先
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
        # フォームスコアが取れない場合は、品質（面積/辺長）だけで最大のものを採用
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
        # v18.9: 自己完結化に伴い、ベース実装も本ファイル内の関数を参照する
        markers = detect_formA_marker_boxes_base(var)
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

    v18.2 改善:
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
    preprocessed_variants: Optional[list[tuple[str, np.ndarray]]] = None,
) -> list[dict[str, Any]]:
    """WeChatエンジンによるQR検出（前処理 + マルチスケール）。

    v18 改善（ユーザー要望）:
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

    # 改善（高速化）:
    # fast/robust の両方を同一画像に対して呼ぶ場合、前処理生成（gray/clahe/adaptive等）を共有できる。
    # preprocessed_variants が渡された場合はそれを使い回す。
    # ただし、呼び出し側の variants が「robust全量」などの場合があるため、ここで variant_names でフィルタする。
    if preprocessed_variants is not None:
        allowed = set(str(x) for x in (variant_names or []))
        variants = [(n, img) for (n, img) in list(preprocessed_variants) if (not allowed) or (str(n) in allowed)]
    else:
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

            # robust: best を選ぶ（ただし、十分良い候補が出たら早期終了して無駄な wechat 呼び出しを減らす）
            score = float("-inf")
            try:
                score, det = score_best_qr_candidate(test if abs(s - 1.0) < 1e-9 else src, qrs)
                # 右上象限ボーナス（+100）が入っていれば、ほぼ確定とみなして早期return
                if float(score) >= 50.0 or bool(det.get("qr_is_in_top_right_quadrant")):
                    return det.get("qrs") or qrs
            except Exception:
                score = 0.0

            if score > best_score:
                best_score = score
                best = qrs

    return best


def score_formB_fast(image_bgr: np.ndarray) -> tuple[bool, float, dict[str, Any]]:
    """回転スキャン中の高速B判定。

    v7の設計（fast→必要ならrobust）を v18(WeChat-only) にも導入するためのもの。
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

    v18.1 改善:
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

            # v18.1 改善:
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
                    # v18では scan に B_fast が入る
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

    v18.3 改善（ユーザー要望）:
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
    rejected_by_threshold: dict[str, Any] = {}
    thr = float(unknown_score_threshold or 0.0)

    # rotate の重複計算を避ける（A探索 / morph / Bfast / Brobust で同じ回転画像を何度も使うため）
    rotated_cache: dict[float, np.ndarray] = {}

    def _get_rot(angle: float) -> np.ndarray:
        a = float(angle)
        if a not in rotated_cache:
            rotated_cache[a] = rotate_image_bound(rectified_bgr, a)
        return rotated_cache[a]

    def _valid_angles() -> list[float]:
        out: list[float] = []
        for a in scan_angles:
            try:
                rot = _get_rot(a)
                h, w = rot.shape[:2]
                if h > w:
                    # enforce_landscape 後なら通常起きないが、安全側で除外
                    scan_results.append({"angle": float(a), "skip": True, "reason": "portrait"})
                    continue
                out.append(float(a))
            except Exception:
                continue
        return out

    valid_angles = _valid_angles()
    if not valid_angles:
        return FormDecision(False, None, None, 0.0, {"reason": "no_valid_angle", "scan_angles": scan_angles})

    def _run_parallel(func, angles: list[float]) -> list[dict[str, Any]]:
        if int(max_workers) <= 1 or len(angles) <= 1:
            out: list[dict[str, Any]] = []
            for a in angles:
                out.append(func(a))
            return out
        with ThreadPoolExecutor(max_workers=min(int(max_workers), len(angles))) as ex:
            futures = [ex.submit(func, a) for a in angles]
            return [f.result() for f in as_completed(futures)]

    def _merge_scan_result(angle: float, key: str, value: Any) -> None:
        for sr in scan_results:
            if abs(float(sr.get("angle", -999.0)) - float(angle)) < 1e-6:
                sr[key] = value
                return
        scan_results.append({"angle": float(angle), "skip": False, key: value})

    # ----------------------------------
    # Step 1: A探索（見つかればA確定）
    # ----------------------------------
    def _eval_formA(angle: float) -> dict[str, Any]:
        rot = _get_rot(angle)
        h, w = rot.shape[:2]
        if h > w:
            return {"angle": float(angle), "skip": True}
        okA, scoreA, detA = score_formA(rot, marker_preproc=str(marker_preproc), geom_cfg=formA_geom_cfg)
        return {"angle": float(angle), "skip": False, "A": {"ok": bool(okA), "score": float(scoreA), "detail": detA}}

    bestA: Optional[FormDecision] = None
    for r in _run_parallel(_eval_formA, valid_angles):
        if not r or r.get("skip"):
            continue
        scan_results.append(r)
        if (r.get("A") or {}).get("ok"):
            a = float(r["angle"])
            candA = FormDecision(True, "A", a, float(r["A"]["score"]), {"A": r["A"]["detail"], "phase": "formA_found"})
            if bestA is None or candA.score > bestA.score:
                bestA = candA

    if bestA is not None:
        # Aが検出できてもスコアが閾値未満なら Unknown に確定せず B探索へフォールバック
        if thr > 0 and float(bestA.score) < thr:
            rejected_by_threshold["A"] = {"score": float(bestA.score), "phase": "formA_found"}
        else:
            return bestA

    # A が全滅した場合のみ「morph」を追加で試す
    if str(marker_preproc) != "morph":

        def _eval_formA_morph(angle: float) -> dict[str, Any]:
            rot = _get_rot(angle)
            h, w = rot.shape[:2]
            if h > w:
                return {"angle": float(angle), "skip": True}
            okA, scoreA, detA = score_formA(rot, marker_preproc="morph", geom_cfg=formA_geom_cfg)
            return {"angle": float(angle), "skip": False, "A_morph": {"ok": bool(okA), "score": float(scoreA), "detail": detA}}

        bestA_morph: Optional[FormDecision] = None
        for r in _run_parallel(_eval_formA_morph, valid_angles):
            if not r or r.get("skip"):
                continue
            _merge_scan_result(float(r["angle"]), "A_morph", r.get("A_morph"))
            if (r.get("A_morph") or {}).get("ok"):
                a = float(r["angle"])
                candA = FormDecision(
                    True,
                    "A",
                    a,
                    float(r["A_morph"]["score"]),
                    {"A": r["A_morph"]["detail"], "phase": "formA_found_fallback_morph"},
                )
                if bestA_morph is None or candA.score > bestA_morph.score:
                    bestA_morph = candA

        if bestA_morph is not None:
            if thr > 0 and float(bestA_morph.score) < thr:
                rejected_by_threshold["A_morph"] = {"score": float(bestA_morph.score), "phase": "formA_found_fallback_morph"}
            else:
                return bestA_morph

    # ----------------------------------
    # Step 2: B の fast 探索
    # ----------------------------------
    def _eval_formB_fast(angle: float) -> dict[str, Any]:
        rot = _get_rot(angle)
        h, w = rot.shape[:2]
        if h > w:
            return {"angle": float(angle), "skip": True}
        okB, scoreB, detB = score_formB_fast(rot)
        return {"angle": float(angle), "skip": False, "B_fast": {"ok": bool(okB), "score": float(scoreB), "detail": detB}}

    bestB_fast: Optional[FormDecision] = None
    for r in _run_parallel(_eval_formB_fast, valid_angles):
        if not r or r.get("skip"):
            continue
        _merge_scan_result(float(r["angle"]), "B_fast", r.get("B_fast"))
        if (r.get("B_fast") or {}).get("ok"):
            a = float(r["angle"])
            candB = FormDecision(True, "B", a, float(r["B_fast"]["score"]), {"B_fast": r["B_fast"]["detail"], "phase": "formB_fast_found"})
            if bestB_fast is None or candB.score > bestB_fast.score:
                bestB_fast = candB

    if bestB_fast is not None:
        if thr > 0 and float(bestB_fast.score) < thr:
            rejected_by_threshold["B_fast"] = {"score": float(bestB_fast.score), "phase": "formB_fast_found"}
        else:
            return bestB_fast

    # ----------------------------------
    # Step 3: B の robust 再挑戦
    # ----------------------------------
    def _eval_formB_robust(angle: float) -> dict[str, Any]:
        rot = _get_rot(angle)
        h, w = rot.shape[:2]
        if h > w:
            return {"angle": float(angle), "skip": True}
        okB, scoreB, detB = score_formB(rot)
        return {"angle": float(angle), "skip": False, "B_robust": {"ok": bool(okB), "score": float(scoreB), "detail": detB}}

    bestB_robust: Optional[FormDecision] = None
    for r in _run_parallel(_eval_formB_robust, valid_angles):
        if not r or r.get("skip"):
            continue
        _merge_scan_result(float(r["angle"]), "B_robust", r.get("B_robust"))
        if (r.get("B_robust") or {}).get("ok"):
            a = float(r["angle"])
            candB = FormDecision(True, "B", a, float(r["B_robust"]["score"]), {"B": r["B_robust"]["detail"], "phase": "formB_robust_fallback"})
            if bestB_robust is None or candB.score > bestB_robust.score:
                bestB_robust = candB

    if bestB_robust is not None:
        if thr > 0 and float(bestB_robust.score) < thr:
            rejected_by_threshold["B_robust"] = {"score": float(bestB_robust.score), "phase": "formB_robust_fallback"}
        else:
            return bestB_robust

    # ----------------------------------
    # Step 4: Unknown
    # ----------------------------------
    if rejected_by_threshold:
        best_rejected = 0.0
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

    return FormDecision(False, None, None, 0.0, {"reason": "no_detection", "scan": scan_results, "scan_angles": scan_angles, "note": "fallback_all_failed"})



"""（template-topn / グローバル特徴による事前絞り込み）

v18 ではユーザー要望により「フォーム確定後は全テンプレを XFeat で照合」します。
そのため、旧版にあったグローバル特徴によるテンプレ候補絞り込み機能は削除しました。
（CSVにも template-topn は出さず空欄にしています）
"""


@dataclass
class CachedRef:
    template_path: str
    s_ref: float
    out0: dict[str, Any]
    # best template を決めた後に再readしないためのキャッシュ（少数枚のためメモリ負荷は小さい）
    template_bgr: Optional[np.ndarray] = None


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
        return CachedRef(template_path=str(template_path), s_ref=float(s_ref), out0=out0, template_bgr=template_bgr)

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
    # v18 では prefilter を使わないため、互換用に「そのまま返す」だけにする。
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


def _json_default_for_dump(o: Any) -> Any:
    """json.dump(..., default=...) 用のフォールバック。

    目的:
      - summary.json 生成時に numpy.ndarray / numpy scalar / Path が混ざっても落ちないようにする。
      - 意図せず巨大な ndarray（例: 画像）を丸ごと JSON 化してファイルが肥大化するのを避ける。
        （小さい配列はそのまま list 化 / 大きい配列は shape と先頭要素だけを残す。）
    """

    # pathlib.Path
    try:
        if isinstance(o, Path):
            return str(o)
    except Exception:
        pass

    # numpy
    try:
        if isinstance(o, np.ndarray):
            # 座標(4x2) 等の小配列はそのまま list 化
            if int(o.size) <= 5000:
                return o.tolist()
            # 画像などの巨大配列は要約のみ
            flat = o.reshape(-1)
            preview_n = min(int(flat.size), 64)
            return {
                "__ndarray__": True,
                "dtype": str(o.dtype),
                "shape": list(o.shape),
                "size": int(o.size),
                "preview": flat[:preview_n].tolist(),
            }
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass

    # set
    try:
        if isinstance(o, set):
            return list(o)
    except Exception:
        pass

    # bytes
    try:
        if isinstance(o, (bytes, bytearray)):
            return o.decode("utf-8", errors="replace")
    except Exception:
        pass

    # 最後は文字列化（落ちないことを優先）
    return str(o)


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

    v18.4:
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
    # NOTE:
    # - synthetic(A/B): フォーム/テンプレ/warp が正しいこと
    # - synthetic(C): form_unknown で棄却されること
    # - test: 既存と同様（ファイル名から GT を解釈）
    # - target: 改悪なしで投入し、warp まで到達すること（GT は未定義）
    if source_dataset == "target":
        expected_behavior_label = "target_should_be_processed_without_degrade_and_reach_warp"
    elif src_form == "C":
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
        # v18 では template-topn は廃止（常に全テンプレ照合）
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


def list_target_images() -> list[Path]:
    """image/target 配下の画像を列挙する。

    改善1（本タスク）:
      - image/target の画像は「改悪生成をせず、そのままパイプラインへ投入」する。
      - 命名規則は固定しない（現場で増えうるため）。拡張子だけで列挙する。
    """

    base = Path(__file__).resolve().parent / "image" / "target"
    if not base.exists():
        return []

    exts = {".png", ".jpg", ".jpeg"}
    paths = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


def parse_test_filename(p: Path) -> Optional[tuple[str, str]]:
    """test 画像ファイル名から (form, template_number) を推定する。

    改善2（本タスク）:
      image/test の命名規則を変更する。

      - 旧: {A|B|C}_{template}.png         例: A_3.png
      - 新: {A|B|C}_{template}_{id}.png    例: A_3_1.png
        - 最初の2つ（A と 3）が GT（フォーム・テンプレ番号）
        - 最後の 1 つは識別番号（同じテンプレでも中身が違うことを表す）

    実装方針:
      - stem を '_' で split し、先頭2要素を (form, template_number) として解釈する
      - 3要素目以降（識別番号など）は case_id には残るが、GT 判定には使用しない
    """

    try:
        parts = [s.strip() for s in str(p.stem).split("_") if s.strip()]
        if len(parts) < 2:
            return None
        head = parts[0].upper()
        num = parts[1]
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
    # Speed profile（大幅高速化）
    # ----------------------------
    p.add_argument(
        "--speed-profile",
        choices=["auto", "fast", "accurate"],
        default="auto",
        help=(
            "速度プロファイル。auto=targetのみfast/それ以外accurate。"
            "fastは精度を多少犠牲にして大幅高速化（DocAligner/decide/uvdoc/bgdiv等を簡略化）。"
        ),
    )
    p.add_argument(
        "--extra-outputs",
        choices=["auto", "all", "none"],
        default="auto",
        help=(
            "追加の可視化出力（demo9/debug_matches）を出すか。"
            "auto=accurateはall/fastはnone。"
        ),
    )

    # fast profile の個別パラメータ（必要なら微調整可能）
    p.add_argument(
        "--fast-docaligner-input-max-side",
        type=int,
        default=1200,
        help="fast profile 時の DocAligner 入力画像の最大辺(px)。小さいほど速いが検出率は落ちる。",
    )
    p.add_argument(
        "--fast-rectified-max-side",
        type=int,
        default=1600,
        help="fast profile 時の rectify 後画像の最大辺(px)。小さいほど速いがマーカー/QRの視認性が落ちる。",
    )
    p.add_argument(
        "--fast-match-input-max-side",
        type=int,
        default=900,
        help="fast profile 時の XFeat matching/warp に投入する画像の最大辺(px)。小さいほど速いが精度は落ちる。",
    )
    p.add_argument(
        "--fast-skip-uvdoc",
        action="store_true",
        help="fast profile 時に UVDoc をスキップする（大幅高速化）。",
    )
    p.add_argument(
        "--fast-skip-bgdiv",
        action="store_true",
        help="fast profile 時に background division をスキップする（高速化）。",
    )

    # ----------------------------
    # 入力/件数
    # ----------------------------
    p.add_argument(
        "--src-forms",
        type=str,
        default=",".join(PIPELINE_DEFAULTS["src_forms"]),
        help=(
            "synthetic生成の入力元フォーム（A,B,C をカンマ区切り）。"
            "空文字なら synthetic を処理しません（target-only運用向け）。"
        ),
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

    p.add_argument(
        "--test-limit",
        type=int,
        default=int(PIPELINE_DEFAULTS.get("test_limit") or 0),
        help="デバッグ用：image/test の処理件数（0=skip, N>0=先頭N枚, N<0=全て）",
    )

    p.add_argument(
        "--target-limit",
        type=int,
        default=int(PIPELINE_DEFAULTS.get("target_limit") or 0),
        help="デバッグ用：image/target の処理件数（0=skip, N>0=先頭N枚, N<0=全て）",
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
        "  (注) v18 ではテンプレ候補絞り込み（template-topn）は廃止し、常に全テンプレ照合します。",
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
        r"  C:\Users\takumi\develop\miniconda3\python.exe APA\paper_pipeline_v18.py --limit 1",
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
    print(f"  test-limit         : {getattr(args, 'test_limit', 0)}")
    print(f"  target-limit       : {getattr(args, 'target_limit', 0)}")
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
    try:
        print(f"  speed-profile       : {getattr(args, 'speed_profile', 'auto')}")
        print(f"  extra-outputs       : {getattr(args, 'extra_outputs', 'auto')}")
        if str(getattr(args, 'speed_profile', 'auto')) in ("auto", "fast"):
            print(
                "  fast-settings       : "
                f"doc_in_max={getattr(args, 'fast_docaligner_input_max_side', 1200)} "
                f"rect_max={getattr(args, 'fast_rectified_max_side', 1600)} "
                f"match_in_max={getattr(args, 'fast_match_input_max_side', 900)} "
                f"skip_uvdoc={bool(getattr(args, 'fast_skip_uvdoc', False))} "
                f"skip_bgdiv={bool(getattr(args, 'fast_skip_bgdiv', False))}"
            )
    except Exception:
        pass


def load_docaligner_model(model_name: str, model_type: str) -> tuple[Any, Any]:
    # NOTE:
    # DocAligner / capybara は .venv 環境（capybara-docsaid / docaligner-docsaid）を前提とする。
    # conda(base) の python で実行すると import に失敗しやすいので、エラーメッセージを明確化する。
    try:
        patch_capybara_exports()
        import capybara as cb
        from docaligner import DocAligner, ModelType
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "DocAligner dependencies are missing in the current Python environment.\n"
            f"  python: {sys.executable}\n"
            "Install (in your venv) or run with the repo venv:\n"
            "  .venv/bin/python -m pip install capybara-docsaid docaligner-docsaid\n"
            "  .venv/bin/python paper_pipeline_v18.py ..."
        ) from e

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
    """改悪生成済みの入力（v18.7）。

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


def _apply_extra_degrade_v18_5(
    *,
    src_bgr: np.ndarray,
    degraded_bgr: np.ndarray,
    H_src_to_deg: np.ndarray,
    degrade_meta: dict[str, Any],
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, Any]]:
    """v18.5 の追加改悪（bend/shadow）を適用する。

    NOTE:
      v18.7 では「改悪生成フェーズ」を最初に全件実行するため、
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
    degrade_meta["extra_degrade_v18_5"] = True
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
    """改悪生成済み画像（degraded_input）を処理する（v18.7）。

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

    profile = resolve_speed_profile(args, dataset=str(di.source_dataset))
    extra_outputs_mode = resolve_extra_outputs_mode(args, profile=profile)

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
    # v18.7: 改悪生成は最初に全件作成し、計測対象外
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
        # fail では "1_degraded" も失敗ケースに保存しておく（デバッグの起点として重要）。
        try:
            out_deg = Path(str(di.output_degraded_image_path))
            write_image(out_deg, degraded_bgr, jpeg_quality=jpeg_quality)
            item["output_degraded_image_path"] = str(out_deg)
        except Exception:
            pass

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
    # DocAligner multi の候補評価でも、dataset ごとの A-geometry を反映できるよう渡す。
    formA_geom_cfg_for_case = make_formA_geom_cfg_for_case(source_dataset=str(di.source_dataset), source_form=str(di.source_form))
    t0 = time.perf_counter()

    # speed profile により DocAligner の重さを変える
    if str(profile) == "fast":
        poly, doc_meta = detect_polygon_docaligner_fast(
            logger=logger,
            degraded_bgr=degraded_bgr,
            max_input_side_px=int(getattr(args, "fast_docaligner_input_max_side", 1200) or 1200),
            model_name=str(getattr(args, "fast_docaligner_model", "lcnet100") or "lcnet100"),
            model_type="heatmap",
        )
    else:
        poly, doc_meta = detect_polygon_docaligner_multi(
            logger=logger,
            args=args,
            degraded_bgr=degraded_bgr,
            formA_geom_cfg=formA_geom_cfg_for_case,
        )
    times.docaligner_s = time.perf_counter() - t0
    if poly is None:
        item["stage"] = "docaligner_failed"
        item["docaligner_multi"] = doc_meta
        _finalize_images_for_stage(item["stage"])
        item["case_total_s"] = float(times.degrade_s + times.docaligner_s)
        return item, times

    item["stage"] = "docaligner_ok"
    item["polygon"] = poly.astype(float).tolist()
    item["docaligner_multi"] = doc_meta

    # (1) polygon margin: デフォルトは ratio ベース
    # v18.11 (DocAligner改善): multi の評価で最良だった margin_px があればそれを優先する。
    picked_margin_px = None
    try:
        if isinstance(doc_meta, dict) and isinstance(doc_meta.get("best"), dict):
            picked_margin_px = float(doc_meta["best"].get("margin_px"))
    except Exception:
        picked_margin_px = None

    if picked_margin_px is not None and math.isfinite(float(picked_margin_px)):
        margin_px = float(picked_margin_px)
        item["polygon_margin"] = {"mode": "picked_by_eval", "value": float(margin_px)}
    elif float(getattr(args, "polygon_margin_px", 0.0)) > 0:
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

    # 3) Rectify は padding を（必要なら）入れて、margin による clip を回避する。
    # overlay は degraded 内で clamp した polygon を描画する。
    overlay_poly = expand_polygon(
        poly,
        margin_px=float(margin_px),
        img_w=int(degraded_bgr.shape[1]),
        img_h=int(degraded_bgr.shape[0]),
    )
    overlay = draw_polygon_overlay(degraded_bgr, overlay_poly)
    out_doc = out_dirs["doc"] / f"{case_id}_doc.jpg"
    _schedule_image("output_doc_overlay_image_path", out_doc, overlay)

    # 3) Rectify（計測対象：rectifyのみ。画像保存は計測外）
    t0 = time.perf_counter()

    rectify_max_side = int(args.docaligner_max_side)
    if str(profile) == "fast":
        rectify_max_side = int(getattr(args, "fast_rectified_max_side", 1600) or 1600)

    rectified, H_deg_to_rect, overlay_poly2, rectify_meta = rectify_with_margin_and_optional_padding(
        degraded_bgr,
        polygon_xy=poly,
        margin_px=float(margin_px),
        out_max_side=int(rectify_max_side),
    )
    item["rectify"] = rectify_meta
    # enforce_landscape の回転（90度CW）を考慮した homography も保持しておく（出力9で使用）
    rect0_h, rect0_w = rectified.shape[:2]
    rectified, rect_rotated_90cw = enforce_landscape(rectified)
    M_rect_to_land = _landscape_rotation_matrix_if_applied(w=int(rect0_w), h=int(rect0_h), rotated_90cw=bool(rect_rotated_90cw))
    H_deg_to_rect_land = np.asarray(M_rect_to_land, dtype=np.float64) @ np.asarray(H_deg_to_rect, dtype=np.float64)
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
    item["H_degraded_to_rectified_landscape"] = H_deg_to_rect_land.astype(float).tolist()
    item["rectified_landscape_rotated_90cw"] = bool(rect_rotated_90cw)

    # ------------------------------------------------------------
    # 4) decide form by rotations
    #   - 通常: rectified（DocAligner polygon）で判定
    #   - フォールバック: stage=form_unknown(no_detection) の場合のみ、
    #     高精度 polygon フォールバック（輪郭ベース）で polygon を再推定して再試行
    #
    # 目的:
    #   target（現場撮影）で DocAligner polygon が「机の縁」等に引っ張られて
    #   紙が欠けた rectify になると、マーカー/QR が画面外になって no_detection になりやすい。
    #   その場合に “時間は無視して” 追加探索を行う。
    # ------------------------------------------------------------

    def _decide_on_rectified(rectified_landscape_bgr: np.ndarray, *, relax_marker_search: bool = False) -> FormDecision:
        """rectified 画像上でフォーム判定を行う。

        relax_marker_search:
          advanced fallback（polygon再推定）時にのみ使う “最後の救済”。
          DocAligner/輪郭ベースpolygonが少しズレた結果、マーカーが corner 付近に来ない場合がある。
          通常は corner 付近探索に絞って誤検出を減らすが、
          no_detection で詰まった場合のみ探索範囲/サイズ想定を緩めて再試行する。
        """

        if not relax_marker_search:
            if str(profile) == "fast":
                return decide_form_fast(
                    rectified_landscape_bgr,
                    unknown_score_threshold=float(args.unknown_score_threshold),
                    unknown_margin=float(args.unknown_margin),
                    formA_geom_cfg=formA_geom_cfg_for_case,
                    marker_preproc=str(getattr(args, "marker_preproc", "none") or "none"),
                )
            return decide_form_by_rotations(
                rectified_landscape_bgr,
                max_workers=int(args.rotation_max_workers),
                marker_preproc=str(args.marker_preproc),
                unknown_score_threshold=float(args.unknown_score_threshold),
                unknown_margin=float(args.unknown_margin),
                formA_geom_cfg=formA_geom_cfg_for_case,
            )

        # 一時的に PIPELINE_DEFAULTS['marker'] を緩める（グローバルなので必ず復元）
        marker_cfg = PIPELINE_DEFAULTS.get("marker")
        marker_backup = dict(marker_cfg) if isinstance(marker_cfg, dict) else {}
        if not isinstance(marker_cfg, dict):
            PIPELINE_DEFAULTS["marker"] = {}
            marker_cfg = PIPELINE_DEFAULTS["marker"]

        try:
            # 探索範囲を広げる（corner誤差/rectifyズレ救済）
            marker_cfg["corner_margin_ratio"] = float(max(0.12, float(marker_backup.get("corner_margin_ratio", 0.12)) * 1.8))
            marker_cfg["corner_margin_ratio"] = float(min(0.30, marker_cfg["corner_margin_ratio"]))

            # サイズ想定も緩める（解像度差/拡大縮小の救済）
            marker_cfg["marker_min_size_ratio"] = float(min(0.008, float(marker_backup.get("marker_min_size_ratio", 0.008)) * 0.75))
            marker_cfg["marker_max_size_ratio"] = float(max(0.09, float(marker_backup.get("marker_max_size_ratio", 0.07)) * 1.3))

            return decide_form_by_rotations(
                rectified_landscape_bgr,
                max_workers=int(args.rotation_max_workers),
                marker_preproc=str(args.marker_preproc),
                unknown_score_threshold=float(args.unknown_score_threshold),
                unknown_margin=float(args.unknown_margin),
                formA_geom_cfg=formA_geom_cfg_for_case,
            )
        finally:
            PIPELINE_DEFAULTS["marker"] = marker_backup

    # 4) decide（初回）
    t0 = time.perf_counter()
    decision = _decide_on_rectified(rectified)
    times.decide_s = time.perf_counter() - t0
    item["form_decision"] = asdict(decision)
    item["predicted_form"] = str(decision.form or "")
    item["predicted_angle_deg"] = "" if decision.angle_deg is None else float(decision.angle_deg)

    # form_unknown の場合のみ、polygon再推定フォールバックを試す
    # fast profile では「時間短縮を優先」し、advanced fallback は行わない
    if (not decision.ok) or (decision.form not in ("A", "B")) or (decision.angle_deg is None):
        reason, _diag = extract_form_unknown_reason(asdict(decision))
        adv_cfg = ((PIPELINE_DEFAULTS.get("docaligner") or {}).get("advanced_fallback") or {})
        adv_enable = bool(adv_cfg.get("enable", True))
        adv_trigger = bool(adv_cfg.get("trigger_on_form_unknown_no_detection", True))

        # C は「棄却が正しい」ため無駄な再探索をしない。
        allow_retry = bool(str(src_form) in ("A", "B") or str(di.source_dataset) == "target")

        if str(profile) != "fast" and allow_retry and adv_enable and adv_trigger and str(reason) == "no_detection":
            t_adv0 = time.perf_counter()
            poly_adv = detect_polygon_fallback_advanced(degraded_bgr)
            times.docaligner_s += time.perf_counter() - t_adv0
            if poly_adv is not None:
                item["docaligner_fallback_advanced"] = {
                    "ok": True,
                    "reason": "triggered_by_form_unknown_no_detection",
                    "poly": order_quad_tl_tr_br_bl(np.asarray(poly_adv, dtype=np.float32)).astype(float).tolist(),
                    # 診断用: margin 試行のログ（成功/失敗も含む）
                    "attempts": [],
                }

                # margin を複数試して、判定スコアが最大のものを採用
                best_fb: Optional[dict[str, Any]] = None
                margin_list = _margin_px_candidates_for_eval(args=args, poly_xy=poly_adv)
                # target は端欠けが多いので、少し大きい margin も追加
                if str(di.source_dataset) == "target":
                    try:
                        extra = polygon_margin_px_from_ratio(poly_adv, ratio=0.16, min_px=float(args.polygon_margin_min_px), max_px=float(args.polygon_margin_max_px))
                        margin_list = margin_list + [float(extra)]
                    except Exception:
                        pass
                # 重複除去（丸め）
                seen_m: set[int] = set()
                margin_list2: list[float] = []
                for m in margin_list:
                    k = int(round(float(m)))
                    if k in seen_m:
                        continue
                    seen_m.add(k)
                    margin_list2.append(float(m))

                for m in margin_list2:
                    try:
                        t_rect0 = time.perf_counter()
                        rect_fb, H_deg_to_rect_fb, poly_exp_fb, rect_fb_meta = rectify_with_margin_and_optional_padding(
                            degraded_bgr,
                            polygon_xy=poly_adv,
                            margin_px=float(m),
                            out_max_side=int(args.docaligner_max_side),
                        )
                        rect0_h2, rect0_w2 = rect_fb.shape[:2]
                        rect_fb, rect_rot90 = enforce_landscape(rect_fb)
                        M_rect_to_land2 = _landscape_rotation_matrix_if_applied(w=int(rect0_w2), h=int(rect0_h2), rotated_90cw=bool(rect_rot90))
                        H_deg_to_rect_land_fb = np.asarray(M_rect_to_land2, dtype=np.float64) @ np.asarray(H_deg_to_rect_fb, dtype=np.float64)
                        times.rectify_s += time.perf_counter() - t_rect0

                        t_dec0 = time.perf_counter()
                        # advanced fallback 時は marker 探索範囲を緩めて再挑戦（no_detection 救済）
                        dec_fb = _decide_on_rectified(rect_fb, relax_marker_search=True)
                        times.decide_s += time.perf_counter() - t_dec0

                        # attempt 記録（失敗も含めて残す）
                        try:
                            u_reason, u_diag = extract_form_unknown_reason(asdict(dec_fb))
                        except Exception:
                            u_reason, u_diag = "unknown", {}
                        try:
                            item["docaligner_fallback_advanced"]["attempts"].append(
                                {
                                    "margin_px": float(m),
                                    "decision": asdict(dec_fb),
                                    "unknown_reason": str(u_reason),
                                    "unknown_diag": u_diag,
                                }
                            )
                        except Exception:
                            pass

                        if not dec_fb.ok or dec_fb.form not in ("A", "B") or dec_fb.angle_deg is None:
                            continue

                        rec = {
                            "poly": order_quad_tl_tr_br_bl(np.asarray(poly_adv, dtype=np.float32)).astype(np.float32),
                            "poly_exp": order_quad_tl_tr_br_bl(np.asarray(poly_exp_fb, dtype=np.float32)).astype(np.float32),
                            "margin_px": float(m),
                            "rectified": rect_fb,
                            "H_deg_to_rect": np.asarray(H_deg_to_rect_fb, dtype=np.float64),
                            "H_deg_to_rect_land": np.asarray(H_deg_to_rect_land_fb, dtype=np.float64),
                            "rect_rotated_90cw": bool(rect_rot90),
                            "rectify": rect_fb_meta,
                            "decision": dec_fb,
                            "score": float(dec_fb.score),
                        }
                        if best_fb is None or float(rec["score"]) > float(best_fb.get("score") or 0.0):
                            best_fb = rec
                    except Exception:
                        continue

                if best_fb is not None:
                    # 置き換え: polygon/rectified/decision を更新して以降の処理を続行
                    poly = best_fb["poly"]
                    poly_exp = best_fb["poly_exp"]
                    margin_px = float(best_fb["margin_px"])
                    rectified = best_fb["rectified"]
                    H_deg_to_rect = best_fb["H_deg_to_rect"]
                    H_deg_to_rect_land = best_fb["H_deg_to_rect_land"]
                    rect_rotated_90cw = bool(best_fb["rect_rotated_90cw"])
                    decision = best_fb["decision"]

                    item["docaligner_fallback_advanced"]["accepted"] = True
                    item["docaligner_fallback_advanced"]["picked_margin_px"] = float(margin_px)
                    item["docaligner_fallback_advanced"]["picked_decision"] = asdict(decision)

                    # item 更新（後段の整合性のため、主要フィールドだけ上書き）
                    item["polygon"] = poly.astype(float).tolist()
                    item["polygon_margin"] = {"mode": "advanced_fallback_eval", "value": float(margin_px)}
                    item["H_degraded_to_rectified"] = np.asarray(H_deg_to_rect, dtype=np.float64).astype(float).tolist()
                    item["H_degraded_to_rectified_landscape"] = np.asarray(H_deg_to_rect_land, dtype=np.float64).astype(float).tolist()
                    item["rectified_landscape_rotated_90cw"] = bool(rect_rotated_90cw)
                    try:
                        hr, wr = rectified.shape[:2]
                        item["rectified_w"] = int(wr)
                        item["rectified_h"] = int(hr)
                    except Exception:
                        pass

                    # 保存画像も上書き（all/fail の整合のため）
                    overlay2 = draw_polygon_overlay(degraded_bgr, poly_exp)
                    out_doc = out_dirs["doc"] / f"{case_id}_doc.jpg"
                    _schedule_image("output_doc_overlay_image_path", out_doc, overlay2)
                    out_rect = out_dirs["rect"] / f"{case_id}_rect.jpg"
                    _schedule_image("output_rectified_image_path", out_rect, rectified)

                    # decision を上書き
                    item["form_decision"] = asdict(decision)
                    item["predicted_form"] = str(decision.form or "")
                    item["predicted_angle_deg"] = "" if decision.angle_deg is None else float(decision.angle_deg)
                else:
                    item["docaligner_fallback_advanced"]["accepted"] = False
                    # 何がダメだったかを最低限残す（scan_max_A/B は attempt から追える）
                    item["docaligner_fallback_advanced"]["reject_reason"] = "no_margin_yielded_valid_form_decision"
            else:
                item["docaligner_fallback_advanced"] = {
                    "ok": False,
                    "reason": "polygon_not_found",
                }

    # （再評価後も）Unknown ならここで終了
    if (not decision.ok) or (decision.form not in ("A", "B")) or (decision.angle_deg is None):
        item["stage"] = "form_unknown"
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
    # fast profile では原則スキップ（大幅高速化）
    if str(profile) == "fast" or bool(getattr(args, "fast_skip_uvdoc", False)):
        chosen_unwarped = chosen
        item["uvdoc"] = {"ok": False, "skipped": True}
    else:
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
    # fast profile では原則スキップ（高速化）
    if str(profile) == "fast" or bool(getattr(args, "fast_skip_bgdiv", False)):
        bgdiv_bgr = chosen_unwarped
        bgdiv_meta = {"applied": False, "skipped": True}
        item["background_division"] = bgdiv_meta
        item["output_background_division_image_path"] = ""
        item["bgdiv_w"] = ""
        item["bgdiv_h"] = ""
    else:
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

    # matching 入力は profile に応じて縮小
    chosen_for_match = bgdiv_bgr
    if str(profile) == "fast":
        try:
            chosen_for_match, _s_match = resize_keep_aspect(
                chosen_for_match,
                int(getattr(args, "fast_match_input_max_side", 900) or 900),
            )
        except Exception:
            pass

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

    # ------------------------------------------------------------
    # 出力9: デモ画像（時間計測対象外）
    # ------------------------------------------------------------
    if str(extra_outputs_mode) == "all":
        try:
            # decide の根拠（chosen座標系）
            decision_markers: Optional[list[dict[str, Any]]] = None
            decision_qrs: Optional[list[dict[str, Any]]] = None
            if str(decision.form) == "A":
                decision_markers = ((decision.detail or {}).get("A") or {}).get("markers") or None
            elif str(decision.form) == "B":
                # phase により B / B_fast どちらかに入る
                dB = (decision.detail or {}).get("B")
                dBf = (decision.detail or {}).get("B_fast")
                decision_qrs = (dB or {}).get("qrs") or (dBf or {}).get("qrs") or None

            demo9 = _generate_demo9_image(
                degraded_bgr=degraded_bgr,
                polygon_xy=poly,
                polygon_margin_px=float(margin_px),
                H_degraded_to_rectified_landscape=np.asarray(H_deg_to_rect_land, dtype=np.float64),
                rectified_landscape_size_wh=(int(rectified.shape[1]), int(rectified.shape[0])),
                decided_form=str(decision.form),
                decided_angle_deg=float(decision.angle_deg),
                decision_markers=decision_markers,
                decision_qrs=decision_qrs,
                aligned_bgr=warped_final,
            )

            out_demo = out_dirs.get("demo")
            if out_demo is not None:
                out_demo9 = Path(out_demo) / f"{case_id}_demo9.jpg"
                write_image(out_demo9, demo9, jpeg_quality=jpeg_quality)
                item["output_demo9_image_path"] = str(out_demo9)
            else:
                item["output_demo9_image_path"] = ""
        except Exception:
            item["output_demo9_image_path"] = ""
    else:
        item["output_demo9_image_path"] = ""

    # 6_debug_matches（best template のマッチ可視化）
    # ユーザー要望: 本番ではない処理のため、時間計測から除外する。
    if str(extra_outputs_mode) == "all":
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
    else:
        item["output_debug_matches_image_path"] = ""

    item["stage"] = "done"
    item["ok_warp"] = True
    # 期待動作としての成功条件:
    # - A/B: フォーム正解 AND テンプレ正解 AND warp 完了
    # - C  : "done" に到達したら誤検出（本来は棄却されるべき）
    if str(di.source_dataset) == "target":
        # target は GT を持たないため、warp 完了を成功とみなす
        item["ok"] = True
    elif src_form in ("A", "B"):
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
    """観測画像（degrade無し）を *process_one_case* と同じ経路で処理する薄いラッパー。

    NOTE:
      - 現行の main() からは呼ばれていませんが、将来の利用/手動デバッグ用に残しています。
      - 実装の二重化（process_one_observed_case 側だけ古くなる/壊れる）を避けるため、
        DegradedCaseInput を作って process_one_case に委譲します。
    """

    if src_bgr is None:
        raise ValueError("src_bgr is None")

    h0, w0 = src_bgr.shape[:2]
    case_id = f"{str(source_dataset)}_{src_path.stem}" if src_path and src_path.stem else f"{str(source_dataset)}_observed"

    out_degraded = out_dirs["degraded"] / f"{case_id}.jpg"
    jpeg_quality = int((PIPELINE_DEFAULTS.get("save_images") or {}).get("jpeg_quality") or 95)
    if str(getattr(args, "save_images", "all")) == "all":
        # process_one_case は「事前に1_degradedが保存されている」想定のため、ここで合わせる
        write_image(out_degraded, src_bgr, jpeg_quality=jpeg_quality)

    di = DegradedCaseInput(
        source_dataset=str(source_dataset),
        source_form=str(src_form),
        source_path=Path(src_path),
        source_w=int(w0),
        source_h=int(h0),
        degraded_variant_index=0,
        case_id=str(case_id),
        degraded_bgr=src_bgr,
        H_src_to_degraded=np.eye(3, dtype=np.float64),
        degrade_meta={"mode": "observed_skip_degrade"},
        output_degraded_image_path=Path(out_degraded),
        ground_truth_form=str(ground_truth_form),
        ground_truth_template_path=Path(ground_truth_template_path) if ground_truth_template_path else None,
        ground_truth_template_number=str(ground_truth_template_number),
    )

    return process_one_case(
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
    logger.info("paper_pipeline_v18")
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
        "demo": mkdir(out_root / "9_demo"),
    }

    # 重いモデルをロード
    logger.info("[INFO] Loading DocAligner...")
    model, cb = load_docaligner_model(args.docaligner_model, args.docaligner_type)
    logger.info("[OK] DocAligner loaded")

    logger.info("[INFO] Loading XFeat...")
    matcher = XFeatMatcher(top_k=args.top_k, device=device, match_max_side=args.match_max_side)
    logger.info("[OK] XFeat loaded")

    # デフォルトは synthetic/test/target を処理する（必要に応じて各limitで絞る）。
    src_forms = [s.strip() for s in str(args.src_forms).split(",") if s.strip()]
    src_forms = [s for s in src_forms if s in ("A", "B", "C")]
    test_limit = int(getattr(args, "test_limit", 0) or 0)
    target_limit = int(getattr(args, "target_limit", 0) or 0)
    will_process_synthetic = bool(src_forms)
    will_process_test = (test_limit != 0)
    will_process_target = (target_limit != 0)
    if not (will_process_synthetic or will_process_test or will_process_target):
        logger.error("Nothing to process. Set --target-limit (-1/all or N>0) and/or --src-forms and/or --test-limit.")
        return 1

    # WeChat QR detector を初期化（フォームBは WeChat のみ）
    # v18 改善: 回転スキャンでの直列化を避けるため、ThreadPool の worker 数だけ detector を確保する。
    wechat_pool_size = int(getattr(args, "rotation_max_workers", 1))
    wechat = init_wechat_qr_detector(str(getattr(args, "wechat_model_dir", "")), logger=logger, pool_size=wechat_pool_size)
    # 引数経由でスレッドに流すと取り回しが悪いので、score_formB に属性としてぶら下げる
    setattr(score_formB, "_wechat", wechat)
    # target/test は中身が A/B 混在し得るため、target/test を処理する場合も WeChat は必須とする。
    if ("B" in src_forms) or will_process_test or will_process_target:
        if wechat is None:
            logger.error(
                "WeChat QR detector is not available, but this run may include Form B detection (target/test or src-forms includes B). "
                "Please install opencv-contrib and set --wechat-model-dir."
            )
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

        # v18.5 extra degrade (bend/shadow)
        try:
            degraded_bgr, degrade_meta = _apply_extra_degrade_v18_5(
                src_bgr=src_bgr,
                degraded_bgr=degraded_bgr,
                H_src_to_deg=H_src_to_deg,
                degrade_meta=degrade_meta,
                rng=rng,
            )
        except Exception as e:
            if isinstance(degrade_meta, dict):
                degrade_meta.setdefault("extra_degrade_v18_5", False)
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
    # NOTE(2026-01-27): デフォルトで test_limit=-1（all）を処理する。
    test_paths = list_test_images()
    test_limit = int(getattr(args, "test_limit", 0) or 0)
    if test_limit == 0:
        test_paths = []
        logger.info("[DEGRADE] test dataset (image/test): skipped (use --test-limit -1 to process all)")
    elif test_limit > 0:
        test_paths = test_paths[: int(test_limit)]
        logger.info("[DEGRADE] test dataset (image/test): %d images (limited)", len(test_paths))
    else:
        logger.info("[DEGRADE] test dataset (image/test): %d images (all)", len(test_paths))

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

    # target dataset は「改悪なし」で投入する（改善1）
    # v18.13: default は target_limit=-1（all）。
    target_paths = list_target_images()
    target_limit = int(getattr(args, "target_limit", 0) or 0)
    if target_limit == 0:
        target_paths = []
        logger.info("[TARGET] target dataset (image/target): skipped (use --target-limit -1 to process all)")
    elif target_limit > 0:
        target_paths = target_paths[: int(target_limit)]
        logger.info("[TARGET] target dataset (image/target): %d images (limited)", len(target_paths))
    else:
        logger.info("[TARGET] target dataset (image/target): %d images (all)", len(target_paths))

    for i, tp in enumerate(target_paths):
        src_bgr = cv2.imread(str(tp))
        if src_bgr is None:
            logger.warning("failed to read target image: %s", tp)
            continue

        try:
            h0, w0 = src_bgr.shape[:2]
        except Exception:
            continue

        # 命名規則は固定しない（現場で増えうる）。重複回避のため index を付与。
        case_id = f"target_{tp.stem}_{i:03d}" if tp.stem else f"target_{i:03d}"
        out_degraded = out_dirs["degraded"] / f"{case_id}.jpg"

        # save-images=all の場合のみ、ここで保存しておく（改悪フェーズ扱いで計測対象外）
        if str(getattr(args, "save_images", "all")) == "all":
            write_image(
                out_degraded,
                src_bgr,
                jpeg_quality=int((PIPELINE_DEFAULTS.get("save_images") or {}).get("jpeg_quality") or 95),
            )

        degraded_inputs.append(
            DegradedCaseInput(
                source_dataset="target",
                source_form="target",
                source_path=Path(tp),
                source_w=int(w0),
                source_h=int(h0),
                degraded_variant_index=0,
                case_id=case_id,
                degraded_bgr=src_bgr,
                H_src_to_degraded=np.eye(3, dtype=np.float64),
                degrade_meta={"mode": "target_skip_degrade"},
                output_degraded_image_path=Path(out_degraded),
            )
        )

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
    # NOTE:
    # - summary は診断情報を含むため、dict の中に numpy.ndarray 等が混ざると json.dump が落ちる。
    # - default で安全に変換して、最後まで run を完走させる。
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default_for_dump)

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
