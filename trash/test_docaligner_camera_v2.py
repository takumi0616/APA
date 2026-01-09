#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DocAligner Real-time Document Detection v2
==========================================

改善版: 手による精度低下を軽減する機能を追加

Features:
1. 時間的平滑化（複数フレームの結果を平均化）
2. モデル選択機能（軽量/高精度）
3. 安定性フィルタ（異常値除去）
4. ROI安定化

Controls:
- 'q' or ESC: 終了
- 's': 画像保存
- 'p': 透視変換プレビュー
- 'm': モデル切り替え（heatmap/point）
- '1': lcnet050 (Point, 最軽量)
- '2': lcnet100 (Heatmap, バランス)
- '3': fastvit_t8 (Heatmap, 軽量)
- '4': fastvit_sa24 (Heatmap, 最高精度)

Created: January 6, 2026
"""

import cv2
import numpy as np
import sys
import os
import warnings
from datetime import datetime
from collections import deque

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suppress warnings
warnings.filterwarnings('ignore')

# Import DocAligner
from docaligner import DocAligner, ModelType
import capybara as cb


class PolygonSmoother:
    """
    時間的平滑化クラス
    複数フレームの結果を平均化して安定性を向上
    """
    def __init__(self, buffer_size=3, outlier_threshold=100):
        self.buffer = deque(maxlen=buffer_size)
        self.outlier_threshold = outlier_threshold
        self.last_valid_polygon = None
        self.no_detect_count = 0
        self.max_no_detect = 10  # この回数検出なしで前回結果をクリア
    
    def update(self, polygon, use_filter=True):
        """新しいポリゴンを追加して平滑化"""
        if polygon is None or len(polygon) < 4:
            self.no_detect_count += 1
            # 一定回数検出なしでリセット
            if self.no_detect_count > self.max_no_detect:
                self.last_valid_polygon = None
                self.buffer.clear()
            elif self.last_valid_polygon is not None:
                return self.last_valid_polygon
            return None
        
        self.no_detect_count = 0
        
        # フィルタが無効の場合は直接返す
        if not use_filter:
            self.last_valid_polygon = polygon
            return polygon
        
        # 異常値チェック（閾値を大きくして手の動きに対応）
        if self.last_valid_polygon is not None:
            diff = np.abs(polygon - self.last_valid_polygon).max()
            if diff > self.outlier_threshold:
                # 大きな変化でも受け入れる（手の位置が変わった可能性）
                self.buffer.clear()  # バッファをクリアして新しい位置に追従
        
        self.buffer.append(polygon.copy())
        self.last_valid_polygon = polygon
        
        if len(self.buffer) < 2:
            return polygon
        
        # 中央値を使用（外れ値に強い）
        stacked = np.stack(list(self.buffer))
        smoothed = np.median(stacked, axis=0)
        
        return smoothed
    
    def reset(self):
        """バッファをリセット"""
        self.buffer.clear()
        self.last_valid_polygon = None
        self.no_detect_count = 0


def set_camera_resolution(cap, width, height, fps):
    """カメラ解像度とFPSを設定"""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    return actual_width, actual_height, actual_fps


def save_frame(frame, save_dir="docaligner_captures_v2"):
    """フレームを保存"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, save_dir)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"doc_{timestamp}.jpg"
    filepath = os.path.join(full_path, filename)
    
    cv2.imwrite(filepath, frame)
    return filepath


def expand_polygon(polygon, margin=20):
    """
    ポリゴンを外側に拡大する
    
    Args:
        polygon: 4点のポリゴン
        margin: 拡大するピクセル数
    
    Returns:
        拡大されたポリゴン
    """
    if polygon is None or len(polygon) < 4:
        return polygon
    
    # 中心点を計算
    center = polygon.mean(axis=0)
    
    # 各点を中心から外側に移動
    expanded = []
    for pt in polygon:
        direction = pt - center
        # 方向を正規化して拡大
        length = np.linalg.norm(direction)
        if length > 0:
            unit_direction = direction / length
            new_pt = pt + unit_direction * margin
        else:
            new_pt = pt
        expanded.append(new_pt)
    
    return np.array(expanded)


def draw_polygon(frame, polygon, color=(0, 255, 0), thickness=3, expand_margin=0):
    """ポリゴンを描画"""
    if polygon is None or len(polygon) < 4:
        return frame
    
    # マージンが指定されている場合は拡大
    if expand_margin > 0:
        polygon = expand_polygon(polygon, expand_margin)
    
    result = frame.copy()
    pts = polygon.astype(np.int32)
    
    # 半透明の塗りつぶし
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.2, result, 0.8, 0, result)
    
    # アウトライン
    cv2.polylines(result, [pts], True, color, thickness)
    
    # コーナーポイント
    for i, pt in enumerate(pts):
        cv2.circle(result, tuple(pt), 8, (0, 0, 255), -1)
        cv2.circle(result, tuple(pt), 10, (255, 255, 255), 2)
        labels = ['TL', 'TR', 'BR', 'BL']
        cv2.putText(result, labels[i], (pt[0] + 15, pt[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result


def draw_info_panel(frame, fps, doc_detected, model_name, smoothing_enabled):
    """情報パネルを描画"""
    result = frame.copy()
    
    # 背景パネル
    cv2.rectangle(result, (5, 5), (400, 130), (0, 0, 0), -1)
    cv2.rectangle(result, (5, 5), (400, 130), (0, 255, 0), 1)
    
    # FPS
    cv2.putText(result, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # 検出状態
    status_color = (0, 255, 0) if doc_detected else (0, 0, 255)
    status_text = "Document: DETECTED" if doc_detected else "Document: NOT FOUND"
    cv2.putText(result, status_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
    
    # モデル情報
    cv2.putText(result, f"Model: {model_name}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # 平滑化状態
    smooth_text = "Smoothing: ON" if smoothing_enabled else "Smoothing: OFF"
    smooth_color = (0, 255, 0) if smoothing_enabled else (100, 100, 100)
    cv2.putText(result, smooth_text, (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, smooth_color, 1)
    
    # キー操作ガイド
    cv2.putText(result, "Keys: q=Exit s=Save m=Model 1-4=Select", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return result


def get_perspective_transform(frame, polygon, output_size=(800, 600)):
    """透視変換を適用"""
    if polygon is None or len(polygon) < 4:
        return None
    
    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(polygon.astype(np.float32), dst)
    warped = cv2.warpPerspective(frame, M, output_size)
    
    return warped


# 利用可能なモデル設定
MODEL_CONFIGS = {
    '1': {'type': ModelType.point, 'cfg': 'lcnet050', 'name': 'lcnet050 (Point, 最軽量)'},
    '2': {'type': ModelType.heatmap, 'cfg': 'lcnet100', 'name': 'lcnet100 (Heatmap, バランス)'},
    '3': {'type': ModelType.heatmap, 'cfg': 'fastvit_t8', 'name': 'fastvit_t8 (Heatmap, 軽量)'},
    '4': {'type': ModelType.heatmap, 'cfg': 'fastvit_sa24', 'name': 'fastvit_sa24 (Heatmap, 最高精度)'},
}


def load_model(key='4'):
    """モデルをロード"""
    config = MODEL_CONFIGS.get(key, MODEL_CONFIGS['4'])
    print(f"Loading model: {config['name']}...")
    model = DocAligner(
        model_type=config['type'],
        model_cfg=config['cfg']
    )
    print(f"[OK] Model loaded: {config['name']}")
    return model, config['name']


def main():
    """メイン関数: リアルタイム書類検出（改善版）"""
    print()
    print("=" * 60)
    print("DocAligner Real-time Document Detection v2")
    print("改善版: 時間的平滑化 + モデル選択")
    print("=" * 60)
    print()
    print(f"OpenCV Version: {cv2.__version__}")
    print()
    
    # 利用可能なモデル表示
    print("利用可能なモデル:")
    for key, config in MODEL_CONFIGS.items():
        print(f"  [{key}] {config['name']}")
    print()
    
    # デフォルトモデルをロード（最高精度）
    current_model_key = '4'
    model, model_name = load_model(current_model_key)
    print()
    
    # カメラを開く
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return 1
    
    width, height, fps = set_camera_resolution(cap, 1280, 720, 30)
    print(f"[OK] Camera opened: {width}x{height} @ {fps:.1f}fps")
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
    
    # 状態変数
    show_perspective = False
    smoothing_enabled = True
    smoother = PolygonSmoother(buffer_size=5, outlier_threshold=50)
    box_margin = 30  # デフォルトマージン（ピクセル）
    
    prev_time = cv2.getTickCount()
    fps_display = 0.0
    frame_count = 0
    
    main_window = "DocAligner v2 - Document Detection"
    perspective_window = "Perspective Correction"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to get frame")
                break
            
            # FPS計算
            current_time = cv2.getTickCount()
            time_diff = (current_time - prev_time) / cv2.getTickFrequency()
            if time_diff >= 1.0:
                fps_display = frame_count / time_diff
                frame_count = 0
                prev_time = current_time
            frame_count += 1
            
            # パディング追加
            padded_frame = cb.pad(frame, 100)
            
            # DocAlignerで検出
            polygon = model(
                img=padded_frame,
                do_center_crop=False
            )
            
            # パディング分を引く
            if polygon is not None:
                polygon = polygon - 100
            
            # 平滑化適用
            if smoothing_enabled:
                polygon = smoother.update(polygon)
            
            doc_detected = polygon is not None and len(polygon) >= 4
            
            # 結果を描画（マージン付き）
            result = frame.copy()
            if doc_detected:
                result = draw_polygon(result, polygon, expand_margin=box_margin)
            
            result = draw_info_panel(result, fps_display, doc_detected, 
                                     model_name, smoothing_enabled)
            
            # マージン表示
            cv2.putText(result, f"Margin: {box_margin}px (+/-)", (width - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow(main_window, result)
            
            # 透視変換ウィンドウ
            if show_perspective and doc_detected:
                warped = get_perspective_transform(frame, polygon)
                if warped is not None:
                    cv2.imshow(perspective_window, warped)
            else:
                try:
                    cv2.destroyWindow(perspective_window)
                except:
                    pass
            
            # キー処理
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                print("\n終了します...")
                break
            elif key == ord('s'):
                filepath = save_frame(result)
                print(f"保存しました: {filepath}")
                if doc_detected:
                    warped = get_perspective_transform(frame, polygon)
                    if warped is not None:
                        warped_path = filepath.replace('.jpg', '_corrected.jpg')
                        cv2.imwrite(warped_path, warped)
                        print(f"補正画像を保存: {warped_path}")
            elif key == ord('p'):
                show_perspective = not show_perspective
                print(f"透視変換プレビュー: {'ON' if show_perspective else 'OFF'}")
            elif key == ord('t'):
                smoothing_enabled = not smoothing_enabled
                if not smoothing_enabled:
                    smoother.reset()
                print(f"平滑化: {'ON' if smoothing_enabled else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                box_margin = min(100, box_margin + 10)
                print(f"マージン: {box_margin}px")
            elif key == ord('-') or key == ord('_'):
                box_margin = max(0, box_margin - 10)
                print(f"マージン: {box_margin}px")
            elif chr(key) in MODEL_CONFIGS:
                new_key = chr(key)
                if new_key != current_model_key:
                    current_model_key = new_key
                    model, model_name = load_model(current_model_key)
                    smoother.reset()
    
    except KeyboardInterrupt:
        print("\n\nCtrl+Cで終了...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print()
    print("=" * 60)
    print("DocAligner Document Detection v2 Complete")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
