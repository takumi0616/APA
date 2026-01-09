#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
フォームA画像の3点マーク検出テストプログラム（改善版）

目的:
- image/Aディレクトリの画像から3点マーク（黒い四角形）を検出
- 上左、上右、下左の3箇所にあるマーカーを検出
- 検出部分にボックスを描画

改善点:
- 複数の二値化方法を試行（適応的閾値、Otsu法など）
- 閾値を緩和して色違いの画像にも対応
- 描画線を太くする
"""

import cv2
import numpy as np
import os
import sys
import time

# Windows環境でUTF-8を強制
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout:
    sys.stdout.reconfigure(encoding='utf-8')


def detect_filled_square_markers(image, target_corners=['top_left', 'top_right', 'bottom_left']):
    """
    塗りつぶしの四角マーカー（3点マーク）を検出する（改善版）
    
    Args:
        image: OpenCV画像（BGR形式）
        target_corners: 検出対象のコーナー
    
    Returns:
        list: 検出された四角マーカーの情報リスト
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = image.shape[:2]
    
    markers = []
    
    # 複数の二値化方法を試行
    binary_images = []
    
    # 方法1: 固定閾値（低め - 薄いマーカー用）
    _, binary1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    binary_images.append(('固定閾値50', binary1))
    
    # 方法2: 固定閾値（中程度）
    _, binary2 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    binary_images.append(('固定閾値80', binary2))
    
    # 方法3: 固定閾値（高め）
    _, binary3 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    binary_images.append(('固定閾値120', binary3))
    
    # 方法4: Otsu法
    _, binary4 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_images.append(('Otsu', binary4))
    
    # 方法5: 適応的閾値（ブロックサイズ小）
    binary5 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 21, 8)
    binary_images.append(('適応的21', binary5))
    
    # 方法6: 適応的閾値（ブロックサイズ大）
    binary6 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 51, 10)
    binary_images.append(('適応的51', binary6))
    
    # 画像の4隅の領域を定義（画像端から15%の範囲）
    corner_margin_x = int(image_width * 0.15)
    corner_margin_y = int(image_height * 0.15)
    
    corners = {
        'top_left': (0, 0, corner_margin_x, corner_margin_y),
        'top_right': (image_width - corner_margin_x, 0, image_width, corner_margin_y),
        'bottom_left': (0, image_height - corner_margin_y, corner_margin_x, image_height),
        'bottom_right': (image_width - corner_margin_x, image_height - corner_margin_y, image_width, image_height)
    }
    
    # マーカーサイズの範囲（画像サイズに基づく）
    min_size = min(image_width, image_height) * 0.005
    max_size = min(image_width, image_height) * 0.08
    min_area = min_size ** 2
    max_area = max_size ** 2
    
    found_corners = {}  # コーナーごとに最良のマーカーを保持
    
    for method_name, binary in binary_images:
        # モルフォロジー演算
        kernel = np.ones((3, 3), np.uint8)
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            contour_area = cv2.contourArea(contour)
            
            if min_area < contour_area < max_area:
                aspect_ratio = float(w) / h if h > 0 else 0
                # アスペクト比を緩和（0.4〜2.5）
                if 0.4 < aspect_ratio < 2.5:
                    cx = x + w // 2
                    cy = y + h // 2
                    
                    corner_name = None
                    for name, (x1, y1, x2, y2) in corners.items():
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            corner_name = name
                            break
                    
                    if corner_name in target_corners:
                        fill_ratio = contour_area / area if area > 0 else 0
                        
                        # 塗りつぶし率を緩和（0.4以上）
                        if fill_ratio > 0.4:
                            mask = np.zeros(gray.shape, dtype=np.uint8)
                            cv2.drawContours(mask, [contour], 0, 255, -1)
                            mean_val = cv2.mean(gray, mask=mask)[0]
                            
                            # 輝度閾値を緩和（180未満）
                            if mean_val < 180:
                                epsilon = 0.05 * cv2.arcLength(contour, True)
                                approx = cv2.approxPolyDP(contour, epsilon, True)
                                
                                # スコア計算
                                aspect_score = 1.0 - abs(aspect_ratio - 1.0) * 0.5
                                fill_score = fill_ratio
                                intensity_score = (180 - mean_val) / 180.0
                                score = aspect_score * 0.25 + fill_score * 0.35 + intensity_score * 0.4
                                
                                marker_info = {
                                    'contour': contour,
                                    'approx': approx,
                                    'area': contour_area,
                                    'center': (cx, cy),
                                    'bbox': (x, y, w, h),
                                    'corner': corner_name,
                                    'aspect_ratio': aspect_ratio,
                                    'fill_ratio': fill_ratio,
                                    'mean_intensity': mean_val,
                                    'vertices': len(approx),
                                    'score': score,
                                    'method': method_name
                                }
                                
                                # 既存のマーカーよりスコアが高ければ更新
                                if corner_name not in found_corners or score > found_corners[corner_name]['score']:
                                    found_corners[corner_name] = marker_info
    
    # 見つかったマーカーをリストに変換
    markers = list(found_corners.values())
    return markers


def draw_detections(image, markers):
    """
    検出結果を画像に描画する（線を太くする）
    """
    result = image.copy()
    
    height, width = image.shape[:2]
    scale = min(width, height) / 1000.0
    font_scale = max(0.8, scale * 1.0)
    # 線の太さを大幅に増加
    thickness = max(8, int(scale * 10))
    font_thickness = max(3, int(scale * 4))
    
    # マーカーの描画（赤色、太い線）
    for i, marker in enumerate(markers):
        x, y, w, h = marker['bbox']
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), thickness)
        
        corner = marker['corner'] if marker['corner'] else 'unknown'
        label = f"M{i+1}:{corner}"
        label_y = max(y - 20, 50)
        cv2.putText(result, label, (x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
    
    return result


def process_image(image_path, output_dir=None):
    """
    画像を処理してマーカーを検出する
    """
    start_time = time.time()
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [エラー] 画像を読み込めませんでした: {image_path}")
        return None
    
    height, width = image.shape[:2]
    
    # 3点マーカー検出（上左、上右、下左）
    expected_corners = ['top_left', 'top_right', 'bottom_left']
    markers = detect_filled_square_markers(image, expected_corners)
    
    process_time = time.time() - start_time
    
    result_image = draw_detections(image, markers)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(output_path, result_image)
    
    return {
        'image_path': image_path,
        'size': (width, height),
        'markers': markers,
        'process_time': process_time,
        'result_image': result_image
    }


def main():
    """メイン関数"""
    print("="*70)
    print("フォームA画像 - 3点マーク検出テスト（改善版）")
    print("="*70)
    
    # image/Aディレクトリのみをテスト
    base_dir = os.path.dirname(__file__)
    image_dir = os.path.join(base_dir, "image", "A")
    output_dir = os.path.join(base_dir, "output", "formA_test")
    
    # 画像ファイルを取得
    all_files = []
    for i in range(1, 7):
        image_path = os.path.join(image_dir, f"{i}.jpg")
        if os.path.exists(image_path):
            all_files.append(image_path)
    
    if not all_files:
        print(f"\n[エラー] 画像が見つかりません: {image_dir}")
        return
    
    print(f"\n検出対象: image/A ディレクトリ（3点マーク検出）")
    print(f"画像数: {len(all_files)}枚")
    print("-"*70)
    
    results = []
    total_start = time.time()
    
    for image_path in all_files:
        filename = os.path.basename(image_path)
        
        result = process_image(image_path, output_dir)
        if result:
            results.append(result)
            
            marker_count = len(result['markers'])
            status = "✓" if marker_count == 3 else f"✗({marker_count}/3)"
            markers_str = ", ".join([m['corner'] for m in result['markers']])
            
            print(f"  {filename}: {result['size'][0]}x{result['size'][1]}, "
                  f"マーカー: {status} [{markers_str}], "
                  f"時間: {result['process_time']*1000:.0f}ms")
    
    total_time = time.time() - total_start
    
    # 結果サマリー
    print("\n" + "="*70)
    print("検出結果サマリー")
    print("="*70)
    
    success_count = sum(1 for r in results if len(r['markers']) == 3)
    avg_time = sum(r['process_time'] for r in results) / len(results) if results else 0
    
    print(f"\n成功: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"平均処理時間: {avg_time*1000:.0f}ms/枚")
    print(f"総処理時間: {total_time:.1f}秒")
    print(f"出力先: {output_dir}")
    
    # 詳細表示
    print("\n【詳細】")
    for result in results:
        filename = os.path.basename(result['image_path'])
        marker_count = len(result['markers'])
        status = "✓ 成功" if marker_count == 3 else f"✗ 失敗({marker_count}/3)"
        markers_str = ", ".join([f"{m['corner']}({m['method']})" for m in result['markers']])
        print(f"  {filename}: {status}")
        if result['markers']:
            print(f"    検出: [{markers_str}]")
    
    print("\n" + "="*70)
    print("テスト完了")
    print("="*70)


if __name__ == "__main__":
    main()
