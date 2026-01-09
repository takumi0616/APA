#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
フォームB画像のQRコード検出テストプログラム（改善版）

目的:
- image/Bディレクトリの画像からQRコードを検出
- 検出部分にボックスを描画

改善点:
- OpenCV QRCodeDetectorのマルチスケール検出
- 様々な解像度で検出を試みる
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


def detect_qr_codes(image):
    """
    QRコードを検出する（OpenCVマルチスケール）
    
    Args:
        image: OpenCV画像（BGR形式）
    
    Returns:
        list: 検出されたQRコードの情報リスト
    """
    qr_results = []
    qr_detector = cv2.QRCodeDetector()
    height, width = image.shape[:2]
    
    # 複数のスケールで検出を試みる
    scales = [0.5, 0.25, 1.0, 0.125, 0.75]
    
    for scale in scales:
        if scale == 1.0:
            test_image = image
        else:
            new_width = int(width * scale)
            new_height = int(height * scale)
            if new_width < 100 or new_height < 100:
                continue
            test_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        try:
            data, points, straight_qrcode = qr_detector.detectAndDecode(test_image)
            if data and points is not None:
                pts = points[0]
                if scale != 1.0:
                    pts = pts / scale
                pts_int = pts.astype(np.int32)
                qr_results.append({
                    'data': data,
                    'points': pts_int,
                    'rect': cv2.boundingRect(pts_int),
                    'scale': scale
                })
                break
        except:
            pass
    
    return qr_results


def draw_detections(image, qr_codes):
    """
    検出結果を画像に描画する（線を太くする）
    """
    result = image.copy()
    
    height, width = image.shape[:2]
    scale = min(width, height) / 1000.0
    font_scale = max(1.0, scale * 1.2)
    # 線の太さを大幅に増加
    thickness = max(10, int(scale * 15))
    font_thickness = max(4, int(scale * 5))
    
    # QRコードの描画（緑色、太い線）
    for qr in qr_codes:
        pts = qr['points'].reshape((-1, 1, 2))
        cv2.polylines(result, [pts], True, (0, 255, 0), thickness)
        
        # バウンディングボックスも描画
        x, y, w, h = qr['rect']
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), thickness)
        
        # ラベル
        data_display = qr['data'][:25] + '...' if len(qr['data']) > 25 else qr['data']
        label_y = max(y - 30, 80)
        cv2.putText(result, f"QR: {data_display}", (x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
    
    return result


def process_image(image_path, output_dir=None):
    """
    画像を処理してQRコードを検出する
    """
    start_time = time.time()
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [エラー] 画像を読み込めませんでした: {image_path}")
        return None
    
    height, width = image.shape[:2]
    
    # QRコード検出
    qr_codes = detect_qr_codes(image)
    
    process_time = time.time() - start_time
    
    result_image = draw_detections(image, qr_codes)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(output_path, result_image)
    
    return {
        'image_path': image_path,
        'size': (width, height),
        'qr_codes': qr_codes,
        'process_time': process_time,
        'result_image': result_image
    }


def main():
    """メイン関数"""
    print("="*70)
    print("フォームB画像 - QRコード検出テスト（改善版）")
    print("="*70)
    print("QR検出: OpenCV QRCodeDetector（マルチスケール）")
    
    # image/Bディレクトリのみをテスト
    base_dir = os.path.dirname(__file__)
    image_dir = os.path.join(base_dir, "image", "B")
    output_dir = os.path.join(base_dir, "output", "formB_test")
    
    # 画像ファイルを取得
    all_files = []
    for i in range(1, 7):
        image_path = os.path.join(image_dir, f"{i}.jpg")
        if os.path.exists(image_path):
            all_files.append(image_path)
    
    if not all_files:
        print(f"\n[エラー] 画像が見つかりません: {image_dir}")
        return
    
    print(f"\n検出対象: image/B ディレクトリ（QRコード検出）")
    print(f"画像数: {len(all_files)}枚")
    print("-"*70)
    
    results = []
    total_start = time.time()
    
    for image_path in all_files:
        filename = os.path.basename(image_path)
        
        result = process_image(image_path, output_dir)
        if result:
            results.append(result)
            
            qr_count = len(result['qr_codes'])
            qr_status = "✓" if qr_count >= 1 else "✗"
            qr_data = result['qr_codes'][0]['data'] if result['qr_codes'] else '-'
            
            print(f"  {filename}: {result['size'][0]}x{result['size'][1]}, "
                  f"QR: {qr_status} ({qr_data}), "
                  f"時間: {result['process_time']*1000:.0f}ms")
    
    total_time = time.time() - total_start
    
    # 結果サマリー
    print("\n" + "="*70)
    print("検出結果サマリー")
    print("="*70)
    
    qr_success = sum(1 for r in results if len(r['qr_codes']) >= 1)
    avg_time = sum(r['process_time'] for r in results) / len(results) if results else 0
    
    print(f"\nQR検出成功: {qr_success}/{len(results)} ({qr_success/len(results)*100:.1f}%)")
    print(f"平均処理時間: {avg_time*1000:.0f}ms/枚")
    print(f"総処理時間: {total_time:.1f}秒")
    print(f"出力先: {output_dir}")
    
    # 詳細表示
    print("\n【詳細】")
    for result in results:
        filename = os.path.basename(result['image_path'])
        qr_count = len(result['qr_codes'])
        
        if qr_count >= 1:
            status = "✓ 成功"
        else:
            status = "✗ 失敗"
        
        qr_data = result['qr_codes'][0]['data'] if result['qr_codes'] else 'なし'
        
        print(f"  {filename}: {status}")
        print(f"    QR: {qr_data}")
    
    print("\n" + "="*70)
    print("テスト完了")
    print("="*70)


if __name__ == "__main__":
    main()
