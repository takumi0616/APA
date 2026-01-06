#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DocAligner Real-time Document Detection
=======================================

This program uses DocAligner for real-time document detection.

Features:
1. Real-time document detection using DocAligner (AI-based)
2. Bounding box display around detected documents
3. 720p camera resolution support
4. Perspective correction preview

Controls:
- 'q' or ESC: Exit program
- 's': Save current frame
- 'p': Toggle perspective correction preview

Created: January 6, 2026
"""

import cv2
import numpy as np
import sys
import os
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Import DocAligner
from docaligner import DocAligner
import capybara as cb


def set_camera_resolution(cap, width, height, fps):
    """Set camera resolution and FPS"""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    return actual_width, actual_height, actual_fps


def save_frame(frame, save_dir="docaligner_captures"):
    """Save current frame"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, save_dir)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"doc_{timestamp}.jpg"
    filepath = os.path.join(full_path, filename)
    
    cv2.imwrite(filepath, frame)
    return filepath


def draw_polygon(frame, polygon, color=(0, 255, 0), thickness=3):
    """Draw polygon on frame"""
    if polygon is None or len(polygon) < 4:
        return frame
    
    result = frame.copy()
    
    # Convert polygon to int points
    pts = polygon.astype(np.int32)
    
    # Draw filled polygon with transparency
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.2, result, 0.8, 0, result)
    
    # Draw polygon outline
    cv2.polylines(result, [pts], True, color, thickness)
    
    # Draw corner points
    for i, pt in enumerate(pts):
        cv2.circle(result, tuple(pt), 8, (0, 0, 255), -1)
        cv2.circle(result, tuple(pt), 10, (255, 255, 255), 2)
        
        # Label corners
        labels = ['TL', 'TR', 'BR', 'BL']
        cv2.putText(result, labels[i], (pt[0] + 15, pt[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result


def draw_info_panel(frame, fps, doc_detected):
    """Draw information panel on frame"""
    result = frame.copy()
    
    # Information panel background
    cv2.rectangle(result, (5, 5), (350, 90), (0, 0, 0), -1)
    cv2.rectangle(result, (5, 5), (350, 90), (0, 255, 0), 1)
    
    # FPS display
    cv2.putText(result, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Document status
    status_color = (0, 255, 0) if doc_detected else (0, 0, 255)
    status_text = "Document: DETECTED" if doc_detected else "Document: NOT FOUND"
    cv2.putText(result, status_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
    
    # Model info
    cv2.putText(result, "Model: DocAligner (fastvit_sa24)", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return result


def get_perspective_transform(frame, polygon, output_size=(800, 600)):
    """Apply perspective transformation"""
    if polygon is None or len(polygon) < 4:
        return None
    
    # Define destination points
    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype="float32")
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(polygon.astype(np.float32), dst)
    
    # Apply the transformation
    warped = cv2.warpPerspective(frame, M, output_size)
    
    return warped


def main():
    """Main function: Real-time document detection with DocAligner"""
    print()
    print("=" * 60)
    print("DocAligner Real-time Document Detection")
    print("=" * 60)
    print()
    print(f"OpenCV Version: {cv2.__version__}")
    print()
    
    # Initialize DocAligner
    print("Loading DocAligner model (fastvit_sa24)...")
    model = DocAligner(model_cfg='fastvit_sa24')
    print("[OK] DocAligner model loaded successfully!")
    print()
    
    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return 1
    
    # Set resolution to 720p
    width, height, fps = set_camera_resolution(cap, 1280, 720, 30)
    print(f"[OK] Camera opened: {width}x{height} @ {fps:.1f}fps")
    print()
    
    print("=" * 60)
    print("Controls:")
    print("  'q' or ESC: Exit")
    print("  's': Save current frame")
    print("  'p': Toggle perspective correction window")
    print("=" * 60)
    print()
    
    # State variables
    show_perspective = False
    prev_time = cv2.getTickCount()
    fps_display = 0.0
    frame_count = 0
    
    # Window names
    main_window = "DocAligner - Document Detection"
    perspective_window = "Perspective Correction"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to get frame")
                break
            
            # FPS calculation
            current_time = cv2.getTickCount()
            time_diff = (current_time - prev_time) / cv2.getTickFrequency()
            if time_diff >= 1.0:
                fps_display = frame_count / time_diff
                frame_count = 0
                prev_time = current_time
            frame_count += 1
            
            # Pad image for corner detection
            padded_frame = cb.pad(frame, 100)
            
            # Detect document using DocAligner
            polygon = model(
                img=padded_frame,
                do_center_crop=False
            )
            
            # Remove padding from polygon coordinates
            if polygon is not None:
                polygon = polygon - 100
            
            # Check if document detected
            doc_detected = polygon is not None and len(polygon) >= 4
            
            # Draw results
            result = frame.copy()
            if doc_detected:
                result = draw_polygon(result, polygon)
            
            # Draw info panel
            result = draw_info_panel(result, fps_display, doc_detected)
            
            # Display main window
            cv2.imshow(main_window, result)
            
            # Perspective correction window
            if show_perspective and doc_detected:
                warped = get_perspective_transform(frame, polygon)
                if warped is not None:
                    cv2.imshow(perspective_window, warped)
            else:
                try:
                    cv2.destroyWindow(perspective_window)
                except:
                    pass
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nExit key pressed")
                break
            elif key == ord('s'):
                filepath = save_frame(result)
                print(f"Frame saved: {filepath}")
                if doc_detected:
                    warped = get_perspective_transform(frame, polygon)
                    if warped is not None:
                        warped_path = filepath.replace('.jpg', '_corrected.jpg')
                        cv2.imwrite(warped_path, warped)
                        print(f"Corrected image saved: {warped_path}")
            elif key == ord('p'):
                show_perspective = not show_perspective
                print(f"Perspective window: {'ON' if show_perspective else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n\nCtrl+C pressed. Exiting...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print()
    print("=" * 60)
    print("DocAligner Document Detection Complete")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
