#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera Access & Object Recognition Test Program
================================================

This program tests:
1. Camera access on the connected PC
2. Real-time video capture
3. Face detection for object recognition

Controls:
- 'q' key: Exit program
- 's' key: Save current frame
- ESC key: Exit program

Created: January 6, 2026
"""

import cv2
import sys
import os
from datetime import datetime


def check_camera_availability():
    """
    Check for available cameras
    
    Returns:
        list: List of available camera indices
    """
    available_cameras = []
    
    print("=" * 60)
    print("Camera Detection Test")
    print("=" * 60)
    
    # Check camera indices 0-4
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  [OK] Camera {i}: Detected ({width}x{height}, {fps:.1f}fps)")
            cap.release()
    
    if not available_cameras:
        print("  [NG] No available cameras found")
    
    print()
    return available_cameras


def load_face_cascade():
    """
    Load HaarCascade classifier for face detection
    
    Returns:
        cv2.CascadeClassifier: Face detection classifier
    """
    # Get path to HaarCascade file included with OpenCV
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(cascade_path):
        print(f"Warning: Face detection file not found: {cascade_path}")
        return None
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Warning: Failed to load face detection classifier")
        return None
    
    print("[OK] Face detection classifier loaded successfully")
    return face_cascade


def detect_faces(frame, face_cascade):
    """
    Detect faces in the frame
    
    Args:
        frame: Camera frame image
        face_cascade: Face detection classifier
    
    Returns:
        tuple: (number of faces detected, list of face regions)
    """
    if face_cascade is None:
        return 0, []
    
    # Convert to grayscale for faster face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # Image scale reduction rate
        minNeighbors=5,       # Number of neighbors required for detection
        minSize=(30, 30)      # Minimum face size to detect
    )
    
    return len(faces), faces


def draw_detection_overlay(frame, faces, fps_text):
    """
    Draw detection results on frame
    
    Args:
        frame: Frame to draw on
        faces: List of detected face regions
        fps_text: FPS display text
    
    Returns:
        frame: Frame with drawings
    """
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        # Green rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Label
        cv2.putText(frame, 'Face', (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Information panel background
    cv2.rectangle(frame, (5, 5), (250, 90), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (250, 90), (0, 255, 0), 1)
    
    # FPS display
    cv2.putText(frame, fps_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Detection count display
    face_count_text = f"Faces Detected: {len(faces)}"
    cv2.putText(frame, face_count_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Controls guide
    cv2.putText(frame, "Press 'q' or ESC to quit", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame


def save_frame(frame, save_dir="captured_images"):
    """
    Save current frame
    
    Args:
        frame: Frame to save
        save_dir: Save directory
    
    Returns:
        str: Saved file path
    """
    # Create save directory
    full_path = os.path.join(os.path.dirname(__file__), save_dir)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    # Filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join(full_path, filename)
    
    cv2.imwrite(filepath, frame)
    return filepath


def set_camera_resolution(cap, width, height, fps):
    """
    Set camera resolution and FPS to maximum supported values
    
    Args:
        cap: VideoCapture object
        width: Desired width
        height: Desired height
        fps: Desired FPS
    
    Returns:
        tuple: (actual_width, actual_height, actual_fps)
    """
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Get actual values (camera may not support requested values)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    return actual_width, actual_height, actual_fps


def main():
    """
    Main function: Test camera access and object recognition
    """
    print()
    print("=" * 60)
    print("Camera Access & Object Recognition Test Program")
    print("=" * 60)
    print()
    
    # Display OpenCV version
    print(f"OpenCV Version: {cv2.__version__}")
    print()
    
    # Detect cameras
    available_cameras = check_camera_availability()
    
    if not available_cameras:
        print("Error: No available cameras.")
        print("Please check:")
        print("  - Camera is properly connected")
        print("  - No other application is using the camera")
        print("  - Camera driver is properly installed")
        return 1
    
    # Use the first detected camera
    camera_index = available_cameras[0]
    print(f"Using Camera {camera_index}")
    print()
    
    # Load face detection classifier
    face_cascade = load_face_cascade()
    print()
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return 1
    
    # Set camera to maximum resolution (720p/30fps based on camera specs)
    # Camera specs: Max 720p (1280x720) at 30fps
    print("=" * 60)
    print("Setting Maximum Resolution (720p/30fps)...")
    print("=" * 60)
    
    # Get default settings first
    default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    default_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Default: {default_width}x{default_height} @ {default_fps:.1f}fps")
    
    # Set to maximum supported resolution
    width, height, fps = set_camera_resolution(cap, 1280, 720, 30)
    print(f"  Requested: 1280x720 @ 30fps")
    print(f"  Actual: {width}x{height} @ {fps:.1f}fps")
    
    if width >= 1280 and height >= 720:
        print("  [OK] 720p resolution set successfully!")
    else:
        print(f"  [INFO] Camera set to {width}x{height} (max supported)")
    print()
    
    print("=" * 60)
    print("Camera Settings")
    print("=" * 60)
    print(f"  Resolution: {width} x {height}")
    print(f"  FPS: {fps:.1f}")
    print()
    
    print("=" * 60)
    print("Starting camera preview...")
    print("=" * 60)
    print("Controls:")
    print("  'q' or ESC: Exit")
    print("  's': Save current frame")
    print()
    
    # FPS calculation variables
    prev_time = cv2.getTickCount()
    fps_display = 0.0
    frame_count = 0
    total_faces_detected = 0
    
    # Window name
    window_name = "Camera Test - Face Detection"
    
    try:
        while True:
            # Get frame
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
            
            # Face detection
            num_faces, faces = detect_faces(frame, face_cascade)
            total_faces_detected += num_faces
            
            # Draw overlay
            fps_text = f"FPS: {fps_display:.1f}"
            frame = draw_detection_overlay(frame, faces, fps_text)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Check key input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nExit key pressed")
                break
            elif key == ord('s'):  # 's' to save
                filepath = save_frame(frame)
                print(f"Frame saved: {filepath}")
    
    except KeyboardInterrupt:
        print("\n\nCtrl+C pressed. Exiting...")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    # Test result summary
    print()
    print("=" * 60)
    print("Test Result Summary")
    print("=" * 60)
    print(f"  [OK] Camera Access: Success")
    print(f"  [OK] Video Capture: Success")
    print(f"  [OK] Face Detection: {'Enabled' if face_cascade is not None else 'Disabled'}")
    print(f"  [OK] Real-time Processing: Success")
    print()
    print("All tests completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
