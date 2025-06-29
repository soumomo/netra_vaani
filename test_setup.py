#!/usr/bin/env python3
"""
Comprehensive test script to validate the MediaPipe Iris Detection setup
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
        
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        
        import mediapipe as mp
        print(f"✓ MediaPipe {mp.__version__}")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
        
        from PIL import Image
        print(f"✓ Pillow (PIL)")
        
        from tqdm import tqdm
        print(f"✓ tqdm")
        
        # Test custom modules
        from libs.face import FaceDetector, FaceLandmarksDetector
        print(f"✓ Custom face detection modules")
        
        from libs.iris import IrisDetector
        print(f"✓ Custom iris detection module")
        
        from libs.helper_func import vid2images, images2vid
        print(f"✓ Helper functions")
        
        print("All imports successful! ✓")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_files():
    """Test if all required data files exist"""
    print("\nTesting data files...")
    required_files = [
        './data/face_landmarks.json',
        './data/iris_landmark.tflite',
        './data/test.npy',
        './data/weights.pkl',
        './examples/01.png'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def test_camera():
    """Test camera availability"""
    print("\nTesting camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera is available")
            cap.release()
            return True
        else:
            print("✗ Camera not available")
            return False
    except Exception as e:
        print(f"✗ Camera test error: {e}")
        return False

def test_model_loading():
    """Test if models can be loaded"""
    print("\nTesting model loading...")
    try:
        from libs.face import FaceLandmarksDetector
        from libs.iris import IrisDetector
        
        print("Loading face landmarks detector...")
        face_detector = FaceLandmarksDetector()
        print("✓ Face landmarks detector loaded")
        
        print("Loading iris detector...")
        iris_detector = IrisDetector()
        print("✓ Iris detector loaded")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("MediaPipe Iris Detection Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Files Test", test_data_files),
        ("Camera Test", test_camera),
        ("Model Loading Test", test_model_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Setup is ready.")
        print("\nYou can now run:")
        print("  python main.py          # Demo mode")
        print("  python main.py --camera # Live camera")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 50)

if __name__ == "__main__":
    main()
