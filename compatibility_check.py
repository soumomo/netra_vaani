#!/usr/bin/env python3
"""
Quick compatibility test for different environments
Run this before the main application to check if your system is ready
"""

import sys
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 12:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python version may have issues. Recommended: Python 3.8-3.12")
        return False

def check_platform():
    """Check platform information"""
    system = platform.system()
    machine = platform.machine()
    print(f"Platform: {system} {machine}")
    
    if system in ['Windows', 'Darwin', 'Linux']:
        print("✓ Platform is supported")
        return True
    else:
        print("? Platform not tested, may work")
        return True

def test_minimal_imports():
    """Test the most basic imports needed"""
    print("\nTesting minimal imports...")
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("✗ NumPy not found - install with: pip install numpy==1.21.6")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not found - install with: pip install opencv-python==4.8.1.78")
        return False
    
    try:
        import PIL
        print(f"✓ Pillow")
    except ImportError:
        print("✗ Pillow not found - install with: pip install Pillow==9.5.0")
        return False
    
    return True

def test_ml_imports():
    """Test ML framework imports"""
    print("\nTesting ML frameworks...")
    
    torch_ok = False
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        torch_ok = True
    except ImportError:
        print("✗ PyTorch not found - install with: pip install torch==1.13.1 torchvision==0.14.1")
    
    tf_ok = False
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        tf_ok = True
    except ImportError:
        print("✗ TensorFlow not found - install with: pip install tensorflow==2.11.0")
    
    mp_ok = False
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe {mp.__version__}")
        mp_ok = True
    except ImportError:
        print("✗ MediaPipe not found - install with: pip install mediapipe==0.9.3.0")
    
    return torch_ok and tf_ok and mp_ok

def main():
    """Run all compatibility checks"""
    print("=" * 50)
    print("MediaPipe Iris - Compatibility Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Platform", check_platform),
        ("Basic Packages", test_minimal_imports),
        ("ML Frameworks", test_ml_imports)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 30)
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"✗ {name} check failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 COMPATIBILITY CHECK PASSED!")
        print("Your system should run the MediaPipe Iris detection.")
        print("\nNext steps:")
        print("1. python extract_iris_landmark_model.py")
        print("2. python test_setup.py")
        print("3. python main.py")
    else:
        print("❌ COMPATIBILITY ISSUES DETECTED")
        print("\nTry installing minimal requirements:")
        print("pip install -r requirements-minimal.txt")
        print("\nOr follow the step-by-step guide in CROSS_PLATFORM_SETUP.md")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
