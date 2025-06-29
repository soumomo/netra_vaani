#!/usr/bin/env python3
"""
Easy deployment script for setting up MediaPipe Iris on any computer
Run this script on a new computer to automatically set everything up
"""

import os
import sys
import subprocess
import platform

def run_command(command, description="Running command"):
    """Run a command and handle errors"""
    print(f"{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and 8 <= version.minor <= 12:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} may have issues")
        print("Recommended: Python 3.8-3.12")
        return False

def install_requirements():
    """Try to install requirements with fallback options"""
    print("\nInstalling dependencies...")
    
    # Try main requirements first
    if run_command("pip install -r requirements.txt", "Installing main requirements"):
        return True
    
    print("Main requirements failed, trying minimal version...")
    if run_command("pip install -r requirements-minimal.txt", "Installing minimal requirements"):
        return True
    
    print("Minimal requirements failed, trying step-by-step...")
    commands = [
        "pip install numpy==1.21.6",
        "pip install opencv-python==4.8.1.78", 
        "pip install Pillow==9.5.0",
        "pip install matplotlib==3.6.3",
        "pip install tqdm==4.64.1",
        "pip install torch==1.13.1 torchvision==0.14.1",
        "pip install tensorflow==2.11.0",
        "pip install mediapipe==0.9.3.0",
        "pip install absl-py==1.4.0",
        "pip install protobuf==3.20.3"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Installing {cmd.split('==')[0].split()[-1]}"):
            print(f"Failed to install {cmd}")
            return False
    
    return True

def setup_models():
    """Generate model weights if needed"""
    if not os.path.exists('./data/weights.pkl'):
        print("\nGenerating model weights...")
        return run_command("python extract_iris_landmark_model.py", "Generating weights")
    else:
        print("✓ Model weights already exist")
        return True

def test_setup():
    """Test the complete setup"""
    print("\nTesting setup...")
    return run_command("python compatibility_check.py", "Running compatibility test")

def main():
    """Main deployment process"""
    print("=" * 60)
    print("MediaPipe Iris Detection - Automatic Setup")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    
    # Step 1: Check Python
    if not check_python():
        print("\n❌ Python version issue. Please install Python 3.8-3.12")
        sys.exit(1)
    
    # Step 2: Install dependencies
    if not install_requirements():
        print("\n❌ Failed to install dependencies")
        print("Please try manual installation following CROSS_PLATFORM_SETUP.md")
        sys.exit(1)
    
    # Step 3: Setup models
    if not setup_models():
        print("\n❌ Failed to setup models")
        sys.exit(1)
    
    # Step 4: Test everything
    if not test_setup():
        print("\n❌ Setup test failed")
        sys.exit(1)
    
    # Success!
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python main.py          # Demo mode")
    print("  python main.py --camera # Live camera detection")
    print("\nFor help, see:")
    print("  SETUP_GUIDE.md")
    print("  CROSS_PLATFORM_SETUP.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
