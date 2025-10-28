#!/usr/bin/env python3
"""
Verification script to check if installation is complete and working.
Run this after installing dependencies to verify everything is set up correctly.
"""

import sys

def check_imports():
    """Check that all required modules can be imported."""
    print("Checking module imports...")
    
    try:
        from lora_finetune import utils, data, training, inference, ui
        print("✓ All lora_finetune modules import successfully")
    except ImportError as e:
        print(f"✗ Failed to import lora_finetune modules: {e}")
        return False
    
    try:
        import gradio
        print(f"✓ Gradio {gradio.__version__} is installed")
    except ImportError:
        print("✗ Gradio is not installed")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} is installed")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA is not available (GPU required for training/inference)")
    except ImportError:
        print("✗ PyTorch is not installed")
        return False
    
    return True


def check_functionality():
    """Check that basic functionality works."""
    print("\nChecking basic functionality...")
    
    from lora_finetune import utils, training
    
    models = utils.get_model_zoo()
    print(f"✓ Model zoo has {len(models)} models: {', '.join(models)}")
    
    loras = utils.get_lora_gallery()
    print(f"✓ LoRA gallery has {len(loras)} LoRAs: {', '.join(loras)}")
    
    cmd, status = training.train_lora('/tmp/test', 'test-model', '/tmp/out')
    print(f"✓ Training command generation works")
    
    return True


def main():
    print("LoRA Fine-Tuning Installation Verification")
    print("=" * 50)
    
    if not check_imports():
        print("\n✗ Installation verification FAILED")
        print("Run: pip install -r requirements.txt")
        return 1
    
    if not check_functionality():
        print("\n✗ Functionality verification FAILED")
        return 1
    
    print("\n" + "=" * 50)
    print("✓ All checks passed! Installation is complete.")
    print("\nTo launch the app, run:")
    print("  python scripts/run_app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
