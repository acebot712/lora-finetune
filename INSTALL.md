# Installation Guide

This guide will walk you through setting up the LoRA Fine-Tuning environment.

## Prerequisites

- Python 3.10 or 3.11 (3.12 not yet supported due to PyTorch compatibility)
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.7 or newer
- 15GB free disk space

## Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/lora-finetune.git
cd lora-finetune

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies (this takes 5-10 minutes)
pip install --upgrade pip
pip install -r requirements-install.txt

# 4. Install diffusers from source (required for LoRA)
pip install git+https://github.com/huggingface/diffusers

# 5. Verify installation
python verify_installation.py

# 6. Launch the app
python scripts/run_app.py
```

## Detailed Installation Steps

### 1. Python Version

Check your Python version:
```bash
python3 --version
```

Should show Python 3.10.x or 3.11.x

### 2. CUDA Verification

Verify CUDA is available:
```bash
nvidia-smi
```

Should show your GPU and CUDA version.

### 3. Virtual Environment

Always use a virtual environment to avoid conflicts:

```bash
# Create venv
python3.11 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify activation (should show venv path)
which python
```

### 4. Install Core Dependencies

```bash
pip install --upgrade pip
pip install -r requirements-install.txt
```

This installs:
- PyTorch 2.0.1 with CUDA support
- Transformers for BLIP captioning
- Accelerate for distributed training
- Gradio for web UI
- All other core dependencies

### 5. Install Diffusers from Source

The released diffusers doesn't have full LoRA support, so install from source:

```bash
pip install git+https://github.com/huggingface/diffusers
```

### 6. Verify Installation

Run the verification script:
```bash
python verify_installation.py
```

Should show all green checkmarks.

## Troubleshooting

### CUDA Not Available

**Problem:** `torch.cuda.is_available()` returns False

**Solution:**
1. Check `nvidia-smi` works
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Out of Memory During Training

**Problem:** CUDA out of memory error

**Solutions:**
- Reduce `--train_batch_size` to 1 (minimum)
- Increase `--gradient_accumulation_steps` to 8 or 16
- Reduce `--resolution` to 256 or 384
- Use a smaller base model (SD 1.5 instead of SDXL)

### Gradio Import Error

**Problem:** `ModuleNotFoundError: No module named 'gradio'`

**Solution:**
```bash
pip install gradio==4.25.0
```

### Slow Image Captioning

**Problem:** BLIP captioning takes >5 seconds per image

**Solution:**
- This is normal on CPU. Use GPU for faster captioning.
- BLIP runs on GPU automatically if CUDA is available.

## Docker Installation

Alternatively, use Docker:

```bash
# Build image
docker build -t lora-finetune .

# Run with GPU support
docker run -p 7860:7860 --gpus all lora-finetune

# Open browser to http://localhost:7860
```

## Development Installation

For development, install additional tools:

```bash
pip install -r requirements-install.txt
pip install pre-commit black isort flake8

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Next Steps

After successful installation:

1. Read the [README](README.md) for usage instructions
2. Try the example workflow with sample images
3. Check the [FAQ](README.md#faq) for common questions

## Getting Help

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Check GitHub Issues for similar problems
3. Open a new issue with:
   - Output of `python verify_installation.py`
   - Output of `nvidia-smi`
   - Python version
   - OS version
