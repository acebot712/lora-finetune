# LoRA Fine-Tuning for Stable Diffusion

A Gradio-based web UI for fine-tuning Stable Diffusion models with LoRA (Low-Rank Adaptation) and running inference with custom adapters.

<figure>
  <img src="https://res.cloudinary.com/dyjvkjts4/image/upload/v1686152045/generated_sample_xjgsxy.png" alt="Generated Sample" style="width:100%">
  <figcaption>Example output after fine-tuning SD with LoRA</figcaption>
</figure>

## What This Does

This tool provides a web interface for:
1. Uploading images and auto-generating captions using BLIP
2. Creating HuggingFace datasets from your images
3. Generating training commands for LoRA fine-tuning
4. Running inference with pre-trained LoRA models

## What This Does NOT Do

- Does not automatically train models (generates command you run separately)
- Does not include ONNX/TensorRT export
- Does not include batch processing or streaming
- Does not include prompt engineering tools
- LoRA strength adjustment is a placeholder (not yet implemented in diffusers)

## Requirements

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (for training and inference)
- CUDA 11.7+
- ~15GB disk space for dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lora-finetune.git
cd lora-finetune

# Install dependencies (this will take 5-10 minutes)
pip install -r requirements.txt

# Install diffusers from source (required for LoRA support)
pip install git+https://github.com/huggingface/diffusers
```

## Quick Start

```bash
# Launch the web UI
python scripts/run_app.py

# Open your browser to http://localhost:7860
```

## Usage

### 1. Upload Images
Upload 10-50 images of your subject. Works best with:
- Consistent subject across images
- Varied poses and angles
- Good lighting and image quality

### 2. Auto-Caption
Click "Auto-caption & Create Dataset" to generate captions using BLIP.
This creates a HuggingFace dataset in your upload directory.

### 3. Train LoRA
1. Select a base model (SD 1.5 or SDXL)
2. Set output directory name
3. Click "Generate Training Command"
4. **Copy and run the command in your terminal** (training takes 30-60 min on T4 GPU)

Example training command:
```bash
accelerate launch --mixed_precision="fp16" sd.py \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
  --train_data_dir=./dataset \
  --output_dir=my_lora \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --checkpointing_steps=500 \
  --seed=1337
```

### 4. Generate Images
Use the inference tab to generate images with pre-trained LoRA models.
Requires ~6GB VRAM for SD 1.5, ~10GB for SDXL.

## Project Structure

```
lora-finetune/
├── lora_finetune/          # Modular Python package
│   ├── data.py            # Image upload & captioning
│   ├── training.py        # Training command generation
│   ├── inference.py       # Image generation
│   ├── utils.py           # Model zoo & helpers
│   └── ui.py              # Gradio UI
├── scripts/
│   └── run_app.py         # Launch script
├── sd.py                  # LoRA training script (modified from HF)
├── blip_captions.py       # Standalone captioning script
├── tests/                 # pytest test suite
└── docs/                  # API documentation
```

## Training Your Own LoRA

1. Prepare 10-50 images of your subject
2. Use the UI or run `blip_captions.py` to generate captions
3. Train with the generated command (adjust `max_train_steps` based on dataset size)
4. Find your LoRA weights in `output_dir/pytorch_lora_weights.bin`
5. Upload to HuggingFace Hub or use locally

**Training time:** ~30-60 minutes for 15K steps on T4 GPU  
**LoRA file size:** ~3-6MB (much smaller than full model fine-tuning)

## Known Issues & Limitations

- Training must be run separately via command line (not integrated in UI)
- LoRA strength slider in UI is not functional (diffusers limitation)
- Auto-captioning can be slow (~1 sec per image)
- Inference requires GPU (CPU inference is extremely slow)
- No validation dataset support yet
- No multi-GPU training support

## Docker

```bash
# Build
docker build -t lora-finetune .

# Run
docker run -p 7860:7860 --gpus all lora-finetune
```

## Development

```bash
# Install dev dependencies
pip install pre-commit pytest

# Run tests
pytest

# Run linting
pre-commit run --all-files
```

## Credits

- Built on [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- Auto-captioning via [Salesforce BLIP](https://github.com/salesforce/BLIP)
- Training script adapted from [diffusers examples](https://github.com/huggingface/diffusers/tree/main/examples)

## License

MIT License - See LICENSE file

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## FAQ

**Q: Can I run this without a GPU?**  
A: No. Training and inference require NVIDIA GPU with CUDA.

**Q: How many images do I need?**  
A: 10-50 images is ideal. More can lead to overfitting.

**Q: Can I use this for commercial projects?**  
A: Yes, but check the licenses of base models you fine-tune.

**Q: Why doesn't training start when I click the button?**  
A: The UI generates a command you must run separately. This is intentional to allow parameter customization.

**Q: My LoRA model doesn't look like my training images**  
A: Try increasing `max_train_steps` to 20000-30000 or adjusting `learning_rate`.
