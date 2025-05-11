# 10x LoRA Fine-Tuning & Inference Platform

A hackathon-ready, Silicon Valley-grade platform for fine-tuning and deploying Stable Diffusion with LoRA adapters—now with a beautiful web UI, one-click workflows, and blazing-fast inference.

## 🚀 Features
- **Gradio Web UI**: Upload images, auto-caption, train, and generate images—all in your browser.
- **One-Click Data Prep**: Auto-captioning and dataset creation from your images.
- **Model Zoo**: Choose from popular base models (SD 1.5, SDXL, etc.) and community LoRA weights.
- **LoRA Merging & Dynamic Strength**: Combine adapters and adjust their influence at inference time.
- **Prompt Engineering Tools**: Prompt generator, negative prompts, and completions.
- **Batch & Fast Inference**: ONNX/TensorRT export, batch image generation, and streaming data support.
- **Docker & pip install**: Run anywhere, deploy instantly.
- **Continuous Integration**: Linting, testing, and deployment with GitHub Actions.
- **Colab/Notebook Tutorials**: Step-by-step guides for every workflow.
- **API & Mobile Ready**: REST API and mobile app starter included.

## 🏁 Quickstart

```bash
# 1. Install (pip or Docker)
pip install -r requirements.txt
# or
docker build -t lora-finetune .

# 2. Launch the Web UI
python scripts/run_app.py
# or
docker run -p 7860:7860 lora-finetune
```

## 🖼️ Demo
- Upload images, auto-caption, and train a LoRA in minutes
- Generate images with your custom LoRA, adjust strength, and merge adapters
- Try community LoRAs and base models from the Model Zoo

## 📚 Documentation
- [Tutorials & Notebooks](notebooks/)
- [API Reference](docs/)
- [Contributing](CONTRIBUTING.md)

## 🏆 Built for Hackathons & Demos
- Lightning-fast, beautiful, and robust
- Designed to impress at any event or pitch
- **Production-grade modular structure**: All logic is split into `lora_finetune/` modules for data, training, inference, UI, and utilities.

---

For more, see the full docs or launch the app and explore!

# Fine-Tuning StableDiffusion with LoRA on custom dataset


<figure>
  <img src="https://res.cloudinary.com/dyjvkjts4/image/upload/v1686152045/generated_sample_xjgsxy.png" alt="Generated Sample" style="width:100%">
  <figcaption>Fig.1 - Generated Sample after finetuning SD with LoRA.</figcaption>
</figure>

Using this repository, you should be able to create your own LoRA models that you can upload to [CivitAI](https://civitai.com/) without using [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

> LoRA models can be made with significantly low compute and memory requirements as opposed to things like Dreambooth, CLIP, or textual inversion (about a few MBs).

## Replicate Machine setup

Setup a T4 instance using [this image](https://cloud.google.com/deep-learning-vm/docs/pytorch_start_instance#creating_a_pytorch_instance_from_the).

## How to Use

Create the environment

1. Install Diffusers from source as we will need to modify SD libraries directly:- `pip install git+https://github.com/huggingface/diffusers`. Comment out the lines that consist of `# black image` to turn off the safety filter.
2. `pip install accelerate transformers datasets evaluate`
3. Assuming torch is already installed; `pip3 install torchvision torchaudio`

## Running fine-tuning

1. Make a folder consisting of images that you want to fine-tune a LoRA model. Don't worry about the folder structure. Make sure the image files are in a format recognized by [Pillow](https://pypi.org/project/Pillow/).
2. Create `(image, text)` image -caption pairs dataset with your images with `python3 blip_captions.py` - (Change any hardcoded values in the file with your requirements first)
3. Update `DATASET_NAME_MAPPING` in `sd.py`
4. The following command is meant to fine-tune your model. Change the parameters' values as necessary.
```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="waifu"
export HUB_MODEL_ID="waifu-lora"

accelerate launch --mixed_precision="fp16"  sd.py   --pretrained_model_name_or_path=$MODEL_NAME   --dataset_name=$DATASET_NAME   --dataloader_num_workers=8   --resolution=512 --center_crop --random_flip   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=15000   --learning_rate=1e-04   --max_grad_norm=1   --lr_scheduler="cosine" --lr_warmup_steps=0   --output_dir=${OUTPUT_DIR}   --push_to_hub   --hub_model_id=${HUB_MODEL_ID}  --checkpointing_steps=500   --validation_prompt="a woman wearing red lipstick with black hair" --train_data_dir="waifu_dataset"  --seed=1337
```
5. To figure out base model name: `python3 sd_test.py`
6. To perform inference using base_model and fine-tuned LoRA weights together. `python3 sd_test2.py` - Change correct model path and prompt.
