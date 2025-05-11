# 10x LoRA Fine-Tuning Platform

Fine-tune and deploy Stable Diffusion with LoRA adapters in minutes. Built for hackathons, demos, and rapid innovation.

## Features
- Web UI: Upload, caption, train, and generate images
- Model Zoo: Use top base models and community LoRAs
- LoRA Merging & Dynamic Strength
- Fast, batch inference (ONNX/TensorRT ready)
- Docker & pip install support

![Demo GIF](demo.gif)

## Try It Now
Run:
```bash
python app.py
```
Or launch with Docker:
```bash
docker build -t lora-finetune .
docker run -p 7860:7860 lora-finetune
```

---

Impress at your next hackathon or pitch! 🚀 