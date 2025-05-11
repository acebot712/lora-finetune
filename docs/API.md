# API Reference

## Modular Architecture
All core logic is organized in the `lora_finetune/` package:
- `data.py`: Data upload and auto-captioning
- `training.py`: LoRA training command generation
- `inference.py`: Inference and image generation
- `utils.py`: Model zoo, LoRA gallery, and helpers
- `ui.py`: Gradio UI orchestration

## lora_finetune.data.upload_images(files)
- **Description:** Uploads images and saves them to a temporary directory.
- **Parameters:**
  - `files`: List of file objects (from Gradio or file upload)
- **Returns:**
  - `temp_dir`: Path to the directory with saved images
  - `status`: Status message
- **Example:**
```python
upload_images([open('img1.png', 'rb'), open('img2.png', 'rb')])
```

## lora_finetune.data.auto_caption_images(image_dir)
- **Description:** Uses BLIP to auto-caption images in a directory and creates a HuggingFace dataset.
- **Parameters:**
  - `image_dir`: Path to directory with images
- **Returns:**
  - `dataset_dir`: Path to saved dataset
  - `preview`: Markdown table of image paths and captions
- **Example:**
```python
auto_caption_images('my_images/')
```

## lora_finetune.training.train_lora(dataset_dir, base_model, output_dir)
- **Description:** Generates a command to train a LoRA model using the specified dataset and base model.
- **Parameters:**
  - `dataset_dir`: Path to dataset
  - `base_model`: Model identifier (e.g., 'runwayml/stable-diffusion-v1-5')
  - `output_dir`: Output directory for LoRA weights
- **Returns:**
  - `command`: Training command string
  - `status`: Status message
- **Example:**
```python
train_lora('waifu_dataset', 'runwayml/stable-diffusion-v1-5', 'waifu')
```

## lora_finetune.inference.generate_image(base_model, lora_model, prompt, lora_alpha, steps, seed)
- **Description:** Loads a base model and LoRA weights, generates an image from a prompt.
- **Parameters:**
  - `base_model`: Model identifier
  - `lora_model`: LoRA weights identifier or path
  - `prompt`: Text prompt
  - `lora_alpha`: LoRA strength (currently placeholder)
  - `steps`: Number of inference steps
  - `seed`: Random seed
- **Returns:**
  - `image`: PIL Image object
- **Example:**
```python
img = generate_image('runwayml/stable-diffusion-v1-5', 'arkayoji/waifu-lora', 'a cat astronaut', 1.0, 50, 42)
img.save('cat.png')
``` 