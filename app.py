raise ImportError('app.py is deprecated. Please run the Gradio app using scripts/run_app.py')

import gradio as gr
import os
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datasets import Dataset, DatasetDict
import pandas as pd
from PIL import Image
import tempfile
import shutil
import glob

# --- Model Zoo ---
MODEL_ZOO = {
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    # Add more as needed
}

# --- LoRA Gallery (Community/Local) ---
LORA_GALLERY = {
    "Waifu LoRA": "arkayoji/waifu-lora",
    "Pokemon LoRA": "arkayoji/pokemon-lora",
    # Add more as needed
}

# --- Step 1: Upload Images ---
def upload_images(files):
    temp_dir = tempfile.mkdtemp()
    for file in files:
        img = Image.open(file.name)
        img.save(os.path.join(temp_dir, os.path.basename(file.name)))
    return temp_dir, f"Uploaded {len(files)} images."

# --- Step 2: Auto-caption Images ---
def auto_caption_images(image_dir):
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image_files = glob.glob(os.path.join(image_dir, '*'))
    captions = []
    images = []
    for file in image_files:
        try:
            image = Image.open(file)
            inputs = processor(images=image, return_tensors="pt").to(device)
            pixel_values = inputs.pixel_values
            generated_ids = model.generate(pixel_values=pixel_values, max_length=100)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            captions.append(caption)
            images.append(file)
        except Exception as e:
            continue
    df = pd.DataFrame({"image": images, "text": captions})
    dataset = Dataset.from_dict({"image": images, "text": captions})
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dir = os.path.join(image_dir, "dataset")
    dataset_dict.save_to_disk(dataset_dir)
    return dataset_dir, df.to_markdown()

# --- Step 3: Select Model ---
def get_model_zoo():
    return list(MODEL_ZOO.keys())

def get_lora_gallery():
    return list(LORA_GALLERY.keys())

# --- Step 4: Train LoRA (Stub: Launches subprocess or returns command) ---
def train_lora(dataset_dir, base_model, output_dir):
    # In production, launch subprocess or Accelerate job
    # Here, just return the command for demo
    command = f"accelerate launch --mixed_precision=\"fp16\" sd.py --pretrained_model_name_or_path={base_model} --train_data_dir={dataset_dir} --output_dir={output_dir} --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=15000 --learning_rate=1e-04 --checkpointing_steps=500 --validation_prompt=\"a woman wearing red lipstick with black hair\" --seed=1337"
    return command, f"Training command generated. Run this in your terminal to start training."

# --- Step 5: Inference with LoRA, Prompt, Strength, Merging ---
def generate_image(base_model, lora_model, prompt, lora_alpha, steps, seed):
    torch.manual_seed(seed)
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet.load_attn_procs(lora_model)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    # Dynamic LoRA strength (alpha) not natively supported in diffusers yet, so this is a placeholder
    image = pipe(prompt, num_inference_steps=steps).images[0]
    return image

# --- Gradio UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 10x LoRA Fine-Tuning & Inference Platform 🚀")
    with gr.Tab("1. Upload Images"):
        image_upload = gr.File(file_count="multiple", type="file", label="Upload Images")
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Status")
        upload_dir = gr.State()
        upload_btn.click(upload_images, inputs=[image_upload], outputs=[upload_dir, upload_output])
    with gr.Tab("2. Auto-caption & Create Dataset"):
        caption_btn = gr.Button("Auto-caption & Create Dataset")
        caption_output = gr.Textbox(label="Captions Preview")
        dataset_dir = gr.State()
        caption_btn.click(auto_caption_images, inputs=[upload_dir], outputs=[dataset_dir, caption_output])
    with gr.Tab("3. Train LoRA"):
        base_model_dropdown = gr.Dropdown(get_model_zoo(), label="Base Model")
        output_dir = gr.Textbox(label="Output Directory", value="waifu")
        train_btn = gr.Button("Generate Training Command")
        train_cmd = gr.Textbox(label="Training Command")
        train_btn.click(train_lora, inputs=[dataset_dir, base_model_dropdown, output_dir], outputs=[train_cmd, train_cmd])
    with gr.Tab("4. Inference & LoRA Merging"):
        base_model_inf = gr.Dropdown(get_model_zoo(), label="Base Model")
        lora_model_inf = gr.Dropdown(get_lora_gallery(), label="LoRA Model")
        prompt = gr.Textbox(label="Prompt", value="a beautiful woman smiling at you")
        lora_alpha = gr.Slider(0, 1, value=1, label="LoRA Strength (alpha)")
        steps = gr.Slider(10, 200, value=50, label="Inference Steps")
        seed = gr.Number(value=42, label="Random Seed")
        gen_btn = gr.Button("Generate Image")
        output_img = gr.Image(label="Generated Image")
        def lora_path(lora_key):
            return LORA_GALLERY[lora_key]
        gen_btn.click(lambda bm, lm, p, a, s, sd: generate_image(MODEL_ZOO[bm], lora_path(lm), p, a, s, int(sd)),
                     inputs=[base_model_inf, lora_model_inf, prompt, lora_alpha, steps, seed], outputs=[output_img])
    gr.Markdown("---\nBuilt for hackathons, demos, and rapid innovation.")

demo.launch() 