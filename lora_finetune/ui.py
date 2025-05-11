import gradio as gr
import logging
from lora_finetune.data import upload_images, auto_caption_images
from lora_finetune.training import train_lora
from lora_finetune.inference import generate_image
from lora_finetune.utils import get_model_zoo, get_lora_gallery, lora_path, model_path

logger = logging.getLogger(__name__)

def build_ui():
    """Build and return the Gradio Blocks UI for the LoRA fine-tuning platform."""
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
            gen_btn.click(
                lambda bm, lm, p, a, s, sd: generate_image(
                    model_path(bm), lora_path(lm), p, a, s, int(sd)
                ),
                inputs=[base_model_inf, lora_model_inf, prompt, lora_alpha, steps, seed],
                outputs=[output_img],
            )
        gr.Markdown("---\nBuilt for hackathons, demos, and rapid innovation.")
    return demo 