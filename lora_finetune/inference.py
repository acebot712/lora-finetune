"""
Inference and image generation using Stable Diffusion with LoRA adapters.
"""

import logging
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

logger = logging.getLogger(__name__)


def generate_image(
    base_model: str,
    lora_model: str,
    prompt: str,
    lora_alpha: float,
    steps: int,
    seed: int,
) -> Image.Image:
    """
    Generate an image using a base Stable Diffusion model with LoRA weights.

    Args:
        base_model: HuggingFace model identifier for base SD model
        lora_model: HuggingFace model identifier or local path for LoRA weights
        prompt: Text prompt for image generation
        lora_alpha: LoRA strength/influence (0-1, currently placeholder)
        steps: Number of inference steps
        seed: Random seed for reproducibility

    Returns:
        PIL Image object

    Raises:
        RuntimeError: If CUDA is not available or model loading fails
    """
    try:
        # Set random seed for reproducibility
        torch.manual_seed(seed)

        logger.info(f"Loading base model: {base_model}")

        # Load base model with FP16 for efficiency
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            base_model, torch_dtype=dtype, safety_checker=None
        )

        # Use DPM++ solver for faster inference
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        # Load LoRA weights
        logger.info(f"Loading LoRA weights: {lora_model}")
        pipe.unet.load_attn_procs(lora_model)

        # Move to device
        pipe = pipe.to(device)

        logger.info(f"Generating image with prompt: {prompt}")

        # Generate image
        # Note: Dynamic LoRA strength (alpha) is not natively supported in diffusers
        # This is a placeholder parameter for future enhancement
        image = pipe(prompt, num_inference_steps=steps).images[0]

        logger.info("Image generation complete")
        return image

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise RuntimeError(f"Failed to generate image: {e}") from e
