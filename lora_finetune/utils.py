"""
Utility functions for model zoo, LoRA gallery, and helper functions.
"""

# Model Zoo - Base models available for fine-tuning
MODEL_ZOO = {
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
}

# LoRA Gallery - Community and local LoRA weights
LORA_GALLERY = {
    "Waifu LoRA": "arkayoji/waifu-lora",
    "Pokemon LoRA": "arkayoji/pokemon-lora",
}


def get_model_zoo():
    """Return list of available base model names."""
    return list(MODEL_ZOO.keys())


def get_lora_gallery():
    """Return list of available LoRA model names."""
    return list(LORA_GALLERY.keys())


def model_path(model_key: str) -> str:
    """
    Get the HuggingFace model path for a given model key.

    Args:
        model_key: Human-readable model name from MODEL_ZOO

    Returns:
        HuggingFace model identifier/path
    """
    return MODEL_ZOO.get(model_key, model_key)


def lora_path(lora_key: str) -> str:
    """
    Get the HuggingFace LoRA path for a given LoRA key.

    Args:
        lora_key: Human-readable LoRA name from LORA_GALLERY

    Returns:
        HuggingFace LoRA identifier/path
    """
    return LORA_GALLERY.get(lora_key, lora_key)
