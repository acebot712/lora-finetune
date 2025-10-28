import os
import tempfile
import glob
import logging
from typing import List, Tuple
from PIL import Image
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch

logger = logging.getLogger(__name__)


def upload_images(files: List) -> Tuple[str, str]:
    """
    Save uploaded image files to a temporary directory.
    Args:
        files: List of file-like objects
    Returns:
        temp_dir: Path to directory with saved images
        status: Status message
    """
    if not files:
        return "", "Error: No files provided"

    temp_dir = tempfile.mkdtemp()
    successful = 0
    failed = 0

    for file in files:
        try:
            img = Image.open(file.name)
            img.save(os.path.join(temp_dir, os.path.basename(file.name)))
            successful += 1
        except Exception as e:
            logger.warning(f"Failed to process {file.name}: {e}")
            failed += 1

    logger.info(f"Uploaded {successful} images to {temp_dir}")

    if successful == 0:
        return "", f"Error: Failed to upload all {len(files)} images"

    status = f"Uploaded {successful} images."
    if failed > 0:
        status += f" ({failed} failed)"

    return temp_dir, status


def auto_caption_images(image_dir: str) -> Tuple[str, str]:
    """
    Auto-caption images in a directory using BLIP and create a HuggingFace dataset.
    Args:
        image_dir: Path to directory with images
    Returns:
        dataset_dir: Path to saved dataset
        preview: Markdown table of image paths and captions
    """
    if not image_dir or not os.path.exists(image_dir):
        return "", f"Error: Directory '{image_dir}' does not exist"

    try:
        logger.info("Loading BLIP model...")
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    except Exception as e:
        logger.error(f"Failed to load BLIP model: {e}")
        return "", f"Error: Failed to load BLIP model: {e}"

    image_files = glob.glob(os.path.join(image_dir, '*'))
    # Filter only image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        return "", f"Error: No image files found in {image_dir}"

    captions = []
    images = []
    failed = 0

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
            logger.warning(f"Failed to caption {file}: {e}")
            failed += 1
            continue

    if not captions:
        return "", "Error: Failed to caption any images"

    df = pd.DataFrame({"image": images, "text": captions})
    dataset = Dataset.from_dict({"image": images, "text": captions})
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dir = os.path.join(image_dir, "dataset")
    dataset_dict.save_to_disk(dataset_dir)

    logger.info(f"Auto-captioned {len(images)} images. Dataset saved to {dataset_dir}")

    status = f"Captioned {len(images)} images"
    if failed > 0:
        status += f" ({failed} failed)"

    return dataset_dir, df.to_markdown() + f"\n\n{status}" 