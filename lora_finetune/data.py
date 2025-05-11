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
    temp_dir = tempfile.mkdtemp()
    for file in files:
        img = Image.open(file.name)
        img.save(os.path.join(temp_dir, os.path.basename(file.name)))
    logger.info(f"Uploaded {len(files)} images to {temp_dir}")
    return temp_dir, f"Uploaded {len(files)} images."


def auto_caption_images(image_dir: str) -> Tuple[str, str]:
    """
    Auto-caption images in a directory using BLIP and create a HuggingFace dataset.
    Args:
        image_dir: Path to directory with images
    Returns:
        dataset_dir: Path to saved dataset
        preview: Markdown table of image paths and captions
    """
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
            logger.warning(f"Failed to caption {file}: {e}")
            continue
    df = pd.DataFrame({"image": images, "text": captions})
    dataset = Dataset.from_dict({"image": images, "text": captions})
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dir = os.path.join(image_dir, "dataset")
    dataset_dict.save_to_disk(dataset_dir)
    logger.info(f"Auto-captioned {len(images)} images. Dataset saved to {dataset_dir}")
    return dataset_dir, df.to_markdown() 