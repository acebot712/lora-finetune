"""
Standalone script for auto-captioning images using BLIP.

Usage:
    python blip_captions.py <image_directory> [--output output_dir] [--max-images N]

Example:
    python blip_captions.py ./my_images --output waifu_dataset --max-images 50
"""

from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import pandas as pd
import glob
from tqdm import tqdm
import random
from datasets import Dataset, DatasetDict
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Auto-caption images using BLIP')
    parser.add_argument('image_dir', type=str, help='Directory containing images')
    parser.add_argument('--output', type=str, default='captioned_dataset',
                        help='Output directory for dataset (default: captioned_dataset)')
    parser.add_argument('--max-images', type=int, default=50,
                        help='Maximum number of images to process (default: 50)')
    args = parser.parse_args()

    if not os.path.exists(args.image_dir):
        print(f"Error: Directory '{args.image_dir}' does not exist")
        return 1

    print(f"Loading BLIP model...")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    directory_paths = [args.image_dir]

    image_files = []
    actual_image_files = []

    for directory_path in directory_paths:
        files = glob.glob(directory_path + "/*.*")
        image_files.extend(files[:args.max_images])

    random.shuffle(image_files)
    print(f"Found {len(image_files)} images to process")

    generated_captions = []

    for file in tqdm(image_files, desc="Processing images"):
        # Load the image
        try:
            image = Image.open(file)
        except Exception as e:
            print(f"\nWarning: Failed to open {file}: {e}")
            continue

        # Prepare image for the model
        inputs = processor(images=image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values

        # Generate caption
        generated_ids = model.generate(pixel_values=pixel_values, max_length=100)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Store the generated caption
        generated_captions.append(generated_caption)
        actual_image_files.append(image)

    # Create a dictionary with image and text features
    data_dict = {"image": actual_image_files, "text": generated_captions}

    df = pd.DataFrame(data_dict)
    print("\nGenerated captions preview:")
    print(df.head())

    # Create a Dataset using the dictionary
    dataset = Dataset.from_dict(data_dict)

    # Create a DatasetDict with the train split
    dataset_dict = DatasetDict({"train": dataset})

    # Save the dataset to disk
    print(f"\nSaving dataset to: {args.output}")
    dataset_dict.save_to_disk(args.output)

    print(f"\nDone! Dataset saved with {len(generated_captions)} captioned images.")
    print(f"You can now use this dataset for training:")
    print(f"  --train_data_dir={args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
