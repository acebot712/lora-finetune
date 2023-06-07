from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import pandas as pd
import glob
from tqdm import tqdm
import random
from datasets import Dataset, DatasetDict, load_dataset
import os
import jsonlines

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

directory_paths = [
    # "HQ_512x512/HQ_512x512/",
    # "DataSet/train/nsfw/",
    "bella/"
    # Add more directory paths here
]

image_files = []
actual_image_files = []

for directory_path in directory_paths:
    files = glob.glob(directory_path + "*.*")
    image_files.extend(files[:50])  # Adjust the number of files as needed

random.shuffle(image_files)

generated_captions = []

for file in tqdm(image_files, desc="Processing images"):
    # Load the image
    try:
        image = Image.open(file)
    except:
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
print(df)

# Create a Dataset using the dictionary
dataset = Dataset.from_dict(data_dict)

# # Create a DatasetDict with the train split
dataset_dict = DatasetDict({"train": dataset})

# Save the dataset to disk
dataset_dict.save_to_disk("waifu_dataset")

# Load the dataset from disk
loaded_dataset_dict = DatasetDict.load_from_disk("waifu_dataset")

# Print the loaded dataset information
print(loaded_dataset_dict)
# print(loaded_dataset_dict["train"][0])
