# Fine-Tuning StableDiffusion with LoRA on custom dataset


<figure>
  <img src="generated_sample.png" alt="Generated Sample" style="width:100%">
  <figcaption>Fig.1 - Generated Sample after finetuning SD with LoRA.</figcaption>
</figure>

Using this repository, you should be able to create your own LoRA models that you can upload to [CivitAI](https://civitai.com/) without using [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

> LoRA models can be made with significantly low compute and memory requirements as opposed to things like Dreambooth, CLIP, or textual inversion (about a few MBs).

## Replicate Machine setup

Setup a T4 instance using [this image](https://cloud.google.com/deep-learning-vm/docs/pytorch_start_instance#creating_a_pytorch_instance_from_the).

## How to Use

Create the environment

1. Install Diffusers from source as we will need to modify SD libraries directly:- `pip install git+https://github.com/huggingface/diffusers`. Comment out the lines that consist of `# black image` to turn off the safety filter.
2. `pip install accelerate transformers datasets evaluate`
3. Assuming torch is already installed; `pip3 install torchvision torchaudio`

## Running fine-tuning

1. Make a folder consisting of images that you want to fine-tune a LoRA model. Don't worry about the folder structure. Make sure the image files are in a format recognized by [Pillow](https://pypi.org/project/Pillow/).
2. Create `(image, text)` image -caption pairs dataset with your images with `python3 blip_captions.py` - (Change any hardcoded values in the file with your requirements first)
3. Update `DATASET_NAME_MAPPING` in `sd.py`
4. The following command is meant to fine-tune your model. Change the parameters' values as necessary.
```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="waifu"
export HUB_MODEL_ID="waifu-lora"

accelerate launch --mixed_precision="fp16"  sd.py   --pretrained_model_name_or_path=$MODEL_NAME   --dataset_name=$DATASET_NAME   --dataloader_num_workers=8   --resolution=512 --center_crop --random_flip   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=15000   --learning_rate=1e-04   --max_grad_norm=1   --lr_scheduler="cosine" --lr_warmup_steps=0   --output_dir=${OUTPUT_DIR}   --push_to_hub   --hub_model_id=${HUB_MODEL_ID}  --checkpointing_steps=500   --validation_prompt="a woman wearing red lipstick with black hair" --train_data_dir="waifu_dataset"  --seed=1337
```
5. To figure out base model name: `python3 sd_test.py`
6. To perform inference using base_model and fine-tuned LoRA weights together. `python3 sd_test2.py` - Change correct model path and prompt.
