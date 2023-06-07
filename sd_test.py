from huggingface_hub import model_info

model_path = "arkayoji/waifu-lora"

info = model_info(model_path)
model_base = info.cardData["base_model"]
print(model_base)