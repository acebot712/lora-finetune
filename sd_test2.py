import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs("arkayoji/waifu-lora")
# pipe.safety_checker = lambda images, clip_input: (images, False)
pipe = pipe.to("cuda")

image = pipe("a woman with blue eyes on the beach smiling", num_inference_steps=200).images[0]
image.save("green_pokemon.png")
