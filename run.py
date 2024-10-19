from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("Alpha-VLLM/Lumina-Next-SFT-diffusers")
pipeline.to("cuda")
print(pipeline)
# Generate the image
image = pipeline("An image of a squirrel in Picasso style", height=512, width=512,).images[0]

# Save the image to disk
image.save("squirrel_picasso_style.png")