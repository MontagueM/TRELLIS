import torch
from diffusers import SanaPipeline

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    variant="fp16",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)

prompt = 'Isometric style ancient ruins with crumbling structures, overgrown vegetation, and hidden artifacts.'
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=5.0,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)[0]

image[0].save("sana.png")