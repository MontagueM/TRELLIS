import gc
import json
import os

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.

import safetensors
import torch
from torchinfo import summary
from trellis.models import SLatMeshDecoder
from trellis.modules import sparse as sp
from safetensors.torch import load_file
from torch.profiler import profile, record_function, ProfilerActivity


# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

torch.no_grad()
sp.SparseTensor.init()

slat = torch.load(r"C:\Users\monta\TRELLIS\slat.pt")
# move the model to the device
# old_data = slat.data
# slat.data = slat.data.replace_feature(slat.data.features.to(torch.float16))
# del old_data
model_json = json.load(open("ckpts_slat_dec_mesh_swin8_B_64l8m256c_fp16.json"))
model = SLatMeshDecoder(**model_json['args'])
model.to('cpu')
model.eval()

model_data = load_file("slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors", device='cpu')
# # quantize the model
# for k, v in model_data.items():
#     v.to("cpu")
#     if 'input_layer' in k or 'out_layer' in k or 'upsample' in k and v.dtype == torch.float32:
#         model_data[k] = v.to(torch.float16)
#         print(f"Quantized {k} to float16")
gc.collect()
torch.cuda.empty_cache()
model.load_state_dict(model_data)

# iterate over model_data and delete all tensors
keys = list(model_data.keys())
for k in keys:
    del model_data[k]
#     gc.collect()
#     torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
# quantize the model
# for k, v in model.named_parameters():
#     if 'input_layer' in k or 'out_layer' in k or 'upsample' in k and v.dtype == torch.float32:
#         old_data = v.data
#         v.data = v.data.to(torch.float16)
#         del old_data
#         print(f"Quantized {k} to float16")


gc.collect()
torch.cuda.empty_cache()

res = model(slat)

a = 0