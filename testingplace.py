import os

# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.

import torch
from trellis.models import SLatMeshDecoder
from trellis.modules import sparse as sp
from safetensors.torch import load_file
import json
import gc
torch.cuda.memory._record_memory_history()

sp.SparseTensor.init()

model_json = json.load(open("ckpts_slat_dec_mesh_swin8_B_64l8m256c_fp16.json"))
model = SLatMeshDecoder(**model_json['args'])

model_data = load_file("slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors", device='cuda')
model.load_state_dict(model_data)
model.to('cuda')
model.eval()

del model_data

gc.collect()
torch.cuda.empty_cache()

slat = torch.load(r"C:\Users\monta\TRELLIS\slat.pt", map_location='cuda')

with torch.no_grad():
    res = model(slat)

del model
gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory._dump_snapshot("snapshot.pickle")

a = 0