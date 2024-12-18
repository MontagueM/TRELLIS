# load the slat_decoder_mesh model
import torch
from safetensors.torch import load_file, save_file

from trellis.models import SLatMeshDecoder
import json
import bitsandbytes as bnb
import torch.nn as nn

def set_nested_attr(obj, attr_path, value):
    """
    Sets a nested attribute on an object given a dotted attribute path.

    Args:
        obj: The root object.
        attr_path (str): The dotted path to the attribute (e.g., 'a.b.c').
        value: The value to set for the attribute.
    """
    attrs = attr_path.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)
    
def quantize_model(model, dtype=bnb.nn.Int8Params):
    """
    Quantize the model parameters using BitsAndBytes.

    Args:
        model (torch.nn.Module): The PyTorch model to quantize.
        dtype: The data type for quantization. Defaults to 8-bit.

    Returns:
        torch.nn.Module: The quantized model.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Replace the parameter with a quantized version
            quantized_param = bnb.nn.Int8Params(param.data, requires_grad=param.requires_grad)
            set_nested_attr(model, name, quantized_param)
    return model

def get_model_size(model):
    """
    Calculate the size of the model in bytes.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: Size in bytes.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    total_size = param_size + buffer_size
    return total_size

model_json = json.load(open("ckpts_slat_dec_mesh_swin8_B_64l8m256c_fp16.json")) 
model = SLatMeshDecoder(**model_json['args'])
state_dict = load_file("slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors", device='cpu')
model.load_state_dict(state_dict)

# ============================
# 3. Calculate Original Model Size
# ============================
original_size = get_model_size(model)
print(f"Original model size: {original_size / 1e6:.2f} MB")

# ============================
# 4. Quantize the Model
# ============================
# quantized_model = quantize_model(model)
quantized_model = torch.quantization.quantize_dynamic(
    model,                      # the original model
    {nn.Linear},                # layers to quantize
    dtype=torch.qint8           # quantization data type
)
print("Model quantized successfully using BitsAndBytes.")

# ============================
# 5. Prepare Quantized State Dict for Saving
# ============================
# Since bitsandbytes uses custom parameter types (e.g., Int8Params),
# we need to convert them back to standard tensors before saving.
quantized_state_dict = {}
for name, param in quantized_model.named_parameters():
    if isinstance(param, bnb.nn.Int8Params):
        # Dequantize to float16 or float32 for saving
        quantized_state_dict[name] = param.dequantize()
    else:
        quantized_state_dict[name] = param.data

# Include buffers if any
for name, buffer in quantized_model.named_buffers():
    quantized_state_dict[name] = buffer

# ============================
# 6. Save the Quantized Model in Safetensors Format
# ============================
quantized_checkpoint_path = 'quantized_model.safetensors'
try:
    save_file(quantized_state_dict, quantized_checkpoint_path)
    print(f"Quantized model saved to {quantized_checkpoint_path}.")
except Exception as e:
    raise RuntimeError(f"Failed to save quantized model: {e}")

# ============================
# 7. Calculate Quantized Model Size
# ============================
quantized_size = get_model_size(quantized_model)
print(f"Quantized model size: {quantized_size / 1e6:.2f} MB")

# ============================
# 8. Calculate and Print Percentage Saved
# ============================
if original_size == 0:
    print("Original model size is zero, cannot compute percentage saved.")
else:
    saved_bytes = original_size - quantized_size
    percentage_saved = (saved_bytes / original_size) * 100
    print(f"Space saved: {percentage_saved:.2f}%")