import importlib
import os
import time
import shutil
import subprocess

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatRadianceFieldDecoder': 'structured_latent_vae',
    'SLatMeshDecoder': 'structured_latent_vae',
    'ElasticSLatEncoder': 'structured_latent_vae',
    'ElasticSLatGaussianDecoder': 'structured_latent_vae',
    'ElasticSLatRadianceFieldDecoder': 'structured_latent_vae',
    'ElasticSLatMeshDecoder': 'structured_latent_vae',
    
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def dl1(path: str):
    """
    Copy a file to the local path and print how long it took.
    
    Args:
        path: The source file path to copy
    """
    start_time = time.time()
    
    # Create a local copy in the current working directory
    filename = os.path.basename(path)
    local_path = f"{filename}.local_copy"
    
    try:
        shutil.copy2(path, local_path)
        end_time = time.time()
        duration = end_time - start_time
        print(f"dl1: Copied {path} to {local_path} in {duration:.4f} seconds")
        return local_path
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"dl1: Failed to copy {path} after {duration:.4f} seconds - Error: {e}")
        return None


def dl2(path: str):
    """
    Download a file from GCS and print how long it took.
    
    Args:
        path: The file path (e.g., /models/trellis-text-xlarge/model.safetensors)
              Will be converted to gs://mont-test-vector/trellis-text-xlarge/model.safetensors
    """
    start_time = time.time()
    
    # Extract the fixed path by removing /models/ prefix if present
    if path.startswith('/models/'):
        fixed_path = path[8:]  # Remove '/models/' (8 characters)
    else:
        fixed_path = path.lstrip('/')  # Remove leading slash if present
    
    gcs_path = f"gs://mont-test-vector/{fixed_path}"
    
    # Save to current working directory
    filename = os.path.basename(path)
    local_download_path = f"{filename}.gcs_download"
    
    try:
        # Use gsutil to download from GCS
        result = subprocess.run(
            ['gsutil', 'cp', gcs_path, local_download_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"dl2: Downloaded {gcs_path} to {local_download_path} in {duration:.4f} seconds")
        return local_download_path
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"dl2: Failed to download {gcs_path} after {duration:.4f} seconds - Error: {e.stderr}")
        return None
    except FileNotFoundError:
        end_time = time.time()
        duration = end_time - start_time
        print(f"dl2: gsutil not found. Failed to download {gcs_path} after {duration:.4f} seconds")
        return None
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"dl2: Unexpected error downloading {gcs_path} after {duration:.4f} seconds - Error: {e}")
        return None


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")
    print("in models.from_pretrained, is_local: ", is_local)
    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    print("in models.from_pretrained, config: ", config)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    print("in models.from_pretrained, model from getattr: ", model)
    download_model1 = dl1(model_file)
    download_model2 = dl2(model_file)
    model.load_state_dict(load_file(model_file))
    print("in models.from_pretrained, model after load_state_dict: ", model)
    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import (
        SparseStructureEncoder, 
        SparseStructureDecoder,
    )
    
    from .sparse_structure_flow import SparseStructureFlowModel
    
    from .structured_latent_vae import (
        SLatEncoder,
        SLatGaussianDecoder,
        SLatRadianceFieldDecoder,
        SLatMeshDecoder,
        ElasticSLatEncoder,
        ElasticSLatGaussianDecoder,
        ElasticSLatRadianceFieldDecoder,
        ElasticSLatMeshDecoder,
    )
    
    from .structured_latent_flow import (
        SLatFlowModel,
        ElasticSLatFlowModel,
    )
