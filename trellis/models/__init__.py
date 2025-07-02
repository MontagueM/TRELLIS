import importlib
import os
import time
from typing import Optional

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

# Global TransferManager instance
_transfer_manager = None

def _get_transfer_manager():
    """Get or create the global TransferManager instance."""
    global _transfer_manager
    if _transfer_manager is None:
        try:
            from google.cloud import storage
            from google.cloud.storage.transfer_manager import TransferManager
            client = storage.Client()
            _transfer_manager = TransferManager(client)
        except ImportError as e:
            print(f"Failed to import Google Cloud Storage: {e}")
            print("Please install google-cloud-storage: pip install google-cloud-storage")
            return None
    return _transfer_manager

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


def download_file_parallel(path: str, max_workers: int = 8, chunk_size: int = 32 * 1024 * 1024) -> Optional[str]:
    """
    Download a file from GCS using parallel chunked downloads with TransferManager.
    
    Args:
        path: The file path to download
        max_workers: Maximum number of parallel download workers for chunks
        chunk_size: Size of each download chunk in bytes (default: 32MB)
        
    Returns:
        Local path of downloaded file, or None if failed
    """
    start_time = time.time()
    
    manager = _get_transfer_manager()
    if manager is None:
        return None
    
    # Extract the fixed path by removing /models/ prefix if present
    if path.startswith('/models/'):
        blob_name = path[8:]  # Remove '/models/' (8 characters)
    else:
        blob_name = path.lstrip('/')  # Remove leading slash if present
    
    bucket_name = "mont-test-vector"
    
    # Save to current working directory
    filename = os.path.basename(path)
    local_download_path = f"{filename}.gcs_download"
    
    try:
        print(f"Starting parallel chunked download: gs://{bucket_name}/{blob_name}")
        print(f"Workers: {max_workers}, Chunk size: {chunk_size // (1024*1024)}MB")
        
        manager.download(
            bucket_name=bucket_name,
            blob_name=blob_name,
            download_path=local_download_path,
            max_workers=max_workers,
            chunk_size=chunk_size
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get file size for reporting
        file_size = os.path.getsize(local_download_path) if os.path.exists(local_download_path) else 0
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"Downloaded gs://{bucket_name}/{blob_name} to {local_download_path}")
        print(f"Size: {file_size_mb:.2f}MB, Time: {duration:.4f}s, Speed: {file_size_mb/duration:.2f}MB/s")
        
        return local_download_path
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"Failed to download gs://{bucket_name}/{blob_name} after {duration:.4f} seconds - Error: {e}")
        return None


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint, downloading from GCS if needed.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import json
    from safetensors.torch import load_file
    
    config_path = f"{path}.json"
    model_path = f"{path}.safetensors"
    
    is_local = os.path.exists(config_path) and os.path.exists(model_path)
    print("in models.from_pretrained, is_local: ", is_local)
    
    if is_local:
        config_file = config_path
        model_file = model_path
    else:
        # Try to download from GCS first using parallel chunked downloads
        print("Attempting to download config file from GCS...")
        downloaded_config = download_file_parallel(config_path)
        
        print("Attempting to download model file from GCS...")
        downloaded_model = download_file_parallel(model_path)
        
        if downloaded_config and downloaded_model:
            # Use downloaded files
            config_file = downloaded_config
            model_file = downloaded_model
            print(f"Using downloaded files: config={config_file}, model={model_file}")
        else:
            # Fallback to Hugging Face Hub
            print("GCS download failed, falling back to Hugging Face Hub...")
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
    
    # Load the model state dict from the appropriate file
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
