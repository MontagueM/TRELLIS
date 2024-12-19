import gc
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..representations import Gaussian, Strivec, MeshExtractResult
from time import time

class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
            self,
            models: dict[str, nn.Module] = None,
            sparse_structure_sampler: samplers.Sampler = None,
            slat_sampler: samplers.Sampler = None,
            slat_normalization: dict = None,
            image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        
        self._device = next(self.models['image_cond_model'].parameters()).device
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)
        self._print_vram_usage("After initialization")
        self.step_callback = None

    @property
    def device(self) -> torch.device:
        """Override device property to ensure it persists"""
        if not hasattr(self, '_device'):
            self._device = torch.device('cuda')
        return self._device

    def _print_vram_usage(self, message: str):
        return
        """
        Print the current VRAM usage with a custom message.

        Args:
            message (str): The message to display alongside VRAM stats.
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            print(f"[VRAM] {message}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        else:
            print("[VRAM] CUDA not available.")

    def load_model(self, model_key: str) -> nn.Module:
        """Move a model to CUDA and return it."""
        # print(f"Loading model '{model_key}' to CUDA.")
        self.models[model_key].to(self.device)
        self._print_vram_usage(f"After loading '{model_key}'")
        return self.models[model_key]

    def unload_models(self, model_keys: List[str], delete=True):
        """Unload models to CPU."""
        for key in model_keys:
            if key in self.models:
                # print(f"Unloading model '{key}' to CPU.")
                self.models[key].to(torch.device("cpu"))
                if delete:
                    del self.models[key]
                gc.collect()
                torch.cuda.empty_cache()
        self._print_vram_usage(f"After unloading models: {model_keys}")

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])
        new_pipeline._print_vram_usage("After loading pretrained pipeline")

        return new_pipeline

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """

        # load slat and execute mesh
        # self._print_vram_usage("Before loading slat and executing mesh")
        # slat = torch.load('slat.pt')
        # # slat = slat.to(self.device)
        # self._print_vram_usage("After loading slat")
        # mesh_decoder = self.load_model('slat_decoder_mesh')
        # # init as its lazy load
        # sp.SparseTensor.init()
        # mesh = mesh_decoder(slat)
        # self._print_vram_usage("After executing mesh")
        
        print(f"Initializing image conditioning model '{name}'.")
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform
        self._print_vram_usage(f"After initializing image_cond_model '{name}'")

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        start_time = time()
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        self._print_vram_usage("After preprocessing image")
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        self._print_vram_usage("Before encoding image")
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.image_cond_model_transform(image).to(self.device)
        self._print_vram_usage("After transforming image")

        # Load the image conditioning model if not already loaded
        image_cond_model = self.load_model('image_cond_model')

        features = image_cond_model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        self._print_vram_usage("After encoding image")
        
        # Unload the image conditioning model
        self.unload_models(['image_cond_model'])
        
        return patchtokens

    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond).to(self.device)
        self._print_vram_usage("After getting conditioning information")
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
            self,
            cond: dict,
            num_samples: int = 1,
            sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        self._print_vram_usage("Before sampling sparse structure")
        # Load the sparse structure flow model
        flow_model = self.load_model('sparse_structure_flow_model')
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        self._print_vram_usage("After sampling sparse structure")

        # Decode occupancy latent
        decoder = self.load_model('sparse_structure_decoder')
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        self.unload_models(['sparse_structure_decoder'])
        self.unload_models(['sparse_structure_flow_model'])

        return coords

    def decode_slat(
            self,
            slat: sp.SparseTensor,
            start_time: float,
            formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        # print slat shape
        # print(f"SLAT shape: {slat.shape}")
    
        # tensor code requires int32 for get_indice_pairs
        # slat.data.indices = slat.data.indices.to(torch.uint8)
        # slat.data = slat.data.replace_feature(slat.data.features.to(torch.float16))
                
        # save entire slat object
        # torch.save(slat, 'slat.pt')
        
        ret = {}
        if 'gaussian' in formats:
            print(f"({time() - start_time:.2f}s) Decoding gaussian...")
            gs_decoder = self.load_model('slat_decoder_gs')
            ret['gaussian'] = gs_decoder(slat)
            self.unload_models(['slat_decoder_gs'])
            self._print_vram_usage("After decoding gaussian")
        if 'radiance_field' in formats:
            print(f"({time() - start_time:.2f}s) Decoding radiance field...")
            rf_decoder = self.load_model('slat_decoder_rf')
            ret['radiance_field'] = rf_decoder(slat)
            self.unload_models(['slat_decoder_rf'])
            self._print_vram_usage("After decoding radiance field")
        if 'mesh' in formats:
            print(f"({time() - start_time:.2f}s) Decoding mesh...")
            mesh_decoder = self.load_model('slat_decoder_mesh')
            ret['mesh'] = mesh_decoder(slat)
            self.unload_models(['slat_decoder_mesh'])
            self._print_vram_usage("After decoding mesh")
            
        self._print_vram_usage("After decoding all formats")
        return ret

    def sample_slat(
            self,
            cond: dict,
            coords: torch.Tensor,
            sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        self._print_vram_usage("Before sampling structured latent")
        # Load the structured latent flow model
        flow_model = self.load_model('slat_flow_model')
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        self.unload_models(['slat_flow_model'])
        self._print_vram_usage("After sampling structured latent")

        std = torch.tensor(self.slat_normalization['std']).to(slat.device).unsqueeze(0)
        mean = torch.tensor(self.slat_normalization['mean']).to(slat.device).unsqueeze(0)
        slat = slat * std + mean
        self._print_vram_usage("After normalizing structured latent")

        return slat

    @torch.no_grad()
    def run(
            self,
            image: Image.Image,
            num_samples: int = 1,
            seed: int = 42,
            sparse_structure_sampler_params: dict = {},
            slat_sampler_params: dict = {},
            formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
            preprocess_image: bool = True,
            step_callback: Optional[Callable[[int], None]] = None,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        self.step_callback = step_callback
        start_time = time()
        # Unload decoders if they are loaded
        # self.unload_models(['slat_decoder_mesh', 'slat_decoder_gs', 'slat_decoder_rf'], delete=False)
        self.unload_models(['sparse_structure_flow_model','sparse_structure_decoder', 'slat_flow_model', 'slat_decoder_mesh', 'slat_decoder_gs', 'slat_decoder_rf'], delete=False)
        self._print_vram_usage("After unloading decoders at start")
        self.step_callback(0)
        if preprocess_image:
            print(f"({time() - start_time:.2f}s) Preprocessing image...")
            image = self.preprocess_image(image)

        self.step_callback(1)
        print(f"({time() - start_time:.2f}s) Getting conditional info of image...")
        cond = self.get_cond([image])

        torch.manual_seed(seed)
        self._print_vram_usage("After setting seed and getting conditioning")

        self.step_callback(2)
        print(f"({time() - start_time:.2f}s) Sampling sparse structure...")
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)

        self.step_callback(3)
        print(f"({time() - start_time:.2f}s) Sampling structured latent...")
        slat = self.sample_slat(cond, coords, slat_sampler_params)

        self.step_callback(4)
        print(f"({time() - start_time:.2f}s) Decoding structured latent...")
        decoded = self.decode_slat(slat, start_time, formats)
        print(f"({time() - start_time:.2f}s) Done.")
        self.step_callback(5)

        self._print_vram_usage("After running pipeline")
        return decoded
