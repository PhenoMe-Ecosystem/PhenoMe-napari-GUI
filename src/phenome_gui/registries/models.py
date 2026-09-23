from typing import TYPE_CHECKING

import torch
from phenome import load_dinov2_model
from phenome.utils import ModelWrapper
from stedfm import get_pretrained_model_v2

from .process_options import OPTIONS_LOADERS
from .transforms import TRANSFORMS_LOADERS

if TYPE_CHECKING:
    import torch

class STEDFMWrapper(ModelWrapper):
    def __init__(self, device: str='cpu', in_channels: int=1, ):
        # Load the pretrained STED-FM model
        model, cfg = get_pretrained_model_v2(
            name = "mae-lightning-small",
            weights = "MAE_SMALL_STED",
            in_channels=in_channels, # For single channel images
            as_classifier=True,
            # global_pool = "patch" # Returns the embeddings for all patches
        )

        # Change input to 3 channels to fit PhenoMe pipeline
        # model.backbone.patch_embed.proj = torch.nn.Conv2d(3, 384, kernel_size=(16,16), stride=(16,16))

        # Initialize the parent class which moves the model to the device and sets it to eval()
        super().__init__(model=model, device=device)

    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_features(tensor)
        return out

def get_dino_model(model_name: str, device: str = "cpu") -> "ModelWrapper":
    """Load a DINOv2 model for embedding extraction.

    Args:
        model_name: Name of the model, as known by load_dinov2_model.
        device: The device to load the model on.

    Returns:
        The loaded model wrapper.
    """
    return load_dinov2_model(model_name, device=device)

def get_stedfm_model(model_name: str, device: str = "cpu") -> "ModelWrapper":
    """Load the STED-FM ViT-S model for embedding extraction.

    Args:
        device: The device to load the model on.

    Returns:
        The loaded model wrapper.
    """
    return STEDFMWrapper(device=device)

MODEL_OPTIONS = {
    "DINOv2- Small": "dinov2_vits14_reg",
    "DINOv2- Base": "dinov2_vitb14_reg",
    "DINOv2- Large": "dinov2_vitl14_reg",
    "DINOv2- Giant": "dinov2_vitg14_reg",
    "STED-FM - Small": "sted-fm"
}

MODEL_LOADERS = {
    "dinov2_vits14_reg": get_dino_model,
    "dinov2_vitb14_reg": get_dino_model,
    "dinov2_vitl14_reg": get_dino_model,
    "dinov2_vitg14_reg": get_dino_model,
    "sted-fm": get_stedfm_model
}

def get_model(model_name: str, device: str = "cpu", **kwargs) -> "ModelWrapper":
    """Load a model by its registry name.

    Args:
        model_name: The name of the model as it appears in MODEL_LOADERS.
        kwargs: Additional keyword arguments forwarded to the loader.

    Returns:
        The loaded model wrapper.

    Raises:
        ValueError: If model_name is not found in the registry.
    """
    try:
        model_loader = MODEL_LOADERS[model_name]
        options_loader = OPTIONS_LOADERS[model_name]
        transforms_loader = TRANSFORMS_LOADERS[model_name]
    except KeyError as err:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available models: {', '.join(MODEL_LOADERS)}"
        ) from err

    return model_loader(model_name, device=device, **kwargs), transforms_loader(), options_loader()
