import torch
from phenome import load_dinov2_model
from phenome.utils import ModelWrapper

from .process_options import OPTIONS_LOADERS
from .stedfm import STEDFM, load_backbone_weights
from .transforms import TRANSFORMS_LOADERS


class STEDFMWrapper(ModelWrapper):
    """ The pretrained STED-FM ViT-S, as PhenoMe extracts embeddings with it.

    The model is the one built in this repository, with the pretrained backbone loaded into it: the
    embedding of an image is the mean of the backbone's patch tokens, its class token left out.
    """

    def __init__(self, device: str = 'cpu', in_channels: int = 1) -> None:
        """ Build the model and load the pretrained backbone into it.

        The weights are downloaded the first time a model is built, and read from the torch hub cache
        afterwards.

        Args:
            device: The device to load the model on.
            in_channels: The number of channels the images are given in, a single one for STED-FM.
        Raises:
            RuntimeError: The weights could not be downloaded, or what was downloaded is not them.
        """
        model = STEDFM(in_channels=in_channels)
        model.backbone.load_state_dict(load_backbone_weights(), strict=True)

        # Moves the model onto the device and puts it in evaluation
        super().__init__(model=model, device=device)

    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        """ The embeddings of a batch of images.

        Args:
            tensor: The images, as a (N, C, H, W) tensor.
        Returns:
            The (N, 384) embeddings.
        """
        return self.model.forward_features(tensor)

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
        model_name: Name of the model, as registered. There is only one STED-FM model.
        device: The device to load the model on.

    Returns:
        The loaded model wrapper.
    """
    return STEDFMWrapper(device=device)

MODEL_OPTIONS = {
    "DINOv2 - Small": "dinov2_vits14_reg",
    "DINOv2 - Base": "dinov2_vitb14_reg",
    "DINOv2 - Large": "dinov2_vitl14_reg",
    "DINOv2 - Giant": "dinov2_vitg14_reg",
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
