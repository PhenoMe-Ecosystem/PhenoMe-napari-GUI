from typing import TYPE_CHECKING

from phenome import load_dinov2_model

if TYPE_CHECKING:
    from phenome import ModelWrapper


def get_dino_model(model_name: str, device: str = "cpu") -> "ModelWrapper":
    """Load a DINOv2 model for embedding extraction.

    Args:
        model_name: Name of the model, as known by load_dinov2_model.
        device: The device to load the model on.

    Returns:
        The loaded model wrapper.
    """
    return load_dinov2_model(model_name, device=device)

MODEL_OPTIONS = {
    "Small (ViT-S/14)": "dinov2_vits14_reg",
    "Base (ViT-B/14)": "dinov2_vitb14_reg",
    "Large (ViT-L/14)": "dinov2_vitl14_reg",
    "Giant (ViT-g/14)": "dinov2_vitg14_reg",
}

MODEL_LOADERS = {
    "dinov2_vits14_reg": get_dino_model,
    "dinov2_vitb14_reg": get_dino_model,
    "dinov2_vitl14_reg": get_dino_model,
    "dinov2_vitg14_reg": get_dino_model,
}

def get_model(model_name: str, **kwargs) -> "ModelWrapper":
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
        loader = MODEL_LOADERS[model_name]
    except KeyError as err:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available models: {', '.join(MODEL_LOADERS)}"
        ) from err

    return loader(model_name, **kwargs)
