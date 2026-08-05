from typing import TYPE_CHECKING

from phenome import load_dinov2_model

if TYPE_CHECKING:
    from phenome import ModelWrapper

MODEL_OPTIONS = {
    "Small (ViT-S/14)": "dinov2_vits14_reg",
    "Base (ViT-B/14)": "dinov2_vitb14_reg",
    "Large (ViT-L/14)": "dinov2_vitl14_reg",
    "Giant (ViT-g/14)": "dinov2_vitg14_reg"
}

def get_dino_model(model_name: str, device: str='cpu') -> ModelWrapper:
    """ Load a dinov2 model for embedding extraction.

    Args:
        model_name: name of the model, as known by load_dinov2_model.
        device: The device to load the model on.
    Returns:
        The loaded model wrapper.
    """

    return load_dinov2_model(model_name, device=device)

def get_model(model_name: str, **kwargs) -> ModelWrapper:
    """ From a given model name, use the corresponding function to load the model.

    Args:
        model_name: The name of the model as known by the model loading function.
        kwargs: Additional keyword arguments to pass to the model loading function.
    Returns:
        The loaded model wrapper.
    """

    models = {
    "dinov2_vits14_reg": get_dino_model,
    "dinov2_vitb14_reg": get_dino_model,
    "dinov2_vitl14_reg": get_dino_model,
    "dinov2_vitg14_reg": get_dino_model
    }

    return models[model_name](model_name, **kwargs)

