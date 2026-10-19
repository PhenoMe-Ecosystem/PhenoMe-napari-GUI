import torchvision.transforms as T
from phenome.utils.transforms import PadToSize

"""
Custom transformations for the different model options.

Every loader takes the resize and pad sizes the run is made with, so each model is brought to the size
asked for in the form, whether it prepares its images itself or leaves that to PhenoMe.
"""

def get_stedfm_transforms(resize_size: int | None = 224, pad_size: int | None = None) -> "T.Compose":
    """ Get the transforms to be used with the STED-FM model.

    Padded then resized, in the order PhenoMe prepares its own images, so the two sizes mean the same
    for every model: the image is padded up to pad_size when smaller, then resized to resize_size.

    Args:
        resize_size: The square size the image is resized to, or None to keep its size.
        pad_size: The size the image is padded up to on each side, or None for no padding.
    Returns:
        T.Compose containing the appropriate transforms.
    """
    steps = [T.ToTensor()]
    if pad_size is not None:
        steps.append(PadToSize(pad_size=pad_size))
    if resize_size is not None:
        steps.append(T.Resize((resize_size, resize_size), interpolation=T.InterpolationMode.NEAREST))
    return T.Compose(steps)

def get_dinov2_transforms(resize_size: int | None = 224, pad_size: int | None = None) -> None:
    """ Get the transforms to be used with a DINOv2 model.

    This seems very redundant but it's meant to load the default transforms while keeping the registry structure.
    The sizes are taken for the registry's sake and left alone: PhenoMe builds its own transforms from
    the same two values, which the run hands it as well.

    Args:
        resize_size: The square size the image is resized to, or None to keep its size.
        pad_size: The size the image is padded up to on each side, or None for no padding.
    Returns:
        None, which turns into the default PhenoMe transforms when fed into process_images
    """
    return None

TRANSFORMS_LOADERS = {
    "dinov2_vits14_reg": get_dinov2_transforms,
    "dinov2_vitb14_reg": get_dinov2_transforms,
    "dinov2_vitl14_reg": get_dinov2_transforms,
    "dinov2_vitg14_reg": get_dinov2_transforms,
    "sted-fm": get_stedfm_transforms
}
