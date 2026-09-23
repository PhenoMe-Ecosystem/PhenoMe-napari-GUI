import torchvision.transforms as T

"""
Custom transformations for the different model options.
"""

def get_stedfm_transforms() -> "T.Compose":
    """ Get the transforms to be used with the STED-FM model.

    Returns:
        T.Compose containing the appropriate transforms.
    """
    return T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
        ])

def get_dinov2_transforms() -> None:
    """ Get the transforms to be used with a DINOv2 model.

    This seems very redundant but it's meant to load the default transforms while keeping the registry structure.

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
