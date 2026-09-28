def get_stedfm_options() -> dict:
    """ Get the process_images arguments that are specific to the STED-FM model.

    Returns:
        dict of kwargs
    """
    return {"force_rgb":False}

def get_dinov2_options() -> dict:
    """ Get the process_images arguments that are specific to the DINOv2 model.

    Returns:
        dict of kwargs
    """
    return {"force_rgb":True}

OPTIONS_LOADERS = {
    "dinov2_vits14_reg": get_dinov2_options,
    "dinov2_vitb14_reg": get_dinov2_options,
    "dinov2_vitl14_reg": get_dinov2_options,
    "dinov2_vitg14_reg": get_dinov2_options,
    "sted-fm": get_stedfm_options
}
