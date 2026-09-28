def get_dinov2_details() -> str:
    """ Describe the DINOv2 models, which the four registered variants share.

    Returns:
        An HTML fragment, for the pane beside the list a model is picked from.
    """
    details = """
    <b>DINOv2</b><br>
    Learning Robust Visual Features without Supervision<br>
    <a href="https://github.com/facebookresearch/dinov2">GitHub</a><br><br>

    <b>Required image input format:</b>
    <ul>
        <li>Channels: 1 or 3</li>
        <li>Type: 16bit int</li>
        <li>Normalization: None</li>
    </ul><br><br>

    <b>Transforms applied at process:</b>
    <ul>
        <li>Duplicate to 3 channels if necessary</li>
        <li>Standardization using ImageNET values</li>
    </ul>
    """

    return details


def get_stedfm_details() -> str:
    """ Describe the STED-FM ViT-S model.

    Returns:
        An HTML fragment, for the pane beside the list a model is picked from.
    """
    details = """
    <b>STED-FM</b><br>
    A Self-Supervised Foundation Model for Robust and Generalizable Representation Learning in STED Microscopy<br>
    <a href="https://github.com/FLClab/STED-FM">GitHub</a><br><br>

    <b>Required image input format:</b>
    <ul>
        <li>Channels: 1</li>
        <li>Type: 32bit float</li>
        <li>Normalization: Min-max</li>
    </ul><br><br>

    <b>Transforms applied at process:</b>
    <ul>
        <li>Resize to 224x224</li>
    </ul>
    """

    return details


DETAILS_LOADERS = {
    "dinov2_vits14_reg": get_dinov2_details,
    "dinov2_vitb14_reg": get_dinov2_details,
    "dinov2_vitl14_reg": get_dinov2_details,
    "dinov2_vitg14_reg": get_dinov2_details,
    "sted-fm": get_stedfm_details
}


def get_model_details(model_name: str) -> str:
    """ Look up what is known about a model, for the pane beside the list it is picked from.

    Answers for a model it does not know rather than raising, unlike get_model(): a description is
    what the window says about a model, not what lets it be picked, and one missing must not keep a
    model from being loaded. A model registered here and nowhere else - or the other way round - is
    caught by the registry tests instead.

    Args:
        model_name: The name of the model as it appears in DETAILS_LOADERS.
    Returns:
        An HTML fragment describing the model, empty for one with no description yet.
    """
    details_loader = DETAILS_LOADERS.get(model_name)

    return details_loader() if details_loader is not None else ''
