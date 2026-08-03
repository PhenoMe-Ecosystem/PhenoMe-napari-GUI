from skimage.io import imread

def display_images(parent, image_path: list[str], mask_path: list[str]=None, row_data: list[dict]=None, names: list[str]=None):
    """
    Display the image and mask in the napari viewer.
    """
    parent._viewer.layers.clear()

    for i, img in enumerate(image_path):
        if row_data:
            metadata = row_data[i]
        else:
            metadata = None

        name = names[i] if names else "Image"

        image = imread(img)
        parent._viewer.add_image(image, name=name, metadata=metadata)

        if mask_path and mask_path[i]:
            mask = imread(mask_path[i])
            parent._viewer.add_labels(mask, name=f"{name} - Mask", metadata=metadata)
