from typing import TYPE_CHECKING

from skimage.io import imread

if TYPE_CHECKING:
    import napari


class ImageViewer:
    """ Image viewer component of the widget.

    Pushes images and masks into the napari viewer. Holds no PhenoMe state.
    """

    def __init__(self, viewer: 'napari.viewer.Viewer') -> None:
        """ Store the napari viewer to display the layers into.

        Args:
            viewer (napari.viewer.Viewer): The napari viewer the widget is docked in.
        Returns:
            None
        """
        self._viewer = viewer

    def display_images(self, image_paths: list[str], mask_paths: list[str] | None = None, row_data: list[dict] | None = None, names: list[str] | None = None) -> None:
        """ Display the images and their masks in the napari viewer.

        The layers already present in the viewer are cleared beforehand.

        Args:
            image_paths (list[str]): Paths of the images to read and display.
            mask_paths (list[str]): Paths of the corresponding masks, added as label layers. None to display the images alone.
            row_data (list[dict]): Metadata attached to each layer, one dict per image. None to attach no metadata.
            names (list[str]): Name given to each image layer, the mask layers being suffixed with ' - Mask'. None to name every layer 'Image'.
        Returns:
            None
        """
        self._viewer.layers.clear()

        for i, img in enumerate(image_paths):
            metadata = row_data[i] if row_data else None
            name = names[i] if names else "Image"

            image = imread(img)
            self._viewer.add_image(image, name=name, metadata=metadata)

            if mask_paths and mask_paths[i]:
                mask = imread(mask_paths[i])
                self._viewer.add_labels(mask, name=f"{name} - Mask", metadata=metadata)
