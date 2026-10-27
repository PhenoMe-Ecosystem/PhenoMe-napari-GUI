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
            viewer: The napari viewer the widget is docked in.
        Returns:
            None
        """
        self._viewer = viewer

    def clear(self) -> None:
        """ Take every layer off the napari viewer.

        What is displayed is one image of the loaded dataset, so it is taken off when that dataset is
        let go of, the same way it is taken off to show the next image.
        """
        self._viewer.layers.clear()

    def display_images(self, image_paths: list[str], mask_paths: list[str] | None = None, row_data: list[dict] | None = None, names: list[str] | None = None) -> None:
        """ Display the images and their masks in the napari viewer.

        The layers already present in the viewer are cleared beforehand.

        Args:
            image_paths: Paths of the images to read and display.
            mask_paths: Paths of the corresponding masks, added as label layers. None to display the images alone.
            row_data: Metadata attached to each layer, one dict per image. None to attach no metadata.
            names: Name given to each image layer, the mask layers being suffixed with ' - Mask'. None to name every layer 'Image'.
        """
        self.clear()

        for i, img in enumerate(image_paths):
            metadata = row_data[i] if row_data else None
            name = names[i] if names else "Image"

            image = imread(img)

            # Stated rather than left to napari, which only guesses RGB when both sides exceed 30 pixels and
            # otherwise shows a small colour image as a greyscale stack of its rows
            is_rgb = image.ndim == 3 and image.shape[-1] in (3, 4)
            self._viewer.add_image(image, name=name, metadata=metadata, rgb=is_rgb)

            if mask_paths and mask_paths[i]:
                mask = imread(mask_paths[i])
                self._viewer.add_labels(mask, name=f"{name} - Mask", metadata=metadata)
