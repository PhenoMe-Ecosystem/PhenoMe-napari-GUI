"""Tests for the image viewer, which reads images off disk and hands them to napari."""

from pathlib import Path

import numpy as np
import pytest
from skimage.io import imsave

from phenome_gui.utils.image_viewer import ImageViewer

from .conftest import RecordingViewer


def written(tmp_path: Path, shape: tuple[int, ...]) -> str:
    """ Write an image of the given shape to disk, as the viewer only reads from files.

    Args:
        tmp_path: The folder to write it into.
        shape: Shape of the image written.
    Returns:
        The path of the written image.
    """
    path = tmp_path / f'image_{"x".join(map(str, shape))}.tif'
    imsave(path, np.zeros(shape, dtype=np.uint8), check_contrast=False)
    return str(path)


# Sides of 30 pixels or fewer are the ones napari would not have guessed as RGB on its own
@pytest.mark.parametrize('shape', [(28, 28, 3), (28, 28, 4), (64, 64, 3)])
def test_a_colour_image_is_displayed_as_rgb(viewer: RecordingViewer, tmp_path: Path, shape: tuple[int, ...]) -> None:
    ImageViewer(viewer).display_images([written(tmp_path, shape)])

    assert viewer.added[0]['rgb'] is True


# Not a 3-plane stack, which tifffile writes as planar RGB and reads back as a colour image
@pytest.mark.parametrize('shape', [(28, 28), (5, 28, 28)])
def test_an_image_without_colour_channels_is_not_displayed_as_rgb(
    viewer: RecordingViewer, tmp_path: Path, shape: tuple[int, ...]
) -> None:
    ImageViewer(viewer).display_images([written(tmp_path, shape)])

    assert viewer.added[0]['rgb'] is False
