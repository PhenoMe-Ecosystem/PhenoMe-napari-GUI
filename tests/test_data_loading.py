"""Tests for which files a load takes for images.

find_files() filters nothing when given no extensions, so a folder's notes, sidecars and checkpoints
were all loaded as images. Real folders throughout: the filtering is PhenoMe's, read off the disk.
"""

import shutil
from pathlib import Path

import pytest
from phenome import PhenoMe

from phenome_gui.utils.data_loading import EXTENSIONS, DataLoading

# Files that sit beside images in a real dataset folder, none of which PhenoMe can read as one
NOT_IMAGES = ['readme.txt', 'data.json', 'checkpoint.hdf5', 'notes.csv', '.DS_Store']


@pytest.fixture
def cluttered(dataset: dict[str, Path], tmp_path: Path) -> Path:
    """ The test dataset's six images, copied into a folder alongside files that are not images.

    Returns:
        The folder, holding the images and the clutter side by side.
    """
    folder = tmp_path / 'images'
    shutil.copytree(dataset['images'], folder)

    for name in NOT_IMAGES:
        (folder / name).write_text('not an image')

    return folder


@pytest.fixture
def loading() -> DataLoading:
    """ The loading component, on a pipeline of its own. """
    return DataLoading(PhenoMe(seed=42))


def found(metadata_df) -> list[str]:
    """ The names of the files a load found, in order. """
    return sorted(Path(path).name for path in metadata_df['file_path'])


def test_a_load_without_metadata_takes_only_the_images(loading: DataLoading, cluttered: Path) -> None:
    """The case that took everything: with no metadata to match against, nothing else filtered."""
    metadata_df = loading.load_data_csv(str(cluttered))

    assert found(metadata_df) == [f'img_{i:03d}.tif' for i in range(6)]


def test_a_load_off_the_paths_takes_only_the_images(loading: DataLoading, cluttered: Path) -> None:
    """A template ending in '.*' matches any file at the right depth, so it filtered nothing either."""
    metadata_df = loading.load_data_path(str(cluttered), template='.../(filename).*')

    assert found(metadata_df) == [f'img_{i:03d}.tif' for i in range(6)]


def test_a_load_with_metadata_still_finds_the_images(
    loading: DataLoading, cluttered: Path, dataset: dict[str, Path]
) -> None:
    """The .csv filtered by accident, a file it did not name being dropped. It still finds them all."""
    metadata_df = loading.load_data_csv(str(cluttered), metadata_csv=dataset['metadata'])

    assert found(metadata_df) == [f'img_{i:03d}.tif' for i in range(6)]


def test_a_folder_without_images_finds_nothing(loading: DataLoading, tmp_path: Path) -> None:
    """Rather than a table of its notes, which is what loading it off the paths used to give."""
    folder = tmp_path / 'notes'
    folder.mkdir()
    for name in NOT_IMAGES:
        (folder / name).write_text('not an image')

    metadata_df = loading.load_data_path(str(folder), template='.../(filename).*')

    assert metadata_df.empty


@pytest.mark.parametrize('extension', ['.hdf5', '.h5', '.czi', '.nd2', '.txt', '.csv'])
def test_files_phenome_cannot_read_are_not_images(extension: str) -> None:
    """A checkpoint saved in the images folder, above all, must not be read back as an image."""
    assert extension not in EXTENSIONS


def test_every_extension_is_written_as_find_files_compares_it() -> None:
    """Compared against os.path.splitext(), which keeps the dot and lower-cases nothing itself."""
    for extension in EXTENSIONS:
        assert extension.startswith('.')
        assert extension == extension.lower()
        assert extension.count('.') == 1
