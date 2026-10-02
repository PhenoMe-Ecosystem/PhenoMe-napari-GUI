"""Tests for the checks of what a path names, which the loading tab outlines its boxes by.

No Qt here: what a path is has nothing to do with the box it was typed into.
"""

from pathlib import Path

from phenome_gui.utils.path_checking import (
    check_csv_file_exists,
    check_directory_exists,
)

# ----------------------------------------------------------------------------
# Directories, which the images and the masks have to be
# ----------------------------------------------------------------------------


def test_a_directory_is_a_directory(tmp_path: Path) -> None:
    assert check_directory_exists(tmp_path)


def test_a_file_is_not_a_directory(tmp_path: Path) -> None:
    """Typed into the images box, it would have read as a good path on existence alone."""
    path = tmp_path / 'image.tif'
    path.write_bytes(b'')

    assert not check_directory_exists(path)


def test_nothing_is_not_a_directory(tmp_path: Path) -> None:
    assert not check_directory_exists(tmp_path / 'missing')


# ----------------------------------------------------------------------------
# .csv files, which the metadata has to be
# ----------------------------------------------------------------------------


def test_a_csv_file_is_one(tmp_path: Path) -> None:
    path = tmp_path / 'metadata.csv'
    path.write_text('filename\n')

    assert check_csv_file_exists(path)


def test_the_suffix_is_read_without_its_case(tmp_path: Path) -> None:
    """The same file to anyone reading it, whatever the filesystem chose to keep."""
    path = tmp_path / 'metadata.CSV'
    path.write_text('filename\n')

    assert check_csv_file_exists(path)


def test_a_file_of_another_kind_is_not_one(tmp_path: Path) -> None:
    path = tmp_path / 'metadata.txt'
    path.write_text('filename\n')

    assert not check_csv_file_exists(path)


def test_a_directory_named_like_one_is_not_one(tmp_path: Path) -> None:
    """The suffix alone is not enough: pandas cannot read a folder."""
    path = tmp_path / 'metadata.csv'
    path.mkdir()

    assert not check_csv_file_exists(path)


def test_nothing_is_not_one(tmp_path: Path) -> None:
    assert not check_csv_file_exists(tmp_path / 'missing.csv')
