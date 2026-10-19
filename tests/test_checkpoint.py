"""Tests for checking the embeddings already held before a run is made to extend them."""

import os
from pathlib import Path

import pytest

from phenome_gui.utils.checkpoint import (
    DATASET_MISMATCH,
    MODEL_MISMATCH,
    image_key,
    inspect_held_embeddings,
)

PARAMS = {
    'channel_mode': 'split', 'channels': None, 'resize_size': 224, 'pad_size': None,
    'force_rgb': True, 'l2_normalize_channels': False,
}

LOADED = ['/data/a.tif', '/data/b.tif', '/data/c.tif']


def inspect(held_params: dict | None = PARAMS, held_paths: list | None = None, current: dict = PARAMS,
            loaded: list = LOADED):
    """ Inspect embeddings held for the first two loaded images unless told otherwise. """
    return inspect_held_embeddings(held_params, LOADED[:2] if held_paths is None else held_paths, current, loaded)


def test_embeddings_made_the_same_way_for_loaded_images_can_be_extended() -> None:
    assert inspect() is None


def test_embeddings_made_under_another_channel_mode_cannot_be_extended() -> None:
    """The two modes answer different questions of the same images, and their embeddings do not mix."""
    found = inspect(current={**PARAMS, 'channel_mode': 'combined'})

    assert found.reason == MODEL_MISMATCH
    assert 'channel_mode' in found.detail


def test_a_legacy_checkpoint_without_the_normalisation_reads_as_normalised() -> None:
    """PhenoMe's own rule: checkpoints written before the key was stored were all normalised."""
    held = {key: value for key, value in PARAMS.items() if key != 'l2_normalize_channels'}

    assert inspect(held_params=held, current={**PARAMS, 'l2_normalize_channels': True}) is None
    assert inspect(held_params=held).reason == MODEL_MISMATCH


def test_sequences_compare_by_their_values() -> None:
    assert inspect(held_params={**PARAMS, 'channels': (0, 1)}, current={**PARAMS, 'channels': [0, 1]}) is None


def test_settings_that_are_not_known_are_not_compared() -> None:
    """Nothing to say about them, rather than everything."""
    assert inspect(held_params=None, current={**PARAMS, 'channel_mode': 'combined'}) is None


def test_an_image_held_but_not_loaded_stops_the_resume() -> None:
    """It would not be skipped, and would be processed again and appended a second time."""
    found = inspect(held_paths=['/data/a.tif', '/elsewhere/z.tif'])

    assert found.reason == DATASET_MISMATCH
    assert found.detail == '1 of the 2 image(s) held are not among the 3 currently loaded.'


def test_the_same_image_reached_another_way_still_matches(tmp_path: Path) -> None:
    image = tmp_path / 'a.tif'
    image.touch()
    link = tmp_path / 'link.tif'
    os.symlink(image, link)

    assert inspect(held_paths=[str(link)], loaded=[str(image)]) is None


@pytest.mark.parametrize('path', ['/data/a.tif', ['/data/a.tif', '/data/a_ch2.tif']])
def test_an_image_read_a_channel_at_a_time_is_known_by_its_first_path(path) -> None:
    assert image_key(path) == image_key('/data/a.tif')
