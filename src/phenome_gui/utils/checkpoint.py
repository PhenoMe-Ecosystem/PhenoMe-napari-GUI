"""Checking the embeddings already held before a run is made to extend them.

A resumed run keeps the embeddings the pipeline holds - read from a checkpoint or left by a stopped run -
and processes only the images missing from them, appending what it computes. That is only sound when
the new embeddings are made the way the held ones were, and when the images held are recognised among
the ones loaded: an image held under a path that names no loaded file would not be skipped, and would
be processed again and appended a second time. Both are checked here, from what is in memory, while the
run can still be called off.
"""

from dataclasses import dataclass
from os.path import realpath
from typing import Any

import numpy as np
from phenome.io import CheckpointManager

# What a mismatch was found in, which decides how the question about it is put
MODEL_MISMATCH = 'model'
DATASET_MISMATCH = 'dataset'


@dataclass(frozen=True)
class CheckpointMismatch:
    """ Embeddings held that the run about to be made could not extend.

    Attributes:
        reason: MODEL_MISMATCH or DATASET_MISMATCH, naming which of the two was found.
        detail: One line naming what differs, to be shown alongside the question.
    """

    reason: str
    detail: str = ''


def image_key(path: Any) -> str:
    """ The form an image's path is compared in, whichever way PhenoMe holds it.

    PhenoMe holds a list of paths rather than one for an image it reads a channel at a time, which is
    known by its first, the way the image viewer finds it.

    Args:
        path: A path, or a list of them for an image read a channel at a time.
    Returns:
        The real path of the image, symbolic links and relative parts resolved.
    """
    if isinstance(path, list | tuple):
        path = path[0]
    return realpath(str(path))


def inspect_held_embeddings(
    held_params: dict | None,
    held_paths: list,
    current_params: dict,
    loaded_paths: list,
) -> CheckpointMismatch | None:
    """ Say whether the embeddings held can be extended by the run about to be made.

    Args:
        held_params: The arguments the held embeddings were made under, as PhenoMe keeps them. None
            when unknown, in which case they are not compared.
        held_paths: The images the embeddings are held for.
        current_params: The arguments the run would be made under.
        loaded_paths: The images currently loaded.
    Returns:
        What was found, or None when the run can extend the held embeddings.
    """
    differing = _differing_params(held_params, current_params)
    if differing:
        return CheckpointMismatch(
            MODEL_MISMATCH,
            ', '.join(f'{key}: {held!r} then, {current!r} now' for key, held, current in differing),
        )

    loaded = {image_key(path) for path in loaded_paths}
    unknown = sum(image_key(path) not in loaded for path in held_paths)
    if unknown:
        return CheckpointMismatch(
            DATASET_MISMATCH,
            f'{unknown} of the {len(held_paths)} image(s) held are not among the {len(loaded_paths)} '
            'currently loaded.'
        )

    return None


def _differing_params(held: dict | None, current: dict) -> list[tuple[str, Any, Any]]:
    """ The processing arguments the held embeddings and the run disagree on.

    Compared over the keys PhenoMe itself validates a checkpoint on, and by its rules: a missing
    channel normalisation reads as on, as in the checkpoints written before it was stored, and
    sequences are compared as lists whatever their type.

    Args:
        held: The arguments the held embeddings were made under, or None when unknown.
        current: The arguments the run would be made under.
    Returns:
        (key, held value, current value) for every key that differs.
    """
    if not held:
        return []

    differing = []
    for key in CheckpointManager._PARAM_KEYS:
        held_value, current_value = held.get(key), current.get(key)

        if key == 'l2_normalize_channels':
            held_value = True if held_value is None else held_value
            current_value = True if current_value is None else current_value

        if isinstance(held_value, list | tuple | np.ndarray):
            held_value = list(held_value)
        if isinstance(current_value, list | tuple | np.ndarray):
            current_value = list(current_value)

        if held_value != current_value:
            differing.append((key, held_value, current_value))

    return differing
