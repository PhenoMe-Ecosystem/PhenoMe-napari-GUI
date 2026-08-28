"""Reading a checkpoint before a run is made against it.

PhenoMe resumes from whatever checkpoint it is pointed at, and raises when what it finds there does not
answer for the run about to be made: images processed under other arguments, or other images
altogether. Both raises come mid-run and from a worker thread, by which point the run is half done and
there is nothing to do but report them. Read here beforehand, they become a question asked while the
run can still be called off.

Read-only throughout, and deliberately shy: a checkpoint that cannot be read, or one this module is
unsure about, is reported as no mismatch at all and left to PhenoMe, which raises on what it will not
resume from anyway. The cost of the two answers is not the same - the run this warns about is offered
a checkpoint's deletion as its way out - so the module says nothing it is not sure of.
"""

from dataclasses import dataclass
from logging import getLogger
from os.path import isfile, realpath
from pathlib import Path

from phenome.io import CheckpointManager

logger = getLogger(__name__)

# What a mismatch was found in, which decides what the caller offers to do about it
MODEL_MISMATCH = 'model'
DATASET_MISMATCH = 'dataset'

# What reading a checkpoint can go wrong with: PhenoMe refuses a file that is not one with a ValueError,
# h5py raises OSError on a file it cannot make sense of, and a checkpoint written by another version can
# leave either a key or a type unaccounted for. Caught together, as none of them is worth stopping a run
# over: the run opens the same file itself, and says so through the usual failure path.
UNREADABLE = (OSError, ValueError, KeyError, TypeError, RuntimeError)


@dataclass(frozen=True)
class CheckpointMismatch:
    """ A checkpoint holding something other than what the run about to be made would put there.

    Attributes:
        reason (str): MODEL_MISMATCH or DATASET_MISMATCH, naming which of the two was found.
        detail (str): One line naming what differs, to be shown alongside the question. Left empty by
            the parameter mismatch, which PhenoMe writes out key by key to the log itself: the reason
            is the whole of what the question needs.
    """

    reason: str
    detail: str = ''


def inspect_checkpoint(
    checkpoint_path: Path | str | None,
    processing_params: dict,
    image_paths: list[str],
) -> CheckpointMismatch | None:
    """ Read an existing checkpoint, and say whether the run about to be made belongs in it.

    Args:
        checkpoint_path (Path | str | None): The checkpoint the run would be written to. A path to no
            file is a run that starts fresh, and has nothing to disagree with.
        processing_params (dict): The arguments the run would be made under, as PhenoMe stores them.
            Compared by PhenoMe's own comparator rather than key by key, which is what the checkpoint
            will be validated with when the run starts.
        image_paths (list[str]): The images the run would process.
    Returns:
        CheckpointMismatch | None: What was found, or None when the checkpoint agrees with the run,
            holds nothing yet, or could not be read.
    """
    if checkpoint_path is None or not isfile(checkpoint_path):
        return None

    try:
        checkpoint = CheckpointManager(str(checkpoint_path), lazy=True)
    except UNREADABLE as error:
        # Only ever opened to ask a question, so a checkpoint that will not open is left to the run
        # itself, which opens it again and reports what it finds through the usual failure path.
        logger.warning("Could not read the checkpoint at '%s': %s", checkpoint_path, error)
        return None

    try:
        mismatches = checkpoint.validate_processing_params(processing_params)
        committed_paths = checkpoint.get_committed_paths() if checkpoint.n_committed else set()
    except UNREADABLE as error:
        logger.warning("Could not read the checkpoint at '%s': %s", checkpoint_path, error)
        return None
    finally:
        # PhenoMe opens the same file moments later, so it is never left held open here
        checkpoint.close()

    if mismatches:
        return CheckpointMismatch(MODEL_MISMATCH)

    if _is_another_dataset(committed_paths, image_paths):
        return CheckpointMismatch(
            DATASET_MISMATCH,
            f'It holds {len(committed_paths)} image(s), none of them among the {len(image_paths)} '
            'currently loaded.'
        )

    return None

def _is_another_dataset(committed_paths: set[str], image_paths: list[str]) -> bool:
    """ Whether a checkpoint's images and the loaded ones have nothing at all in common.

    Held to nothing in common rather than to anything missing, a checkpoint being filled one run at a
    time: a dataset half processed is the ordinary way to meet one, and is what resuming is for. Only
    a checkpoint sharing no image with the dataset loaded was filled from another one.

    Args:
        committed_paths (set[str]): The images the checkpoint holds, as PhenoMe resolved them.
        image_paths (list[str]): The images currently loaded.
    Returns:
        bool: True only when both are known and no image is in both.
    """
    if not committed_paths or not image_paths:
        return False

    committed = {realpath(path) for path in committed_paths}
    loaded = {realpath(path) for path in image_paths}

    return committed.isdisjoint(loaded)
