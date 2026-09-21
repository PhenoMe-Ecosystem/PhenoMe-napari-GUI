"""Tests for reading a checkpoint before a run is made against it.

No real checkpoint is written: filling one means embeddings, which means a model. The CheckpointManager
is stood in for instead, which is all the module ever reads a checkpoint through.
"""

from pathlib import Path

import pytest

from phenome_gui.utils import checkpoint as checkpoint_module
from phenome_gui.utils.checkpoint import (
    DATASET_MISMATCH,
    MODEL_MISMATCH,
    inspect_checkpoint,
)

PARAMS = {'force_rgb': True, 'l2_normalize_channels': True}


class StubCheckpoint:
    """ Stands in for the PhenoMe CheckpointManager, answering the three things the module asks it. """

    def __init__(self, mismatches: list | None = None, paths: list[str] | None = None) -> None:
        """ Build a checkpoint holding given images, and disagreeing on given parameters.

        Args:
            mismatches (list | None): The (key, stored, current) triples its validation returns.
            paths (list[str] | None): The images it holds as committed.
        """
        self._mismatches = mismatches or []
        self._paths = paths or []
        self.closed = False
        self.validated_against = None

    def validate_processing_params(self, current_params: dict) -> list:
        """ Return the parameters this checkpoint disagrees with, as PhenoMe's comparator does.

        Args:
            current_params (dict): The parameters of the run. Kept rather than read, the comparison
                being PhenoMe's: what matters here is that the run's own arguments reach it.
        Returns:
            list: The mismatches it was built with.
        """
        self.validated_against = current_params
        return self._mismatches

    @property
    def n_committed(self) -> int:
        """ How many images the checkpoint holds. """
        return len(self._paths)

    def get_committed_paths(self) -> set[str]:
        """ The images the checkpoint holds. """
        return set(self._paths)

    def close(self) -> None:
        """ Take note that the file was let go of. """
        self.closed = True


@pytest.fixture
def existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """ Return a hook putting a stubbed checkpoint behind a real file path.

    The file has to exist, the module refusing to open a path to none: a CheckpointManager creates the
    file it is pointed at when there is none there, so inspecting one would otherwise leave an empty
    checkpoint behind it.

    Returns:
        Callable[..., StubCheckpoint]: Called as existing(mismatches=..., paths=...), returning the stub.
    """
    def _place(**kwargs) -> StubCheckpoint:
        path = tmp_path / 'checkpoint.hdf5'
        path.write_bytes(b'')

        stub = StubCheckpoint(**kwargs)
        monkeypatch.setattr(checkpoint_module, 'CheckpointManager', lambda *args, **kw: stub)
        stub.path = path
        return stub

    return _place


# ----------------------------------------------------------------------------
# Nothing to report
# ----------------------------------------------------------------------------


def test_a_checkpoint_that_does_not_exist_is_no_mismatch(tmp_path: Path) -> None:
    """The usual first run, which starts fresh and has nothing to disagree with."""
    assert inspect_checkpoint(tmp_path / 'nothing.hdf5', PARAMS, ['/data/a.tif']) is None


def test_no_checkpoint_at_all_is_no_mismatch() -> None:
    assert inspect_checkpoint(None, PARAMS, ['/data/a.tif']) is None


def test_a_checkpoint_agreeing_on_everything_is_no_mismatch(existing) -> None:
    existing(paths=['/data/a.tif', '/data/b.tif'])

    assert inspect_checkpoint('/x/checkpoint.hdf5', PARAMS, ['/data/a.tif', '/data/b.tif']) is None


def test_an_empty_checkpoint_is_no_mismatch(existing) -> None:
    """Holding no image, it answers for no dataset and cannot be filled from another one."""
    stub = existing(paths=[])

    assert inspect_checkpoint(stub.path, PARAMS, ['/data/a.tif']) is None


def test_a_dataset_half_processed_is_no_mismatch(existing) -> None:
    """Resuming is what a checkpoint is for, so images it has yet to reach are the ordinary case."""
    stub = existing(paths=['/data/a.tif'])

    result = inspect_checkpoint(stub.path, PARAMS, ['/data/a.tif', '/data/b.tif', '/data/c.tif'])

    assert result is None


def test_no_loaded_images_is_no_mismatch(existing) -> None:
    """With nothing loaded there is nothing to compare the committed images against."""
    stub = existing(paths=['/data/a.tif'])

    assert inspect_checkpoint(stub.path, PARAMS, []) is None


# ----------------------------------------------------------------------------
# The two mismatches
# ----------------------------------------------------------------------------


def test_differing_parameters_are_a_model_mismatch(existing) -> None:
    stub = existing(mismatches=[('force_rgb', True, False)], paths=['/data/a.tif'])

    result = inspect_checkpoint(stub.path, PARAMS, ['/data/a.tif'])

    assert result.reason == MODEL_MISMATCH


def test_a_differing_channel_mode_is_a_mismatch(existing) -> None:
    """The first of the keys PhenoMe compares, and the one the processing form can now change."""
    stub = existing(mismatches=[('channel_mode', 'split', 'combined')], paths=['/data/a.tif'])

    result = inspect_checkpoint(stub.path, PARAMS, ['/data/a.tif'])

    assert result.reason == MODEL_MISMATCH


def test_the_run_s_own_arguments_are_what_the_checkpoint_is_compared_against(existing) -> None:
    """PhenoMe makes the comparison, so what is handed to it is the whole of this module's part in it."""
    stub = existing(paths=['/data/a.tif'])
    params = {**PARAMS, 'channel_mode': 'combined'}

    inspect_checkpoint(stub.path, params, ['/data/a.tif'])

    assert stub.validated_against == params


def test_the_model_mismatch_adds_no_detail_of_its_own(existing) -> None:
    """PhenoMe writes the comparison out key by key to the log, so the reason is the whole answer.

    Built all the same rather than left half made: the reason alone is what the question reads, and a
    mismatch that cannot be built is a mismatch that reaches nobody.
    """
    stub = existing(mismatches=[('force_rgb', True, False)], paths=['/data/a.tif'])

    assert inspect_checkpoint(stub.path, PARAMS, ['/data/a.tif']).detail == ''


def test_images_with_nothing_in_common_are_a_dataset_mismatch(existing) -> None:
    stub = existing(paths=['/elsewhere/x.tif', '/elsewhere/y.tif'])

    result = inspect_checkpoint(stub.path, PARAMS, ['/data/a.tif', '/data/b.tif'])

    assert result.reason == DATASET_MISMATCH


def test_the_dataset_mismatch_counts_both_sides(existing) -> None:
    stub = existing(paths=['/elsewhere/x.tif', '/elsewhere/y.tif'])

    detail = inspect_checkpoint(stub.path, PARAMS, ['/data/a.tif']).detail

    assert '2' in detail
    assert '1' in detail


def test_the_parameters_are_read_before_the_images(existing) -> None:
    """Both wrong is reported as the parameters, which PhenoMe is also the first to refuse."""
    stub = existing(mismatches=[('force_rgb', True, False)], paths=['/elsewhere/x.tif'])

    assert inspect_checkpoint(stub.path, PARAMS, ['/data/a.tif']).reason == MODEL_MISMATCH


def test_paths_are_compared_resolved(existing) -> None:
    """The same image reached by two spellings is one image, not two datasets."""
    stub = existing(paths=['/data/nested/../a.tif'])

    assert inspect_checkpoint(stub.path, PARAMS, ['/data/a.tif']) is None


# ----------------------------------------------------------------------------
# The file itself
# ----------------------------------------------------------------------------


def test_the_checkpoint_is_let_go_of(existing) -> None:
    """PhenoMe opens the same file moments later, so it is never left held open here."""
    stub = existing(paths=['/data/a.tif'])

    inspect_checkpoint(stub.path, PARAMS, ['/data/a.tif'])

    assert stub.closed


def test_a_checkpoint_that_will_not_open_is_left_to_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only ever opened to ask a question, so one that cannot be read asks nothing and blocks nothing."""
    path = tmp_path / 'checkpoint.hdf5'
    path.write_bytes(b'not a checkpoint')

    def _refuse(*args, **kwargs):
        raise ValueError("Unrecognised checkpoint format for 'checkpoint.hdf5'. Expected HDF5.")

    monkeypatch.setattr(checkpoint_module, 'CheckpointManager', _refuse)

    assert inspect_checkpoint(path, PARAMS, ['/data/a.tif']) is None


def test_a_checkpoint_that_will_not_be_read_is_let_go_of(existing) -> None:
    """The file is closed whether or not the reading of it got anywhere."""
    stub = existing(paths=['/data/a.tif'])
    stub.get_committed_paths = lambda: (_ for _ in ()).throw(OSError('truncated file'))

    assert inspect_checkpoint(stub.path, PARAMS, ['/data/a.tif']) is None
    assert stub.closed
