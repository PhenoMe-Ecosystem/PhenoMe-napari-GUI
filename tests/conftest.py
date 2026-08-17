"""Fixtures shared by the whole suite.

No test may load a model or extract embeddings: both are slow and need network access. Tests that need
the pipeline to look processed stub its results instead, through the `processed` fixture.
"""

import os

# Must be set before anything imports qtpy, so the tests never try to open a window. setdefault leaves a
# developer free to force a real one when debugging.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from skimage.io import imsave  # noqa: E402

from phenome_gui._widget import PhenoMeWidget  # noqa: E402
from phenome_gui.utils import analysis  # noqa: E402
from phenome_gui.utils.phenome_interface import PhenoMeInterface  # noqa: E402

# The dataset the fixtures below build. Two grouping columns, so the tests can tell a resolved grouping
# key apart from a fallback, and one drug with a single image to keep group sizes uneven.
DRUGS = ['ctrl', 'ctrl', 'ctrl', 'drugA', 'drugA', 'drugB']
TIMES = [0, 0, 24, 0, 24, 24]


@pytest.fixture(scope='session')
def dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """ Write a tiny readable dataset to disk, with images, masks and a .csv metadata file.

    The images have to be real files rather than stubs, as the image viewer reads them with
    skimage.io.imread when a row is selected.

    Returns:
        dict[str, Path]: The images directory, the masks directory and the metadata file.
    """
    root = tmp_path_factory.mktemp('dataset')
    images_dir = root / 'images'
    masks_dir = root / 'masks'
    images_dir.mkdir()
    masks_dir.mkdir()

    rng = np.random.default_rng(seed=0)
    names = [f'img_{i:03d}' for i in range(len(DRUGS))]

    for name in names:
        imsave(images_dir / f'{name}.tif', rng.integers(0, 255, (8, 8), dtype=np.uint8), check_contrast=False)
        imsave(masks_dir / f'{name}.tif', np.ones((8, 8), dtype=np.uint8), check_contrast=False)

    metadata = pd.DataFrame({'filename': names, 'drug_name': DRUGS, 'time': TIMES})
    metadata_csv = root / 'metadata.csv'
    metadata.to_csv(metadata_csv, index=False)

    return {'images': images_dir, 'masks': masks_dir, 'metadata': metadata_csv}


@pytest.fixture(scope='session')
def path_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """ The same images laid out so the metadata can be read off the paths instead of a .csv.

    Returns:
        dict[str, Path]: The images directory, the masks directory and the matching template.
    """
    root = tmp_path_factory.mktemp('path_dataset')
    rng = np.random.default_rng(seed=1)

    # Two samples share a drug and a time, so the name has to carry the index to stay unique
    for sample, (drug, time) in enumerate(zip(DRUGS, TIMES, strict=True)):
        for index, kind in enumerate(('images', 'masks')):
            folder = root / kind / drug / str(time)
            folder.mkdir(parents=True, exist_ok=True)
            data = rng.integers(0, 255, (8, 8), dtype=np.uint8) if index == 0 else np.ones((8, 8), np.uint8)
            imsave(folder / f'img_{sample:03d}.tif', data, check_contrast=False)

    return {
        'images': root / 'images',
        'masks': root / 'masks',
        'template': '.../(drug_name)/(time)/(filename).*',
    }


@pytest.fixture
def phenome(dataset: dict[str, Path]) -> PhenoMeInterface:
    """ A PhenoMe interface with its paths pointed at the test dataset, holding no data yet.

    Function scoped, as the tests edit its parameters and its metadata.
    """
    interface = PhenoMeInterface(seed=42)
    interface.params.paths.images = dataset['images']
    interface.params.paths.masks = dataset['masks']
    interface.params.paths.metadata = dataset['metadata']
    return interface


@pytest.fixture
def loaded(phenome: PhenoMeInterface) -> PhenoMeInterface:
    """ The same interface, with the test dataset loaded. """
    phenome.load_data_csv()
    return phenome


class StubResults:
    """ Stands in for the PhenoMe results, so a pipeline can look processed without running a model. """

    def __init__(self, n_images: int = 0) -> None:
        self.n_images = n_images
        self.metadata: list[dict] = []

        # One dict per image once the properties have been computed, which is where PhenoMe reads the
        # property keys the explorer offers as colouring columns
        self.properties: list[dict] = []

        # Path of each processed image, which is what a projected point is resolved back to
        self.img_path: list[str] = []


@pytest.fixture
def processed():
    """ Return a hook making an interface report a given number of processed images.

    Returns:
        Callable[[PhenoMeInterface, int], StubResults]: Called as processed(interface, n_images).
    """
    def _apply(interface: PhenoMeInterface, n_images: int = 6) -> StubResults:
        results = StubResults(n_images)
        interface.pheno.results = results
        return results

    return _apply


class FakeLayers(list):
    """ Stands in for the napari layer list, which the image viewer only ever clears. """


class RecordingViewer:
    """ Stands in for the napari viewer, recording what would have been displayed.

    Lets the display assertions be exact without building a real viewer, which is slow and needs Qt to
    actually paint. One integration test covers the real thing.
    """

    def __init__(self) -> None:
        self.layers = FakeLayers()
        self.added: list[dict[str, Any]] = []

    def add_image(self, data, name=None, metadata=None) -> None:
        """ Record an image layer. """
        self.added.append({'kind': 'image', 'data': data, 'name': name, 'metadata': metadata})

    def add_labels(self, data, name=None, metadata=None) -> None:
        """ Record a labels layer. """
        self.added.append({'kind': 'labels', 'data': data, 'name': name, 'metadata': metadata})

    @property
    def names(self) -> list[str]:
        """ The names of the recorded layers, in order. """
        return [layer['name'] for layer in self.added]


@pytest.fixture
def viewer() -> RecordingViewer:
    """ A recording stand-in for the napari viewer. """
    return RecordingViewer()


@pytest.fixture
def widget(viewer: RecordingViewer, dataset: dict[str, Path], qtbot) -> PhenoMeWidget:
    """ A fully built plugin widget on the recording viewer, with its paths pointed at the test dataset. """
    widget = PhenoMeWidget(viewer)
    qtbot.addWidget(widget)

    widget._phenome.params.paths.images = dataset['images']
    widget._phenome.params.paths.masks = dataset['masks']
    widget._phenome.params.paths.metadata = dataset['metadata']
    return widget


class Spy:
    """ Records the keyword arguments of the calls made to it, and returns a canned result. """

    def __init__(self, result: Any = None) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs) -> Any:
        self.calls.append(kwargs)
        return self.result

    @property
    def last(self) -> dict[str, Any]:
        """ The keyword arguments of the most recent call. """
        return self.calls[-1]

    @property
    def called(self) -> bool:
        """ Whether the spy has been called at all. """
        return bool(self.calls)


@pytest.fixture
def spy():
    """ Return the Spy class, so a test can make one with its own canned result. """
    return Spy


class ReductionSpy:
    """ Stands in for PhenoMe's run_dimensionality_reduction.

    A free function taking the pipeline as its first argument rather than a pipeline method, so it
    cannot be stubbed by assigning onto the pipeline the way the other PhenoMe calls are.
    """

    def __init__(self, frame: pd.DataFrame | None, data_type: str = 'Embeddings') -> None:
        self.frame = frame
        self.data_type = data_type
        self.calls: list[dict[str, Any]] = []
        self.pipelines: list[Any] = []

        # Called while the projection is running, for a test watching what the GUI does meanwhile
        self.on_call = None

    def __call__(self, pipeline, **kwargs) -> tuple:
        self.pipelines.append(pipeline)
        self.calls.append(kwargs)
        if self.on_call is not None:
            self.on_call()

        # PhenoMe answers with three Nones when it is left with no data to project
        if self.frame is None:
            return None, None, None
        return self.frame, self.data_type, None

    @property
    def last(self) -> dict[str, Any]:
        """ The keyword arguments of the most recent call. """
        return self.calls[-1]

    @property
    def called(self) -> bool:
        """ Whether the reduction has been run at all. """
        return bool(self.calls)


@pytest.fixture
def reduction(monkeypatch: pytest.MonkeyPatch):
    """ Return a hook replacing PhenoMe's dimensionality reduction with a recording stand-in.

    No test may run a real projection: t-SNE and UMAP are slow, and the pipeline holds no embeddings
    to project anyway.

    Returns:
        Callable[[pd.DataFrame | None, str], ReductionSpy]: Called as reduction(frame).
    """
    def _replace(frame: pd.DataFrame | None = None, data_type: str = 'Embeddings') -> ReductionSpy:
        stand_in = ReductionSpy(frame, data_type)
        monkeypatch.setattr(analysis, 'run_dimensionality_reduction', stand_in)
        return stand_in

    return _replace


class _Signal:
    """ Minimal stand-in for the `returned` and `errored` signals of a napari worker. """

    def __init__(self) -> None:
        self._slots = []

    def connect(self, slot) -> None:
        self._slots.append(slot)

    def emit(self, *args) -> None:
        for slot in self._slots:
            slot(*args)


class _SyncWorker:
    """ Stands in for a napari thread_worker, running its body inline.

    The panels start their long steps on a worker thread, which would make the tests racy. Swapping the
    worker factory for this one keeps the rest of the flow - the finish handler included - identical.

    A raising step is reported the way the real worker reports it, through `errored` rather than out of
    start(), so a handler waiting on it is exercised as it would be in the running GUI.
    """

    def __init__(self, function) -> None:
        self._function = function
        self.returned = _Signal()
        self.errored = _Signal()

    def start(self) -> None:
        try:
            self._function()
        except Exception as error:  # noqa: BLE001 - the real worker catches everything too
            self.errored.emit(error)
        else:
            self.returned.emit()


@pytest.fixture
def sync_worker():
    """ Return a hook replacing a panel's worker factory with one that runs inline.

    Returns:
        Callable[[QWidget, str, Callable], None]: Called as sync_worker(panel, '_prototypes_worker', fn).
    """
    def _replace(panel, worker_name: str, function) -> None:
        setattr(panel, worker_name, lambda: _SyncWorker(function))

    return _replace


@pytest.fixture
def dialog_result(monkeypatch: pytest.MonkeyPatch):
    """ Return a hook driving a modal parameter form without showing it.

    Patches the class's exec_ so it applies the given widget values and then reports accepted or rejected,
    which is what tells the caller whether to keep the edit.

    Returns:
        Callable[[type, bool], None]: Called as dialog_result(DialogClass, accepted=True, _spin=3).
    """
    def _drive(dialog_class: type, accepted: bool, **edits) -> None:
        def _exec(dialog) -> int:
            for attribute, value in edits.items():
                getattr(dialog, attribute).value = value
            return int(accepted)

        monkeypatch.setattr(dialog_class, 'exec_', _exec)

    return _drive
