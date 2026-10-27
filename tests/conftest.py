"""Fixtures shared by the whole suite.

No test may load a model or extract embeddings: both are slow and need network access. Tests that need
the pipeline to look processed stub its results instead, through the `processed` fixture.
"""

import os

# Must be set before anything imports qtpy, so the tests never try to open a window. setdefault leaves a
# developer free to force a real one when debugging.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import shutil  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from phenome.core.pipeline_results import PhenoMeResults  # noqa: E402
from phenome.io import CheckpointManager  # noqa: E402
from qtpy.QtWidgets import QMainWindow, QMessageBox  # noqa: E402
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
        The images directory, the masks directory and the metadata file.
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


@pytest.fixture
def copied_dataset(dataset: dict[str, Path], tmp_path: Path):
    """ Return a hook copying the test dataset elsewhere, whole, in part, or as another dataset.

    A copy left whole is the same dataset moved to another folder, which a checkpoint still answers
    for. A part of it is a subset of the images, and a renamed copy is images of another experiment
    altogether, whose names and metadata no checkpoint of the original answers for.

    Returns:
        Called as copied_dataset(indices=[0, 1, 2], renamed=False), returning the
        images directory, the masks directory and the metadata file of the copy.
    """
    copies = []

    def _copy(indices: list[int] | None = None, renamed: bool = False) -> dict[str, Path]:
        root = tmp_path / f'copy_{len(copies)}'
        copies.append(root)
        images_dir = root / 'images'
        masks_dir = root / 'masks'
        images_dir.mkdir(parents=True)
        masks_dir.mkdir()

        metadata = pd.read_csv(dataset['metadata'])
        if indices is not None:
            metadata = metadata.iloc[indices].reset_index(drop=True)

        names = list(metadata['filename'])
        if renamed:
            metadata['filename'] = [f'other_{i:03d}' for i in range(len(metadata))]
            metadata['drug_name'] = 'unrelated'

        for old, new in zip(names, metadata['filename'], strict=True):
            shutil.copy(dataset['images'] / f'{old}.tif', images_dir / f'{new}.tif')
            shutil.copy(dataset['masks'] / f'{old}.tif', masks_dir / f'{new}.tif')

        metadata_csv = root / 'metadata.csv'
        metadata.to_csv(metadata_csv, index=False)

        return {'images': images_dir, 'masks': masks_dir, 'metadata': metadata_csv}

    return _copy


@pytest.fixture(scope='session')
def path_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """ The same images laid out so the metadata can be read off the paths instead of a .csv.

    Returns:
        The images directory, the masks directory and the matching template.
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


class Confirmation:
    """ Stands in for the prompt asking whether the loaded results may be dropped.

    Answers Yes unless a test sets `accepted` to False, and records whether it was asked at all, which
    is how a test tells a silent load apart from a confirmed one.
    """

    def __init__(self) -> None:
        self.accepted = True
        self.asked = 0

    def __call__(self, *args, **kwargs) -> QMessageBox.StandardButton:
        self.asked += 1
        return QMessageBox.StandardButton.Yes if self.accepted else QMessageBox.StandardButton.No


@pytest.fixture(autouse=True)
def confirmation(monkeypatch: pytest.MonkeyPatch) -> Confirmation:
    """ Answer the prompt shown when a load would drop the results of the loaded dataset.

    Autouse, as a real one is modal: any test loading over existing results would otherwise wait on a
    window nobody can answer.

    Returns:
        The stand-in, whose `accepted` a test sets to answer No.
    """
    stand_in = Confirmation()
    monkeypatch.setattr(QMessageBox, 'question', stand_in)
    return stand_in


def _unexpected_message_box(parent, title, text, *args, **kwargs) -> None:
    """ Fail on a message box no test asked for, rather than wait on it.

    Args:
        parent: The widget the message would be shown over.
        title: Its title bar.
        text: What it says, which is what the failure reports.
        *args: The buttons offered, of no interest here.
        **kwargs: As above.
    Raises:
        AssertionError: Always, naming the message that would have been shown.
    """
    raise AssertionError(f'Unexpected message box {title!r}: {text}')


@pytest.fixture(autouse=True)
def no_unexpected_message_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """ Turn an error or warning box no test was waiting for into a failure.

    A real one is modal, so a test reaching one by mistake would hang the whole suite instead of
    failing. A test expecting one replaces it with a stand-in of its own, which takes precedence.
    """
    monkeypatch.setattr(QMessageBox, 'critical', _unexpected_message_box)
    monkeypatch.setattr(QMessageBox, 'warning', _unexpected_message_box)


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

    @property
    def has_properties(self) -> bool:
        """ Whether the results carry properties, as PhenoMe reads its own. """
        return any(self.properties)


@pytest.fixture
def processed():
    """ Return a hook making an interface report a given number of processed images.

    Returns:
        Called as processed(interface, n_images).
    """
    def _apply(interface: PhenoMeInterface, n_images: int = 6) -> StubResults:
        results = StubResults(n_images)
        interface.pheno.results = results
        return results

    return _apply


# The properties a written checkpoint carries, one dict per image. Two of them, so a test can tell a
# frame of properties from a frame of one.
CHECKPOINT_PROPERTIES = ('area', 'intensity_mean')

# Width of the embeddings written. Nothing reads their values, only that they are there and how many.
EMBEDDING_DIM = 8


def results_for(interface: PhenoMeInterface, properties: bool = True) -> PhenoMeResults:
    """ Real PhenoMe results for the dataset an interface currently holds, as an extraction leaves them.

    No model is needed: embeddings are an array like any other. The dataset has to have been loaded
    first, the paths and the metadata being read off it.

    Args:
        interface: The interface whose loaded dataset the results answer for.
        properties: Whether the results carry properties, as a run computing them leaves them.
    Returns:
        One row per loaded image, with embeddings and, if asked for, properties.
    """
    metadata_df = interface.metadata_df
    paths = list(metadata_df['file_path'])

    # The metadata as PhenoMe holds it, which is what a load matches its own files against: a
    # checkpoint whose keys do not answer for them is read as holding none of them.
    keys = [column for column in metadata_df.columns if column not in ('file_path', 'mask_path')]
    metadata = [{key: row[key] for key in keys} for _, row in metadata_df.iterrows()]

    return PhenoMeResults(
        img_path=paths,
        metadata=metadata,
        properties=[
            {name: float(row) for name in CHECKPOINT_PROPERTIES} for row in range(len(paths))
        ] if properties else [],
        embeddings=np.random.default_rng(0).random((len(paths), EMBEDDING_DIM)).astype('float32'),
    )


@pytest.fixture
def extracted():
    """ Return a hook leaving real results in an interface's pipeline, as a run without a checkpoint does.

    Returns:
        Called as extracted(interface, properties=False).
    """
    def _apply(interface: PhenoMeInterface, properties: bool = True) -> None:
        interface.pheno.results = results_for(interface, properties)

    return _apply


@pytest.fixture
def written_checkpoint(tmp_path: Path):
    """ Return a hook writing a real checkpoint for the dataset an interface currently holds.

    Written rather than stubbed wherever what is being checked is the reading itself - PhenoMe
    resolves the paths a checkpoint stores against the files currently found, which no stand-in can
    answer for.

    Returns:
        Called as written_checkpoint(interface, properties=False), returning
        the path it was written to.
    """
    def _write(interface: PhenoMeInterface, properties: bool = True, name: str = 'previous.hdf5') -> Path:
        path = tmp_path / name

        # The arguments the checkpoint is written under are the ones the interface would run with, so
        # it agrees with the run and the extraction has nothing to ask about
        CheckpointManager.write_results_to_hdf5(
            results_for(interface, properties), str(path), processing_params=interface.processing_params()
        )
        return path

    return _write


class FakeLayers(list):
    """ Stands in for the napari layer list, which the image viewer only ever clears.

    The clears are counted, as they are all the list ever sees: the recorded layers are kept on the
    viewer itself, so an emptied list would otherwise be indistinguishable from one never filled.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cleared = 0

    def clear(self) -> None:
        """ Take note that the layers were taken off. """
        self.cleared += 1
        super().clear()


class RecordingViewer:
    """ Stands in for the napari viewer, recording what would have been displayed.

    Lets the display assertions be exact without building a real viewer, which is slow and needs Qt to
    actually paint. One integration test covers the real thing.
    """

    def __init__(self) -> None:
        self.layers = FakeLayers()
        self.added: list[dict[str, Any]] = []

    def add_image(self, data, name=None, metadata=None, rgb=None) -> None:
        """ Record an image layer. """
        self.added.append({'kind': 'image', 'data': data, 'name': name, 'metadata': metadata, 'rgb': rgb})

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

    # The boxes were filled from the defaults when the widget was built, which name a dataset on
    # one developer's machine. Shown again from the parameters, so the paths they hold - and the
    # checks run against them - are ones that exist wherever the suite runs.
    widget.loading_panel._on_structure_changed()
    return widget


@pytest.fixture
def in_main_window(qtbot):
    """ Return a hook building a panel inside the main window napari would have put it in.

    A docked table only knows the way back into its area when a main window is above it, Qt
    refusing to redock a dock it finds in no area of the main window it belongs to. A panel built
    on its own would therefore pass whether its table comes back or not.

    Returns:
        Called as
        in_main_window(lambda parent: SomePanel(..., parent)), returning the built panel.
    """
    # The windows outlive the calls that built them: qtbot holds only a weak reference, and a
    # collected window takes the panel down with it.
    windows = []

    def _build(build_panel):
        window = QMainWindow()
        panel = build_panel(window)
        window.setCentralWidget(panel)
        qtbot.addWidget(window)
        windows.append(window)
        return panel

    return _build


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
        Called as reduction(frame).
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


@pytest.fixture(autouse=True)
def inline_loads(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """ Run every dataset load inline rather than on a worker thread.

    The loading tab loads on a worker so the window keeps painting, which leaves most tests asserting
    on a table not yet filled. Patched on the class, so it covers every way a panel is built. A test
    marked real_load_worker keeps the real thread, for what only a real thread can show.
    """
    if request.node.get_closest_marker('real_load_worker'):
        return

    from phenome_gui.ui.loading_panel import LoadingPanel

    monkeypatch.setattr(
        LoadingPanel, '_load_data_worker', lambda self, load: _SyncWorker(load)
    )


@pytest.fixture
def sync_worker():
    """ Return a hook replacing a panel's worker factory with one that runs inline.

    Whatever the panel passes the factory is passed on to the function, so a step started under a flag
    is exercised under the flag it would really be given.

    Returns:
        Called as sync_worker(panel, '_prototypes_worker', fn).
    """
    def _replace(panel, worker_name: str, function) -> None:
        def _factory(*args, **kwargs) -> _SyncWorker:
            return _SyncWorker(lambda: function(*args, **kwargs))

        setattr(panel, worker_name, _factory)

    return _replace


@pytest.fixture
def dialog_result(monkeypatch: pytest.MonkeyPatch):
    """ Return a hook driving a modal parameter form without showing it.

    Patches the class's exec_ so it applies the given widget values and then reports accepted or rejected,
    which is what tells the caller whether to keep the edit.

    Returns:
        Called as dialog_result(DialogClass, accepted=True, _spin=3).
    """
    def _drive(dialog_class: type, accepted: bool, **edits) -> None:
        def _exec(dialog) -> int:
            for attribute, value in edits.items():
                getattr(dialog, attribute).value = value
            return int(accepted)

        monkeypatch.setattr(dialog_class, 'exec_', _exec)

    return _drive
