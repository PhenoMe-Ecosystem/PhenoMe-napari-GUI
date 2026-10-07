"""Tests for the image processing panel.

Nothing here loads a model or extracts embeddings: the pipeline state is stubbed instead, the model
being a sentinel object, and the load and the extraction stand-ins where a run is driven.
"""

import threading
from pathlib import Path

import pandas as pd
import pytest
from qtpy.QtWidgets import QFileDialog, QMessageBox

from phenome_gui.registries.models import MODEL_OPTIONS
from phenome_gui.ui.advanced_processing_params import AdvancedProcessingParams
from phenome_gui.ui.model_window import ModelWindow
from phenome_gui.ui.processing_panel import (
    CHECKPOINT_TEXT,
    NO_CHECKPOINT_TEXT,
    PROGRESS_TEXT,
    ProcessingPanel,
    format_elapsed,
)
from phenome_gui.ui.styles import GROUP_STYLESHEETS
from phenome_gui.utils import phenome_interface
from phenome_gui.utils.checkpoint import (
    DATASET_MISMATCH,
    MODEL_MISMATCH,
    CheckpointMismatch,
)
from phenome_gui.utils.phenome_interface import AUTOSAVE_CHECKPOINT
from phenome_gui.utils.progress import TaskStopped

# Two of the models the GUI offers, read off the registry: spelling them out here is what let these
# tests drift from it when they were renamed.
FIRST_MODEL = next(iter(MODEL_OPTIONS))
ANOTHER_MODEL = list(MODEL_OPTIONS)[2]


@pytest.fixture
def panel(widget) -> ProcessingPanel:
    """ The processing panel of a built widget. """
    return widget.processing_panel


@pytest.fixture
def ready(panel: ProcessingPanel, widget, processed, tmp_path: Path) -> ProcessingPanel:
    """ The panel with data loaded, a checkpoint selected, a model in place and images processed. """
    widget.loading_panel._on_load_data_csv()
    panel._phenome.model = object()
    panel._phenome.params.paths.checkpoint = tmp_path / 'run.hdf5'
    processed(panel._phenome, 6)
    panel.refresh_buttons()
    return panel


class Reported:
    """ Stands in for the message shown when a task raised, which is modal and would hang a test. """

    def __init__(self) -> None:
        self.shown = 0
        self.last = ''

    def __call__(self, parent, title, text, *args, **kwargs) -> None:
        """ Take note of a message, and of what it said.

        Args:
            parent: The widget the message would be shown over.
            title: Its title bar.
            text: What it says, which is what the tests read.
            *args: Whatever buttons the caller passed, which are of no interest here.
            **kwargs: As above.
        """
        self.shown += 1
        self.last = text


@pytest.fixture
def reported(monkeypatch: pytest.MonkeyPatch) -> Reported:
    """ Catch the message a failed task shows, rather than opening a window nobody can close.

    Returns:
        The stand-in, holding what was said.
    """
    stand_in = Reported()
    monkeypatch.setattr(QMessageBox, 'critical', stand_in)
    return stand_in


class Mismatched:
    """ Stands in for both halves of the resume question: what is found, and what is answered. """

    def __init__(self) -> None:
        self.asked = 0
        self.last = ''
        self.accepted = False

    def __call__(self, mismatch, accepted: bool = False) -> None:
        """ Have the checkpoint report a mismatch, and have the user answer it a given way.

        Args:
            mismatch: What reading the checkpoint is to report.
            accepted: Whether the question about it is to be answered yes.
        """
        self._mismatch = mismatch
        self.accepted = accepted

    def inspect(self, *args) -> object:
        """ Stand in for checking the embeddings held against the run.

        Args:
            *args: The held settings and images, the run's settings and the loaded images, of no interest here.
        Returns:
            Whatever the test asked to be found.
        """
        return self._mismatch

    def ask(self, parent, title, text, *args, **kwargs) -> QMessageBox.StandardButton:
        """ Stand in for the question put to the user, and answer it as the test asked.

        Args:
            parent: The widget the question would be shown over.
            title: Its title bar.
            text: What it asks, which is what the tests read.
            *args: The buttons offered, of no interest here.
            **kwargs: As above.
        Returns:
            Yes or No, as the test set it.
        """
        self.asked += 1
        self.last = text
        return QMessageBox.StandardButton.Yes if self.accepted else QMessageBox.StandardButton.No


@pytest.fixture
def mismatched(panel: ProcessingPanel, monkeypatch: pytest.MonkeyPatch) -> Mismatched:
    """ Decide what checking the held embeddings finds, and how the question about it is answered.

    Only the check is replaced, so the interface still decides whether there is anything to check at
    all - nothing, unless the run resumes.

    Returns:
        The stand-in, called as mismatched(what_is_found, accepted=...).
    """
    stand_in = Mismatched()
    stand_in(None)

    monkeypatch.setattr(phenome_interface, 'inspect_held_embeddings', stand_in.inspect)
    monkeypatch.setattr(QMessageBox, 'warning', stand_in.ask)
    return stand_in


# ----------------------------------------------------------------------------
# The process button, which waits on the data and a model
# ----------------------------------------------------------------------------


def test_extract_waits_on_the_data(panel: ProcessingPanel) -> None:
    """The model is not waited on: the default counts as selected, and the run loads it itself."""
    assert not panel._process_button.isEnabled()
    assert panel._process_button.toolTip() == 'Please load data first.'


def test_extract_opens_on_the_data_alone(panel: ProcessingPanel, widget) -> None:
    """No checkpoint is waited on: the run keeps its results in memory until they are saved."""
    widget.loading_panel._on_load_data_csv()

    assert panel._phenome.params.paths.checkpoint is None
    assert panel._process_button.isEnabled()
    assert panel._process_button.toolTip() == ''


def test_extract_waits_on_a_model_the_registry_no_longer_offers(ready: ProcessingPanel) -> None:
    """The parameter outlives the registry, and a run on a model nobody can load would only fail."""
    ready._phenome.params.process.model = 'Some - Retired Model'
    ready.refresh_buttons()

    assert not ready._process_button.isEnabled()
    assert ready._process_button.toolTip() == 'Please select a model first.'


def test_extract_opens_without_a_loaded_model(ready: ProcessingPanel) -> None:
    """The run is what loads the model, so waiting on a loaded one would never open the button."""
    ready._phenome.model = None
    ready.refresh_buttons()

    assert ready._process_button.isEnabled()


# ----------------------------------------------------------------------------
# The running lock
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('running', [True, False])
def test_the_lock_is_announced(panel: ProcessingPanel, qtbot, running: bool) -> None:
    with qtbot.waitSignal(panel.task_running_changed, timeout=1000) as blocker:
        panel._set_processing_task_running(running)

    assert blocker.args == [running]


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------


def test_choosing_cuda_writes_the_torch_name(panel: ProcessingPanel) -> None:
    """Lowercase matters: the string goes straight to torch, which does not know 'CUDA'."""
    panel._cuda_radio.setEnabled(True)  # off without a GPU, so force it to drive the toggle
    panel._cuda_radio.setChecked(True)

    assert panel._phenome.params.process.device == 'cuda'


def test_going_back_to_the_cpu(panel: ProcessingPanel) -> None:
    panel._cuda_radio.setEnabled(True)
    panel._cuda_radio.setChecked(True)
    panel._cpu_radio.setChecked(True)

    assert panel._phenome.params.process.device == 'cpu'


def test_the_cuda_choice_follows_the_hardware(panel: ProcessingPanel) -> None:
    assert panel._cuda_radio.isEnabled() is panel._phenome.cuda_available


@pytest.fixture
def picking_model(panel: ProcessingPanel, monkeypatch: pytest.MonkeyPatch):
    """ Return a hook answering the model window with a pick, without showing it.

    Its exec_ is modal and would hang the test, so it is answered outright. The window is built
    either way, which is what keeps this a test of the panel driving a real one.

    Returns:
        Called as picking_model(ANOTHER_MODEL), or with None for the window being cancelled.
    """
    def _answer(model: str | None) -> None:
        def _exec(window) -> int:
            if model is None:
                return 0
            window._select(model)
            return 1

        monkeypatch.setattr(ModelWindow, 'exec_', _exec)

    return _answer


def test_choosing_a_model_writes_the_parameter(panel: ProcessingPanel, picking_model) -> None:
    picking_model(ANOTHER_MODEL)

    panel._select_model_button.click()

    assert panel._phenome.params.process.model == ANOTHER_MODEL


def test_choosing_a_model_does_not_load_it(
    panel: ProcessingPanel, picking_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The download is left to the extraction, a pick being only a choice."""
    loads = []
    monkeypatch.setattr(panel._phenome, 'load_model', lambda: loads.append(True))
    picking_model(ANOTHER_MODEL)

    panel._select_model_button.click()

    assert loads == []
    assert panel._phenome.model is None
    assert panel._phenome.needs_model_load


def test_a_cancelled_window_changes_nothing(panel: ProcessingPanel, picking_model) -> None:
    """A model looked at and thought better of leaves the parameters as they were."""
    before = panel._phenome.params.process.model
    picking_model(None)

    panel._select_model_button.click()

    assert panel._phenome.params.process.model == before


def test_choosing_a_preset_writes_the_parameter(panel: ProcessingPanel) -> None:
    panel._props_selection_combo.setCurrentText('Complete')

    assert panel._phenome.params.properties.preset == 'Complete'


# ----------------------------------------------------------------------------
# The advanced parameters form
# ----------------------------------------------------------------------------


@pytest.fixture
def advanced_form(monkeypatch: pytest.MonkeyPatch):
    """ Return a hook answering the advanced form without showing it.

    Its exec_ is modal and would hang the test, so it fills in the fields it is given and settles at
    once. The form is built either way, which is what keeps this a test of the panel driving a real one.

    Returns:
        Called as advanced_form(accepted=True, batch_size=64, num_workers=2, channel_mode='combined',
        resize_size=None, pad_size=300, l2_normalize_channels=True), any field left out keeping what
        the form opened on, and a size of None unticking its box. Returns the values each opened form
        showed.
    """
    shown = []

    def _answer(accepted: bool, **fields) -> list[dict]:
        def _exec(dialog: AdvancedProcessingParams) -> int:
            shown.append({
                'channel_mode': dialog._channel_mode_combo.currentText(),
                'batch_size': dialog._batch_size_spinbox.value(),
                'num_workers': dialog._num_workers_spinbox.value(),
            })
            if 'channel_mode' in fields:
                dialog._channel_mode_combo.setCurrentText(fields['channel_mode'])
            if 'batch_size' in fields:
                dialog._batch_size_spinbox.setValue(fields['batch_size'])
            if 'num_workers' in fields:
                dialog._num_workers_spinbox.setValue(fields['num_workers'])
            for name, checkbox, spinbox in (
                ('resize_size', dialog._resize_checkbox, dialog._resize_spinbox),
                ('pad_size', dialog._pad_checkbox, dialog._pad_spinbox),
            ):
                if name in fields:
                    checkbox.setChecked(fields[name] is not None)
                    if fields[name] is not None:
                        spinbox.setValue(fields[name])
            if 'l2_normalize_channels' in fields:
                dialog._l2_checkbox.setChecked(fields['l2_normalize_channels'])
            return int(accepted)

        monkeypatch.setattr(AdvancedProcessingParams, 'exec_', _exec)
        return shown

    return _answer


def test_confirming_the_advanced_form_applies_it(panel: ProcessingPanel, advanced_form) -> None:
    """Only the fields it shows: the model and the device are the panel's own to pick."""
    model = panel._phenome.params.process.model
    advanced_form(
        accepted=True, channel_mode='combined', batch_size=128, num_workers=4,
        resize_size=None, pad_size=300, l2_normalize_channels=True,
    )

    panel._advanced_parameters_button.click()

    process = panel._phenome.params.process
    assert (process.channel_mode, process.batch_size, process.num_workers) == ('combined', 128, 4)
    assert (process.resize_size, process.pad_size, process.l2_normalize_channels) == (None, 300, True)
    assert process.model == model


def test_cancelling_the_advanced_form_changes_nothing(panel: ProcessingPanel, advanced_form) -> None:
    before = panel._phenome.params.process.model_copy(deep=True)
    advanced_form(
        accepted=False, channel_mode='combined', batch_size=128, num_workers=4,
        resize_size=None, pad_size=300, l2_normalize_channels=True,
    )

    panel._advanced_parameters_button.click()

    assert panel._phenome.params.process == before


def test_the_advanced_form_opens_on_what_the_run_would_use(panel: ProcessingPanel, advanced_form) -> None:
    """Built anew on every open, so a cancelled edit is never shown again."""
    panel._phenome.params.process.channel_mode = 'combined'
    panel._phenome.params.process.batch_size = 64
    panel._phenome.params.process.num_workers = 6
    shown = advanced_form(accepted=False, batch_size=512)
    panel._advanced_parameters_button.click()

    panel._advanced_parameters_button.click()

    assert shown == [{'channel_mode': 'combined', 'batch_size': 64, 'num_workers': 6}] * 2


# ----------------------------------------------------------------------------
# The progress bar
# ----------------------------------------------------------------------------


def test_a_known_total_drives_the_bar(panel: ProcessingPanel) -> None:
    panel._on_progress(3, 10, 'Processing batches')

    assert panel._progress_bar.maximum() == 10
    assert panel._progress_bar.value() == 3
    assert panel._progress_bar.format() == 'Processing batches: %v/%m (%p%)'


def test_an_unknown_total_puts_the_bar_in_busy_mode(panel: ProcessingPanel) -> None:
    """PhenoMe documents a total of 0 as 'unknown'; a 0..0 range is Qt's busy indicator."""
    panel._on_progress(0, 0, 'Preparing')

    assert panel._progress_bar.maximum() == 0
    assert panel._progress_bar.format() == 'Preparing'


# ----------------------------------------------------------------------------
# Letting go of the previous run when another dataset is loaded
# ----------------------------------------------------------------------------


def test_clearing_lets_go_of_the_checkpoint(ready: ProcessingPanel) -> None:
    """The interface unselects it, a checkpoint answering only for its own images, and the label follows."""
    ready._phenome.params.paths.checkpoint = Path('previous.hdf5')
    ready._phenome.clear_results()

    ready.clear_results()

    assert ready._phenome.params.paths.checkpoint is None
    assert ready._current_checkpoint_label.text() == NO_CHECKPOINT_TEXT


def test_clearing_settles_the_progress_bar(ready: ProcessingPanel) -> None:
    ready._on_progress(3, 10, 'Processing batches')

    ready.clear_results()

    assert ready._progress_bar.value() == 0
    assert ready._progress_bar.maximum() == 1
    assert ready._progress_bar.format() == ''


def test_clearing_keeps_the_selected_model(ready: ProcessingPanel) -> None:
    """A model is not read off the images, so the label naming it, and the network loaded for it, stay."""
    ready._current_model_label.setText(FIRST_MODEL)

    ready._phenome.clear_results()
    ready.clear_results()

    assert ready._current_model_label.text() == FIRST_MODEL
    assert ready._phenome.model is not None


# ----------------------------------------------------------------------------
# A task that raises, which must never leave the tab locked
# ----------------------------------------------------------------------------


def test_a_failed_extraction_is_reported(ready: ProcessingPanel, reported) -> None:
    ready._on_process_failed(RuntimeError('Checkpoint parameter mismatch'))

    assert 'Checkpoint parameter mismatch' in reported.last


def test_a_failed_extraction_settles_the_bar(ready: ProcessingPanel, reported) -> None:
    """A task raising while the bar spins would otherwise leave it spinning for good."""
    ready._on_progress(0, 0, 'Preparing...')

    ready._on_process_failed(RuntimeError('nope'))

    assert ready._progress_bar.maximum() == 1
    assert ready._progress_bar.format() == 'Failed'


def test_a_failed_extraction_announces_the_task_is_over(ready: ProcessingPanel, reported) -> None:
    """The analysis steps lock themselves on this too, and would otherwise stay locked."""
    announced = []
    ready.task_running_changed.connect(announced.append)

    ready._on_process_failed(RuntimeError('nope'))

    assert announced[-1] is False


def test_a_running_task_shuts_the_model_and_device_choices(ready: ProcessingPanel) -> None:
    """The run loads the model it starts with, so a pick made during it would name one it never used."""
    ready._set_processing_task_running(True)

    assert not ready._select_model_button.isEnabled()
    assert not ready._cpu_radio.isEnabled()
    assert not ready._cuda_radio.isEnabled()


def test_a_failed_extraction_reopens_the_model_and_device_choices(ready: ProcessingPanel, reported) -> None:
    """A failed model load arrives as a failed extraction, and the next try may want another model."""
    ready._set_processing_task_running(True)

    ready._on_process_failed(OSError('connection refused'))

    assert ready._select_model_button.isEnabled()
    assert ready._cpu_radio.isEnabled()
    assert ready._cuda_radio.isEnabled() is ready._phenome.cuda_available


# ----------------------------------------------------------------------------
# Stopping a running task
# ----------------------------------------------------------------------------


def test_there_is_nothing_to_stop_before_a_run(panel: ProcessingPanel) -> None:
    assert not panel._stop_button.isEnabled()
    assert panel._stop_button.toolTip() == 'No task is running.'


def test_there_is_still_nothing_to_stop_once_everything_is_loaded(ready: ProcessingPanel) -> None:
    """Data, a model and processed images open the two task buttons, none of which is a running task."""
    assert not ready._stop_button.isEnabled()


def test_a_stopped_extraction_is_not_reported_as_a_failure(ready: ProcessingPanel, reported) -> None:
    """It was asked for, and what it computed is kept."""
    ready._on_process_failed(TaskStopped())

    assert reported.shown == 0


def test_a_stopped_extraction_settles_the_bar(ready: ProcessingPanel, reported) -> None:
    """A task stopped while it prepares would otherwise leave the bar spinning for good."""
    ready._on_progress(0, 0, 'Preparing...')

    ready._on_process_failed(TaskStopped())

    assert ready._progress_bar.maximum() == 1
    assert ready._progress_bar.format() == 'Stopped'


# ----------------------------------------------------------------------------
# A run of the Process button, driven end to end
# ----------------------------------------------------------------------------


class Pipeline:
    """ Stands in for the PhenoMe calls a run makes, recording them in the order they come.

    Stubbed where the GUI hands over to PhenoMe, so everything in front of it - the panel, the
    interface, the progress throttle and the stop - runs as it does for real. Each step can be given
    something to do while it runs, for a test watching the panel mid-run or making the step fail.
    """

    def __init__(self) -> None:
        self.steps: list[str] = []

        # The keyword arguments each extraction and properties step was handed, in the order they came
        self.extractions: list[dict] = []
        self.properties: list[dict] = []

        # Where an interrupted extraction was autosaved
        self.autosaved: list[Path] = []

        self.on_load = None
        self.on_process = None
        self.on_properties = None

    def load_model(self, model_name: str, device: str = 'cpu') -> tuple:
        """ Stand in for downloading a network.

        Args:
            model_name: The registry name of the model.
            device: The device it would be moved onto.
        Returns:
            A model, PhenoMe's own transformations, and no arguments of its own.
        """
        self.steps.append('load')
        if self.on_load is not None:
            self.on_load()
        return object(), None, {}

    def process_images(self, model, progress_callback=None, **kwargs) -> None:
        """ Stand in for the extraction, handing its progress callback to the test.

        Args:
            model: The loaded network.
            progress_callback: What the extraction reports its batches through.
            **kwargs: The run's other arguments, recorded.
        """
        self.steps.append('process')
        self.extractions.append(kwargs)
        if self.on_process is not None:
            self.on_process(progress_callback)

    def compute_properties(self, property_preset: str, progress_callback=None, **kwargs) -> pd.DataFrame:
        """ Stand in for the properties step, handing its progress callback to the test.

        Args:
            property_preset: The preset, as PhenoMe names it.
            progress_callback: What the step reports its images through.
            **kwargs: The step's other arguments, recorded.
        Returns:
            A frame of properties, one image's worth.
        """
        self.steps.append('properties')
        self.properties.append(kwargs)
        if self.on_properties is not None:
            self.on_properties(progress_callback)
        return pd.DataFrame({'area': [1.0]})

    def save_results(self, checkpoint_path: str) -> None:
        """ Take note of an interrupted run being autosaved.

        Args:
            checkpoint_path: Where it would be written.
        """
        self.autosaved.append(Path(checkpoint_path))


@pytest.fixture
def stubbed_phenome(ready: ProcessingPanel, mismatched, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    """ Put a stand-in for PhenoMe under the Process button, leaving its worker as it is.

    A resumed run finds the embeddings held matching unless a test says otherwise through `mismatched`.
    An interrupted run keeps whatever the pipeline holds, the stand-in results having no buffer of
    embeddings to stack, and is autosaved into the stand-in rather than to disk.

    Returns:
        The stand-in, recording the steps the run made.
    """
    stand_in = Pipeline()
    image_processing = ready._phenome.image_processing
    monkeypatch.setattr(image_processing, 'load_model', stand_in.load_model)
    monkeypatch.setattr(image_processing, 'process_images', stand_in.process_images)
    monkeypatch.setattr(ready._phenome.pheno, 'compute_properties', stand_in.compute_properties)
    monkeypatch.setattr(image_processing, 'keep_stopped_results', lambda: ready._phenome.has_processed_images)
    monkeypatch.setattr(image_processing, 'save_results', stand_in.save_results)
    return stand_in


@pytest.fixture
def pipeline(ready: ProcessingPanel, stubbed_phenome: Pipeline, sync_worker) -> Pipeline:
    """ Have the Process button run its real worker inline, on a stand-in for PhenoMe.

    Inline rather than on a thread, so a test can watch the panel from inside each step.

    Returns:
        The stand-in, recording the steps the run made.
    """
    sync_worker(
        ready,
        '_process_images_worker',
        lambda *args: ProcessingPanel._process_images_worker.__wrapped__(ready, *args),
    )
    return stubbed_phenome


@pytest.fixture
def resume(ready: ProcessingPanel, processed) -> None:
    """ Tick Resume progress over embeddings held for half of the loaded images. """
    processed(ready._phenome, 3)
    ready._resume_checkbox.click()


def fail(error: Exception):
    """ A step that raises, the way PhenoMe raises out of one.

    Args:
        error: What the step raises.
    Returns:
        Something to hand a Pipeline as one of its steps.
    """
    def _raise(*args) -> None:
        raise error

    return _raise


def test_a_run_loads_the_model_extracts_then_computes_the_properties(
    ready: ProcessingPanel, pipeline: Pipeline
) -> None:
    ready._process_button.click()

    assert pipeline.steps == ['load', 'process', 'properties']


def test_a_run_without_properties_ends_with_the_embeddings(ready: ProcessingPanel, pipeline: Pipeline) -> None:
    ready._props_checkbox.click()

    ready._process_button.click()

    assert pipeline.steps == ['load', 'process']
    assert ready._progress_bar.format() == 'Done!'


def test_the_model_held_is_not_loaded_again(ready: ProcessingPanel, pipeline: Pipeline) -> None:
    """A second download of the same network would be hundreds of megabytes for nothing."""
    ready._process_button.click()

    ready._process_button.click()

    assert pipeline.steps.count('load') == 1
    assert pipeline.steps.count('process') == 2


def test_picking_another_model_loads_it_on_the_next_run(
    ready: ProcessingPanel, pipeline: Pipeline, picking_model
) -> None:
    ready._process_button.click()
    picking_model(ANOTHER_MODEL)
    ready._select_model_button.click()

    ready._process_button.click()

    assert pipeline.steps.count('load') == 2


def test_the_button_follows_the_run_from_step_to_step(ready: ProcessingPanel, pipeline: Pipeline) -> None:
    """Each step takes a while of its own, and the button is what says which one is under way."""
    said = []

    def _note(*args) -> None:
        said.append(ready._process_button.text())

    pipeline.on_load = pipeline.on_process = pipeline.on_properties = _note

    ready._process_button.click()

    assert said == ['Loading model...', 'Processing images...', 'Calculating properties...']
    assert ready._process_button.text() == 'Process Images'


def test_the_tab_is_locked_while_the_run_goes(ready: ProcessingPanel, pipeline: Pipeline, widget) -> None:
    """The run reads the model, the preset and the checkbox as it goes, and only the stop is offered."""
    during = {}

    def _look(callback) -> None:
        during.update(
            process=ready._process_button.isEnabled(),
            model=ready._select_model_button.isEnabled(),
            preset=ready._props_selection_combo.isEnabled(),
            compute=ready._props_checkbox.isEnabled(),
            resume=ready._resume_checkbox.isEnabled(),
            stop=ready._stop_button.isEnabled(),
            analysis=widget.analysis_panel._run_clustering_button.isEnabled(),
        )

    pipeline.on_process = _look

    ready._process_button.click()

    assert during == {
        'process': False, 'model': False, 'preset': False, 'compute': False, 'resume': False, 'stop': True,
        'analysis': False,
    }


def test_a_finished_run_gives_the_tab_back(ready: ProcessingPanel, pipeline: Pipeline, widget) -> None:
    ready._process_button.click()

    assert ready._process_button.isEnabled()
    assert ready._props_checkbox.isEnabled()
    assert ready._resume_checkbox.isEnabled()
    assert not ready._stop_button.isEnabled()
    assert widget.analysis_panel._run_clustering_button.isEnabled()
    assert ready._progress_bar.value() == ready._progress_bar.maximum()


def test_what_phenome_reports_reaches_the_bar(ready: ProcessingPanel, pipeline: Pipeline) -> None:
    seen = []

    def _report(callback) -> None:
        callback(3, 10, 'Processing batches')
        seen.append((ready._progress_bar.value(), ready._progress_bar.maximum()))

    pipeline.on_process = _report

    ready._process_button.click()

    assert seen == [(3, 10)]


# ----------------------------------------------------------------------------
# The outline of each step's group, following the run from step to step
# ----------------------------------------------------------------------------


def outline(group) -> str:
    """ How a group is outlined: 'ok', 'wait', 'bad', or 'none' for one left unstyled. """
    names = {
        GROUP_STYLESHEETS['STYLE_OK']: 'ok',
        GROUP_STYLESHEETS['STYLE_WAIT']: 'wait',
        GROUP_STYLESHEETS['STYLE_ERROR']: 'bad',
    }
    return names.get(group.styleSheet(), 'none')


def outlines(panel: ProcessingPanel) -> dict[str, str]:
    """ How the embeddings and the properties groups are outlined. """
    return {
        'embeddings': outline(panel._embeddings_group),
        'properties': outline(panel._properties_group),
    }


def test_the_groups_follow_the_run_from_step_to_step(ready: ProcessingPanel, pipeline: Pipeline) -> None:
    """The step under way is outlined as waiting, and the one before it as done."""
    during = {}
    pipeline.on_process = lambda callback: during.update(extracting=outlines(ready))
    pipeline.on_properties = lambda callback: during.update(computing=outlines(ready))

    ready._process_button.click()

    assert during['extracting']['embeddings'] == 'wait'
    assert during['computing'] == {'embeddings': 'ok', 'properties': 'wait'}


def test_a_finished_run_outlines_both_groups_as_done(ready: ProcessingPanel, pipeline: Pipeline) -> None:
    ready._process_button.click()

    assert outlines(ready) == {'embeddings': 'ok', 'properties': 'ok'}


def test_a_run_without_properties_never_outlines_them_as_waiting(
    ready: ProcessingPanel, pipeline: Pipeline
) -> None:
    seen = []
    pipeline.on_process = lambda callback: seen.append(outline(ready._properties_group))
    ready._props_checkbox.click()

    ready._process_button.click()

    assert 'wait' not in seen
    assert outline(ready._properties_group) != 'wait'
    assert outline(ready._embeddings_group) == 'ok'


def test_a_failed_extraction_outlines_the_embeddings_as_failed(
    ready: ProcessingPanel, pipeline: Pipeline, reported
) -> None:
    pipeline.on_process = fail(RuntimeError('nope'))

    ready._process_button.click()

    assert outline(ready._embeddings_group) == 'bad'


def test_a_failed_properties_step_outlines_only_the_properties_as_failed(
    ready: ProcessingPanel, pipeline: Pipeline, reported
) -> None:
    """The embeddings were extracted before the properties step began, and are done."""
    pipeline.on_properties = fail(ValueError('unknown property'))

    ready._process_button.click()

    assert outlines(ready) == {'embeddings': 'ok', 'properties': 'bad'}


def test_a_stopped_properties_step_is_outlined_as_stopped(
    ready: ProcessingPanel, pipeline: Pipeline, reported
) -> None:
    pipeline.on_properties = reporting(10, stop_after=2, panel=ready, done=[])

    ready._process_button.click()

    assert outline(ready._properties_group) == 'bad'


@pytest.mark.parametrize(('kept', 'saveable'), [(3, True), (0, False)])
def test_a_stopped_extraction_is_outlined_as_stopped_and_keeps_its_embeddings(
    ready: ProcessingPanel, pipeline: Pipeline, reported, processed, kept: int, saveable: bool
) -> None:
    """Whatever it kept, which can be saved meanwhile and is counted in the progress."""
    stop = reporting(10, stop_after=2, panel=ready, done=[])

    def _extract(callback) -> None:
        # PhenoMe empties the results as it starts, and the interface keeps what it computed on a stop
        processed(ready._phenome, kept)
        stop(callback)

    pipeline.on_process = _extract

    ready._process_button.click()

    assert outline(ready._embeddings_group) == 'bad'
    assert ready._save_checkpoint_button.isEnabled() is saveable
    assert ready._progress_value_label.text() == PROGRESS_TEXT.format(held=kept, loaded=6)


def test_a_run_on_its_own_thread_leaves_the_widgets_to_the_gui_thread(
    ready: ProcessingPanel, stubbed_phenome: Pipeline, qtbot, qtlog
) -> None:
    """On the real worker thread: a widget may only be touched from the GUI thread.

    Restyling a group from the worker is what made Qt warn about timers started from another thread,
    but only for some of them, so each restyle is also checked for the thread it is made on.
    """
    restyled_from = []
    for group in (ready._embeddings_group, ready._properties_group):
        restyle = group.setStyleSheet

        def _noting(sheet: str, restyle=restyle) -> None:
            restyled_from.append(threading.current_thread() is threading.main_thread())
            restyle(sheet)

        group.setStyleSheet = _noting

    ready._process_button.click()

    qtbot.waitUntil(lambda: ready._process_button.isEnabled(), timeout=5000)

    assert stubbed_phenome.steps == ['load', 'process', 'properties']
    assert restyled_from
    assert all(restyled_from)
    assert [record.message for record in qtlog.records if 'another thread' in record.message] == []


# ----------------------------------------------------------------------------
# Processing again over results already there
# ----------------------------------------------------------------------------


@pytest.fixture
def clustered(ready: ProcessingPanel) -> ProcessingPanel:
    """ The panel over a session that has also been clustered, the labels sitting in the metadata. """
    paths = list(ready._phenome.metadata_df['file_path'])
    ready._phenome.clustering_df = pd.DataFrame({'image_path': paths, 'cluster': [0, 1] * 3})
    ready._phenome.metadata_df['cluster'] = pd.array([0, 1] * 3, dtype='Int64')
    return ready


def test_a_run_over_existing_embeddings_asks_first(
    ready: ProcessingPanel, pipeline: Pipeline, confirmation
) -> None:
    ready._process_button.click()

    assert confirmation.asked == 1
    assert pipeline.steps == ['load', 'process', 'properties']


def test_a_first_run_asks_nothing(ready: ProcessingPanel, pipeline: Pipeline, confirmation, processed) -> None:
    """There is nothing yet for it to replace."""
    processed(ready._phenome, 0)
    ready.refresh_buttons()

    ready._process_button.click()

    assert confirmation.asked == 0
    assert pipeline.steps == ['load', 'process', 'properties']


def test_refusing_runs_nothing_and_keeps_the_results(
    clustered: ProcessingPanel, pipeline: Pipeline, confirmation, qtbot
) -> None:
    confirmation.accepted = False

    with qtbot.assertNotEmitted(clustered.analysis_cleared):
        clustered._process_button.click()

    assert pipeline.steps == []
    assert clustered._phenome.clustering_df is not None
    assert 'cluster' in clustered._phenome.metadata_df.columns
    assert clustered._process_button.isEnabled()
    assert not clustered._stop_button.isEnabled()


def test_confirming_drops_the_analysis_before_the_run(clustered: ProcessingPanel, pipeline: Pipeline) -> None:
    """Computed from the embeddings the run replaces, so they would answer for none it produces."""
    announced = []
    clustered.analysis_cleared.connect(lambda: announced.append(list(pipeline.steps)))

    clustered._process_button.click()

    assert announced == [[]]
    assert clustered._phenome.clustering_df is None
    assert 'cluster' not in clustered._phenome.metadata_df.columns
    assert pipeline.steps == ['load', 'process', 'properties']


@pytest.mark.usefixtures('resume')
def test_resuming_asks_nothing_but_drops_the_analysis(
    clustered: ProcessingPanel, pipeline: Pipeline, confirmation, qtbot
) -> None:
    """Nothing is replaced, only added to, but the analysis no longer answers for every embedding there."""
    with qtbot.waitSignal(clustered.analysis_cleared):
        clustered._process_button.click()

    assert confirmation.asked == 0
    assert clustered._phenome.clustering_df is None
    assert pipeline.steps == ['load', 'process', 'properties']
    assert pipeline.extractions[0]['append'] is True


def test_resume_progress_with_nothing_held_is_a_run_from_scratch(
    ready: ProcessingPanel, pipeline: Pipeline, confirmation, processed
) -> None:
    processed(ready._phenome, 0)
    ready._resume_checkbox.click()

    ready._process_button.click()

    assert confirmation.asked == 0
    assert 'append' not in pipeline.extractions[0]


@pytest.mark.usefixtures('resume')
@pytest.mark.parametrize('reason', [MODEL_MISMATCH, DATASET_MISMATCH])
def test_refusing_a_restart_runs_nothing_and_keeps_the_results(
    clustered: ProcessingPanel, pipeline: Pipeline, mismatched, confirmation, reason: str
) -> None:
    mismatched(CheckpointMismatch(reason, 'what differs'), accepted=False)

    clustered._process_button.click()

    assert mismatched.asked == 1
    assert confirmation.asked == 0
    assert pipeline.steps == []
    assert clustered._phenome.clustering_df is not None
    assert clustered._process_button.isEnabled()
    assert not clustered._stop_button.isEnabled()


@pytest.mark.usefixtures('resume')
def test_a_confirmed_restart_processes_every_image(ready: ProcessingPanel, pipeline: Pipeline, mismatched) -> None:
    """The answer is the consent to replace the embeddings, so nothing more is asked."""
    mismatched(CheckpointMismatch(MODEL_MISMATCH, 'channel_mode differs'), accepted=True)

    ready._process_button.click()

    assert mismatched.asked == 1
    assert pipeline.steps == ['load', 'process', 'properties']
    assert 'append' not in pipeline.extractions[0]


@pytest.mark.usefixtures('resume')
def test_the_restart_question_says_what_was_found(ready: ProcessingPanel, pipeline: Pipeline, mismatched) -> None:
    mismatched(CheckpointMismatch(DATASET_MISMATCH, '1 of the 3 image(s) held are not among the 6 currently loaded.'))

    ready._process_button.click()

    assert '1 of the 3 image(s) held are not among the 6 currently loaded.' in mismatched.last


# ----------------------------------------------------------------------------
# Where a run writes: nowhere, unless it is interrupted
# ----------------------------------------------------------------------------


def test_a_run_never_writes_to_a_checkpoint(ready: ProcessingPanel, pipeline: Pipeline) -> None:
    """The results stay in memory until saved, even with a checkpoint selected from a save or a load."""
    assert ready._phenome.params.paths.checkpoint is not None

    ready._process_button.click()

    for passed in pipeline.extractions + pipeline.properties:
        assert not {'checkpoint_path', 'lazy_checkpoint', 'force_update', 'force_reprocess'} & set(passed)
    assert pipeline.autosaved == []


def test_a_run_from_scratch_asks_nothing_of_the_embeddings_held(
    ready: ProcessingPanel, pipeline: Pipeline, mismatched
) -> None:
    """They are replaced rather than extended, so what they were made under is no reason to stop."""
    mismatched(CheckpointMismatch(DATASET_MISMATCH, 'no image in common'), accepted=False)

    ready._process_button.click()

    assert mismatched.asked == 0
    assert pipeline.steps == ['load', 'process', 'properties']


@pytest.fixture
def working_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """ Run the test from a directory of its own, where an autosave would be written. """
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_a_stopped_run_is_autosaved_without_becoming_the_current_checkpoint(
    ready: ProcessingPanel, pipeline: Pipeline, reported, working_directory: Path
) -> None:
    previous = ready._phenome.params.paths.checkpoint
    pipeline.on_process = fail(TaskStopped())

    ready._process_button.click()

    autosave = working_directory / AUTOSAVE_CHECKPOINT
    assert pipeline.autosaved == [autosave]
    assert ready._phenome.params.paths.checkpoint == previous
    assert ready._current_checkpoint_label.text() == CHECKPOINT_TEXT.format(name=previous.name)
    assert ready._progress_bar.format() == 'Stopped - progress autosaved'
    assert str(autosave) in ready._progress_bar.toolTip()
    assert reported.shown == 0


def test_a_failed_extraction_says_where_its_progress_was_saved(
    ready: ProcessingPanel, pipeline: Pipeline, reported, working_directory: Path
) -> None:
    pipeline.on_process = fail(RuntimeError('out of memory'))

    ready._process_button.click()

    assert reported.shown == 1
    assert 'out of memory' in reported.last
    assert str(working_directory / AUTOSAVE_CHECKPOINT) in reported.last


def test_a_stopped_properties_step_is_not_autosaved(
    ready: ProcessingPanel, pipeline: Pipeline, reported, working_directory: Path
) -> None:
    """The embeddings are all there, to be saved like any finished run's."""
    pipeline.on_properties = fail(TaskStopped())

    ready._process_button.click()

    assert pipeline.autosaved == []
    assert ready._progress_bar.format() == 'Stopped'


def test_the_next_run_forgets_the_last_autosave(
    ready: ProcessingPanel, pipeline: Pipeline, reported, working_directory: Path
) -> None:
    pipeline.on_process = fail(TaskStopped())
    ready._process_button.click()

    pipeline.on_process = None
    pipeline.on_properties = fail(TaskStopped())
    ready._process_button.click()

    assert ready._progress_bar.format() == 'Stopped'
    assert ready._progress_bar.toolTip() == ''


def test_embeddings_for_part_of_the_images_are_shown_as_waiting_to_be_resumed(
    ready: ProcessingPanel, processed
) -> None:
    processed(ready._phenome, 3)

    ready.refresh_buttons()

    assert ready._embeddings_group.styleSheet() == GROUP_STYLESHEETS['STYLE_WAIT']
    assert ready._progress_value_label.text() == PROGRESS_TEXT.format(held=3, loaded=6)
    assert ready._current_checkpoint_label.text() == CHECKPOINT_TEXT.format(name='run.hdf5')


def test_embeddings_for_every_image_are_shown_as_done(ready: ProcessingPanel) -> None:
    ready.refresh_buttons()

    assert ready._embeddings_group.styleSheet() == GROUP_STYLESHEETS['STYLE_OK']
    assert ready._progress_value_label.text() == PROGRESS_TEXT.format(held=6, loaded=6)


def test_the_progress_counts_the_loaded_images_before_any_is_processed(panel: ProcessingPanel, widget) -> None:
    """Nothing loaded is nothing to count; a dataset loaded is all of it still to process."""
    assert panel._progress_value_label.text() == PROGRESS_TEXT.format(held=0, loaded=0)

    widget.loading_panel._on_load_data_csv()

    assert panel._progress_value_label.text() == PROGRESS_TEXT.format(held=0, loaded=6)


def test_the_resume_checkbox_turns_resuming_on_and_off(ready: ProcessingPanel) -> None:
    assert ready._phenome.params.process.resume_progress is False

    ready._resume_checkbox.click()
    assert ready._phenome.params.process.resume_progress is True

    ready._resume_checkbox.click()
    assert ready._phenome.params.process.resume_progress is False


# ----------------------------------------------------------------------------
# A step of the run that raises
# ----------------------------------------------------------------------------


def test_a_failed_model_load_extracts_nothing(ready: ProcessingPanel, pipeline: Pipeline, reported) -> None:
    """The next try may want another model, so the choice of one is given back with the rest."""
    pipeline.on_load = fail(OSError('connection refused'))

    ready._process_button.click()

    assert pipeline.steps == ['load']
    assert reported.shown == 1
    assert 'connection refused' in reported.last
    assert ready._process_button.isEnabled()
    assert ready._select_model_button.isEnabled()
    assert ready._phenome.needs_model_load


def test_a_failed_extraction_never_reaches_the_properties(
    ready: ProcessingPanel, pipeline: Pipeline, reported
) -> None:
    """Properties computed over embeddings that are not all there would answer for no run at all."""
    pipeline.on_process = fail(RuntimeError('Checkpoint has committed rows but none match'))

    ready._process_button.click()

    assert pipeline.steps == ['load', 'process']
    assert 'embeddings' in reported.last
    assert 'Checkpoint has committed rows but none match' in reported.last
    assert ready._progress_bar.format() == 'Failed'
    assert ready._process_button.isEnabled()


def test_a_failed_properties_step_is_reported_as_one(
    ready: ProcessingPanel, pipeline: Pipeline, reported
) -> None:
    """The embeddings are in the checkpoint by then, so what failed is the properties alone."""
    pipeline.on_properties = fail(ValueError('unknown property'))

    ready._process_button.click()

    assert reported.shown == 1
    assert 'properties' in reported.last
    assert 'unknown property' in reported.last
    assert ready._progress_bar.format() == 'Failed'
    assert ready._process_button.isEnabled()


# ----------------------------------------------------------------------------
# Stopping the run
# ----------------------------------------------------------------------------


def reporting(total: int, stop_after: int | None = None, panel: ProcessingPanel | None = None, done=None):
    """ A step reporting each of its units the way PhenoMe does, and pressing Stop partway through.

    Args:
        total: Units the step reports.
        stop_after: The unit after which Stop is pressed. None to let the step run out.
        panel: The panel whose Stop is pressed.
        done: Filled with each unit reported, as the step gets past it.
    Returns:
        Something to hand a Pipeline as one of its steps.
    """
    def _step(callback) -> None:
        for unit in range(1, total + 1):
            callback(unit, total, 'Processing batches')
            done.append(unit)
            if unit == stop_after:
                panel._stop_button.click()

    return _step


def test_stopping_ends_the_extraction_at_its_next_report(
    ready: ProcessingPanel, pipeline: Pipeline, reported
) -> None:
    """End to end: the stop is read from the callback PhenoMe reports through, and lands on the panel."""
    batches = []
    pipeline.on_process = reporting(10, stop_after=3, panel=ready, done=batches)

    ready._process_button.click()

    assert batches == [1, 2, 3]
    assert pipeline.steps == ['load', 'process']
    assert reported.shown == 0
    assert ready._progress_bar.format() == 'Stopped - progress autosaved'
    assert ready._process_button.isEnabled()


def test_a_stop_cannot_be_asked_for_twice(ready: ProcessingPanel, pipeline: Pipeline) -> None:
    """It is taken at the next report PhenoMe makes, and a second click would do nothing."""
    during = {}

    def _stop(callback) -> None:
        ready._stop_button.click()
        during.update(enabled=ready._stop_button.isEnabled(), text=ready._stop_button.text())

    pipeline.on_process = _stop

    ready._process_button.click()

    assert during == {'enabled': False, 'text': 'Stopping...'}
    assert ready._stop_button.text() == 'Stop'


def test_a_run_stopped_after_its_last_report_still_finishes(ready: ProcessingPanel, pipeline: Pipeline) -> None:
    """The stop is only read on a report; one that never comes leaves a result as good as any other."""
    pipeline.on_properties = lambda callback: ready._stop_button.click()

    ready._process_button.click()

    assert ready._progress_bar.format() == 'Done!'
    assert ready._process_button.isEnabled()


def test_a_new_run_is_not_stopped_by_the_one_before(
    ready: ProcessingPanel, pipeline: Pipeline, reported
) -> None:
    """Every run gets a stop of its own, or the one before it would end this one at its first report."""
    pipeline.on_process = reporting(10, stop_after=3, panel=ready, done=[])
    ready._process_button.click()

    batches = []
    pipeline.on_process = reporting(10, done=batches)
    ready._process_button.click()

    assert batches == list(range(1, 11))
    assert ready._progress_bar.format() == 'Done!'


# ----------------------------------------------------------------------------
# The elapsed time of a running task
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(('seconds', 'shown'), [(0, '0:00'), (65.9, '1:05'), (3599, '59:59'), (3723, '1:02:03')])
def test_a_duration_reads_as_a_stopwatch_shows_it(seconds: float, shown: str) -> None:
    assert format_elapsed(seconds) == shown


@pytest.fixture
def clock(panel: ProcessingPanel) -> list[float]:
    """ Have the panel read the elapsed time off a clock the test moves by hand.

    Returns:
        A one-item list holding the current time, which the test advances.
    """
    now = [1000.0]
    panel._clock = lambda: now[0]
    return now


def test_starting_a_task_resets_the_elapsed_time(ready: ProcessingPanel, clock: list[float]) -> None:
    """Whatever the task before it took is not this one's."""
    ready._set_processing_task_running(True)
    clock[0] += 125
    ready._set_processing_task_running(False)

    ready._set_processing_task_running(True)

    assert ready._timer_value_label.text() == format_elapsed(0)
    assert ready._elapsed_timer.isActive()


def test_the_elapsed_time_follows_a_running_task(ready: ProcessingPanel, clock: list[float]) -> None:
    ready._set_processing_task_running(True)

    clock[0] += 42
    ready._elapsed_timer.timeout.emit()

    assert ready._timer_value_label.text() == format_elapsed(42)


def test_a_finished_run_leaves_its_total(ready: ProcessingPanel, pipeline: Pipeline, clock: list[float]) -> None:
    """The total is read as the task ends, rather than left at the last whole tick."""
    def _take_a_while(callback) -> None:
        clock[0] += 61.5

    pipeline.on_process = _take_a_while

    ready._process_button.click()

    assert not ready._elapsed_timer.isActive()
    assert ready._timer_value_label.text() == format_elapsed(61.5)


@pytest.mark.parametrize('error', [TaskStopped(), RuntimeError('out of memory')])
def test_an_interrupted_run_leaves_its_total(
    ready: ProcessingPanel, pipeline: Pipeline, reported, clock: list[float], error: Exception
) -> None:
    def _interrupted(callback) -> None:
        clock[0] += 30
        raise error

    pipeline.on_process = _interrupted

    ready._process_button.click()

    assert not ready._elapsed_timer.isActive()
    assert ready._timer_value_label.text() == format_elapsed(30)


# ----------------------------------------------------------------------------
# The Compute Properties button, running the properties alone over the embeddings there
# ----------------------------------------------------------------------------


@pytest.fixture
def computing(ready: ProcessingPanel, pipeline: Pipeline, sync_worker) -> Pipeline:
    """ Have the Compute Properties button run its real worker inline, on a stand-in for PhenoMe.

    The Process button's worker is inlined too, through `pipeline`, so a test can use both.

    Returns:
        The stand-in, recording the steps the runs made.
    """
    sync_worker(
        ready,
        '_compute_properties_worker',
        lambda *args: ProcessingPanel._compute_properties_worker.__wrapped__(ready, *args),
    )
    return pipeline


def test_computing_properties_waits_on_the_embeddings(panel: ProcessingPanel, widget) -> None:
    """Data alone is not enough: the properties are computed over images already processed."""
    widget.loading_panel._on_load_data_csv()
    panel.refresh_buttons()

    assert not panel._compute_properties_button.isEnabled()


def test_computing_properties_opens_once_there_are_embeddings(ready: ProcessingPanel) -> None:
    assert ready._compute_properties_button.isEnabled()


def test_a_checkpoint_bringing_embeddings_opens_the_properties(
    loaded: ProcessingPanel, picking, processed, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _load(path) -> None:
        loaded._phenome.params.paths.checkpoint = Path(path)
        processed(loaded._phenome, 6)

    monkeypatch.setattr(loaded._phenome, 'load_checkpoint', _load)
    picking(tmp_path / 'previous.hdf5')
    assert not loaded._compute_properties_button.isEnabled()

    loaded._load_checkpoint_button.click()

    assert loaded._compute_properties_button.isEnabled()


def test_computing_properties_runs_only_the_properties(
    ready: ProcessingPanel, computing: Pipeline, confirmation
) -> None:
    """No model is loaded, no image extracted again, and nothing asked."""
    ready._compute_properties_button.click()

    assert computing.steps == ['properties']
    assert confirmation.asked == 0
    assert ready._phenome.has_properties


def test_computing_properties_ignores_the_after_embeddings_checkbox(
    ready: ProcessingPanel, computing: Pipeline
) -> None:
    ready._props_checkbox.click()

    ready._compute_properties_button.click()

    assert computing.steps == ['properties']


def test_computing_properties_keeps_the_analysis(clustered: ProcessingPanel, computing: Pipeline, qtbot) -> None:
    """The embeddings stay, and so do the results computed from them."""
    with qtbot.assertNotEmitted(clustered.analysis_cleared):
        clustered._compute_properties_button.click()

    assert clustered._phenome.clustering_df is not None
    assert 'cluster' in clustered._phenome.metadata_df.columns


def test_computing_properties_locks_the_tab_until_done(ready: ProcessingPanel, computing: Pipeline) -> None:
    during = {}

    def _note(callback) -> None:
        during.update(
            process=ready._process_button.isEnabled(),
            compute=ready._compute_properties_button.isEnabled(),
            stop=ready._stop_button.isEnabled(),
            outlines=outlines(ready),
        )

    computing.on_properties = _note

    ready._compute_properties_button.click()

    assert during == {
        'process': False,
        'compute': False,
        'stop': True,
        'outlines': {'embeddings': 'ok', 'properties': 'wait'},
    }
    assert ready._process_button.isEnabled()
    assert ready._compute_properties_button.isEnabled()
    assert outline(ready._properties_group) == 'ok'


def test_computing_properties_leaves_the_process_button_alone(
    ready: ProcessingPanel, computing: Pipeline
) -> None:
    """Only the full run names its steps on the Process button."""
    before = ready._process_button.text()
    during = []
    computing.on_properties = lambda callback: during.append(ready._process_button.text())

    ready._compute_properties_button.click()

    assert during == [before]


def test_a_failed_properties_computation_is_reported_and_gives_the_tab_back(
    ready: ProcessingPanel, computing: Pipeline, reported
) -> None:
    computing.on_properties = fail(ValueError('unknown property'))

    ready._compute_properties_button.click()

    assert reported.shown == 1
    assert 'unknown property' in reported.last
    assert outlines(ready) == {'embeddings': 'ok', 'properties': 'bad'}
    assert ready._compute_properties_button.isEnabled()
    assert ready._progress_bar.format() == 'Failed'


def test_a_stopped_properties_computation_is_not_reported_as_a_failure(
    ready: ProcessingPanel, computing: Pipeline, reported
) -> None:
    images = []
    computing.on_properties = reporting(10, stop_after=2, panel=ready, done=images)

    ready._compute_properties_button.click()

    assert images == [1, 2]
    assert reported.shown == 0
    assert ready._progress_bar.format() == 'Stopped'
    assert ready._compute_properties_button.isEnabled()


# ----------------------------------------------------------------------------
# Loading a checkpoint, which picks a dataset up where it was left
# ----------------------------------------------------------------------------


@pytest.fixture
def loaded(panel: ProcessingPanel, widget) -> ProcessingPanel:
    """ The panel with a dataset loaded and no checkpoint come by yet. """
    widget.loading_panel._on_load_data_csv()
    return panel


@pytest.fixture
def picking(monkeypatch: pytest.MonkeyPatch):
    """ Return a hook answering the load dialog with a path, without showing it.

    Returns:
        Called as picking(tmp_path / 'run.hdf5'), or with '' for the
        dialog being closed without a name.
    """
    def _answer(file_path) -> None:
        monkeypatch.setattr(
            QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (str(file_path), '')
        )

    return _answer


@pytest.fixture
def naming(monkeypatch: pytest.MonkeyPatch):
    """ Return a hook answering the save dialog with a name, without showing it.

    Returns:
        Called as naming(tmp_path / 'new.hdf5'), or with '' for the
        dialog being closed without a name.
    """
    def _answer(file_path) -> None:
        monkeypatch.setattr(
            QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (str(file_path), '')
        )

    return _answer


@pytest.fixture
def reading(panel: ProcessingPanel, monkeypatch: pytest.MonkeyPatch):
    """ Return a hook standing in for the interface reading a checkpoint off disk.

    Returns:
        Called as reading() or reading(raises=...), returning the list of paths
        the interface was asked to load.
    """
    def _stub(raises: Exception | None = None) -> list:
        asked = []

        def _load(path):
            asked.append(path)
            if raises is not None:
                raise raises
            panel._phenome.params.paths.checkpoint = Path(path)

        monkeypatch.setattr(panel._phenome, 'load_checkpoint', _load)
        return asked

    return _stub


def test_a_loaded_checkpoint_is_named_on_the_label(
    loaded: ProcessingPanel, picking, reading, tmp_path: Path
) -> None:
    checkpoint = tmp_path / 'previous.hdf5'
    reading()
    picking(checkpoint)

    loaded._load_checkpoint_button.click()

    assert loaded._phenome.params.paths.checkpoint == checkpoint
    assert loaded._current_checkpoint_label.text() == CHECKPOINT_TEXT.format(name='previous.hdf5')


def test_closing_the_load_dialog_reads_nothing(
    loaded: ProcessingPanel, picking, reading
) -> None:
    asked = reading()
    picking('')

    loaded._load_checkpoint_button.click()

    assert asked == []
    assert loaded._phenome.params.paths.checkpoint is None


def test_a_checkpoint_is_loaded_without_being_asked_about(
    loaded: ProcessingPanel, picking, reading, mismatched, tmp_path: Path
) -> None:
    """Whatever it was filled under, it loads: the settings are the run's to compare, not the load's."""
    checkpoint = tmp_path / 'other.hdf5'
    mismatched(CheckpointMismatch(DATASET_MISMATCH, 'no image in common'))
    asked = reading()
    picking(checkpoint)

    loaded._load_checkpoint_button.click()

    assert asked == [checkpoint]
    assert mismatched.asked == 0
    assert loaded._phenome.params.paths.checkpoint == checkpoint


def test_a_checkpoint_that_will_not_open_is_reported(
    loaded: ProcessingPanel, picking, reading, reported, tmp_path: Path
) -> None:
    """A file that is not a checkpoint, or one another process holds: neither may reach a slot as a traceback."""
    reading(raises=OSError('file is already open'))
    picking(tmp_path / 'broken.hdf5')

    loaded._load_checkpoint_button.click()

    assert reported.shown == 1
    assert 'file is already open' in reported.last
    assert loaded._phenome.params.paths.checkpoint is None


def test_a_loaded_checkpoint_is_announced(
    loaded: ProcessingPanel, picking, reading, qtbot, tmp_path: Path
) -> None:
    """It can carry a previous run's embeddings, which nothing else would tell the analysis steps about."""
    reading()
    picking(tmp_path / 'previous.hdf5')

    with qtbot.waitSignal(loaded.checkpoint_loaded, timeout=100):
        loaded._load_checkpoint_button.click()


def test_a_checkpoint_that_will_not_open_is_not_announced(
    loaded: ProcessingPanel, picking, reading, reported, qtbot, tmp_path: Path
) -> None:
    """Nothing was loaded, so the steps running on a checkpoint's embeddings have nothing to hear."""
    reading(raises=OSError('file is already open'))
    picking(tmp_path / 'broken.hdf5')

    with qtbot.assertNotEmitted(loaded.checkpoint_loaded):
        loaded._load_checkpoint_button.click()


def test_a_checkpoint_of_another_dataset_is_reported_and_not_loaded(
    loaded: ProcessingPanel, picking, reported, written_checkpoint, copied_dataset, qtbot
) -> None:
    """End to end, on real files: PhenoMe would load it as holding none of the images, without a word."""
    path = written_checkpoint(loaded._phenome)
    other = copied_dataset(renamed=True)
    loaded._phenome.params.paths.images = other['images']
    loaded._phenome.params.paths.masks = other['masks']
    loaded._phenome.params.paths.metadata = other['metadata']
    loaded._phenome.load_data_csv()
    loaded.refresh_buttons()
    picking(path)

    with qtbot.assertNotEmitted(loaded.checkpoint_loaded):
        loaded._load_checkpoint_button.click()

    assert reported.shown == 1
    assert 'different dataset' in reported.last
    assert loaded._phenome.params.paths.checkpoint is None
    assert loaded._current_checkpoint_label.text() == NO_CHECKPOINT_TEXT
    assert not loaded._phenome.has_processed_images
    assert not loaded._save_checkpoint_button.isEnabled()


def test_loading_a_checkpoint_waits_on_the_data(panel: ProcessingPanel) -> None:
    """It resolves the paths it stores against the images currently found, so there have to be some."""
    assert not panel._load_checkpoint_button.isEnabled()
    assert panel._load_checkpoint_button.toolTip() == 'Please load data first.'


# ----------------------------------------------------------------------------
# What the Save button waits on: something extracted to save, and no task running
# ----------------------------------------------------------------------------


def test_loading_data_opens_the_load_but_not_the_save(loaded: ProcessingPanel) -> None:
    """There is a dataset to resolve a checkpoint against, but nothing extracted yet to save."""
    assert loaded._load_checkpoint_button.isEnabled()
    assert not loaded._save_checkpoint_button.isEnabled()
    assert loaded._save_checkpoint_button.toolTip() == 'Please extract embeddings first.'


def test_a_finished_run_opens_the_save(ready: ProcessingPanel, pipeline: Pipeline, processed) -> None:
    processed(ready._phenome, 0)
    ready.refresh_buttons()
    assert not ready._save_checkpoint_button.isEnabled()
    pipeline.on_process = lambda callback: processed(ready._phenome, 6)

    ready._process_button.click()

    assert ready._save_checkpoint_button.isEnabled()
    assert ready._save_checkpoint_button.toolTip() == ''


def test_a_loaded_checkpoint_opens_the_save(loaded: ProcessingPanel, written_checkpoint) -> None:
    """Its embeddings are results like any other, and can be saved to another file."""
    loaded._load_checkpoint(written_checkpoint(loaded._phenome))

    assert loaded._save_checkpoint_button.isEnabled()


def test_a_running_task_holds_the_save(ready: ProcessingPanel) -> None:
    """The run would be replacing the very results being written out."""
    ready._set_processing_task_running(True)

    assert not ready._save_checkpoint_button.isEnabled()
    assert ready._save_checkpoint_button.toolTip() == 'Please wait for the running task to finish.'


# ----------------------------------------------------------------------------
# Saving a checkpoint
# ----------------------------------------------------------------------------


class Saved:
    """ Stands in for PhenoMe writing a checkpoint, recording where it was asked to write.

    Stubbed where the GUI hands over to PhenoMe, so the panel, its worker and the interface run as
    they do for real. The write itself is covered against real files in the interface tests.
    """

    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.on_save = None

    def save_results(self, checkpoint_path: str) -> None:
        """ Take note of a checkpoint written, doing whatever the test asked meanwhile.

        Args:
            checkpoint_path: Where the checkpoint would be written.
        """
        self.paths.append(Path(checkpoint_path))
        if self.on_save is not None:
            self.on_save()


@pytest.fixture
def saving(ready: ProcessingPanel, sync_worker, monkeypatch: pytest.MonkeyPatch) -> Saved:
    """ Have the Save button run its real worker inline, on a stand-in for PhenoMe's write.

    Returns:
        The stand-in, recording where each save was written.
    """
    stand_in = Saved()
    monkeypatch.setattr(ready._phenome.image_processing, 'save_results', stand_in.save_results)
    sync_worker(
        ready,
        '_save_checkpoint_worker',
        lambda path: ProcessingPanel._save_checkpoint_worker.__wrapped__(ready, path),
    )
    return stand_in


def test_saving_writes_to_the_picked_file(ready: ProcessingPanel, saving: Saved, naming, tmp_path: Path) -> None:
    naming(tmp_path / 'saved.hdf5')

    ready._save_checkpoint_button.click()

    assert saving.paths == [tmp_path / 'saved.hdf5']


def test_the_saved_file_becomes_the_selected_checkpoint(
    ready: ProcessingPanel, saving: Saved, naming, tmp_path: Path
) -> None:
    """The label names the file alone, and the whole path is a hover away: runs kept apart by their
    directory would otherwise read alike."""
    path = tmp_path / 'saved.hdf5'
    naming(path)

    ready._save_checkpoint_button.click()

    assert ready._phenome.params.paths.checkpoint == path
    assert ready._current_checkpoint_label.text() == CHECKPOINT_TEXT.format(name='saved.hdf5')
    assert ready._current_checkpoint_label.toolTip() == str(path)


def test_a_name_given_without_an_extension_is_spelled_out(
    ready: ProcessingPanel, saving: Saved, naming, tmp_path: Path
) -> None:
    """Not every dialog applies the filter, and PhenoMe refuses to read back a file without one."""
    naming(tmp_path / 'bare')

    ready._save_checkpoint_button.click()

    assert saving.paths == [tmp_path / 'bare.hdf5']


def test_closing_the_save_dialog_saves_nothing(ready: ProcessingPanel, saving: Saved, naming) -> None:
    """Reaching for a file and thinking better of it must not cost the checkpoint already selected."""
    before = ready._phenome.params.paths.checkpoint
    naming('')

    ready._save_checkpoint_button.click()

    assert saving.paths == []
    assert ready._phenome.params.paths.checkpoint == before


@pytest.mark.parametrize('accepted', [False, True], ids=['refused', 'confirmed'])
def test_a_completed_name_landing_on_a_file_is_asked_about(
    ready: ProcessingPanel, saving: Saved, naming, confirmation, tmp_path: Path, accepted: bool
) -> None:
    """The dialog asked about 'already', where the extension put the save on 'already.hdf5' - another file."""
    existing = tmp_path / 'already.hdf5'
    existing.write_bytes(b'')
    confirmation.accepted = accepted
    naming(tmp_path / 'already')

    ready._save_checkpoint_button.click()

    assert confirmation.asked == 1
    assert saving.paths == ([existing] if accepted else [])


def test_a_file_the_dialog_asked_about_is_not_asked_about_again(
    ready: ProcessingPanel, saving: Saved, naming, confirmation, tmp_path: Path
) -> None:
    """The dialog asked whether to replace it and was told yes, which is the whole question."""
    existing = tmp_path / 'already.hdf5'
    existing.write_bytes(b'')
    naming(existing)

    ready._save_checkpoint_button.click()

    assert confirmation.asked == 0
    assert saving.paths == [existing]


def test_the_tab_is_held_while_saving(
    ready: ProcessingPanel, saving: Saved, naming, widget, tmp_path: Path
) -> None:
    """A run started meanwhile would replace the results being written, and a save cannot be stopped halfway."""
    during = {}
    saving.on_save = lambda: during.update(
        process=ready._process_button.isEnabled(),
        save=ready._save_checkpoint_button.isEnabled(),
        model=ready._select_model_button.isEnabled(),
        stop=ready._stop_button.isEnabled(),
        analysis=widget.analysis_panel._run_clustering_button.isEnabled(),
    )
    naming(tmp_path / 'saved.hdf5')

    ready._save_checkpoint_button.click()

    assert during == {'process': False, 'save': False, 'model': False, 'stop': False, 'analysis': False}
    assert ready._process_button.isEnabled()
    assert ready._save_checkpoint_button.isEnabled()
    assert widget.analysis_panel._run_clustering_button.isEnabled()


def test_a_failed_save_is_reported_and_keeps_the_previous_checkpoint(
    ready: ProcessingPanel, saving: Saved, naming, reported, tmp_path: Path
) -> None:
    previous = ready._phenome.params.paths.checkpoint
    saving.on_save = fail(OSError('disk full'))
    naming(tmp_path / 'saved.hdf5')

    ready._save_checkpoint_button.click()

    assert reported.shown == 1
    assert 'disk full' in reported.last
    assert ready._phenome.params.paths.checkpoint == previous
    assert ready._current_checkpoint_label.text() == CHECKPOINT_TEXT.format(name=previous.name)
    assert ready._progress_bar.format() == 'Failed'
    assert ready._process_button.isEnabled()
    assert ready._save_checkpoint_button.isEnabled()


def test_a_save_leaves_the_next_run_writing_nowhere(
    ready: ProcessingPanel, saving: Saved, naming, pipeline: Pipeline, tmp_path: Path
) -> None:
    """Saving is a step of its own: the runs after it still keep their results in memory."""
    naming(tmp_path / 'saved.hdf5')
    ready._save_checkpoint_button.click()

    ready._process_button.click()

    assert 'checkpoint_path' not in pipeline.extractions[0]
