"""Tests for the image processing panel.

Nothing here loads a model or extracts embeddings: the pipeline state is stubbed instead, the model
being a sentinel object, and the load and the extraction stand-ins where a run is driven.
"""

from pathlib import Path

import pandas as pd
import pytest
from qtpy.QtWidgets import QFileDialog, QMessageBox

from phenome_gui.registries.models import MODEL_OPTIONS
from phenome_gui.ui.model_window import ModelWindow
from phenome_gui.ui.processing_panel import (
    CHECKPOINT_TEXT,
    NO_CHECKPOINT_TEXT,
    ProcessingPanel,
)
from phenome_gui.utils.checkpoint import (
    DATASET_MISMATCH,
    MODEL_MISMATCH,
    CheckpointMismatch,
)
from phenome_gui.utils.params import (
    CHANNEL_MODE_CHOICES,
    MAX_WORKERS,
)
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
    """ Stands in for both halves of the checkpoint question: what is found, and what is answered. """

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

    def inspect(self) -> object:
        """ Stand in for reading the checkpoint, which no test lets reach a real file.

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
    """ Decide what reading the checkpoint finds, and how the question about it is answered.

    Reading is stood in for rather than pointed at a written file: the default checkpoint path is a
    relative one, and a test left to resolve it would read whatever sits in the working directory.

    Returns:
        The stand-in, called as mismatched(what_is_found, accepted=...).
    """
    stand_in = Mismatched()
    stand_in(None)

    monkeypatch.setattr(panel._phenome, 'inspect_checkpoint', stand_in.inspect)
    monkeypatch.setattr(QMessageBox, 'warning', stand_in.ask)
    return stand_in


# ----------------------------------------------------------------------------
# The extract embeddings button, which waits on the data, a checkpoint and a model
# ----------------------------------------------------------------------------


def test_extract_needs_the_data_and_a_checkpoint(panel: ProcessingPanel) -> None:
    """The model is not among them: the default counts as selected, and the run loads it itself."""
    assert not panel._process_button.isEnabled()
    assert panel._process_button.toolTip() == (
        'Please load data and create or load a checkpoint first.'
    )


def test_extract_still_waits_on_data_after_the_checkpoint(panel: ProcessingPanel, tmp_path: Path) -> None:
    panel._phenome.params.paths.checkpoint = tmp_path / 'run.hdf5'
    panel.refresh_buttons()

    assert not panel._process_button.isEnabled()
    assert panel._process_button.toolTip() == 'Please load data first.'


def test_extract_still_waits_on_a_checkpoint(panel: ProcessingPanel, widget) -> None:
    """The embeddings are written to it, so a run without one would have nowhere to put them."""
    widget.loading_panel._on_load_data_csv()
    panel.refresh_buttons()

    assert not panel._process_button.isEnabled()
    assert panel._process_button.toolTip() == 'Please create or load a checkpoint first.'


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


def test_extract_opens_once_all_three_are_there(ready: ProcessingPanel) -> None:
    assert ready._process_button.isEnabled()
    assert ready._process_button.toolTip() == ''


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


def test_the_checkpoint_path_writes_the_parameter(
    panel: ProcessingPanel, naming, tmp_path: Path
) -> None:
    naming(tmp_path / 'run.hdf5')

    panel._on_new_checkpoint_pressed()

    assert panel._phenome.params.paths.checkpoint == tmp_path / 'run.hdf5'


# ----------------------------------------------------------------------------
# The advanced parameters form
# ----------------------------------------------------------------------------


def drive_advanced_form(
    panel: ProcessingPanel,
    batch_size: int,
    accepted: bool,
    num_workers: int | None = None,
    channel_mode: str | None = None,
) -> None:
    """ Open the advanced form with its exec_ replaced, so it fills the fields and settles at once.

    The panel builds the form on first use and keeps it, so it is built here and its instance patched,
    leaving every other dialog alone.

    Args:
        panel: The panel whose form is driven.
        batch_size: The batch size to type into the form.
        accepted: Whether the form is confirmed or cancelled.
        num_workers: The worker count to type in, or None to leave the field as loaded.
        channel_mode: The channel mode to pick, or None to leave the field as loaded.
    """
    panel._advanced_parameters_dialog = panel._build_advanced_parameters_dialog()

    def _exec() -> int:
        panel._batch_size_spinbox.setValue(batch_size)
        if num_workers is not None:
            panel._num_workers_spinbox.setValue(num_workers)
        if channel_mode is not None:
            panel._channel_mode_combo.setCurrentText(channel_mode)
        return int(accepted)

    panel._advanced_parameters_dialog.exec_ = _exec
    panel._on_advanced_parameters_clicked()


def test_the_advanced_form_applies_the_batch_size(panel: ProcessingPanel) -> None:
    drive_advanced_form(panel, 128, accepted=True)

    assert panel._phenome.params.process.batch_size == 128


def test_the_advanced_form_discards_a_cancelled_edit(panel: ProcessingPanel) -> None:
    before = panel._phenome.params.process.batch_size

    drive_advanced_form(panel, 256, accepted=False)

    assert panel._phenome.params.process.batch_size == before


def test_the_advanced_form_applies_the_worker_count(panel: ProcessingPanel) -> None:
    drive_advanced_form(panel, 32, accepted=True, num_workers=4)

    assert panel._phenome.params.process.num_workers == 4


def test_the_advanced_form_discards_a_cancelled_worker_count(panel: ProcessingPanel) -> None:
    before = panel._phenome.params.process.num_workers

    drive_advanced_form(panel, 32, accepted=False, num_workers=4)

    assert panel._phenome.params.process.num_workers == before


def test_the_worker_field_offers_no_more_than_the_model_takes(panel: ProcessingPanel) -> None:
    """A count the form can offer and the model rejects would raise on a confirmed edit."""
    panel._advanced_parameters_dialog = panel._build_advanced_parameters_dialog()

    assert panel._num_workers_spinbox.minimum() == 0
    assert panel._num_workers_spinbox.maximum() == MAX_WORKERS


def test_the_form_shows_the_worker_count_it_would_run_under(panel: ProcessingPanel) -> None:
    """The fields are reloaded on every open, so a cancelled edit is never shown again."""
    panel._phenome.params.process.num_workers = 6

    drive_advanced_form(panel, 32, accepted=False)

    assert panel._num_workers_spinbox.value() == 6


# ----------------------------------------------------------------------------
# The channel mode, which decides what is embedded rather than how quickly
# ----------------------------------------------------------------------------


def test_the_channel_mode_offers_the_values_the_model_accepts(panel: ProcessingPanel) -> None:
    """Offering anything else would raise on assignment, the field being typed as a literal."""
    panel._advanced_parameters_dialog = panel._build_advanced_parameters_dialog()

    combo = panel._channel_mode_combo
    assert [combo.itemText(row) for row in range(combo.count())] == CHANNEL_MODE_CHOICES


def test_the_advanced_form_applies_the_channel_mode(panel: ProcessingPanel) -> None:
    drive_advanced_form(panel, 32, accepted=True, channel_mode='combined')

    assert panel._phenome.params.process.channel_mode == 'combined'


def test_the_advanced_form_discards_a_cancelled_channel_mode(panel: ProcessingPanel) -> None:
    before = panel._phenome.params.process.channel_mode

    drive_advanced_form(panel, 32, accepted=False, channel_mode='combined')

    assert panel._phenome.params.process.channel_mode == before


def test_the_form_shows_the_channel_mode_it_would_run_under(panel: ProcessingPanel) -> None:
    """The fields are reloaded on every open, so the combo opens on the stored mode rather than the first."""
    panel._phenome.params.process.channel_mode = 'combined'

    drive_advanced_form(panel, 32, accepted=False)

    assert panel._channel_mode_combo.currentText() == 'combined'


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
# A preset changed after the properties were computed
# ----------------------------------------------------------------------------


def test_loading_another_dataset_forgets_the_changed_preset(ready: ProcessingPanel) -> None:
    """The checkpoint went with the results, so there are no properties left to compute over."""
    ready._props_selection_combo.setCurrentText('Complete')

    ready._phenome.clear_results()
    ready.clear_results()

    assert not ready._preset_changed


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
    """It was asked for, and the checkpoint holds what it had committed."""
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
    """ Stands in for the three PhenoMe calls a run makes, recording them in the order they come.

    Stubbed where the GUI hands over to PhenoMe, so everything in front of it - the panel, the
    interface, the progress throttle and the stop - runs as it does for real. Each step can be given
    something to do while it runs, for a test watching the panel mid-run or making the step fail.
    """

    def __init__(self) -> None:
        self.steps: list[str] = []
        self.force_reprocess: list[bool] = []
        self.force_update: list[bool] = []
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

    def process_images(
        self, model, checkpoint_path, progress_callback=None, force_reprocess: bool = False, **kwargs
    ) -> None:
        """ Stand in for the extraction, handing its progress callback to the test.

        Args:
            model: The loaded network.
            checkpoint_path: The checkpoint the embeddings would be written to.
            progress_callback: What the extraction reports its batches through.
            force_reprocess: Whether the checkpoint would be deleted and filled again.
            **kwargs: The run's other arguments, of no interest here.
        """
        self.steps.append('process')
        self.force_reprocess.append(force_reprocess)
        if self.on_process is not None:
            self.on_process(progress_callback)

    def compute_properties(
        self, property_preset: str, checkpoint_path=None, progress_callback=None, force_update: bool = False
    ) -> pd.DataFrame:
        """ Stand in for the properties step, handing its progress callback to the test.

        Args:
            property_preset: The preset, as PhenoMe names it.
            checkpoint_path: The checkpoint the properties would be written to.
            progress_callback: What the step reports its images through.
            force_update: Whether the properties the checkpoint holds would be computed again.
        Returns:
            A frame of properties, one image's worth.
        """
        self.steps.append('properties')
        self.force_update.append(force_update)
        if self.on_properties is not None:
            self.on_properties(progress_callback)
        return pd.DataFrame({'area': [1.0]})


@pytest.fixture
def pipeline(
    ready: ProcessingPanel, mismatched, sync_worker, monkeypatch: pytest.MonkeyPatch
) -> Pipeline:
    """ Have the Process button run its real worker inline, on a stand-in for PhenoMe.

    The checkpoint reads as matching unless a test says otherwise through `mismatched`.

    Returns:
        The stand-in, recording the steps the run made.
    """
    stand_in = Pipeline()
    monkeypatch.setattr(ready._phenome.image_processing, 'load_model', stand_in.load_model)
    monkeypatch.setattr(ready._phenome.image_processing, 'process_images', stand_in.process_images)
    monkeypatch.setattr(ready._phenome.pheno, 'compute_properties', stand_in.compute_properties)
    sync_worker(
        ready,
        '_process_images_worker',
        lambda *args: ProcessingPanel._process_images_worker.__wrapped__(ready, *args),
    )
    return stand_in


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
            stop=ready._stop_button.isEnabled(),
            analysis=widget.analysis_panel._run_clustering_button.isEnabled(),
        )

    pipeline.on_process = _look

    ready._process_button.click()

    assert during == {
        'process': False, 'model': False, 'preset': False, 'compute': False, 'stop': True, 'analysis': False,
    }


def test_a_finished_run_gives_the_tab_back(ready: ProcessingPanel, pipeline: Pipeline, widget) -> None:
    ready._process_button.click()

    assert ready._process_button.isEnabled()
    assert ready._props_checkbox.isEnabled()
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
# The checkpoint, read before the run is let go of
# ----------------------------------------------------------------------------


def test_a_matching_checkpoint_is_resumed_without_a_question(
    ready: ProcessingPanel, pipeline: Pipeline, mismatched
) -> None:
    """The usual run, which has to stay a single click."""
    ready._process_button.click()

    assert mismatched.asked == 0
    assert pipeline.force_reprocess == [False]


def test_a_confirmed_overwrite_runs_from_scratch(ready: ProcessingPanel, pipeline: Pipeline, mismatched) -> None:
    mismatched(CheckpointMismatch(MODEL_MISMATCH), accepted=True)

    ready._process_button.click()

    assert mismatched.asked == 1
    assert pipeline.force_reprocess == [True]


def test_a_refused_overwrite_runs_nothing_at_all(ready: ProcessingPanel, pipeline: Pipeline, mismatched) -> None:
    """The checkpoint is deleted by the run, so refusing has to stop it before even the model is loaded."""
    mismatched(CheckpointMismatch(DATASET_MISMATCH, 'no image in common'), accepted=False)

    ready._process_button.click()

    assert pipeline.steps == []
    assert ready._process_button.isEnabled()
    assert not ready._stop_button.isEnabled()


def test_the_question_says_what_was_found(ready: ProcessingPanel, pipeline: Pipeline, mismatched) -> None:
    """It is answered by deleting a checkpoint, so it has to say what is wrong with this one."""
    mismatched(CheckpointMismatch(DATASET_MISMATCH, 'It holds 3 image(s), none of them loaded.'))

    ready._process_button.click()

    assert 'It holds 3 image(s), none of them loaded.' in mismatched.last


def test_a_question_with_nothing_to_add_leaves_no_gap(
    ready: ProcessingPanel, pipeline: Pipeline, mismatched
) -> None:
    """The parameter mismatch adds no detail, which must not leave an empty paragraph behind it."""
    mismatched(CheckpointMismatch(MODEL_MISMATCH))

    ready._process_button.click()

    assert mismatched.asked == 1
    assert '\n\n\n' not in mismatched.last


# ----------------------------------------------------------------------------
# The properties preset, which a resumed checkpoint would otherwise ignore
# ----------------------------------------------------------------------------


def test_an_unchanged_preset_resumes_the_properties(ready: ProcessingPanel, pipeline: Pipeline) -> None:
    ready._process_button.click()

    assert pipeline.force_update == [False]


def test_a_changed_preset_is_computed_again_once(ready: ProcessingPanel, pipeline: Pipeline) -> None:
    """Once computed it answers for what the checkpoint holds, so the run after may resume again."""
    ready._props_selection_combo.setCurrentText('Complete')

    ready._process_button.click()
    ready._process_button.click()

    assert pipeline.force_update == [True, False]


def test_turning_the_properties_back_on_computes_them_again(ready: ProcessingPanel, pipeline: Pipeline) -> None:
    ready._props_checkbox.click()
    ready._props_checkbox.click()

    ready._process_button.click()

    assert pipeline.force_update == [True]


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


def test_a_failed_properties_step_keeps_the_changed_preset_owed(
    ready: ProcessingPanel, pipeline: Pipeline, reported
) -> None:
    """Nothing was computed, so the preset still has to be forced onto the next run."""
    ready._props_selection_combo.setCurrentText('Complete')
    pipeline.on_properties = fail(ValueError('unknown property'))
    ready._process_button.click()

    pipeline.on_properties = None
    ready._process_button.click()

    assert pipeline.force_update == [True, True]


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
    assert ready._progress_bar.format() == 'Stopped'
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


def test_stopping_the_properties_lets_the_next_run_resume_them(
    ready: ProcessingPanel, pipeline: Pipeline, reported
) -> None:
    """PhenoMe clears the old properties before its first report, so the preset has been applied:
    forcing it again would wipe the partial run the next click is meant to carry on from."""
    ready._props_selection_combo.setCurrentText('Complete')
    pipeline.on_properties = reporting(10, stop_after=1, panel=ready, done=[])
    ready._process_button.click()

    pipeline.on_properties = None
    ready._process_button.click()

    assert reported.shown == 0
    assert pipeline.force_update == [True, False]


# ----------------------------------------------------------------------------
# Coming by a checkpoint, which is what everything else waits on
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


def test_both_ways_to_a_checkpoint_wait_on_the_data(panel: ProcessingPanel) -> None:
    """Both read the images currently found, so neither has anything to work against before a load."""
    for button in (panel._new_checkpoint_button, panel._load_checkpoint_button):
        assert not button.isEnabled()
        assert button.toolTip() == 'Please load data first.'


def test_loading_data_opens_both_ways_to_a_checkpoint(loaded: ProcessingPanel) -> None:
    for button in (loaded._new_checkpoint_button, loaded._load_checkpoint_button):
        assert button.isEnabled()
        assert button.toolTip() == ''


def test_a_free_path_is_taken_up_as_it_stands(
    loaded: ProcessingPanel, naming, tmp_path: Path
) -> None:
    """The ordinary way one is created: PhenoMe writes the file itself when the first run reaches it."""
    checkpoint = tmp_path / 'new.hdf5'
    naming(checkpoint)

    loaded._new_checkpoint_button.click()

    assert loaded._phenome.params.paths.checkpoint == checkpoint
    assert loaded._current_checkpoint_label.text() == CHECKPOINT_TEXT.format(name='new.hdf5')


def test_the_whole_path_is_a_hover_away(
    loaded: ProcessingPanel, naming, tmp_path: Path
) -> None:
    """The label names the file alone, where two runs kept apart by their directory would read alike."""
    checkpoint = tmp_path / 'new.hdf5'
    naming(checkpoint)

    loaded._new_checkpoint_button.click()

    assert loaded._current_checkpoint_label.toolTip() == str(checkpoint)


def test_a_name_given_without_an_extension_is_spelled_out(
    loaded: ProcessingPanel, naming, tmp_path: Path
) -> None:
    """Not every dialog applies the filter, and PhenoMe refuses to read back a file without one."""
    naming(tmp_path / 'bare')

    loaded._new_checkpoint_button.click()

    assert loaded._phenome.params.paths.checkpoint == tmp_path / 'bare.hdf5'


def test_closing_the_save_dialog_selects_nothing(loaded: ProcessingPanel, naming) -> None:
    naming('')

    loaded._new_checkpoint_button.click()

    assert loaded._phenome.params.paths.checkpoint is None
    assert loaded._current_checkpoint_label.text() == NO_CHECKPOINT_TEXT


def test_a_created_checkpoint_opens_the_extraction(
    loaded: ProcessingPanel, naming, tmp_path: Path
) -> None:
    loaded._phenome.model = object()
    loaded.refresh_buttons()
    assert not loaded._process_button.isEnabled()

    naming(tmp_path / 'new.hdf5')
    loaded._new_checkpoint_button.click()

    assert loaded._process_button.isEnabled()


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


# ----------------------------------------------------------------------------
# Naming a path that already holds a checkpoint
# ----------------------------------------------------------------------------


def existing_checkpoint(tmp_path: Path, name: str = 'already.hdf5') -> Path:
    """ A file standing where a checkpoint would be, which is all the panel reads of it. """
    path = tmp_path / name
    path.write_bytes(b'')
    return path


def test_naming_an_existing_file_replaces_it(
    loaded: ProcessingPanel, naming, tmp_path: Path
) -> None:
    """The save dialog asked whether to replace it and was told yes, which is the whole question."""
    checkpoint = existing_checkpoint(tmp_path)
    naming(checkpoint)

    loaded._new_checkpoint_button.click()

    assert not checkpoint.exists()
    assert loaded._phenome.params.paths.checkpoint == checkpoint


def test_naming_a_checkpoint_asks_nothing_of_its_own(
    loaded: ProcessingPanel, naming, mismatched, tmp_path: Path
) -> None:
    """The dialog's own question is the only one: reading an existing one is the Load button's job."""
    naming(existing_checkpoint(tmp_path))

    loaded._new_checkpoint_button.click()

    assert mismatched.asked == 0


def test_a_completed_name_leaves_the_file_it_lands_on_alone(
    loaded: ProcessingPanel, naming, tmp_path: Path
) -> None:
    """The dialog asked about 'already', where the extension put us on 'already.hdf5' - another file."""
    checkpoint = existing_checkpoint(tmp_path)
    naming(tmp_path / 'already')

    loaded._new_checkpoint_button.click()

    assert checkpoint.exists()
    assert loaded._phenome.params.paths.checkpoint == checkpoint


def test_cancelling_leaves_the_file_and_the_selection_alone(
    loaded: ProcessingPanel, naming, tmp_path: Path
) -> None:
    """Closing the dialog is refusing the replacement along with everything else."""
    checkpoint = existing_checkpoint(tmp_path)
    naming('')

    loaded._new_checkpoint_button.click()

    assert checkpoint.exists()
    assert loaded._phenome.params.paths.checkpoint is None


def test_cancelling_leaves_the_checkpoint_in_hand_selected(
    loaded: ProcessingPanel, naming, tmp_path: Path
) -> None:
    """Reaching for another file and thinking better of it must not cost the one being written to."""
    selected = tmp_path / 'chosen.hdf5'
    naming(selected)
    loaded._new_checkpoint_button.click()

    naming('')
    loaded._new_checkpoint_button.click()

    assert loaded._phenome.params.paths.checkpoint == selected
    assert loaded._current_checkpoint_label.text() == CHECKPOINT_TEXT.format(name='chosen.hdf5')
