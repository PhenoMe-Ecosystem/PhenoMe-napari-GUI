"""Tests for the image processing panel.

Nothing here loads a model or extracts embeddings: the pipeline state is stubbed instead, and the model
is a sentinel object, since all the panel ever asks is whether one is present.
"""

from pathlib import Path

import pytest
from qtpy.QtWidgets import QMessageBox

from phenome_gui.ui.processing_panel import ProcessingPanel
from phenome_gui.utils.checkpoint import (
    DATASET_MISMATCH,
    MODEL_MISMATCH,
    CheckpointMismatch,
)
from phenome_gui.utils.params import DEFAULT_CHECKPOINT


@pytest.fixture
def panel(widget) -> ProcessingPanel:
    """ The processing panel of a built widget. """
    return widget.processing_panel


@pytest.fixture
def ready(panel: ProcessingPanel, widget, processed) -> ProcessingPanel:
    """ The panel with data loaded, a model in place and images processed. """
    widget.loading_panel._on_load_data_csv()
    panel._phenome.model = object()
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
        Reported: The stand-in, holding what was said.
    """
    stand_in = Reported()
    monkeypatch.setattr(QMessageBox, 'critical', stand_in)
    return stand_in


class Started:
    """ Stands in for a worker, recording what it was asked to run rather than running it.

    Both processing tasks are started the same way, so one of these answers for either: what is
    recorded is the flag each was given, which is the whole of what the tests are checking.
    """

    def __init__(self) -> None:
        self.runs = []

    def __call__(self, force: bool = False) -> "Started":
        """ Stand in for the worker factory, taking the flag the panel passes it.

        Args:
            force (bool): force_reprocess for the extraction, force_update for the properties.
        Returns:
            Started: Itself, standing in for the worker it would have built.
        """
        self._pending = force
        return self

    def connect(self, handler) -> None:
        """ Take a handler the panel connects to one of the worker's signals, and drop it.

        Args:
            handler: What the panel would be called back on, which nothing here ever calls.
        """

    @property
    def returned(self) -> "Started":
        """ The signal the panel connects its completion handler to. """
        return self

    @property
    def errored(self) -> "Started":
        """ The signal the panel connects its failure handler to. """
        return self

    def start(self) -> None:
        """ Take note that a run was started, and under which flag. """
        self.runs.append(self._pending)


@pytest.fixture
def started(panel: ProcessingPanel, monkeypatch: pytest.MonkeyPatch) -> Started:
    """ Stand in for both task workers, so a click records a run instead of making one.

    Returns:
        Started: The stand-in, holding the flag each started run was given.
    """
    stand_in = Started()
    monkeypatch.setattr(panel, '_process_images_worker', stand_in)
    monkeypatch.setattr(panel, '_compute_properties_worker', stand_in)
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
            mismatch (CheckpointMismatch | None): What reading the checkpoint is to report.
            accepted (bool): Whether the question about it is to be answered yes.
        """
        self._mismatch = mismatch
        self.accepted = accepted

    def inspect(self) -> object:
        """ Stand in for reading the checkpoint, which no test lets reach a real file.

        Returns:
            CheckpointMismatch | None: Whatever the test asked to be found.
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
            QMessageBox.StandardButton: Yes or No, as the test set it.
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
        Mismatched: The stand-in, called as mismatched(what_is_found, accepted=...).
    """
    stand_in = Mismatched()
    stand_in(None)

    monkeypatch.setattr(panel._phenome, 'inspect_checkpoint', stand_in.inspect)
    monkeypatch.setattr(QMessageBox, 'warning', stand_in.ask)
    return stand_in


# ----------------------------------------------------------------------------
# The extract embeddings button, which waits on both data and a model
# ----------------------------------------------------------------------------


def test_extract_needs_both_data_and_a_model(panel: ProcessingPanel) -> None:
    assert not panel._process_button.isEnabled()
    assert panel._process_button.toolTip() == 'Please load data and load a model first.'


def test_extract_still_waits_on_a_model_after_loading(panel: ProcessingPanel, widget) -> None:
    widget.loading_panel._on_load_data_csv()

    assert not panel._process_button.isEnabled()
    assert panel._process_button.toolTip() == 'Please load a model first.'


def test_extract_still_waits_on_data_after_the_model(panel: ProcessingPanel) -> None:
    panel._phenome.model = object()
    panel.refresh_buttons()

    assert not panel._process_button.isEnabled()
    assert panel._process_button.toolTip() == 'Please load data first.'


def test_extract_opens_once_both_are_there(ready: ProcessingPanel) -> None:
    assert ready._process_button.isEnabled()
    assert ready._process_button.toolTip() == ''


# ----------------------------------------------------------------------------
# The properties button, which waits on the embeddings
# ----------------------------------------------------------------------------


def test_properties_wait_on_the_embeddings(panel: ProcessingPanel) -> None:
    assert not panel._props_button.isEnabled()
    assert panel._props_button.toolTip() == 'Please extract embeddings first.'


def test_properties_open_once_the_images_are_processed(panel: ProcessingPanel, processed) -> None:
    processed(panel._phenome, 6)
    panel.refresh_buttons()

    assert panel._props_button.isEnabled()
    assert panel._props_button.toolTip() == ''


# ----------------------------------------------------------------------------
# The running lock
# ----------------------------------------------------------------------------


def test_a_running_task_locks_both_buttons(ready: ProcessingPanel) -> None:
    ready._set_processing_task_running(True)

    for button in (ready._process_button, ready._props_button):
        assert not button.isEnabled()
        assert button.toolTip() == 'Please wait for the running task to finish.'


def test_finishing_releases_both_buttons(ready: ProcessingPanel) -> None:
    ready._set_processing_task_running(True)
    ready._set_processing_task_running(False)

    assert ready._process_button.isEnabled()
    assert ready._props_button.isEnabled()


@pytest.mark.parametrize('running', [True, False])
def test_the_lock_is_announced(panel: ProcessingPanel, qtbot, running: bool) -> None:
    with qtbot.waitSignal(panel.task_running_changed, timeout=1000) as blocker:
        panel._set_processing_task_running(running)

    assert blocker.args == [running]


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------


def test_the_device_defaults_to_the_cpu(panel: ProcessingPanel) -> None:
    assert panel._cpu_radio.isChecked()
    assert panel._phenome.params.process.device == 'cpu'


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


def test_choosing_a_model_writes_the_parameter(panel: ProcessingPanel) -> None:
    panel._model_selection_combo.setCurrentText('Large (ViT-L/14)')

    assert panel._phenome.params.process.model == 'Large (ViT-L/14)'


def test_choosing_a_preset_writes_the_parameter(panel: ProcessingPanel) -> None:
    panel._props_selection_combo.setCurrentText('Complete')

    assert panel._phenome.params.properties.preset == 'Complete'


def test_the_checkpoint_path_writes_the_parameter(panel: ProcessingPanel, tmp_path) -> None:
    panel._checkpoint_path_dialog.value = tmp_path / 'run.hdf5'

    assert panel._phenome.params.paths.checkpoint == tmp_path / 'run.hdf5'


# ----------------------------------------------------------------------------
# The advanced parameters form
# ----------------------------------------------------------------------------


def drive_advanced_form(panel: ProcessingPanel, batch_size: int, accepted: bool) -> None:
    """ Open the advanced form with its exec_ replaced, so it sets a batch size and settles at once.

    The panel builds the form on first use and keeps it, so it is built here and its instance patched,
    leaving every other dialog alone.
    """
    panel._advanced_parameters_dialog = panel._build_advanced_parameters_dialog()

    def _exec() -> int:
        panel._batch_size_spinbox.setValue(batch_size)
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


def test_the_bar_settles_when_the_task_finishes(panel: ProcessingPanel) -> None:
    """A warm checkpoint can return without ever reporting progress, so the end is set here."""
    panel._on_progress(0, 0, 'Preparing')

    panel._on_finish_process()

    assert panel._progress_bar.maximum() == 1
    assert panel._progress_bar.value() == 1
    assert panel._progress_bar.format() == 'Done!'
    assert panel._process_button.text() == 'Extract Embeddings'


# ----------------------------------------------------------------------------
# Letting go of the previous run when another dataset is loaded
# ----------------------------------------------------------------------------


def test_clearing_reads_the_checkpoint_back_from_the_parameters(ready: ProcessingPanel) -> None:
    """The interface puts it back to its default, a checkpoint answering only for its own images."""
    ready._checkpoint_path_dialog.value = 'previous.hdf5'
    ready._phenome.clear_results()

    ready.clear_results()

    assert Path(ready._checkpoint_path_dialog.value) == Path(DEFAULT_CHECKPOINT).resolve()
    assert ready._phenome.params.paths.checkpoint == Path(DEFAULT_CHECKPOINT).resolve()


def test_clearing_settles_the_progress_bar(ready: ProcessingPanel) -> None:
    ready._on_progress(3, 10, 'Processing batches')

    ready.clear_results()

    assert ready._progress_bar.value() == 0
    assert ready._progress_bar.maximum() == 1
    assert ready._progress_bar.format() == ''


def test_clearing_closes_the_properties_button_back_down(ready: ProcessingPanel) -> None:
    assert ready._props_button.isEnabled()

    ready._phenome.clear_results()
    ready.clear_results()

    assert not ready._props_button.isEnabled()
    assert ready._props_button.toolTip() == 'Please extract embeddings first.'


def test_clearing_keeps_the_loaded_model(ready: ProcessingPanel) -> None:
    """A downloaded network is not read off the images, so the label naming it stays true."""
    ready._current_model_label.setText('Small (ViT-S/14)')

    ready._phenome.clear_results()
    ready.clear_results()

    assert ready._current_model_label.text() == 'Small (ViT-S/14)'
    assert ready._phenome.model is not None


# ----------------------------------------------------------------------------
# A task that raises, which must never leave the tab locked
# ----------------------------------------------------------------------------


def test_a_failed_extraction_reopens_the_buttons(ready: ProcessingPanel, reported) -> None:
    """Both buttons are held shut for the duration of a task, and a raise has to reopen them."""
    ready._set_processing_task_running(True)

    ready._on_process_failed(RuntimeError('Checkpoint has committed rows but none match'))

    assert ready._process_button.isEnabled()
    assert ready._props_button.isEnabled()
    assert ready._process_button.text() == 'Extract Embeddings'


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


def test_a_failed_properties_run_reopens_the_buttons(ready: ProcessingPanel, reported) -> None:
    ready._set_processing_task_running(True)

    ready._on_properties_failed(RuntimeError('nope'))

    assert ready._props_button.isEnabled()
    assert ready._process_button.isEnabled()
    assert ready._props_button.text() == 'Calculate Properties'


def test_a_failed_properties_run_keeps_the_changed_preset(ready: ProcessingPanel, reported) -> None:
    """Nothing was computed, so the preset still has to be forced onto the next run."""
    ready._props_selection_combo.setCurrentText('Complete')

    ready._on_properties_failed(RuntimeError('nope'))

    assert ready._preset_changed


def test_a_failed_model_load_reopens_the_model_controls(panel: ProcessingPanel, reported) -> None:
    """A failed download otherwise leaves the button and the combo shut with no way to try again."""
    panel._load_model_button.setEnabled(False)
    panel._model_selection_combo.setEnabled(False)

    panel._on_load_model_failed(RuntimeError('connection refused'))

    assert panel._load_model_button.isEnabled()
    assert panel._model_selection_combo.isEnabled()
    assert panel._load_model_button.text() == 'Load Model'


# ----------------------------------------------------------------------------
# Reading the checkpoint before the run is made against it
# ----------------------------------------------------------------------------


def test_a_matching_checkpoint_is_not_asked_about(ready: ProcessingPanel, started, mismatched) -> None:
    """The usual run, which has to stay a single click."""
    mismatched(None)

    ready._on_process_clicked()

    assert mismatched.asked == 0
    assert started.runs == [False]


def test_a_mismatched_checkpoint_is_asked_about(ready: ProcessingPanel, started, mismatched) -> None:
    mismatched(CheckpointMismatch(MODEL_MISMATCH, 'force_rgb differs'))

    ready._on_process_clicked()

    assert mismatched.asked == 1


def test_a_confirmed_overwrite_runs_from_scratch(ready: ProcessingPanel, started, mismatched) -> None:
    mismatched(CheckpointMismatch(MODEL_MISMATCH, 'force_rgb differs'), accepted=True)

    ready._on_process_clicked()

    assert started.runs == [True]


def test_a_refused_overwrite_runs_nothing_at_all(ready: ProcessingPanel, started, mismatched) -> None:
    """The checkpoint is deleted by the run, so refusing has to stop it before it starts."""
    mismatched(CheckpointMismatch(DATASET_MISMATCH, 'no image in common'), accepted=False)

    ready._on_process_clicked()

    assert started.runs == []


def test_a_refused_overwrite_leaves_the_buttons_open(ready: ProcessingPanel, started, mismatched) -> None:
    mismatched(CheckpointMismatch(DATASET_MISMATCH, 'no image in common'), accepted=False)

    ready._on_process_clicked()

    assert ready._process_button.isEnabled()
    assert ready._process_button.text() == 'Extract Embeddings'


def test_the_question_names_what_was_found(ready: ProcessingPanel, started, mismatched) -> None:
    """It is answered by overwriting a checkpoint, so it has to say what is wrong with this one."""
    mismatched(CheckpointMismatch(DATASET_MISMATCH, 'It holds 3 image(s), none of them loaded.'))

    ready._on_process_clicked()

    assert 'different dataset' in mismatched.last
    assert 'It holds 3 image(s), none of them loaded.' in mismatched.last


def test_the_question_reads_without_a_detail(ready: ProcessingPanel, started, mismatched) -> None:
    """The parameter mismatch adds none, and must not leave an empty paragraph behind it."""
    mismatched(CheckpointMismatch(MODEL_MISMATCH))

    ready._on_process_clicked()

    assert 'a different model than the one currently loaded' in mismatched.last
    assert '\n\n\n' not in mismatched.last


# ----------------------------------------------------------------------------
# A preset changed after the properties were computed
# ----------------------------------------------------------------------------


def test_an_unchanged_preset_resumes_the_checkpoint(ready: ProcessingPanel, started) -> None:
    ready._on_properties_clicked()

    assert started.runs == [False]


def test_a_changed_preset_is_computed_again(ready: ProcessingPanel, started) -> None:
    """PhenoMe otherwise resumes the properties of the preset picked before, and only warns about it."""
    ready._props_selection_combo.setCurrentText('Complete')

    ready._on_properties_clicked()

    assert started.runs == [True]


def test_a_computed_preset_is_not_forced_twice(ready: ProcessingPanel, started) -> None:
    """Once computed it answers for what the checkpoint holds, so the next run may resume again."""
    ready._props_selection_combo.setCurrentText('Complete')
    ready._on_finish_properties()

    ready._on_properties_clicked()

    assert started.runs == [False]


def test_loading_another_dataset_forgets_the_changed_preset(ready: ProcessingPanel) -> None:
    """The checkpoint went with the results, so there are no properties left to compute over."""
    ready._props_selection_combo.setCurrentText('Complete')

    ready._phenome.clear_results()
    ready.clear_results()

    assert not ready._preset_changed
