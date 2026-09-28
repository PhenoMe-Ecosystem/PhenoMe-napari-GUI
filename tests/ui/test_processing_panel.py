"""Tests for the image processing panel.

Nothing here loads a model or extracts embeddings: the pipeline state is stubbed instead, and the model
is a sentinel object, since all the panel ever asks is whether one is present.
"""

from pathlib import Path

import pytest
from qtpy.QtWidgets import QFileDialog, QFormLayout, QMessageBox

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
            force: force_reprocess for the extraction, force_update for the properties.
        Returns:
            Itself, standing in for the worker it would have built.
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
        The stand-in, holding the flag each started run was given.
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


def test_extract_needs_the_data_a_checkpoint_and_a_model(panel: ProcessingPanel) -> None:
    assert not panel._process_button.isEnabled()
    assert panel._process_button.toolTip() == (
        'Please load data and create or load a checkpoint and load a model first.'
    )


def test_extract_still_waits_on_the_rest_after_loading(panel: ProcessingPanel, widget) -> None:
    widget.loading_panel._on_load_data_csv()

    assert not panel._process_button.isEnabled()
    assert panel._process_button.toolTip() == (
        'Please create or load a checkpoint and load a model first.'
    )


def test_extract_still_waits_on_data_after_the_model(panel: ProcessingPanel, tmp_path: Path) -> None:
    panel._phenome.model = object()
    panel._phenome.params.paths.checkpoint = tmp_path / 'run.hdf5'
    panel.refresh_buttons()

    assert not panel._process_button.isEnabled()
    assert panel._process_button.toolTip() == 'Please load data first.'


def test_extract_still_waits_on_a_checkpoint(panel: ProcessingPanel, widget) -> None:
    """The embeddings are written to it, so a run without one would have nowhere to put them."""
    widget.loading_panel._on_load_data_csv()
    panel._phenome.model = object()
    panel.refresh_buttons()

    assert not panel._process_button.isEnabled()
    assert panel._process_button.toolTip() == 'Please create or load a checkpoint first.'


def test_extract_opens_once_all_three_are_there(ready: ProcessingPanel) -> None:
    assert ready._process_button.isEnabled()
    assert ready._process_button.toolTip() == ''


# ----------------------------------------------------------------------------
# The properties button, which waits on the embeddings and a checkpoint
# ----------------------------------------------------------------------------


def test_properties_wait_on_a_checkpoint(panel: ProcessingPanel) -> None:
    assert not panel._props_button.isEnabled()
    assert panel._props_button.toolTip() == 'Please create or load a checkpoint first.'


def test_properties_wait_on_the_embeddings(panel: ProcessingPanel, tmp_path: Path) -> None:
    panel._phenome.params.paths.checkpoint = tmp_path / 'run.hdf5'
    panel.refresh_buttons()

    assert not panel._props_button.isEnabled()
    assert panel._props_button.toolTip() == 'Please extract embeddings first.'


def test_properties_open_once_the_images_are_processed(
    panel: ProcessingPanel, processed, tmp_path: Path
) -> None:
    panel._phenome.params.paths.checkpoint = tmp_path / 'run.hdf5'
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


@pytest.fixture
def picking_model(panel: ProcessingPanel, monkeypatch: pytest.MonkeyPatch):
    """ Return a hook answering the model window with a pick, without showing it.

    Its exec_ is modal and would hang the test, so it is answered outright. The window is built
    either way, which is what keeps this a test of the panel driving a real one.

    Returns:
        Called as picking_model(ANOTHER_MODEL), or with None for
        the window being cancelled. Returns the list of loads the panel then started.
    """
    started = []
    monkeypatch.setattr(panel, '_load_model', lambda: started.append(panel._phenome.params.process.model))

    def _answer(model: str | None) -> list:
        def _exec(window) -> int:
            if model is None:
                return 0
            window._select(model)
            return 1

        monkeypatch.setattr(ModelWindow, 'exec_', _exec)
        return started

    return _answer


def test_choosing_a_model_writes_the_parameter(panel: ProcessingPanel, picking_model) -> None:
    picking_model(ANOTHER_MODEL)

    panel._select_model_button.click()

    assert panel._phenome.params.process.model == ANOTHER_MODEL


def test_choosing_a_model_loads_it(panel: ProcessingPanel, picking_model) -> None:
    """Picking and loading are the one gesture now: the window is what the download is started from."""
    started = picking_model(ANOTHER_MODEL)

    panel._select_model_button.click()

    assert started == [ANOTHER_MODEL]


def test_a_cancelled_window_changes_nothing(panel: ProcessingPanel, picking_model) -> None:
    """A model looked at and thought better of leaves the parameters as they were."""
    before = panel._phenome.params.process.model
    started = picking_model(None)

    panel._select_model_button.click()

    assert panel._phenome.params.process.model == before
    assert started == []


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


def test_the_worker_count_starts_at_none_of_them(panel: ProcessingPanel) -> None:
    """Every worker is a wait before the run starts, which is not worth paying unasked."""
    assert panel._phenome.params.process.num_workers == 0


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


def test_the_channel_mode_is_the_first_thing_asked(panel: ProcessingPanel) -> None:
    """It decides what is embedded, where the fields under it only decide how quickly."""
    panel._advanced_parameters_dialog = panel._build_advanced_parameters_dialog()

    form = panel._advanced_parameters_dialog.layout()
    assert form.itemAt(0, QFormLayout.ItemRole.FieldRole).widget() is panel._channel_mode_combo
    assert form.itemAt(0, QFormLayout.ItemRole.LabelRole).widget().text() == 'Channel mode'


def test_the_channel_mode_starts_split(panel: ProcessingPanel) -> None:
    """PhenoMe's own default, a channel keeping its own place in the embedding."""
    assert panel._phenome.params.process.channel_mode == 'split'


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


def test_clearing_closes_the_properties_button_back_down(ready: ProcessingPanel) -> None:
    """The checkpoint went with the embeddings, and is the first of the two to be come by again."""
    assert ready._props_button.isEnabled()

    ready._phenome.clear_results()
    ready.clear_results()

    assert not ready._props_button.isEnabled()
    assert ready._props_button.toolTip() == 'Please create or load a checkpoint first.'


def test_clearing_keeps_the_loaded_model(ready: ProcessingPanel) -> None:
    """A downloaded network is not read off the images, so the label naming it stays true."""
    ready._current_model_label.setText(FIRST_MODEL)

    ready._phenome.clear_results()
    ready.clear_results()

    assert ready._current_model_label.text() == FIRST_MODEL
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
    """A failed download otherwise leaves the one way to a model shut, with no way to try again."""
    panel._select_model_button.setEnabled(False)
    panel._select_model_button.setText('Downloading model...')

    panel._on_load_model_failed(RuntimeError('connection refused'))

    assert panel._select_model_button.isEnabled()
    assert panel._select_model_button.text() == 'Select Model...'


# ----------------------------------------------------------------------------
# Reading the checkpoint before the run is made against it
# ----------------------------------------------------------------------------


def test_a_matching_checkpoint_is_not_asked_about(ready: ProcessingPanel, started, mismatched) -> None:
    """The usual run, which has to stay a single click."""
    mismatched(None)

    ready._on_process_clicked()

    assert mismatched.asked == 0
    assert started.runs == [False]


def test_a_complete_checkpoint_is_still_handed_to_phenome(
    ready: ProcessingPanel, started, mismatched
) -> None:
    """PhenoMe loads rather than processes when it already holds every image, so the panel leaves it to."""
    mismatched(None)

    ready._on_process_clicked()

    assert started.runs == [False]


def test_a_mismatched_checkpoint_is_asked_about(ready: ProcessingPanel, started, mismatched) -> None:
    """The settings it compares can be picked again after the checkpoint was come by."""
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


def test_the_overwrite_question_names_what_was_found(
    ready: ProcessingPanel, started, mismatched
) -> None:
    """It is answered by deleting a checkpoint, so it has to say what is wrong with this one."""
    mismatched(CheckpointMismatch(DATASET_MISMATCH, 'It holds 3 image(s), none of them loaded.'))

    ready._on_process_clicked()

    assert 'different dataset' in mismatched.last
    assert 'It holds 3 image(s), none of them loaded.' in mismatched.last


def test_the_overwrite_question_says_the_file_goes(
    ready: ProcessingPanel, started, mismatched
) -> None:
    """PhenoMe removes the file rather than writing over it, properties and all."""
    mismatched(CheckpointMismatch(MODEL_MISMATCH))

    ready._on_process_clicked()

    assert 'deletes the file' in mismatched.last


def test_the_overwrite_question_reads_without_a_detail(
    ready: ProcessingPanel, started, mismatched
) -> None:
    """The parameter mismatch adds none, and must not leave an empty paragraph behind it."""
    mismatched(CheckpointMismatch(MODEL_MISMATCH))

    ready._on_process_clicked()

    assert 'different processing settings' in mismatched.last
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


# ----------------------------------------------------------------------------
# Stopping a running task
# ----------------------------------------------------------------------------


def test_there_is_nothing_to_stop_before_a_run(panel: ProcessingPanel) -> None:
    assert not panel._stop_button.isEnabled()
    assert panel._stop_button.toolTip() == 'No task is running.'


def test_there_is_still_nothing_to_stop_once_everything_is_loaded(ready: ProcessingPanel) -> None:
    """Data, a model and processed images open the two task buttons, none of which is a running task."""
    assert not ready._stop_button.isEnabled()


@pytest.mark.parametrize('click', ['_on_process_clicked', '_on_properties_clicked'])
def test_either_started_task_can_be_stopped(ready: ProcessingPanel, started, mismatched, click: str) -> None:
    mismatched(None)

    getattr(ready, click)()

    assert ready._stop_button.isEnabled()
    assert ready._stop_button.text() == 'Stop'


def test_a_finished_task_leaves_nothing_to_stop(ready: ProcessingPanel, started, mismatched) -> None:
    mismatched(None)
    ready._on_process_clicked()

    ready._on_finish_process()

    assert not ready._stop_button.isEnabled()
    assert ready._stop_button.text() == 'Stop'


def test_stopping_asks_the_task_to_stop(ready: ProcessingPanel, started, mismatched) -> None:
    mismatched(None)
    ready._on_process_clicked()

    ready._stop_button.click()

    assert ready._stop_event.is_set()


def test_a_stop_cannot_be_asked_for_twice(ready: ProcessingPanel, started, mismatched) -> None:
    """It is taken at the next report PhenoMe makes, and a second click would do nothing."""
    mismatched(None)
    ready._on_process_clicked()

    ready._stop_button.click()

    assert not ready._stop_button.isEnabled()
    assert ready._stop_button.text() == 'Stopping...'
    assert ready._stop_button.toolTip() == 'Waiting for the current step to finish.'


def test_a_new_run_may_be_stopped_again(ready: ProcessingPanel, started, mismatched) -> None:
    """Every run gets an event of its own, or the one before it would stop this one at once."""
    mismatched(None)
    ready._on_process_clicked()
    ready._stop_button.click()
    ready._on_process_failed(TaskStopped())

    ready._on_process_clicked()

    assert ready._stop_button.isEnabled()
    assert not ready._stop_event.is_set()


def test_a_stopped_extraction_reopens_the_buttons(ready: ProcessingPanel, reported) -> None:
    ready._set_processing_task_running(True)

    ready._on_process_failed(TaskStopped())

    assert ready._process_button.isEnabled()
    assert ready._props_button.isEnabled()
    assert ready._process_button.text() == 'Extract Embeddings'


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


def test_a_stopped_properties_run_reopens_the_buttons(ready: ProcessingPanel, reported) -> None:
    ready._set_processing_task_running(True)

    ready._on_properties_failed(TaskStopped())

    assert ready._props_button.isEnabled()
    assert ready._process_button.isEnabled()
    assert ready._props_button.text() == 'Calculate Properties'


def test_a_stopped_properties_run_is_not_reported_as_a_failure(ready: ProcessingPanel, reported) -> None:
    ready._on_properties_failed(TaskStopped())

    assert reported.shown == 0
    assert ready._progress_bar.format() == 'Stopped'


def test_a_stopped_properties_run_resumes_rather_than_starting_over(
    ready: ProcessingPanel, started, reported
) -> None:
    """PhenoMe clears the old properties before the first report, so the preset has been applied:
    forcing it again would wipe the partial run the next click is meant to carry on from."""
    ready._props_selection_combo.setCurrentText('Complete')

    ready._on_properties_failed(TaskStopped())
    ready._on_properties_clicked()

    assert not ready._preset_changed
    assert started.runs == [False]


def test_a_task_stopped_between_reports_still_finishes(ready: ProcessingPanel, started, mismatched) -> None:
    """The stop is only read on a report; one that never comes leaves a result as good as any other."""
    mismatched(None)
    ready._on_process_clicked()
    ready._stop_button.click()

    ready._on_finish_process()

    assert ready._process_button.isEnabled()
    assert ready._progress_bar.format() == 'Done!'


def test_a_stopped_run_is_unwound_out_of_phenome(
    ready: ProcessingPanel, sync_worker, mismatched, reported, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the stop is read from the callback PhenoMe reports through, and lands on the panel."""
    mismatched(None)
    batches = []

    def _extract(progress_callback, force_reprocess: bool = False) -> None:
        """ Report ten batches the way PhenoMe does, the fourth of which never comes. """
        for batch in range(1, 11):
            progress_callback(batch, 10, 'Processing batches')
            batches.append(batch)
            if batch == 3:
                ready._stop_button.click()

    monkeypatch.setattr(ready._phenome, 'process_images', _extract)
    sync_worker(
        ready,
        '_process_images_worker',
        lambda force: ProcessingPanel._process_images_worker.__wrapped__(ready, force),
    )

    ready._on_process_clicked()

    assert batches == [1, 2, 3]
    assert reported.shown == 0
    assert ready._progress_bar.format() == 'Stopped'
    assert ready._process_button.isEnabled()


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


def test_the_checkpoint_controls_are_shown(panel: ProcessingPanel) -> None:
    """Built but never laid out is built for nobody, whatever they are connected to."""
    for widget in (
        panel._new_checkpoint_button,
        panel._load_checkpoint_button,
        panel._current_checkpoint_label,
    ):
        assert widget.isVisibleTo(panel)


def test_nothing_is_selected_to_begin_with(panel: ProcessingPanel) -> None:
    assert panel._phenome.params.paths.checkpoint is None
    assert panel._current_checkpoint_label.text() == NO_CHECKPOINT_TEXT


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
