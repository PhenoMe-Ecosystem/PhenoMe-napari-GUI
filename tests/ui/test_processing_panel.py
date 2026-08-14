"""Tests for the image processing panel.

Nothing here loads a model or extracts embeddings: the pipeline state is stubbed instead, and the model
is a sentinel object, since all the panel ever asks is whether one is present.
"""

import pytest

from phenome_gui.ui.processing_panel import ProcessingPanel


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
