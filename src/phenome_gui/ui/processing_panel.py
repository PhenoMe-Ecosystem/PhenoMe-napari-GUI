import logging
import threading
from pathlib import Path

from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..utils.checkpoint import MODEL_MISMATCH, CheckpointMismatch
from ..utils.params import CHANNEL_MODE_CHOICES, MAX_WORKERS, PRESET_CHOICES
from ..utils.phenome_interface import PhenoMeInterface
from ..utils.progress import StoppableProgress, TaskStopped, ThrottledProgress
from .model_window import ModelWindow

logger = logging.getLogger("main_logger")

# What the label reads, before a checkpoint has been come by and once there is one. Named rather than
# written where they are shown, the tests reading the label to say which state the panel is in.
NO_CHECKPOINT_TEXT = 'Current checkpoint: none'
CHECKPOINT_TEXT = 'Current checkpoint: {name}'

# The file a checkpoint is picked from or written to, as both dialogs name the type, and what a name
# given without one is spelled out under
CHECKPOINT_FILTER = 'HDF5 Files (*.hdf5)'
CHECKPOINT_SUFFIX = '.hdf5'


class ProcessingPanel(QWidget):
    """ The image processing tab of the PhenoMe widget.

    Holds the model loading, the embedding extraction and the properties calculation, along with the
    progress bar the last two report into.
    """

    # PhenoMe runs its progress callback on the worker thread, which must not touch widgets. Emitting
    # this signal from there hands (current, total, desc) back to the main thread through a queued
    # connection. Its signature matches phenome.utils.ProgressCallback, so `emit` is itself a valid callback.
    progress_updated = Signal(int, int, str)

    # Emitted when a processing task starts and again when it finishes, so the steps running on what the
    # task produces can lock themselves for the duration.
    task_running_changed = Signal(bool)

    # Emitted when a checkpoint has been loaded, which can bring embeddings and properties with it: the
    # steps running on those have no other way of hearing that they are suddenly there.
    checkpoint_loaded = Signal()

    def __init__(
        self,
        phenome: PhenoMeInterface,
        parent: QWidget | None = None
    ) -> None:
        """ Build the panel and its widgets.

        Args:
            phenome (PhenoMeInterface): The interface every call to PhenoMe goes through.
            parent (QWidget): The widget the panel is nested in.
        """
        super().__init__(parent)
        self._phenome = phenome

        # Both processing tasks work on the same pipeline object and share the progress bar, so only one
        # of them may run at a time. Read by the button refresh methods to lock the other button.
        self._processing_task_running = False

        # Whether the properties preset has been changed since the last run. PhenoMe resumes a
        # checkpoint's properties rather than recomputing them, and only warns about the keys the new
        # preset asks for and cannot find, so a changed preset has to say so or it is quietly ignored.
        self._preset_changed = False

        # Set from here to ask the task running on the worker thread to stop, and read there from the
        # progress callback. A fresh one is made for every run, so a stop never carries into the next.
        self._stop_event: threading.Event | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        """ Build the image processing tab and its widgets.

        Includes widgets to select processing and properties parameters and running the embedding task and the properties calculation.
        """
        layout = QVBoxLayout()

        # Checkpoint group
        checkpoint_group = QGroupBox("Checkpoint")
        checkpoint_layout = QVBoxLayout()
        checkpoint_group.setLayout(checkpoint_layout)

        # The two ways to come by a checkpoint, to save both embeddings AND properties. Named from a
        # dialog either way rather than typed: a path is only ever meant to come from one.
        self._new_checkpoint_button = QPushButton("New checkpoint...")
        self._new_checkpoint_button.clicked.connect(self._on_new_checkpoint_pressed)
        checkpoint_layout.addWidget(self._new_checkpoint_button)

        self._load_checkpoint_button = QPushButton("Load checkpoint...")
        self._load_checkpoint_button.clicked.connect(self._on_load_checkpoint_pressed)
        checkpoint_layout.addWidget(self._load_checkpoint_button)

        # Which of the two was chosen, and which file it landed on. Under both buttons rather than
        # beside either: it answers for whichever way the checkpoint was come by.
        self._current_checkpoint_label = QLabel(NO_CHECKPOINT_TEXT)
        checkpoint_layout.addWidget(self._current_checkpoint_label)

        # Closes both ways to a checkpoint until a dataset has been loaded, which is what they read
        self._refresh_checkpoint()

        # Group for embedding extraction
        embeddings_group = QGroupBox("Embeddings")
        embeddings_layout = QVBoxLayout()
        embeddings_group.setLayout(embeddings_layout)

        # Device selection radio buttons
        device_group = QGroupBox("Processing Device")
        device_layout = QHBoxLayout()
        self._cpu_radio = QRadioButton("CPU")
        self._cpu_radio.setChecked(True)  # Default selection
        self._cuda_radio = QRadioButton("CUDA")
        self._cuda_radio.setEnabled(self._phenome.cuda_available)
        self._cpu_radio.toggled.connect(self._on_device_changed)
        self._cuda_radio.toggled.connect(self._on_device_changed)
        device_layout.addWidget(self._cpu_radio)
        device_layout.addWidget(self._cuda_radio)
        device_group.setLayout(device_layout)
        embeddings_layout.addWidget(device_group)

        # Opens the window the model is picked from, which is also what starts the download: the
        # models differ in more than their names, and a list on its own could not say how.
        self._select_model_button = QPushButton("Select Model...")
        self._select_model_button.clicked.connect(self._on_select_model_clicked)
        embeddings_layout.addWidget(self._select_model_button)

        # Current loaded model
        current_model_row = QHBoxLayout()
        current_model_row.addWidget(QLabel("Loaded model:"))
        self._current_model_label = QLabel("None")
        self._current_model_label.setAlignment(Qt.AlignRight)
        current_model_row.addWidget(self._current_model_label)
        embeddings_layout.addLayout(current_model_row)

        # Advanced parameters button, opening a form built on first use
        self._advanced_parameters_dialog = None
        self._advanced_parameters_button = QPushButton("Advanced options...")
        self._advanced_parameters_button.clicked.connect(self._on_advanced_parameters_clicked)
        embeddings_layout.addWidget(self._advanced_parameters_button)

        # Extract embeddings button, enabled only once both data and a model are available
        self._process_button = QPushButton("Extract Embeddings")
        self._process_button.clicked.connect(self._on_process_clicked)
        embeddings_layout.addWidget(self._process_button)
        self._refresh_process_button()

        # Group for properties calculation
        properties_group = QGroupBox("Properties")
        properties_layout = QVBoxLayout()
        properties_group.setLayout(properties_layout)

        # Properties selection combo box
        props_row = QHBoxLayout()
        props_row.addWidget(QLabel("Properties preset"))
        self._props_selection_combo = QComboBox()
        self._props_selection_combo.addItems(PRESET_CHOICES)
        self._props_selection_combo.setCurrentText(self._phenome.params.properties.preset)
        self._props_selection_combo.currentTextChanged.connect(self._on_properties_changed)

        props_row.addWidget(self._props_selection_combo)
        properties_layout.addLayout(props_row)

        # Properties calculation button, enabled only once the images have been processed
        self._props_button = QPushButton('Calculate Properties')
        self._props_button.clicked.connect(self._on_properties_clicked)
        properties_layout.addWidget(self._props_button)
        self._refresh_props_button()

        # Stop button, calling off whichever of the two tasks is running. Sits with the progress bar
        # rather than in either group, being the one control that answers for both of them.
        self._stop_button = QPushButton('Stop')
        self._stop_button.clicked.connect(self._on_stop_clicked)
        self._refresh_stop_button()

        # Progress bar, driven from the worker thread through the progress_updated signal
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat('')
        self.progress_updated.connect(self._on_progress)

        layout.addWidget(checkpoint_group)
        layout.addWidget(embeddings_group,1)
        layout.addWidget(properties_group,1)
        layout.addStretch(1)
        layout.addWidget(self._stop_button)
        layout.addWidget(self._progress_bar)
        self.setLayout(layout)

    def _on_device_changed(self) -> None:
        """ Update the device parameter when the selection is changed. """
        device = 'cpu' if self._cpu_radio.isChecked() else 'cuda'
        self._phenome.params.process.device = device

    def _on_select_model_clicked(self) -> None:
        """ Open the window a model is picked from, and load whichever it is closed on.

        The window opens on the model currently chosen and is rebuilt on every use, the descriptions
        it shows being read as it is constructed. Nothing is written until it is confirmed: a model
        looked at and thought better of leaves the parameters as they were, the way every other form
        here behaves.
        """
        window = ModelWindow(self, self._phenome.params.process.model)

        if not window.exec_():
            logger.info('No model was picked, leaving the loaded one as it is.')
            return

        self._phenome.params.process.model = window.selected_model()
        self._load_model()

    def _load_model(self) -> None:
        """ Start the download of the chosen model, on a worker thread.

        Hundreds of megabytes over the network, so the button is shut for the duration and says what
        is keeping it: the one control the model is picked and loaded through.
        """
        worker = self._load_model_worker()
        worker.returned.connect(self._finish_load_model)
        worker.errored.connect(self._on_load_model_failed)
        worker.start()

        self._select_model_button.setText('Downloading model...')
        self._select_model_button.setEnabled(False)

    def _finish_load_model(self) -> None:
        """ Called when model is finished loading. Reopens the button and displays which model is loaded. """
        self._select_model_button.setText('Select Model...')
        self._current_model_label.setText(self._phenome.params.process.model)
        self._select_model_button.setEnabled(True)
        self._refresh_process_button()

    def _on_load_model_failed(self, error: Exception) -> None:
        """ Reopen the button when the load raised, so another model can be picked.

        A download that fails otherwise leaves the one way to a model shut, with the label still
        naming whichever model was loaded before it, and no way to try again.

        Args:
            error (Exception): The exception the load raised, as the worker caught it.
        """
        self._select_model_button.setText('Select Model...')
        self._select_model_button.setEnabled(True)

        logger.error("Could not load '%s': %s", self._phenome.params.process.model, error)
        QMessageBox.critical(
            self,
            'Model loading failed',
            f"Could not load '{self._phenome.params.process.model}'.\n\n{error}"
        )

    def _set_processing_task_running(self, running: bool) -> None:
        """ Mark a processing task as running or finished, and lock the buttons accordingly.

        Both processing buttons go through here, so starting either one disables the other until it is
        done. The change is announced, as the analysis steps run on the embeddings the tasks produce and
        have to lock themselves for the duration too.

        Args:
            running (bool): True when a task has just been started, False once it has finished.
        """
        self._processing_task_running = running
        self.refresh_buttons()
        self.task_running_changed.emit(running)

    def clear_results(self) -> None:
        """ Let go of what the dataset loaded before produced, a new one having taken its place.

        The checkpoint is let go of along with them: the interface unselects it when it drops the
        results, a checkpoint answering only for the images it was filled from, and the field and the
        label follow it back to naming nothing. The loaded model is not a result and stays where it
        is, along with the label naming it.
        """
        # The checkpoint went with the results, so there are no properties left for a changed preset
        # to have to be computed over
        self._preset_changed = False

        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat('')

        self.refresh_buttons()

    def refresh_buttons(self) -> None:
        """ Bring the checkpoint controls and both processing buttons in line with the state of the pipeline.

        Called whenever a task starts or finishes, whenever a checkpoint is come by, and from the
        outside once data has been loaded, which is what the checkpoint controls themselves wait on.
        """
        self._refresh_checkpoint()
        self._refresh_process_button()
        self._refresh_props_button()
        self._refresh_stop_button()

    def _refresh_stop_button(self) -> None:
        """ Offer the stop only while there is a task to stop, and only once.

        A stop that has been asked for cannot be asked for again: it is honoured at the next report
        PhenoMe makes, and the button says so rather than taking a second click that would do nothing.
        """
        stop_requested = self._stop_event is not None and self._stop_event.is_set()
        stoppable = self._processing_task_running and not stop_requested

        self._stop_button.setEnabled(stoppable)
        self._stop_button.setText('Stopping...' if stop_requested and self._processing_task_running else 'Stop')

        if stoppable:
            self._stop_button.setToolTip('Stop the running task. What it has computed so far is kept.')
        elif stop_requested and self._processing_task_running:
            self._stop_button.setToolTip('Waiting for the current step to finish.')
        else:
            self._stop_button.setToolTip('No task is running.')

    def _on_stop_clicked(self) -> None:
        """ Ask the running task to stop, and wait for it to unwind.

        Nothing is asked of the worker itself: it cannot be killed, and telling it to abort would only
        keep it from reporting that it was done, leaving the buttons shut for good. The stop is read
        from the progress callback instead, so it takes effect at the next batch or image, which is
        also the last one the checkpoint holds.

        A task that happens to finish between here and its next report finishes normally, its result
        being as good as any other.
        """
        if self._stop_event is None:
            return

        logger.info('Stopping the running task. It stops at its next report, which the checkpoint holds.')
        self._stop_event.set()
        self._refresh_stop_button()

    def _refresh_process_button(self) -> None:
        """ Enable the embedding extraction button only when the data, a checkpoint and a model are in place, and no task is running.

        Reads the state from the interface rather than tracking it separately, so it stays correct
        whichever loading tab was used. Called after each loading step, whenever a checkpoint is come
        by, and whenever a task starts or finishes.
        """
        metadata_df = self._phenome.metadata_df
        has_data = metadata_df is not None and not metadata_df.empty
        has_checkpoint = self._phenome.params.paths.checkpoint is not None
        has_model = self._phenome.model is not None

        self._process_button.setEnabled(
            has_data and has_checkpoint and has_model and not self._processing_task_running
        )

        if self._processing_task_running:
            self._process_button.setToolTip('Please wait for the running task to finish.')
            return

        # Named in the order they are come by, so the tooltip reads as the way through the tab
        missing = []
        if not has_data:
            missing.append('load data')
        if not has_checkpoint:
            missing.append('create or load a checkpoint')
        if not has_model:
            missing.append('load a model')
        self._process_button.setToolTip(f"Please {' and '.join(missing)} first." if missing else '')

    @thread_worker
    def _load_model_worker(self) -> None:
        """ Create worker to load model on a parallel thread. """
        self._phenome.load_model()

    def _on_new_checkpoint_pressed(self) -> None:
        """ Name the checkpoint the runs are to be written to.

        A save dialog, so it puts its own question when the name it is given is already a file:
        answering it is agreeing to replace what is there, and the file goes. Nothing is asked here
        on top of that - reading an existing checkpoint is what the Load button is for.

        A path with no file at it is taken up as it stands, that being the ordinary way a checkpoint
        is created: PhenoMe writes the file itself when the first run reaches it.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'New checkpoint', '', CHECKPOINT_FILTER
        )

        if not file_path:
            return

        named = Path(file_path)

        # Not every platform's dialog spells the filter's extension out onto the name, and a
        # checkpoint named without one is a file PhenoMe will refuse to read back later
        path = named if named.suffix else named.with_suffix(CHECKPOINT_SUFFIX)

        # Replaced rather than resumed: the dialog asked whether to replace it and was told yes,
        # where PhenoMe resumes whatever it finds at a path. Only the name the dialog asked about may
        # be removed - where the extension was filled in above, whatever now sits at that path is a
        # file nobody was asked about, and it is left alone and resumed like any other.
        if path == named and path.is_file():
            logger.warning("Replacing the checkpoint at '%s', which is to be filled again.", path)
            path.unlink()

        self._select_checkpoint(path)

    def _on_load_checkpoint_pressed(self) -> None:
        """ Read an existing checkpoint, picked from a dialog, and work on from where it was left.

        A read dialog rather than the field's save one: this names a file that has to be there, where
        the field names one that mostly is not yet.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load checkpoint', '', CHECKPOINT_FILTER
        )

        if not file_path:
            return

        self._load_checkpoint(Path(file_path))

    def _load_checkpoint(self, path: Path) -> None:
        """ Hand a checkpoint to the interface, and say so when it could not be read.

        Nothing is compared: a checkpoint filled under other settings loads all the same, and the
        disagreement is found when a run is made against it, which is the only moment the settings
        being compared are the ones that will be used.

        Run on this thread rather than on a worker: the embeddings are left on disk and read as they
        are needed, so what is read here is the metadata and the properties alone.

        Args:
            path (Path): The checkpoint file to read.
        """
        try:
            self._phenome.load_checkpoint(path)
        except (OSError, RuntimeError, ValueError) as error:
            # A file that is not a checkpoint, one another process holds open, or a dataset that was
            # never loaded: PhenoMe raises on all three, and none of them may reach the user as a
            # traceback out of a slot.
            logger.error("Could not load the checkpoint at '%s': %s", path, error)
            QMessageBox.critical(
                self, 'Checkpoint loading failed', f"Could not load the checkpoint at {path}.\n\n{error}"
            )
            self._refresh_checkpoint()
            return

        # The properties may have come with it, so a preset picked before is no longer unanswered for
        self._preset_changed = False

        self.refresh_buttons()

        # The embeddings it holds are what the analysis steps run on, and nothing else would tell them
        self.checkpoint_loaded.emit()

    def _select_checkpoint(self, path: Path) -> None:
        """ Take up a checkpoint to be written to, whether or not there is a file there yet.

        Args:
            path (Path): The checkpoint the runs are to be written to.
        """
        self._phenome.params.paths.checkpoint = path
        logger.info("Writing this run to the checkpoint at '%s'.", path)

        self.refresh_buttons()

    def _refresh_checkpoint(self) -> None:
        """ Say which checkpoint is current, and offer the two ways to come by one only once there is data.

        Both ways read the images currently found - one to check the file answers for them, the other
        to resolve the paths it stores against them - so neither is offered before a dataset has been
        loaded.
        """
        checkpoint = self._phenome.params.paths.checkpoint
        metadata_df = self._phenome.metadata_df
        has_data = metadata_df is not None and not metadata_df.empty

        self._current_checkpoint_label.setText(
            NO_CHECKPOINT_TEXT if checkpoint is None else CHECKPOINT_TEXT.format(name=checkpoint.name)
        )

        # The label names the file alone, where two runs kept apart by their directory would read as
        # the same one. The whole path is a hover away rather than on screen, being long and rarely asked.
        self._current_checkpoint_label.setToolTip('' if checkpoint is None else str(checkpoint))

        tooltip = '' if has_data else 'Please load data first.'
        for button in (self._new_checkpoint_button, self._load_checkpoint_button):
            button.setEnabled(has_data)
            button.setToolTip(tooltip)

    def _on_advanced_parameters_clicked(self) -> None:
        """ Opens a parameter form for advanced users.

        The form is built on first use and kept afterwards, but its fields are reloaded from the current
        parameters every time it opens, so the values of a cancelled edit are never kept nor shown again.
        The parameters are only written back when the form is confirmed.

        For now, the advanced parameters include the channel mode, the batch size and the number of
        loader workers.
        TODO: Think about which parameters should be here. Some parameters would probably instead be tied to the model used (resize, force_rgb, etc)?
        """
        if self._advanced_parameters_dialog is None:
            self._advanced_parameters_dialog = self._build_advanced_parameters_dialog()

        self._channel_mode_combo.setCurrentText(self._phenome.params.process.channel_mode)
        self._batch_size_spinbox.setValue(self._phenome.params.process.batch_size)
        self._num_workers_spinbox.setValue(self._phenome.params.process.num_workers)

        if self._advanced_parameters_dialog.exec_():
            self._phenome.params.process.channel_mode = self._channel_mode_combo.currentText()
            self._phenome.params.process.batch_size = self._batch_size_spinbox.value()
            self._phenome.params.process.num_workers = self._num_workers_spinbox.value()

    def _build_advanced_parameters_dialog(self) -> QDialog:
        """ Build the advanced processing parameters window.

        Only builds the form, without filling it. The fields are loaded from the parameters by the
        caller, which also reads them back when the form is confirmed.

        Returns:
            QDialog: The parameter form, parented to the panel so it closes with it.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Processing parameters")
        dialog.setFixedSize(300, 160)

        form_layout = QFormLayout()

        # What a multi-channel image is embedded as. First of the three, being the only one that
        # decides what is embedded rather than how quickly: the other two leave the same embeddings.
        self._channel_mode_combo = QComboBox()
        self._channel_mode_combo.addItems(CHANNEL_MODE_CHOICES)
        self._channel_mode_combo.setToolTip(
            "How a multi-channel image reaches the model. 'split' embeds each channel on its own and "
            "joins the results; 'combined' feeds the channels as one image. A checkpoint filled under "
            "one is not resumed under the other."
        )
        form_layout.addRow("Channel mode", self._channel_mode_combo)

        # Number of images PhenoMe feeds to the model at once
        self._batch_size_spinbox = QSpinBox()
        self._batch_size_spinbox.setRange(1, 1024)
        form_layout.addRow("Batch size", self._batch_size_spinbox)

        # Number of processes the images are read in, none of them by default: a worker is a Python
        # started from nothing, which imports torch before it reads its first image, and the run waits
        # on all of them before its first batch. Worth raising where starting a process is cheap and
        # the images are slow to read.
        self._num_workers_spinbox = QSpinBox()
        self._num_workers_spinbox.setRange(0, MAX_WORKERS)
        self._num_workers_spinbox.setToolTip(
            'Processes the images are read in. 0 reads them in the processing thread, which starts '
            'right away; more can read faster once started, at a wait before every run.'
        )
        form_layout.addRow("Num. workers", self._num_workers_spinbox)

        # OK closes the form with an accepted result, which is what tells the caller to apply the values
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        form_layout.addRow(button_box)

        dialog.setLayout(form_layout)
        return dialog

    def _on_process_clicked(self) -> None:
        """ Begins the image processing/embedding extraction task.

        The checkpoint is read before anything is started: PhenoMe resumes from the one it is pointed
        at, and raises on one it cannot resume from, which is a long way into a run to find out. Asked
        here, the same disagreement is put to the user while the run can still be called off.

        Only the disagreement is asked about. A checkpoint that agrees is handed to PhenoMe as it is,
        which resumes it - processing the images it does not yet hold, or loading it and processing
        nothing at all when it already holds them all.
        """
        mismatch = self._phenome.inspect_checkpoint()

        force_reprocess = False
        if mismatch is not None:
            if not self._confirm_overwrite(mismatch):
                logger.info('Extraction was called off, leaving the checkpoint as it was.')
                return
            force_reprocess = True

        self._stop_event = threading.Event()

        worker = self._process_images_worker(force_reprocess)
        worker.returned.connect(self._on_finish_process)
        worker.errored.connect(self._on_process_failed)

        # Put the tab in its running state before the run is let go of, rather than after: the lock is
        # what offers the stop, and the run is what the stop is offered on.
        self._process_button.setText('Processing images...')
        self._set_processing_task_running(True)

        # PhenoMe stays silent while it builds the dataloader and scans the checkpoint, so show a busy
        # indicator until the first batch is reported.
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat('Preparing...')

        worker.start()

    def _confirm_overwrite(self, mismatch: CheckpointMismatch) -> bool:
        """ Ask whether a checkpoint that answers for another run may be deleted and filled again.

        The only way past a checkpoint PhenoMe will not resume from, short of naming another file: it
        removes the file rather than writing over it, so a long extraction is lost with it, and the
        properties in the same file go with it. Asked with the refusal as its default answer, the way
        dropping the loaded results is asked.

        Args:
            mismatch (CheckpointMismatch): What the checkpoint holds that this run would not put there.
        Returns:
            bool: True when the checkpoint may be deleted and the run made from scratch.
        """
        checkpoint = self._phenome.params.paths.checkpoint
        filled_with = (
            'under different processing settings, or with a different model'
            if mismatch.reason == MODEL_MISMATCH else 'from a different dataset'
        )

        # Only the dataset mismatch has anything to add: what a run's parameters disagree on goes to
        # the log, PhenoMe writing the whole comparison out there key by key itself.
        detail = f'{mismatch.detail}\n\n' if mismatch.detail else ''

        logger.warning(
            "The checkpoint at '%s' was filled %s. %s", checkpoint, filled_with, mismatch.detail
        )

        answer = QMessageBox.warning(
            self,
            'Checkpoint does not match',
            f"The checkpoint at {checkpoint} was filled {filled_with}.\n\n"
            f"{detail}"
            'Overwriting deletes the file, and the properties it holds along with the embeddings.\n\n'
            'Overwrite the checkpoint?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        return answer == QMessageBox.StandardButton.Yes

    def _on_progress(self, current: int, total: int, desc: str) -> None:
        """ Update the progress bar from a PhenoMe progress report.

        Runs on the main thread via the progress_updated signal. Keyed off desc rather than the step it
        comes from, so the embedding and properties steps can share it.

        Args:
            current: Units completed so far. Batches for embedding extraction, images for properties.
            total: Expected total number of units. PhenoMe documents 0 as "unknown".
            desc: Short stage label, e.g. "Processing batches".
        """
        if total > 0:
            if self._progress_bar.maximum() != total:
                self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(current)
            self._progress_bar.setFormat(f"{desc}: %v/%m (%p%)")
        else:
            self._progress_bar.setRange(0, 0)
            self._progress_bar.setFormat(desc)

    def _on_finish_process(self) -> None:
        """ Called when processing is done. Re-enables the button and changes text. """
        # Releasing the lock also re-enables the properties button, whose step runs on the images
        # processed here and only becomes available now.
        self._process_button.setText('Extract Embeddings')
        self._set_processing_task_running(False)

        # A warm checkpoint can make PhenoMe return without ever reporting progress, so settle the bar
        # here instead of relying on a final callback.
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(1)
        self._progress_bar.setFormat('Done!')

    def _on_process_failed(self, error: Exception) -> None:
        """ Reopen the buttons when the extraction raised, so another run can be attempted.

        The worker swallows the exception into this signal, and both buttons are held shut for the
        duration of a task: with nobody listening, a raise leaves them shut for good and the tab has to
        be restarted to be used again. PhenoMe raises here for a checkpoint it cannot resume from,
        among other things, which the run before it was asked about but which can still arrive from a
        checkpoint written between the question and the run.

        A stop arrives the same way, the run being called off by raising out of the progress callback,
        and is not reported as a failure: it was asked for, and it is picked up by asking for the run
        again.

        Args:
            error (Exception): The exception the extraction raised, as the worker caught it.
        """
        self._process_button.setText('Extract Embeddings')
        self._set_processing_task_running(False)

        if isinstance(error, TaskStopped):
            self._settle_bar('Stopped')
            logger.info(
                'The extraction was stopped. The checkpoint holds the batches it had committed; '
                'extract the embeddings again to carry on from there.'
            )
            return

        self._settle_bar('Failed')

        logger.error('Could not extract the embeddings: %s', error)
        QMessageBox.critical(self, 'Extraction failed', f'Could not extract the embeddings.\n\n{error}')

    def _settle_bar(self, status: str) -> None:
        """ Take the progress bar out of the busy state a task that did not run to its end left it spinning in.

        Args:
            status (str): What the bar is to read, naming how the task ended.
        """
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat(status)

    @thread_worker
    def _process_images_worker(self, force_reprocess: bool = False) -> None:
        """ Create worker to process images on a parallel thread.

        PhenoMe reports after every batch, which the throttle thins out before it reaches the bar: a
        report costs a hop to the main thread and a repaint, and the step is only worth taking when it
        is big enough to show. The stop is read outside the throttle, every report being a chance to
        take it and only a hundredth of them reaching the bar.

        Args:
            force_reprocess (bool): Whether to delete the checkpoint and process every image again,
                which is only ever passed for a checkpoint the user has been asked about.
        """
        self._phenome.process_images(
            progress_callback=StoppableProgress(
                ThrottledProgress(self.progress_updated.emit), self._stop_event
            ),
            force_reprocess=force_reprocess
        )

    def _on_properties_changed(self) -> None:
        """ Update the properties preset parameter when the selection changes.

        A changed preset is noted, as the next run has to be told to compute its properties again
        rather than resume the ones the checkpoint holds, which were computed under the preset picked
        before it. Picking a preset is the whole of the signal, so nothing has to be asked.
        """
        preset = self._props_selection_combo.currentText()
        self._phenome.params.properties.preset = preset
        self._preset_changed = True

    def _refresh_props_button(self) -> None:
        """ Enable the properties calculation button only once there are processed images and a checkpoint to write to, and no task is running.

        PhenoMe computes the properties over the images held by the pipeline, and raises when there are
        none, so the state is read from the interface. Called when building the tab, whenever a
        checkpoint is come by, and whenever a task starts or finishes.

        The two conditions come apart only for a moment: images are processed into a checkpoint, and
        loading one brings its images with it, so a run with images and no checkpoint is one whose
        dataset has just been dropped.
        """
        has_processed_images = self._phenome.has_processed_images
        has_checkpoint = self._phenome.params.paths.checkpoint is not None

        self._props_button.setEnabled(
            has_processed_images and has_checkpoint and not self._processing_task_running
        )

        if self._processing_task_running:
            self._props_button.setToolTip('Please wait for the running task to finish.')
        elif not has_checkpoint:
            self._props_button.setToolTip('Please create or load a checkpoint first.')
        else:
            self._props_button.setToolTip('' if has_processed_images else 'Please extract embeddings first.')

    def _on_properties_clicked(self) -> None:
        """ Begins the properties calculation task """
        self._stop_event = threading.Event()

        worker = self._compute_properties_worker(self._preset_changed)
        worker.returned.connect(self._on_finish_properties)
        worker.errored.connect(self._on_properties_failed)

        # As in the extraction: the tab is locked, and the stop offered with it, before the run starts
        self._props_button.setText('Calculating properties...')
        self._set_processing_task_running(True)

        # PhenoMe stays silent while it resolves the image paths and scans the checkpoint, so show a busy
        # indicator until the first image is reported.
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat('Preparing...')

        worker.start()

    def _on_finish_properties(self) -> None:
        """ Called when the properties calculation is done. Re-enables the button and changes text. """
        self._props_button.setText('Calculate Properties')
        self._set_processing_task_running(False)

        # The preset now answers for what the checkpoint holds, this run having just computed it
        self._preset_changed = False

        # A warm checkpoint can make PhenoMe return without ever reporting progress, so settle the bar
        # here instead of relying on a final callback.
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(1)
        self._progress_bar.setFormat('Done!')

    def _on_properties_failed(self, error: Exception) -> None:
        """ Reopen the buttons when the properties calculation raised, so another run can be attempted.

        The twin of the extraction's failure handler, and needed for the same reason: the buttons are
        held shut for the duration of a task, and a raise nobody listens for never reopens them. The
        changed preset is left standing, this run not having computed it.

        A stop is not one of those failures, and answers for the preset differently. PhenoMe clears
        the properties the checkpoint holds before it computes the first one, so a run that got as far
        as reporting has already replaced them with the preset it was started under: the change has
        been made, and saying otherwise would have the next run clear the partial properties it is
        meant to carry on from.

        Args:
            error (Exception): The exception the calculation raised, as the worker caught it.
        """
        self._props_button.setText('Calculate Properties')
        self._set_processing_task_running(False)

        if isinstance(error, TaskStopped):
            self._preset_changed = False
            self._settle_bar('Stopped')
            logger.info(
                'The properties calculation was stopped. The checkpoint holds the images it had '
                'committed; calculate the properties again to carry on from there.'
            )
            return

        self._settle_bar('Failed')

        logger.error('Could not compute the properties: %s', error)
        QMessageBox.critical(self, 'Properties failed', f'Could not compute the properties.\n\n{error}')

    @thread_worker
    def _compute_properties_worker(self, force_update: bool = False) -> None:
        """ Create worker to compute the properties on a parallel thread.

        PhenoMe reports after every image here, which is far more often than the bar can be drawn, so
        the throttle lets only a hundredth of the run through and the computation is left to run at
        its own pace.

        Args:
            force_update (bool): Whether to drop the properties the checkpoint holds and compute them
                again, which a preset changed since the last run needs. The embeddings in the same
                file are left where they are.
        """
        self._phenome.compute_properties(
            progress_callback=StoppableProgress(
                ThrottledProgress(self.progress_updated.emit), self._stop_event
            ),
            force_update=force_update
        )
