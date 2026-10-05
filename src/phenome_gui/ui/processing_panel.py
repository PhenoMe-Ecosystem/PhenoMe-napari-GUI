import logging
import threading
from pathlib import Path

from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
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

from ..registries.models import MODEL_OPTIONS
from ..utils.checkpoint import MODEL_MISMATCH, CheckpointMismatch
from ..utils.params import CHANNEL_MODE_CHOICES, MAX_WORKERS, PRESET_CHOICES
from ..utils.phenome_interface import PhenoMeInterface
from ..utils.progress import StoppableProgress, TaskStopped, ThrottledProgress
from .model_window import ModelWindow
from .styles import GROUP_STYLESHEETS

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

    Holds the checkpoint the results are saved to or loaded from, the model selection, the embedding
    extraction and the properties calculation, along with the progress bar the tasks report into.

    A run keeps its results in memory unless lazy checkpointing is on, in which case it writes them to
    the selected checkpoint as it goes; saving them is otherwise a step of its own.
    """

    # PhenoMe runs its progress callback on the worker thread, which must not touch widgets. Emitting
    # this signal from there hands (current, total, desc) back to the main thread through a queued
    # connection. Its signature matches phenome.utils.ProgressCallback, so `emit` is itself a valid callback.
    progress_updated = Signal(int, int, str)

    # Emitted from the worker thread once the extraction has its model, loaded or kept, so the button
    # can stop saying it is loading one. Queued like progress_updated, so it lands before the first batch.
    model_ready = Signal()

    # Emitted from the worker thread once the embeddings are extracted and the properties are next, so
    # the button can say so and the end of the task be read against the right step.
    properties_started = Signal()

    # Emitted when a processing task starts and again when it finishes, so the steps running on what the
    # task produces can lock themselves for the duration.
    task_running_changed = Signal(bool)

    # Emitted when a checkpoint has been loaded, which can bring embeddings and properties with it: the
    # steps running on those have no other way of hearing that they are suddenly there.
    checkpoint_loaded = Signal()

    # Emitted when a run is about to replace embeddings that were already there, the analysis results
    # computed from them having just been dropped: the panels and windows showing those results have
    # to let go of their own share of them.
    analysis_cleared = Signal()

    def __init__(
        self,
        phenome: PhenoMeInterface,
        parent: QWidget | None = None
    ) -> None:
        """ Build the panel and its widgets.

        Args:
            phenome: The interface every call to PhenoMe goes through.
            parent: The widget the panel is nested in.
        """
        super().__init__(parent)
        self._phenome = phenome

        # Whether the processing task is running: the extraction, and the properties after it. Read by
        # the refresh methods to lock the controls a run reads from.
        self._processing_task_running = False

        # Whether the running task has moved on from the embeddings to the properties, set from the
        # properties_started signal. Read when the task ends, a failure or a stop answering differently
        # for each of the two steps.
        self._computing_properties = False

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
        self._checkpoint_group = QGroupBox("Checkpoint")
        self._checkpoint_group.setObjectName("parentGroup")
        checkpoint_layout = QVBoxLayout()
        self._checkpoint_group.setLayout(checkpoint_layout)

        # The two ways to a checkpoint: writing out the embeddings and properties extracted here, or
        # reading back ones extracted before. Named from a dialog either way rather than typed: a path
        # is only ever meant to come from one.
        self._save_checkpoint_button = QPushButton("Save checkpoint...")
        self._save_checkpoint_button.clicked.connect(self._on_save_checkpoint_pressed)
        checkpoint_layout.addWidget(self._save_checkpoint_button)

        self._load_checkpoint_button = QPushButton("Load checkpoint...")
        self._load_checkpoint_button.clicked.connect(self._on_load_checkpoint_pressed)
        checkpoint_layout.addWidget(self._load_checkpoint_button)

        # Which of the two was chosen, and which file it landed on. Under both buttons rather than
        # beside either: it answers for whichever way the checkpoint was come by.
        self._current_checkpoint_label = QLabel(NO_CHECKPOINT_TEXT)
        checkpoint_layout.addWidget(self._current_checkpoint_label)

        # Closes the load until a dataset has been loaded, and the save until there is something to save
        self._refresh_checkpoint()

        # Group for embedding extraction
        self._embeddings_group = QGroupBox("Embeddings")
        self._embeddings_group.setObjectName("parentGroup")
        embeddings_layout = QVBoxLayout()
        self._embeddings_group.setLayout(embeddings_layout)

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

        # Model selection row
        model_row = QHBoxLayout()

        # Opens the window the model is picked from: the models differ in more than their names, and a
        # list on its own could not say how. Picking only chooses; the download waits for the run.
        self._select_model_button = QPushButton("Select Model...")
        self._select_model_button.clicked.connect(self._on_select_model_clicked)
        model_row.addWidget(self._select_model_button)

        # Advanced parameters button, opening a form built on first use
        self._advanced_parameters_dialog = None
        self._advanced_parameters_button = QPushButton("Advanced options...")
        self._advanced_parameters_button.clicked.connect(self._on_advanced_parameters_clicked)
        model_row.addWidget(self._advanced_parameters_button)
        embeddings_layout.addLayout(model_row)

        # Current selected model, the parameters' default until another is picked
        current_model_row = QHBoxLayout()
        current_model_row.addWidget(QLabel("Selected model:"))
        self._current_model_label = QLabel(self._phenome.params.process.model)
        self._current_model_label.setAlignment(Qt.AlignRight)
        current_model_row.addWidget(self._current_model_label)
        embeddings_layout.addLayout(current_model_row)

        # Group for properties calculation
        self._properties_group = QGroupBox("Properties")
        self._properties_group.setObjectName("parentGroup")
        properties_layout = QVBoxLayout()
        self._properties_group.setLayout(properties_layout)

        # Properties selection combo box
        props_row = QHBoxLayout()
        props_row.addWidget(QLabel("Properties preset"))
        self._props_selection_combo = QComboBox()
        self._props_selection_combo.addItems(PRESET_CHOICES)
        self._props_selection_combo.setCurrentText(self._phenome.params.properties.preset)
        self._props_selection_combo.currentTextChanged.connect(self._on_properties_changed)

        # Checkbox for properties calculation, computed after the embeddings when Process Images runs
        self._props_checkbox = QCheckBox("Compute properties")
        self._props_checkbox.setChecked(self._phenome.params.properties.compute)
        self._props_checkbox.clicked.connect(self._on_properties_checkbox_changed)

        props_row.addWidget(self._props_selection_combo)
        properties_layout.addLayout(props_row)
        properties_layout.addWidget(self._props_checkbox)

        # Process / Stop row
        process_row = QHBoxLayout()

        # Process button, enabled only once data and a model are available
        self._process_button = QPushButton("Process Images")
        self._process_button.clicked.connect(self._on_process_clicked)
        self._refresh_process_button()

        # Stop button, calling off whichever of the two tasks is running. Sits with the progress bar
        # rather than in either group, being the one control that answers for both of them.
        self._stop_button = QPushButton('Stop')
        self._stop_button.clicked.connect(self._on_stop_clicked)
        self._refresh_stop_button()

        process_row.addWidget(self._process_button)
        process_row.addWidget(self._stop_button)

        # Progress bar, driven from the worker thread through the progress_updated signal
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat('')
        self.progress_updated.connect(self._on_progress)
        self.model_ready.connect(self._on_model_ready)
        self.properties_started.connect(self._on_properties_started)

        layout.addWidget(self._checkpoint_group)
        layout.addWidget(self._embeddings_group,1)
        layout.addWidget(self._properties_group,1)
        layout.addStretch(1)
        layout.addLayout(process_row)
        layout.addWidget(self._progress_bar)
        self.setLayout(layout)

    def _on_device_changed(self) -> None:
        """ Update the device parameter when the selection is changed. """
        device = 'cpu' if self._cpu_radio.isChecked() else 'cuda'
        self._phenome.params.process.device = device

    def _on_select_model_clicked(self) -> None:
        """ Open the window a model is picked from, and select whichever it is closed on.

        The window opens on the model currently chosen and is rebuilt on every use, the descriptions
        it shows being read as it is constructed. Nothing is written until it is confirmed: a model
        looked at and thought better of leaves the parameters as they were, the way every other form
        here behaves.

        Only the parameter is written. The model is loaded when the embeddings are extracted, the
        download being some hundreds of megabytes that a pick alone does not call for.
        """
        window = ModelWindow(self, self._phenome.params.process.model)

        if not window.exec_():
            logger.info('No model was picked, leaving the selected one as it is.')
            return

        self._phenome.params.process.model = window.selected_model()
        self._current_model_label.setText(self._phenome.params.process.model)
        self.refresh_buttons()

    def _set_processing_task_running(self, running: bool) -> None:
        """ Mark a processing task as running or finished, and lock the buttons accordingly.

        Both processing buttons go through here, so starting either one disables the other until it is
        done. The change is announced, as the analysis steps run on the embeddings the tasks produce and
        have to lock themselves for the duration too.

        Args:
            running: True when a task has just been started, False once it has finished.
        """
        self._processing_task_running = running
        self.refresh_buttons()
        self.task_running_changed.emit(running)

    def clear_results(self) -> None:
        """ Let go of what the dataset loaded before produced, a new one having taken its place.

        The checkpoint is let go of along with them: the interface unselects it when it drops the
        results, a checkpoint answering only for the images it was filled from, and the field and the
        label follow it back to naming nothing. The selected model is not a result and stays where
        it is, along with the label naming it and the network loaded for it.
        """
        # The checkpoint went with the results, so there are no properties left for a changed preset
        # to have to be computed over
        self._preset_changed = False

        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat('')

        self.refresh_buttons()

    def refresh_buttons(self) -> None:
        """ Bring the checkpoint controls and the task buttons in line with the state of the pipeline.

        Called whenever a task starts or finishes, whenever a checkpoint is come by, and from the
        outside once data has been loaded, which is what loading a checkpoint waits on.
        """
        self._refresh_groups()
        self._refresh_checkpoint()
        self._refresh_settings_controls()
        self._refresh_process_button()
        self._refresh_stop_button()

    def _refresh_settings_controls(self) -> None:
        """ Shut the choices a run reads from while it runs.

        The extraction loads the model it starts with, so a pick made during the run would name a
        model it never used. The properties step reads its preset and whether to run at all only once
        the embeddings are done, so a change made meanwhile would apply halfway through. The CUDA
        choice also stays shut where the hardware has none.
        """
        idle = not self._processing_task_running

        self._select_model_button.setEnabled(idle)
        self._cpu_radio.setEnabled(idle)
        self._cuda_radio.setEnabled(idle and self._phenome.cuda_available)
        self._props_selection_combo.setEnabled(idle)
        self._props_checkbox.setEnabled(idle)

    def _refresh_stop_button(self) -> None:
        """ Offer the stop only while there is a task to stop, and only once.

        A stop that has been asked for cannot be asked for again: it is honoured at the next report
        PhenoMe makes, and the button says so rather than taking a second click that would do nothing.
        Only a run has a stop to ask for: a checkpoint being saved holds the tab too, but cannot be
        called off halfway without leaving a broken file behind.
        """
        stop_requested = self._stop_event is not None and self._stop_event.is_set()
        stoppable = self._processing_task_running and self._stop_event is not None and not stop_requested

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

        logger.info('Stopping the running task. It stops at its next report.')
        self._stop_event.set()
        self._refresh_stop_button()

    def _refresh_process_button(self) -> None:
        """ Enable the embedding extraction button only when the data and a model are in place, and no task is running.

        No checkpoint is waited on: a run keeps its results in memory, and saving them is a step of
        its own once they are there.

        Reads the state from the interface rather than tracking it separately, so it stays correct
        whichever loading tab was used. Called after each loading step, whenever a checkpoint is come
        by, and whenever a task starts or finishes.
        """
        metadata_df = self._phenome.metadata_df
        has_data = metadata_df is not None and not metadata_df.empty
        # Selected rather than loaded: the run loads it itself
        has_model = self._phenome.params.process.model in MODEL_OPTIONS

        self._process_button.setEnabled(has_data and has_model and not self._processing_task_running)

        if self._processing_task_running:
            self._process_button.setToolTip('Please wait for the running task to finish.')
            return

        # Named in the order they are come by, so the tooltip reads as the way through the tab
        missing = []
        if not has_data:
            missing.append('load data')
        if not has_model:
            missing.append('select a model')
        self._process_button.setToolTip(f"Please {' and '.join(missing)} first." if missing else '')

    def _on_save_checkpoint_pressed(self) -> None:
        """ Write the embeddings and the properties extracted so far to a checkpoint, named from a dialog.

        A save dialog, so it puts its own question when the name it is given is already a file, and
        answering it is agreeing to replace what is there. Where the extension had to be filled in
        below, the file it lands on is one the dialog never asked about, so the question is put here.

        Written on a worker, the embeddings being as large as the dataset, with the tab held shut
        meanwhile like it is for a run: a run started during the save would replace the results being
        written.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save checkpoint', '', CHECKPOINT_FILTER
        )

        if not file_path:
            return

        named = Path(file_path)

        # Not every platform's dialog spells the filter's extension out onto the name, and a
        # checkpoint named without one is a file PhenoMe will refuse to read back later
        path = named if named.suffix else named.with_suffix(CHECKPOINT_SUFFIX)

        if path != named and path.is_file() and not self._confirm_replace(path):
            logger.info("Saving was called off, leaving the file at '%s' as it was.", path)
            return

        worker = self._save_checkpoint_worker(path)
        worker.returned.connect(self._on_checkpoint_saved)
        worker.errored.connect(lambda error: self._on_save_failed(path, error))

        # No stop for a save: it cannot be called off halfway without leaving a broken file behind
        self._stop_event = None
        self._set_processing_task_running(True)

        # PhenoMe writes the file in one go, with nothing to report along the way
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat('Saving checkpoint...')

        worker.start()

    def _confirm_replace(self, path: Path) -> bool:
        """ Ask whether a file the save dialog never asked about may be replaced.

        Args:
            path: The file the save would write over.
        Returns:
            True when the file may be replaced.
        """
        answer = QMessageBox.question(
            self,
            'Replace checkpoint',
            f'{path} already exists.\n\nReplace it?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    @thread_worker
    def _save_checkpoint_worker(self, path: Path) -> None:
        """ Create worker to write the results to a checkpoint on a parallel thread.

        Args:
            path: The checkpoint file to write.
        """
        self._phenome.save_checkpoint(path)

    def _on_checkpoint_saved(self) -> None:
        """ Give the tab back once the checkpoint is written, the label now naming it. """
        self._set_processing_task_running(False)
        self._settle_bar('Checkpoint saved')

    def _on_save_failed(self, path: Path, error: Exception) -> None:
        """ Give the tab back when the checkpoint could not be written, and say why.

        The previous checkpoint stays selected, the interface only adopting a file once it is written.

        Args:
            path: The checkpoint file that was to be written.
            error: The exception the save raised, as the worker caught it.
        """
        self._set_processing_task_running(False)
        self._settle_bar('Failed')

        logger.error("Could not save the checkpoint at '%s': %s", path, error)
        QMessageBox.critical(
            self, 'Checkpoint saving failed', f'Could not save the checkpoint at {path}.\n\n{error}'
        )

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
            path: The checkpoint file to read.
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

    def _refresh_checkpoint(self) -> None:
        """ Say which checkpoint is current, and offer the two ways to one only once each has what it needs.

        Loading resolves the paths a checkpoint stores against the images currently found, so it waits
        on a dataset. Saving writes out the embeddings extracted, so it waits on them, and on any
        running task, which would be replacing what is being written.
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

        self._load_checkpoint_button.setEnabled(has_data)
        self._load_checkpoint_button.setToolTip('' if has_data else 'Please load data first.')

        has_embeddings = self._phenome.has_processed_images
        self._save_checkpoint_button.setEnabled(has_embeddings and not self._processing_task_running)
        if self._processing_task_running:
            self._save_checkpoint_button.setToolTip('Please wait for the running task to finish.')
        else:
            self._save_checkpoint_button.setToolTip('' if has_embeddings else 'Please extract embeddings first.')

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
            The parameter form, parented to the panel so it closes with it.
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
        """ Begins the processing task: the embedding extraction, then the properties when they are asked for.

        With lazy checkpointing on, the checkpoint is read before anything is started: PhenoMe resumes
        from the one it is pointed at, and raises on one it cannot resume from, which is a long way into
        a run to find out. Asked here, the same disagreement is put to the user while the run can still
        be called off. Without it the run writes to no checkpoint, and there is nothing to ask.

        Only the disagreement is asked about. A checkpoint that agrees is handed to PhenoMe as it is,
        which resumes it - processing the images it does not yet hold, or loading it and processing
        nothing at all when it already holds them all.

        Before either, a run over embeddings already there - extracted here or read from a checkpoint -
        asks whether they may be replaced, as the analysis results computed from them go with them.
        Those are only dropped once every question has been answered yes, so a run called off at any
        of them leaves everything as it was.
        """
        replacing = self._phenome.has_processed_images
        if replacing and not self._confirm_replace_results():
            logger.info('Processing was called off, leaving the current results as they were.')
            return

        mismatch = self._phenome.inspect_checkpoint()

        force_reprocess = False
        if mismatch is not None:
            if not self._confirm_overwrite(mismatch):
                logger.info('Extraction was called off, leaving the checkpoint as it was.')
                return
            force_reprocess = True

        # Computed from the embeddings the run replaces, so they would answer for none it produces
        if replacing:
            self._phenome.clear_analysis()
            self.analysis_cleared.emit()

        self._stop_event = threading.Event()
        self._computing_properties = False

        # A preset changed since the last run is forced onto the properties step, which would
        # otherwise resume the properties the checkpoint holds under the preset before it

        worker = self._process_images_worker(force_reprocess, self._preset_changed)
        worker.returned.connect(self._on_finish_process)
        worker.errored.connect(self._on_process_failed)

        # Read before the run is let go of, the worker being what loads the model and settles this
        status = 'Loading model...' if self._phenome.needs_model_load else 'Processing images...'

        # Put the tab in its running state before the run is let go of, rather than after: the lock is
        # what offers the stop, and the run is what the stop is offered on.
        self._process_button.setText(status)
        self._set_processing_task_running(True)

        # Neither the download nor PhenoMe, while it builds the dataloader and scans the checkpoint,
        # reports anything, so show a busy indicator until the first batch is reported.
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat('Loading model...' if status == 'Loading model...' else 'Preparing...')

        worker.start()

    def _on_model_ready(self) -> None:
        """ Say the run has moved on from its model to the images, once the worker has the model.

        Runs on the main thread via the model_ready signal. Repeats the state the click left where no
        load was needed, which is harmless. The embeddings group is outlined as waiting until they are
        extracted.
        """
        self._embeddings_group.setStyleSheet(GROUP_STYLESHEETS['STYLE_WAIT'])
        self._process_button.setText('Processing images...')
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat('Preparing...')

    def _on_properties_started(self) -> None:
        """ Say the run has moved on from the embeddings to the properties.

        Runs on the main thread via the properties_started signal, which lands before the first
        property report and before any failure of the step, so the end of the task is read against it.
        The embeddings are done by then, and their group is outlined as such; the properties group is
        outlined as waiting until they are computed.
        """
        self._embeddings_group.setStyleSheet(GROUP_STYLESHEETS['STYLE_OK'])
        self._properties_group.setStyleSheet(GROUP_STYLESHEETS['STYLE_WAIT'])
        self._computing_properties = True
        self._process_button.setText('Calculating properties...')

        # PhenoMe stays silent while it resolves the image paths and scans the checkpoint, so show a
        # busy indicator until the first image is reported.
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat('Preparing...')

    def _confirm_replace_results(self) -> bool:
        """ Ask whether the embeddings already there, and the results computed from them, may be replaced.

        Asked with the refusal as its default answer, the way loading another dataset over the
        current results is asked.

        Returns:
            True when the run may replace them.
        """
        answer = QMessageBox.question(
            self,
            'Overwrite results?',
            'Processing the images again will overwrite the current embeddings and properties, and discard all results.\n\nProceed?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_overwrite(self, mismatch: CheckpointMismatch) -> bool:
        """ Ask whether a checkpoint that answers for another run may be deleted and filled again.

        The only way past a checkpoint PhenoMe will not resume from, short of naming another file: it
        removes the file rather than writing over it, so a long extraction is lost with it, and the
        properties in the same file go with it. Asked with the refusal as its default answer, the way
        dropping the loaded results is asked.

        Args:
            mismatch: What the checkpoint holds that this run would not put there.
        Returns:
            True when the checkpoint may be deleted and the run made from scratch.
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
        self._process_button.setText('Process Images')
        self._set_processing_task_running(False)

        # The preset now answers for what the checkpoint holds, this run having just computed it
        if self._computing_properties:
            self._preset_changed = False

        # A warm checkpoint can make PhenoMe return without ever reporting progress, so settle the bar
        # here instead of relying on a final callback.
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(1)
        self._progress_bar.setFormat('Done!')

    def _on_process_failed(self, error: Exception) -> None:
        """ Reopen the controls when the task raised, so another run can be attempted.

        The worker swallows the exception into this signal, and the controls are held shut for the
        duration of a task: with nobody listening, a raise leaves them shut for good and the tab has to
        be restarted to be used again. PhenoMe raises here for a checkpoint it cannot resume from,
        among other things, which the run before it was asked about but which can still arrive from a
        checkpoint written between the question and the run.

        Read against the step the task had reached. A failed properties step leaves the changed preset
        standing, this run not having computed it.

        A stop arrives the same way, the run being called off by raising out of the progress callback,
        and is not reported as a failure: it was asked for, and it is picked up by asking for the run
        again. A stop in the properties step answers for the preset differently: PhenoMe clears the
        properties the checkpoint holds before it computes the first one, so a run that got as far as
        reporting has already replaced them with the preset it was started under, and saying otherwise
        would have the next run clear the partial properties it is meant to carry on from.

        Args:
            error: The exception the task raised, as the worker caught it.
        """
        self._process_button.setText('Process Images')
        self._set_processing_task_running(False)

        # The step the run was on did not finish, whether it was stopped or failed. Outlined after the
        # unlock, which outlines the groups from what the pipeline holds.
        unfinished = self._properties_group if self._computing_properties else self._embeddings_group
        unfinished.setStyleSheet(GROUP_STYLESHEETS['STYLE_ERROR'])

        if isinstance(error, TaskStopped):
            self._settle_bar('Stopped')
            if self._computing_properties:
                self._preset_changed = False
                logger.info('The properties calculation was stopped.')
            else:
                logger.info('The extraction was stopped.')
            return

        self._settle_bar('Failed')

        if self._computing_properties:
            logger.error('Could not compute the properties: %s', error)
            QMessageBox.critical(self, 'Properties failed', f'Could not compute the properties.\n\n{error}')
        else:
            logger.error('Could not extract the embeddings: %s', error)
            QMessageBox.critical(self, 'Extraction failed', f'Could not extract the embeddings.\n\n{error}')

    def _settle_bar(self, status: str) -> None:
        """ Take the progress bar out of the busy state a task that did not run to its end left it spinning in.

        Args:
            status: What the bar is to read, naming how the task ended.
        """
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat(status)

    @thread_worker
    def _process_images_worker(self, force_reprocess: bool = False, force_update: bool = False) -> None:
        """ Create worker to load the selected model, process images and compute their properties on a parallel thread.

        The model is loaded first, unless the one held already answers for the selected model and
        device: a download of some hundreds of megabytes, which the button and the bar name until
        model_ready says it is done. A failed load raises out of the worker before that, like a failed
        extraction, and is reported the same way.

        The properties follow the embeddings when they are asked for, read from the parameters only
        once the extraction is done. A stopped or failed extraction raises before them, so they are
        never computed over embeddings that are not all there.

        PhenoMe reports after every batch, and after every image for the properties, which the
        throttle thins out before it reaches the bar: a report costs a hop to the main thread and a
        repaint, and the step is only worth taking when it is big enough to show. The stop is read
        outside the throttle, every report being a chance to take it and only a hundredth of them
        reaching the bar.

        Args:
            force_reprocess: Whether to delete the checkpoint and process every image again,
                which is only ever passed for a checkpoint the user has been asked about.
            force_update: Whether to drop the properties the checkpoint holds and compute them
                again, which a preset changed since the last run needs. The embeddings in the same
                file are left where they are.
        """
        # No widget is touched from here: this runs off the GUI thread, where Qt cannot restyle or
        # repaint one. Each step is announced through a signal, and the panel shows it on its own thread.
        self._phenome.ensure_model_loaded()
        self.model_ready.emit()

        self._phenome.process_images(
            progress_callback=StoppableProgress(
                ThrottledProgress(self.progress_updated.emit), self._stop_event
            ),
            force_reprocess=force_reprocess
        )

        if not self._phenome.params.properties.compute:
            return

        self.properties_started.emit()
        self._phenome.compute_properties(
            progress_callback=StoppableProgress(
                ThrottledProgress(self.progress_updated.emit), self._stop_event
            ),
            force_update=force_update
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

    def _on_properties_checkbox_changed(self) -> None:
        """ Update the properties calculation parameter when the checkbox is toggled.

        A changed preset is noted, as the next run has to be told to compute its properties again
        rather than resume the ones the checkpoint holds, which were computed under the preset picked
        before it. Toggling the checkbox is the whole of the signal, so nothing has to be asked.
        """
        self._phenome.params.properties.compute = self._props_checkbox.isChecked()
        self._preset_changed = True

    def _refresh_groups(self) -> None:
        """ Update the style of the embedding and properties groups to reflect their state.

        This is mainly used for loading an existing checkpoint, or when a new dataset is loaded and results are cleared.
        If embeddings or properties are present, the group is styled with a green border. Otherwise, the default style is used.
        """

        if self._phenome.has_processed_images:
            self._embeddings_group.setStyleSheet(GROUP_STYLESHEETS['STYLE_OK'])
        else:
            self._embeddings_group.setStyleSheet("")

        if self._phenome.has_properties:
            self._properties_group.setStyleSheet(GROUP_STYLESHEETS['STYLE_OK'])
        else:
            self._properties_group.setStyleSheet("")
