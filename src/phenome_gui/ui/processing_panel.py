import logging

from magicgui.widgets import create_widget
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
from ..utils.params import MAX_WORKERS, PRESET_CHOICES
from ..utils.phenome_interface import PhenoMeInterface
from ..utils.progress import ThrottledProgress

logger = logging.getLogger("main_logger")


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

        self._build_ui()

    def _build_ui(self) -> None:
        """ Build the image processing tab and its widgets.

        Includes widgets to select processing and properties parameters and running the embedding task and the properties calculation.
        """
        layout = QVBoxLayout()

        # Checkpoint path selection, used to save both embeddings AND properties
        self._checkpoint_path_dialog = create_widget(
            label='Checkpoint path', annotation=str, widget_type='FileEdit',
            options={'mode': 'w', 'filter': 'HDF5 Files (*.hdf5)','value':self._phenome.params.paths.checkpoint}
        )
        self._checkpoint_path_dialog.choose_btn.text = 'Checkpoint path...'
        self._checkpoint_path_dialog.changed.connect(self._on_checkpoint_path_changed)
        layout.addWidget(self._checkpoint_path_dialog.native, 0)

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

        # Model selection combo box
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model to load"))
        self._model_selection_combo = QComboBox()
        self._model_selection_combo.addItems(MODEL_OPTIONS.keys())
        self._model_selection_combo.setCurrentText(self._phenome.params.process.model)
        self._model_selection_combo.currentTextChanged.connect(self._on_model_selection_changed)
        model_row.addWidget(self._model_selection_combo)
        embeddings_layout.addLayout(model_row)

        # Current loaded model
        current_model_row = QHBoxLayout()
        current_model_row.addWidget(QLabel("Loaded model:"))
        self._current_model_label = QLabel("None")
        self._current_model_label.setAlignment(Qt.AlignRight)
        current_model_row.addWidget(self._current_model_label)
        embeddings_layout.addLayout(current_model_row)

        # Load model button
        self._load_model_button = QPushButton("Load Model")
        self._load_model_button.clicked.connect(self._on_load_model_clicked)
        embeddings_layout.addWidget(self._load_model_button)

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

        # Progress bar, driven from the worker thread through the progress_updated signal
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat('')
        self.progress_updated.connect(self._on_progress)

        layout.addWidget(embeddings_group,1)
        layout.addWidget(properties_group,1)
        layout.addStretch(1)
        layout.addWidget(self._progress_bar)
        self.setLayout(layout)

    def _on_device_changed(self) -> None:
        """ Update the device parameter when the selection is changed. """
        device = 'cpu' if self._cpu_radio.isChecked() else 'cuda'
        self._phenome.params.process.device = device

    def _on_model_selection_changed(self) -> None:
        """ Update the model parameter when the selection changes. """
        model_name = self._model_selection_combo.currentText()
        self._phenome.params.process.model = model_name

    def _on_load_model_clicked(self) -> None:
        """ Load the model for embedding extraction when the button is clicked. """
        worker = self._load_model_worker()
        worker.returned.connect(self._finish_load_model)
        worker.errored.connect(self._on_load_model_failed)
        worker.start()

        self._load_model_button.setText('Downloading model...')
        self._load_model_button.setEnabled(False)
        self._model_selection_combo.setEnabled(False)

    def _finish_load_model(self) -> None:
        """ Called when model is finished loading. Re-enables the button and changes text, and displays which model is loaded. """
        self._load_model_button.setText('Load Model')
        self._current_model_label.setText(self._phenome.params.process.model)
        self._load_model_button.setEnabled(True)
        self._model_selection_combo.setEnabled(True)
        self._refresh_process_button()

    def _on_load_model_failed(self, error: Exception) -> None:
        """ Reopen the model controls when the load raised, so another model can be picked.

        A download that fails otherwise leaves the button and the combo shut, with the label still
        naming whichever model was loaded before it, and no way to try again.

        Args:
            error (Exception): The exception the load raised, as the worker caught it.
        """
        self._load_model_button.setText('Load Model')
        self._load_model_button.setEnabled(True)
        self._model_selection_combo.setEnabled(True)

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

        The checkpoint field is read back from the parameters rather than left as it was: the interface
        puts it back to its default when it drops the results, a checkpoint answering only for the
        images it was filled from. The loaded model is not a result and stays where it is, along with
        the label naming it.
        """
        self._checkpoint_path_dialog.value = str(self._phenome.params.paths.checkpoint)

        # The checkpoint went back to its default along with the results, so there are no properties
        # left for a changed preset to have to be computed over
        self._preset_changed = False

        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat('')

        self.refresh_buttons()

    def refresh_buttons(self) -> None:
        """ Bring both processing buttons in line with the state of the pipeline.

        Called whenever a task starts or finishes, and from the outside once data has been loaded, which
        is one of the two things the embedding extraction waits on.
        """
        self._refresh_process_button()
        self._refresh_props_button()

    def _refresh_process_button(self) -> None:
        """ Enable the embedding extraction button only when both data and a model are loaded, and no task is running.

        Reads the state from the interface rather than tracking it separately, so it stays correct
        whichever loading tab was used. Called after each loading step and whenever a task starts or finishes.
        """
        metadata_df = self._phenome.metadata_df
        has_data = metadata_df is not None and not metadata_df.empty
        has_model = self._phenome.model is not None

        self._process_button.setEnabled(has_data and has_model and not self._processing_task_running)

        if self._processing_task_running:
            self._process_button.setToolTip('Please wait for the running task to finish.')
            return

        missing = []
        if not has_data:
            missing.append('load data')
        if not has_model:
            missing.append('load a model')
        self._process_button.setToolTip(f"Please {' and '.join(missing)} first." if missing else '')

    @thread_worker
    def _load_model_worker(self) -> None:
        """ Create worker to load model on a parallel thread. """
        self._phenome.load_model()

    def _on_checkpoint_path_changed(self) -> None:
        """ Change the checkpoint path parameter when the value is changed. """
        self._phenome.params.paths.checkpoint = self._checkpoint_path_dialog.value

    def _on_advanced_parameters_clicked(self) -> None:
        """ Opens a parameter form for advanced users.

        The form is built on first use and kept afterwards, but its fields are reloaded from the current
        parameters every time it opens, so the values of a cancelled edit are never kept nor shown again.
        The parameters are only written back when the form is confirmed.

        For now, the advanced parameters include the batch size and the number of loader workers.
        TODO: Think about which parameters should be here. Some parameters would probably instead be tied to the model used (resize, force_rgb, etc)?
        """
        if self._advanced_parameters_dialog is None:
            self._advanced_parameters_dialog = self._build_advanced_parameters_dialog()

        self._batch_size_spinbox.setValue(self._phenome.params.process.batch_size)
        self._num_workers_spinbox.setValue(self._phenome.params.process.num_workers)

        if self._advanced_parameters_dialog.exec_():
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
        dialog.setFixedSize(300, 130)

        form_layout = QFormLayout()

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
        """
        mismatch = self._phenome.inspect_checkpoint()

        force_reprocess = False
        if mismatch is not None:
            if not self._confirm_overwrite(mismatch):
                logger.info('Extraction was called off, leaving the checkpoint as it was.')
                return
            force_reprocess = True

        worker = self._process_images_worker(force_reprocess)
        worker.returned.connect(self._on_finish_process)
        worker.errored.connect(self._on_process_failed)
        worker.start()

        self._process_button.setText('Processing images...')
        self._set_processing_task_running(True)

        # PhenoMe stays silent while it builds the dataloader and scans the checkpoint, so show a busy
        # indicator until the first batch is reported.
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat('Preparing...')

    def _confirm_overwrite(self, mismatch: CheckpointMismatch) -> bool:
        """ Ask whether a checkpoint that answers for another run may be deleted and filled again.

        The only way past a checkpoint PhenoMe will not resume from, short of naming another file: it
        removes the file rather than writing over it, so a long extraction is lost with it. Asked with
        the refusal as its default answer, the way dropping the loaded results is asked.

        Args:
            mismatch (CheckpointMismatch): What the checkpoint holds that this run would not put there.
        Returns:
            bool: True when the checkpoint may be deleted and the run made from scratch.
        """
        filled_with = (
            'using a different model than the one currently loaded' if mismatch.reason == MODEL_MISMATCH
            else 'from a different dataset'
        )

        # Only the dataset mismatch has anything to add: what a run's parameters disagree on goes to
        # the log, PhenoMe writing the whole comparison out there key by key itself.
        detail = f'{mismatch.detail}\n\n' if mismatch.detail else ''

        logger.warning(
            "The checkpoint at '%s' was filled %s. %s",
            self._phenome.params.paths.checkpoint, filled_with, mismatch.detail
        )

        answer = QMessageBox.warning(
            self,
            'Checkpoint does not match',
            f"The checkpoint at {self._phenome.params.paths.checkpoint} was filled {filled_with}.\n\n"
            f"{detail}"
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

        Args:
            error (Exception): The exception the extraction raised, as the worker caught it.
        """
        self._process_button.setText('Extract Embeddings')
        self._set_processing_task_running(False)
        self._settle_failed_bar()

        logger.error('Could not extract the embeddings: %s', error)
        QMessageBox.critical(self, 'Extraction failed', f'Could not extract the embeddings.\n\n{error}')

    def _settle_failed_bar(self) -> None:
        """ Take the progress bar out of the busy state a failed task left it spinning in. """
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat('Failed')

    @thread_worker
    def _process_images_worker(self, force_reprocess: bool = False) -> None:
        """ Create worker to process images on a parallel thread.

        PhenoMe reports after every batch, which the throttle thins out before it reaches the bar: a
        report costs a hop to the main thread and a repaint, and the step is only worth taking when it
        is big enough to show.

        Args:
            force_reprocess (bool): Whether to delete the checkpoint and process every image again,
                which is only ever passed for a checkpoint the user has been asked about.
        """
        self._phenome.process_images(
            progress_callback=ThrottledProgress(self.progress_updated.emit),
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
        """ Enable the properties calculation button only once the images have been processed, and no task is running.

        PhenoMe computes the properties over the images held by the pipeline, and raises when there are
        none, so the state is read from the interface. Called when building the tab and whenever a task
        starts or finishes.
        """
        has_processed_images = self._phenome.has_processed_images

        self._props_button.setEnabled(has_processed_images and not self._processing_task_running)

        if self._processing_task_running:
            self._props_button.setToolTip('Please wait for the running task to finish.')
        else:
            self._props_button.setToolTip('' if has_processed_images else 'Please extract embeddings first.')

    def _on_properties_clicked(self) -> None:
        """ Begins the properties calculation task """
        worker = self._compute_properties_worker(self._preset_changed)
        worker.returned.connect(self._on_finish_properties)
        worker.errored.connect(self._on_properties_failed)
        worker.start()

        self._props_button.setText('Calculating properties...')
        self._set_processing_task_running(True)

        # PhenoMe stays silent while it resolves the image paths and scans the checkpoint, so show a busy
        # indicator until the first image is reported.
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat('Preparing...')

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

        Args:
            error (Exception): The exception the calculation raised, as the worker caught it.
        """
        self._props_button.setText('Calculate Properties')
        self._set_processing_task_running(False)
        self._settle_failed_bar()

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
            progress_callback=ThrottledProgress(self.progress_updated.emit),
            force_update=force_update
        )
