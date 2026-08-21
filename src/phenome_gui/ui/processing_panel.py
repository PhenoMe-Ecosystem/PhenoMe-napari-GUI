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
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..registries.models import MODEL_OPTIONS
from ..utils.params import PRESET_CHOICES
from ..utils.phenome_interface import PhenoMeInterface

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

        For now, the advanced parameters include batch size only.
        TODO: Think about which parameters should be here. Some parameters would probably instead be tied to the model used (resize, force_rgb, etc)?
        """
        if self._advanced_parameters_dialog is None:
            self._advanced_parameters_dialog = self._build_advanced_parameters_dialog()

        self._batch_size_spinbox.setValue(self._phenome.params.process.batch_size)

        if self._advanced_parameters_dialog.exec_():
            self._phenome.params.process.batch_size = self._batch_size_spinbox.value()

    def _build_advanced_parameters_dialog(self) -> QDialog:
        """ Build the advanced processing parameters window.

        Only builds the form, without filling it. The fields are loaded from the parameters by the
        caller, which also reads them back when the form is confirmed.

        Returns:
            QDialog: The parameter form, parented to the panel so it closes with it.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Processing parameters")
        dialog.setFixedSize(300, 100)

        form_layout = QFormLayout()

        # Number of images PhenoMe feeds to the model at once
        self._batch_size_spinbox = QSpinBox()
        self._batch_size_spinbox.setRange(1, 1024)
        form_layout.addRow("Batch size", self._batch_size_spinbox)

        # OK closes the form with an accepted result, which is what tells the caller to apply the values
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        form_layout.addRow(button_box)

        dialog.setLayout(form_layout)
        return dialog

    def _on_process_clicked(self) -> None:
        """ Begins the image processing/embedding extraction task """
        worker = self._process_images_worker()
        worker.returned.connect(self._on_finish_process)
        worker.start()

        self._process_button.setText('Processing images...')
        self._set_processing_task_running(True)

        # PhenoMe stays silent while it builds the dataloader and scans the checkpoint, so show a busy
        # indicator until the first batch is reported.
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat('Preparing...')

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

    @thread_worker
    def _process_images_worker(self) -> None:
        """ Create worker to process images on a parallel thread. """
        self._phenome.process_images(progress_callback=self.progress_updated.emit)

    def _on_properties_changed(self) -> None:
        """ Update the properties preset parameter when the selection changes. """
        preset = self._props_selection_combo.currentText()
        self._phenome.params.properties.preset = preset

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
        worker = self._compute_properties_worker()
        worker.returned.connect(self._on_finish_properties)
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

        # A warm checkpoint can make PhenoMe return without ever reporting progress, so settle the bar
        # here instead of relying on a final callback.
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(1)
        self._progress_bar.setFormat('Done!')

    @thread_worker
    def _compute_properties_worker(self) -> None:
        """ Create worker to compute the properties on a parallel thread. """
        self._phenome.compute_properties(progress_callback=self.progress_updated.emit)
