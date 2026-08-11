import logging
from typing import TYPE_CHECKING

import pandas as pd
from magicgui.widgets import create_widget
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
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
    QSizePolicy,
    QSpinBox,
    QStackedLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .ui.clustering_params import ClusterParamsWindow
from .utils.image_viewer import ImageViewer
from .utils.models import MODEL_OPTIONS
from .utils.phenome_interface import PhenoMeInterface

logger = logging.getLogger("main_logger")
logging.basicConfig(filename='phenome_log.log', level=logging.DEBUG, filemode='w')

if TYPE_CHECKING:
    import napari

class PhenoMeWidget(QWidget):
    """ The napari dock widget of the PhenoMe plugin.

    Holds one tab per step of the workflow: data loading, processing and analysis. Every call to PhenoMe goes through the PhenoMeInterface object it instantiates.
    """

    # PhenoMe runs its progress callback on the worker thread, which must not touch widgets. Emitting
    # this signal from there hands (current, total, desc) back to the main thread through a queued
    # connection. Its signature matches phenome.utils.ProgressCallback, so `emit` is itself a valid callback.
    progress_updated = Signal(int, int, str)

    def __init__(self, viewer: 'napari.viewer.Viewer') -> None:
        """ Build the widget, its interface to PhenoMe and its three tabs.

        Args:
            viewer (napari.viewer.Viewer): The napari viewer the widget is docked in.
        """
        super().__init__()
        self._viewer = viewer

        # Interface to PhenoMe, holding the pipeline object and its parameters
        self._phenome = PhenoMeInterface(seed=42)
        self._image_viewer = ImageViewer(viewer)

        # Both processing tasks work on the same pipeline object and share the progress bar, so only one
        # of them may run at a time. Read by the button refresh methods to lock the other button.
        self._processing_task_running = False

        # Build the panel structure of the widget
        self._tabs = QTabWidget()

        # Panel 1: Data loading and table view of metadata
        self._tab_loading = QWidget()
        self._tabs.addTab(self._tab_loading, "Data loading")
        self._build_loading_tab()

        # Panel 2: Image processing and properties computing
        self._tab_process = QWidget()
        self._tabs.addTab(self._tab_process, "Processing")
        self._build_process_tab()

        # Panel 3: Analysis and plotting
        self._tab_analysis = QWidget()
        self._tabs.addTab(self._tab_analysis, "Analysis")
        self._build_analysis_tab()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._tabs)

    def _build_loading_tab(self) -> None:
        """ Builds the data loading tab and its widgets.

        Includes radio buttons to select the metadata structure, file dialogs to select the directories and files, a button to load the data, and a table to display the metadata.
        """
        layout = QVBoxLayout()

        # Group for radio buttons to select data structure
        structure_group = QGroupBox("Metadata structure")
        structure_layout = QHBoxLayout()

        self._csv_radio = QRadioButton(".CSV file")
        self._csv_radio.setChecked(True)  # Default selection
        self._path_radio = QRadioButton("Path-based")

        structure_layout.addWidget(self._csv_radio)
        structure_layout.addWidget(self._path_radio)
        structure_group.setLayout(structure_layout)

        # Group for data loading params, holding one page per metadata
        # structure and showing the one matching the checked radio button.
        loading_group = QGroupBox("Data loading")
        self._loading_stack = QStackedLayout()

        # .csv metadata widgets
        self._image_dir_dialog = create_widget(
            label='Images Directory...', annotation=str, widget_type='FileEdit', options={'mode': 'd', 'value':str(self._phenome.get_path('images'))}
        )
        self._mask_dir_dialog = create_widget(
            label='Masks Directory...', annotation=str, widget_type='FileEdit', options={'mode': 'd', 'value':str(self._phenome.get_path('masks'))}
        )
        self._metadata_dir_dialog = create_widget(
            label='Metadata Directory', annotation=str, widget_type='FileEdit', options={'mode':'r', 'filter':'CSV Files (*.csv)', 'value':str(self._phenome.get_path('metadata'))}
        )

        # path-based metadata widgets
        self._path_image_dir_dialog = create_widget(
            label='Images Directory...', annotation=str, widget_type='FileEdit',
            options={'mode': 'd', 'value':self._phenome.get_path('images')}
        )
        self._path_mask_dir_dialog = create_widget(
            label='Masks Directory...', annotation=str, widget_type='FileEdit',
            options={'mode': 'd', 'value':self._phenome.get_path('masks')}
        )
        self._path_template = create_widget(
            label='Template', annotation=str, widget_type='LineEdit', options={'value':self._phenome.get_path('template')}
        )
        self._path_template_label = QLabel('Template')

        self._image_dir_dialog.choose_btn.text = 'Images Directory...'
        self._mask_dir_dialog.choose_btn.text = 'Masks Directory...'
        self._metadata_dir_dialog.choose_btn.text = 'Metadata Path...'
        self._path_image_dir_dialog.choose_btn.text = 'Images Directory...'
        self._path_mask_dir_dialog.choose_btn.text = 'Masks Directory...'
        self._path_template_label.text = 'Template'

        self._csv_load_data_btn = QPushButton('Load Data')
        self._path_load_data_btn = QPushButton('Load Data')

        self._build_loading_group()
        loading_group.setLayout(self._loading_stack)

        self._csv_radio.toggled.connect(self._on_structure_changed)
        self._path_radio.toggled.connect(self._on_structure_changed)
        self._on_structure_changed()

        # Connect signals to update parameters when directories change
        self._image_dir_dialog.changed.connect(self._on_directory_changed)
        self._mask_dir_dialog.changed.connect(self._on_directory_changed)
        self._path_image_dir_dialog.changed.connect(self._on_directory_changed)
        self._path_mask_dir_dialog.changed.connect(self._on_directory_changed)
        self._metadata_dir_dialog.changed.connect(self._on_directory_changed)
        self._path_template.changed.connect(self._on_directory_changed)

        # Connect buttons to load data
        self._csv_load_data_btn.clicked.connect(self._on_load_data_csv)
        self._path_load_data_btn.clicked.connect(self._on_load_data_path)

        # Data table to view metadata
        self._metadata_table = create_widget(
            label='Metadata Table', widget_type='Table', options={'value': None}
        )
        self._metadata_table.native.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers) # Prevent editing
        self._metadata_table.native.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._metadata_table.native.selectionModel().selectionChanged.connect(self._on_table_selection_changed)

        layout.addWidget(structure_group, 0)
        layout.addWidget(loading_group, 0)
        layout.addWidget(self._metadata_table.native, 1)

        self._on_directory_changed()
        self._tab_loading.setLayout(layout)

    def _build_loading_group(self) -> None:
        """ Builds the parameter widgets for the two possible data structures. """

        # .csv metadata structure
        csv_page = QWidget()
        csv_loading_layout = QVBoxLayout(csv_page)
        csv_loading_layout.addWidget(self._image_dir_dialog.native)
        csv_loading_layout.addWidget(self._mask_dir_dialog.native)
        csv_loading_layout.addWidget(self._metadata_dir_dialog.native)
        csv_loading_layout.addStretch(0)
        csv_loading_layout.addWidget(self._csv_load_data_btn)

        # path-based metadata structure
        path_page = QWidget()
        path_loading_layout = QVBoxLayout(path_page)
        path_loading_layout.addWidget(self._path_image_dir_dialog.native)
        path_loading_layout.addWidget(self._path_mask_dir_dialog.native)
        template_row = QHBoxLayout()
        template_row.addWidget(self._path_template.native)
        template_row.addWidget(self._path_template_label)
        path_loading_layout.addLayout(template_row)
        path_loading_layout.addStretch(0)
        path_loading_layout.addWidget(self._path_load_data_btn)

        self._loading_stack.insertWidget(0, csv_page)
        self._loading_stack.insertWidget(1, path_page)

    def _on_structure_changed(self) -> None:
        """ Update the displayed page of the loading stack when the metadata structure selection changes. """
        # Change text of corresponding widgets
        self._image_dir_dialog.value = str(self._phenome.get_path('images'))
        self._mask_dir_dialog.value = str(self._phenome.get_path('masks'))
        self._path_image_dir_dialog.value = str(self._phenome.get_path('images'))
        self._path_mask_dir_dialog.value = str(self._phenome.get_path('masks'))
        self._metadata_dir_dialog.value = str(self._phenome.get_path('metadata'))

        # Change page
        page = 0 if self._csv_radio.isChecked() else 1
        self._loading_stack.setCurrentIndex(page)

    def _on_directory_changed(self) -> None:
        """ Update the parameters when a directory is changed. """
        if self._csv_radio.isChecked():
            self._phenome.set_path('images', self._image_dir_dialog.value)
            self._phenome.set_path('masks', self._mask_dir_dialog.value)
            self._phenome.set_path('metadata', self._metadata_dir_dialog.value)
        else:
            self._phenome.set_path('images', self._path_image_dir_dialog.value)
            self._phenome.set_path('masks', self._path_mask_dir_dialog.value)
            self._phenome.set_path('template', self._path_template.value)

    def _on_load_data_csv(self) -> None:
        """ Load the data using a .csv metadata file and show it in the table. """
        self._update_metadata_table(self._phenome.load_data_csv())

    def _on_load_data_path(self) -> None:
        """ Load the data using path-based metadata and show it in the table. """
        self._update_metadata_table(self._phenome.load_data_path())

    def _update_metadata_table(self, metadata_df: pd.DataFrame | None) -> None:
        """ Update the metadata table in the GUI with the given metadata dataframe.

        Args:
            metadata_df (pd.DataFrame): The metadata dataframe to display in the table.
        Raises:
            ValueError: If the dataframe is None or empty, the table being cleared beforehand.
        """
        if metadata_df is None or metadata_df.empty:
            self._metadata_table.value = pd.DataFrame()  # Clear the table if no data is loaded
            self._refresh_process_button()
            self._refresh_group_by_combo()
            raise ValueError("Could not load data. Please check the provided paths and files.")
        else:
            self._metadata_table.value = metadata_df
            self._refresh_process_button()
            self._refresh_group_by_combo()

    def _on_table_selection_changed(self) -> None:
        """ Display corresponding image and mask when a row is selected from the table. """
        row = self._metadata_table.native.currentRow()
        if row < 0:
            return

        image_paths, mask_paths, row_data = self._phenome.get_row_display_data(row)
        self._image_viewer.display_images(image_paths, mask_paths, row_data, names=[f"Image {row}"])

    def _build_process_tab(self) -> None:
        """ Build the image processing tab and its widgets.

        Includes widgets to select processing and properties parameters and running the embedding task and the properties calculation.
        """
        layout = QVBoxLayout()

        # Checkpoint path selection, used to save both embeddings AND properties
        self._checkpoint_path_dialog = create_widget(
            label='Checkpoint path', annotation=str, widget_type='FileEdit',
            options={'mode': 'w', 'filter': 'HDF5 Files (*.hdf5)','value':self._phenome.get_path('checkpoint')}
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
        device_layout.addWidget(self._cpu_radio)
        device_layout.addWidget(self._cuda_radio)
        device_group.setLayout(device_layout)
        embeddings_layout.addWidget(device_group)

        # Model selection combo box
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model to load"))
        self._model_selection_combo = QComboBox()
        self._model_selection_combo.addItems(MODEL_OPTIONS.keys())
        self._model_selection_combo.setCurrentText(self._phenome.get_param('process', 'model'))
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
        self._advanced_parameters_button = QPushButton("Avanced parameters...")
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
        props_row.addWidget(QLabel("Properties"))
        self._props_selection_combo = QComboBox()
        self._props_selection_combo.addItems(['Basic', 'Standard', 'Complete', 'Shape', 'Intensity'])
        self._props_selection_combo.setCurrentText(self._phenome.get_param('properties', 'preset'))
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
        self._tab_process.setLayout(layout)

    def _on_device_changed(self) -> None:
        """ Update the device parameter when the selection is changed. """
        device = 'cpu' if self._cpu_radio.isChecked else 'CUDA'
        self._phenome.set_param('process', 'device', device)

    def _on_model_selection_changed(self) -> None:
        """ Update the model parameter when the selection changes. """
        model_name = self._model_selection_combo.currentText()
        self._phenome.set_param('process', 'model', model_name)

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
        self._current_model_label.setText(self._phenome.get_param('process', 'model'))
        self._load_model_button.setEnabled(True)
        self._model_selection_combo.setEnabled(True)
        self._refresh_process_button()

    def _set_processing_task_running(self, running: bool) -> None:
        """ Mark a processing task as running or finished, and lock the buttons accordingly.

        Both processing buttons go through here, so starting either one disables the other until it is
        done. The clustering button is refreshed too, as it runs on the embeddings the tasks produce.

        Args:
            running (bool): True when a task has just been started, False once it has finished.
        """
        self._processing_task_running = running
        self._refresh_process_button()
        self._refresh_props_button()
        self._refresh_clustering_button()

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
        self._phenome.set_path('checkpoint', self._checkpoint_path_dialog.value)

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

        self._batch_size_spinbox.setValue(self._phenome.get_param('process', 'batch_size'))

        if self._advanced_parameters_dialog.exec_():
            self._phenome.set_param('process', 'batch_size', self._batch_size_spinbox.value())

    def _build_advanced_parameters_dialog(self) -> QDialog:
        """ Build the advanced processing parameters window.

        Only builds the form, without filling it. The fields are loaded from the parameters by the
        caller, which also reads them back when the form is confirmed.

        Returns:
            QDialog: The parameter form, parented to the widget so it closes with it.
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
        self._phenome.set_param('properties', 'preset', preset)

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

    def _build_analysis_tab(self) -> None:
        """ Build the analysis tab and its widgets.

        Includes a group to configure and run the clustering, and one to view the prototype and outlier
        images of a chosen metadata group.
        """
        layout = QVBoxLayout()

        # Group for the clustering step
        clustering_group = QGroupBox("Clustering")
        clustering_layout = QHBoxLayout()
        clustering_group.setLayout(clustering_layout)

        # Clustering parameters button, opening a form rebuilt on every use
        self._clustering_params_dialog = None
        self._cluster_options_button = QPushButton("Options...")
        self._cluster_options_button.clicked.connect(self._on_cluster_options_clicked)
        clustering_layout.addWidget(self._cluster_options_button)

        # Clustering button, enabled only once the images have been processed
        self._run_clustering_button = QPushButton("Compute Clustering")
        self._run_clustering_button.clicked.connect(self._on_run_clustering_clicked)
        clustering_layout.addWidget(self._run_clustering_button)
        self._refresh_clustering_button()

        # Group for the image inspection steps
        images_group = QGroupBox("View Images")
        images_layout = QVBoxLayout()
        images_group.setLayout(images_layout)

        # Metadata column the images are grouped by, filled once the data is loaded
        group_by_row = QHBoxLayout()
        group_by_row.addWidget(QLabel("Group by:"))
        self._group_by_combo = QComboBox()
        group_by_row.addWidget(self._group_by_combo)
        images_layout.addLayout(group_by_row)

        # Number of images shown for each group
        no_images_row = QHBoxLayout()
        no_images_row.addWidget(QLabel("Images per Group:"))
        self._no_images_spinbox = QSpinBox()
        self._no_images_spinbox.setRange(1, 100)
        self._no_images_spinbox.setValue(self._phenome.get_param('view_images', 'no_images'))
        self._no_images_spinbox.valueChanged.connect(self._on_no_images_changed)
        no_images_row.addWidget(self._no_images_spinbox)
        images_layout.addLayout(no_images_row)

        self._prototype_button = QPushButton("View Prototype Images")
        self._prototype_button.clicked.connect(self._on_prototype_clicked)
        images_layout.addWidget(self._prototype_button)

        self._outlier_button = QPushButton("View Outlier Images")
        self._outlier_button.clicked.connect(self._on_outlier_clicked)
        images_layout.addWidget(self._outlier_button)

        self._refresh_group_by_combo()

        layout.addWidget(clustering_group, 0)
        layout.addWidget(images_group, 0)
        layout.addStretch(1)
        self._tab_analysis.setLayout(layout)

    def _refresh_clustering_button(self) -> None:
        """ Enable the clustering button only once the images have been processed, and no task is running.

        PhenoMe clusters the embeddings held by the pipeline, and raises when there are none, so the state
        is read from the interface the same way the properties button does. Called when building the tab
        and whenever a processing task starts or finishes.
        """
        has_processed_images = self._phenome.has_processed_images

        self._run_clustering_button.setEnabled(has_processed_images and not self._processing_task_running)

        if self._processing_task_running:
            self._run_clustering_button.setToolTip('Please wait for the running task to finish.')
        else:
            self._run_clustering_button.setToolTip('' if has_processed_images else 'Please extract embeddings first.')

    def _refresh_group_by_combo(self) -> None:
        """ Fill the group by combo box with the metadata columns of the loaded data.

        The columns are only known once the data has been loaded, so the combo box is filled here rather
        than when the tab is built. Called when building the tab and after every loading step.
        """
        columns = self._phenome.get_metadata_columns()
        previous = self._group_by_combo.currentText()

        # Repopulating clears the selection, which would otherwise be reported as a user choice
        self._group_by_combo.blockSignals(True)
        self._group_by_combo.clear()
        self._group_by_combo.addItems(columns)
        if previous in columns:
            self._group_by_combo.setCurrentText(previous)
        self._group_by_combo.blockSignals(False)

        self._group_by_combo.setEnabled(bool(columns))
        self._group_by_combo.setToolTip('' if columns else 'Please load data first.')

    def _on_no_images_changed(self) -> None:
        """ Update the number of images parameter when the value changes. """
        self._phenome.set_param('view_images', 'no_images', self._no_images_spinbox.value())

    def _on_cluster_options_clicked(self) -> None:
        """ Open the clustering parameter form.

        The form is rebuilt on every use, as its fields are filled when it is constructed, so a cancelled
        edit is never kept nor shown again. The parameters are only written back when the form is confirmed.
        """
        self._clustering_params_dialog = ClusterParamsWindow(self, self._phenome.params['clustering'])

        if self._clustering_params_dialog.exec_():
            for key, value in self._clustering_params_dialog.get_params().items():
                self._phenome.set_param('clustering', key, value)

    def _on_run_clustering_clicked(self) -> None:
        """ Begin the clustering task."""
        logger.info("Compute Clustering clicked, with parameters %s", self._phenome.params['clustering'])
        worker = self._compute_clustering_worker()
        worker.returned.connect(self._on_finish_clustering)
        worker.start()

    @thread_worker
    def _compute_clustering_worker(self) -> None:
        """ Create worker to compute the properties on a parallel thread. """
        self._phenome.compute_clustering()

    def _on_finish_clustering(self) -> None:
        """ Once clustering is finished, add the cluster number to group_by options"""
        self._refresh_group_by_combo()

    def _on_prototype_clicked(self) -> None:
        """ Display the prototype images of each group.

        TODO: Run PhenoMe.find_prototypes() with cluster_col set to the selected column, and display the
        returned images through the image viewer rather than through the matplotlib figures PhenoMe plots.
        """
        logger.info("View Prototype Images clicked, grouped by '%s'", self._group_by_combo.currentText())

    def _on_outlier_clicked(self) -> None:
        """ Display the outlier images of each group.

        TODO: Run PhenoMe.detect_outliers() with group_by set to the selected column, and display the
        returned images through the image viewer rather than through the matplotlib figures PhenoMe plots.
        """
        logger.info("View Outlier Images clicked, grouped by '%s'", self._group_by_combo.currentText())
