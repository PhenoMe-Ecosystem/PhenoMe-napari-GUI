import logging
from typing import TYPE_CHECKING

import pandas as pd
from magicgui.widgets import create_widget
from napari.qt.threading import thread_worker
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStackedLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

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
        Returns:
            None
        """
        super().__init__()
        self._viewer = viewer

        # Interface to PhenoMe, holding the pipeline object and its parameters
        self._phenome = PhenoMeInterface(seed=42)
        self._image_viewer = ImageViewer(viewer)

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

        Args:
            None
        Returns:
            None
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
            label='Images Directory...', annotation=str, widget_type='FileEdit', options={'label':'test', 'mode': 'd', 'value':self._phenome.get_path('images')}
        )
        self._mask_dir_dialog = create_widget(
            label='Masks Directory...', annotation=str, widget_type='FileEdit', options={'mode': 'd', 'value':self._phenome.get_path('masks')}
        )
        self._metadata_dir_dialog = create_widget(
            label='Metadata Directory', annotation=str, widget_type='FileEdit', options={'mode':'r', 'filter':'CSV Files (*.csv)', 'value':self._phenome.get_path('metadata')}
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
        self._tab_loading.setLayout(layout)

    def _build_loading_group(self) -> None:
        """ Builds the parameter widgets for the two possible data structures.

        Args:
            None
        Returns:
            None
        """

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
        """ Update the displayed page of the loading stack when the metadata structure selection changes.

        Args:
            None
        Returns:
            None
        """
        # Change text of corresponding widgets
        self._image_dir_dialog.value = self._phenome.get_path('images')
        self._mask_dir_dialog.value = self._phenome.get_path('masks')
        self._path_image_dir_dialog.value = self._phenome.get_path('images')
        self._path_mask_dir_dialog.value = self._phenome.get_path('masks')

        # Change page
        page = 0 if self._csv_radio.isChecked() else 1
        self._loading_stack.setCurrentIndex(page)

    def _on_directory_changed(self) -> None:
        """ Update the parameters when a directory is changed.

        Args:
            None
        Returns:
            None
        """
        if self._csv_radio.isChecked():
            self._phenome.set_path('images', self._image_dir_dialog.value)
            self._phenome.set_path('masks', self._mask_dir_dialog.value)
            self._phenome.set_path('metadata', self._metadata_dir_dialog.value)
        else:
            self._phenome.set_path('images', self._path_image_dir_dialog.value)
            self._phenome.set_path('masks', self._path_mask_dir_dialog.value)
            self._phenome.set_path('template', self._path_template.value)

    def _on_load_data_csv(self) -> None:
        """ Load the data using a .csv metadata file and show it in the table.

        Args:
            None
        Returns:
            None
        """
        self._update_metadata_table(self._phenome.load_data_csv())

    def _on_load_data_path(self) -> None:
        """ Load the data using path-based metadata and show it in the table.

        Args:
            None
        Returns:
            None
        """
        self._update_metadata_table(self._phenome.load_data_path())

    def _update_metadata_table(self, metadata_df: pd.DataFrame | None) -> None:
        """ Update the metadata table in the GUI with the given metadata dataframe.

        Args:
            metadata_df (pd.DataFrame): The metadata dataframe to display in the table.
        Returns:
            None
        Raises:
            ValueError: If the dataframe is None or empty, the table being cleared beforehand.
        """
        if metadata_df is None or metadata_df.empty:
            self._metadata_table.value = pd.DataFrame()  # Clear the table if no data is loaded
            self._refresh_process_button()
            raise ValueError("Could not load data. Please check the provided paths and files.")
        else:
            self._metadata_table.value = metadata_df
            self._refresh_process_button()

    def _on_table_selection_changed(self) -> None:
        """ Display corresponding image and mask when a row is selected from the table.

        Args:
            None
        Returns:
            None
        """
        row = self._metadata_table.native.currentRow()
        if row < 0:
            return

        image_paths, mask_paths, row_data = self._phenome.get_row_display_data(row)
        self._image_viewer.display_images(image_paths, mask_paths, row_data, names=[f"Image {row}"])

    def _build_process_tab(self) -> None:
        """ Build the image processing tab and its widgets.

        Includes widgets to select processing and properties parameters and running the embedding task and the properties calculation.

        Args:
            None
        Returns:
            None
        """
        layout = QVBoxLayout()

        # Group for embedding extraction
        embeddings_group = QGroupBox("Embeddings")
        embeddings_layout = QVBoxLayout()

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
        model_row.addWidget(QLabel("Model"))
        self._model_selection_combo = QComboBox()
        self._model_selection_combo.addItems(MODEL_OPTIONS.keys())
        self._model_selection_combo.setCurrentText(self._phenome.get_param('process', 'model'))
        self._model_selection_combo.currentTextChanged.connect(self._on_model_selection_changed)
        model_row.addWidget(self._model_selection_combo)
        embeddings_layout.addLayout(model_row)
        embeddings_group.setLayout(embeddings_layout)
        # Load model button
        self._load_model_button = QPushButton("Load Model")
        self._load_model_button.clicked.connect(self._on_load_model_clicked)
        embeddings_layout.addWidget(self._load_model_button)

        # Checkpoint path selection
        self._checkpoint_path_dialog = create_widget(
            label='Checkpoint path', annotation=str, widget_type='FileEdit',
            options={'mode': 'w', 'filter': 'HDF5 Files (*.hdf5)','value':self._phenome.get_path('checkpoint')}
        )
        self._checkpoint_path_dialog.choose_btn.text = 'Checkpoint path...'
        self._checkpoint_path_dialog.changed.connect(self._on_checkpoint_path_changed)
        embeddings_layout.addWidget(self._checkpoint_path_dialog.native)

        # Advanced parameters button
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

        # Progress bar, driven from the worker thread through the progress_updated signal
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat('')
        self.progress_updated.connect(self._on_progress)

        layout.addWidget(embeddings_group)
        layout.addWidget(properties_group)
        layout.addWidget(self._progress_bar)
        self._tab_process.setLayout(layout)

    def _on_device_changed(self) -> None:
        """ Update the device parameter when the selection is changed.

        Args:
            None
        Returns:
            None
        """
        device = 'cpu' if self._cpu_radio.isChecked else 'CUDA'
        self._phenome.set_param('process', 'device', device)

    def _on_model_selection_changed(self) -> None:
        """ Update the model parameter when the selection changes.

        Args:
            None
        Returns:
            None
        """
        model_name = self._model_selection_combo.currentText()
        self._phenome.set_param('process', 'model', model_name)

    def _on_load_model_clicked(self) -> None:
        """ Load the model for embedding extraction when the button is clicked.

        Args:
            None
        Returns:
            None
        """
        worker = self._load_model_worker()
        worker.returned.connect(self._finish_load_model)
        worker.start()

        self._load_model_button.setText('Downloading model...')
        self._load_model_button.setEnabled(False)

    def _finish_load_model(self) -> None:
        """ Called when model is finished loading. Re-enables the button and changes text, and displays which model is loaded.

        Args:
            None
        Returns:
            None
        """
        self._load_model_button.setText('Load Model')
        self._load_model_button.setEnabled(True)
        self._refresh_process_button()

    def _refresh_process_button(self) -> None:
        """ Enable the embedding extraction button only when both data and a model are loaded.

        Reads the state from the interface rather than tracking it separately, so it stays correct
        whichever loading tab was used. Called after each loading step and once processing finishes.

        Args:
            None
        Returns:
            None
        """
        metadata_df = self._phenome.metadata_df
        has_data = metadata_df is not None and not metadata_df.empty
        has_model = self._phenome.model is not None

        self._process_button.setEnabled(has_data and has_model)

        missing = []
        if not has_data:
            missing.append('load data')
        if not has_model:
            missing.append('load a model')
        self._process_button.setToolTip(f"Please {' and '.join(missing)} first." if missing else '')

    @thread_worker
    def _load_model_worker(self) -> None:
        """ Create worker to load model on a parallel thread.

        Args:
            None
        Returns:
            None
        """
        self._phenome.load_model()

    def _on_checkpoint_path_changed(self) -> None:
        """ Change the checkpoint path parameter when the value is changed.

        Args:
            None
        Returns:
            None
        """
        self._phenome.set_path('checkpoint', self._checkpoint_path_dialog.value)

    def _on_advanced_parameters_clicked(self) -> None:
        """ Opens a parameter form for advanced users.

        Args:
            None
        Returns:
            None
        """
        return

    def _on_process_clicked(self) -> None:
        """ Begins the image processing/embedding extraction task

        Args:
            None
        Returns:
            None
        """
        worker = self._process_images_worker()
        worker.returned.connect(self._on_finish_process)
        worker.start()

        self._process_button.setText('Processing images...')
        self._process_button.setEnabled(False)

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
        Returns:
            None
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
        """ Called when processing is done. Re-enables the button and changes text.

        Args:
            None
        Returns:
            None
        """
        self._process_button.setText('Extract Embeddings')
        self._refresh_process_button()

        # A warm checkpoint can make PhenoMe return without ever reporting progress, so settle the bar
        # here instead of relying on a final callback.
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(1)
        self._progress_bar.setFormat('Done!')

    @thread_worker
    def _process_images_worker(self) -> None:
        """ Create worker to process images on a parallel thread.

        Args:
            None
        Returns:
            None
        """
        self._phenome.process_images(progress_callback=self.progress_updated.emit)

    def _build_analysis_tab(self) -> None:
        """ Build the analysis tab and its widgets.

        Args:
            None
        Returns:
            None
        """
        layout = QVBoxLayout()
        self._tab_analysis.setLayout(layout)
