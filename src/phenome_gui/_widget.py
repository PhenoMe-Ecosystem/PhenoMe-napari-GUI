import logging
from typing import TYPE_CHECKING

import pandas as pd
from magicgui.widgets import create_widget
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
            raise ValueError("Could not load data. Please check the provided paths and files.")
        else:
            self._metadata_table.value = metadata_df

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

        # Group for properties calculation
        properties_group = QGroupBox("Properties")

        # Progress bar
        self._progress_bar = QProgressBar()


        layout.addWidget(embeddings_group)
        layout.addWidget(properties_group)
        layout.addWidget(self._progress_bar)
        self._tab_process.setLayout(layout)

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
        self._phenome.load_model()

    def _build_analysis_tab(self) -> None:
        """ Build the analysis tab and its widgets.

        Args:
            None
        Returns:
            None
        """
        layout = QVBoxLayout()
        self._tab_analysis.setLayout(layout)
