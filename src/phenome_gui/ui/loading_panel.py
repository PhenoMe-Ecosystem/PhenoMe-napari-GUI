import logging
from collections.abc import Callable

import pandas as pd
from magicgui.widgets import create_widget
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from ..utils.image_viewer import ImageViewer
from ..utils.phenome_interface import PhenoMeInterface
from .table_dock import TableDockArea

logger = logging.getLogger("main_logger")


class LoadingPanel(QWidget):
    """ The data loading tab of the PhenoMe widget.

    Holds the metadata structure choice, the file dialogs feeding the loading parameters, the load button
    and the table showing the loaded metadata.
    """

    # Emitted whenever the metadata table has been rewritten, so the steps running on the loaded data can
    # refresh. Fires on a failed load too, as that clears the table and closes those steps back down.
    data_loaded = Signal()

    # Emitted once a load has dropped everything computed from the dataset before it, so whoever shows
    # those results can let go of them. Kept apart from data_loaded, which is also emitted when the table
    # is redrawn from the metadata a step has just added to: that is not a load, and drops nothing.
    results_cleared = Signal()

    def __init__(
        self,
        phenome: PhenoMeInterface,
        image_viewer: ImageViewer,
        parent: QWidget | None = None
    ) -> None:
        """ Build the panel and its widgets.

        Args:
            phenome: The interface every call to PhenoMe goes through.
            image_viewer: The viewer the selected image is displayed in.
            parent: The widget the panel is nested in.
        """
        super().__init__(parent)
        self._phenome = phenome
        self._image_viewer = image_viewer

        self._build_ui()

    def _build_ui(self) -> None:
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
            label='Images Directory...', annotation=str, widget_type='FileEdit', options={'mode': 'd', 'value':str(self._phenome.params.paths.images)}
        )
        self._mask_dir_dialog = create_widget(
            label='Masks Directory...', annotation=str, widget_type='FileEdit', options={'mode': 'd', 'value':str(self._phenome.params.paths.masks)}
        )
        self._metadata_dir_dialog = create_widget(
            label='Metadata Directory', annotation=str, widget_type='FileEdit', options={'mode':'r', 'filter':'CSV Files (*.csv)', 'value':str(self._phenome.params.paths.metadata)}
        )

        # path-based metadata widgets
        self._path_image_dir_dialog = create_widget(
            label='Images Directory...', annotation=str, widget_type='FileEdit',
            options={'mode': 'd', 'value':self._phenome.params.paths.images}
        )
        self._path_mask_dir_dialog = create_widget(
            label='Masks Directory...', annotation=str, widget_type='FileEdit',
            options={'mode': 'd', 'value':self._phenome.params.paths.masks}
        )
        self._path_template = create_widget(
            label='Template', annotation=str, widget_type='LineEdit', options={'value':self._phenome.params.paths.template}
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

        # Data table to view metadata, in a dock area it can be popped out of and back into
        self._table_area = TableDockArea('Metadata table', self)
        self._table_dock = self._table_area.dock

        self._metadata_table = create_widget(
            label='Metadata Table', widget_type='Table', options={'value': None}
        )
        self._metadata_table.native.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers) # Prevent editing
        self._metadata_table.native.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._metadata_table.native.selectionModel().selectionChanged.connect(self._on_table_selection_changed)

        self._table_area.setWidget(self._metadata_table.native)
        layout.addWidget(structure_group, 0)
        layout.addWidget(loading_group, 0)
        layout.addWidget(self._table_area, 1)

        self._on_directory_changed()
        self.setLayout(layout)

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
        self._image_dir_dialog.value = str(self._phenome.params.paths.images)
        self._mask_dir_dialog.value = str(self._phenome.params.paths.masks)
        self._path_image_dir_dialog.value = str(self._phenome.params.paths.images)
        self._path_mask_dir_dialog.value = str(self._phenome.params.paths.masks)
        self._metadata_dir_dialog.value = str(self._phenome.params.paths.metadata)

        # Change page
        page = 0 if self._csv_radio.isChecked() else 1
        self._loading_stack.setCurrentIndex(page)

    def _on_directory_changed(self) -> None:
        """ Update the parameters when a directory is changed. """
        if self._csv_radio.isChecked():
            self._phenome.params.paths.images = self._image_dir_dialog.value
            self._phenome.params.paths.masks = self._mask_dir_dialog.value
            self._phenome.params.paths.metadata = self._metadata_dir_dialog.value
        else:
            self._phenome.params.paths.images = self._path_image_dir_dialog.value
            self._phenome.params.paths.masks = self._path_mask_dir_dialog.value
            self._phenome.params.paths.template = self._path_template.value

    def _on_load_data_csv(self) -> None:
        """ Load the data using a .csv metadata file and show it in the table. """
        self._load_data(self._phenome.load_data_csv)

    def _on_load_data_path(self) -> None:
        """ Load the data using path-based metadata and show it in the table. """
        self._load_data(self._phenome.load_data_path)

    def _load_data(self, load: Callable[[], pd.DataFrame | None]) -> None:
        """ Ask before dropping what is there, load the dataset, and show it in the table.

        The two metadata structures differ only in the call they load through, so they share this: the
        confirmation, the announcement and the table are the same either way.

        Args:
            load: The interface method loading the dataset, which drops the previous results itself.
        Raises:
            ValueError: Raised on through _update_metadata_table() when no file was found.
        """
        if not self._confirm_clearing():
            logger.info('Loading was called off, leaving the previous results as they were.')
            return

        try:
            metadata_df = load()
        finally:
            # The interface drops the previous results before it reads a single file, so whoever shows
            # them is told whether or not the load itself got as far as finding any.
            self.results_cleared.emit()

        self._update_metadata_table(metadata_df)

    def _confirm_clearing(self) -> bool:
        """ Ask whether the results of the loaded dataset may be dropped, when there are any.

        Loading a dataset drops what was computed from the one before it, an embedding run being long
        enough that losing one to a misplaced click is worth a question. Nothing is asked when there is
        nothing to lose, so the usual first load goes straight through.

        Returns:
            True when the load may go ahead.
        """
        if not self._phenome.has_results:
            return True

        answer = QMessageBox.question(
            self,
            "Load new data?",
            "Loading a new dataset will discard all current results. \n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        return answer == QMessageBox.StandardButton.Yes

    def refresh_table(self) -> None:
        """ Redraw the table from the metadata currently held by the interface.

        Lets the other panels show what they added to the metadata, the cluster labels being the one case
        so far. Reads the interface rather than taking a dataframe, so the caller needs to know nothing
        about how the table is filled.
        """
        self._update_metadata_table(self._phenome.get_metadata_table())

    def _update_metadata_table(self, metadata_df: pd.DataFrame | None) -> None:
        """ Update the metadata table in the GUI with the given metadata dataframe.

        Args:
            metadata_df: The metadata dataframe to display in the table.
        Raises:
            ValueError: If the dataframe is None or empty, the table being cleared beforehand.
        """
        if metadata_df is None or metadata_df.empty:
            self._metadata_table.value = pd.DataFrame()  # Clear the table if no data is loaded
            self.data_loaded.emit()
            raise ValueError("Could not load data. Please check the provided paths and files.")
        else:
            self._metadata_table.value = metadata_df
            self.data_loaded.emit()

    def _on_table_selection_changed(self) -> None:
        """ Display corresponding image and mask when a row is selected from the table. """
        row = self._metadata_table.native.currentRow()
        if row < 0:
            return

        image_paths, mask_paths, row_data = self._phenome.get_row_display_data(row)
        self._image_viewer.display_images(image_paths, mask_paths, row_data, names=[f"Image {row}"])
