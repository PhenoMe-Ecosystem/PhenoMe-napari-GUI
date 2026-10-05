import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from magicgui.widgets import create_widget
from napari.qt.threading import thread_worker
from napari.utils import notifications
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
from ..utils.path_checking import check_csv_file_exists, check_directory_exists
from ..utils.phenome_interface import PhenoMeInterface
from .path_widget import DirectoryWidget, FileWidget, PathWidget
from .styles import GROUP_STYLESHEETS, STYLESHEETS
from .table_dock import TableDockArea

logger = logging.getLogger("main_logger")


def _chosen(widget: PathWidget) -> Path | None:
    """ The path a box names, or None when it has been left empty.

    Args:
        widget: The path box the path is read from.
    Returns:
        The path named, or None for an empty box.
    """
    text = widget.path.strip()

    return Path(text) if text else None


def _show(widget: PathWidget, path: Path | None) -> None:
    """ Put a path in a box, leaving it empty for one that names nothing.

    Args:
        widget: The path box the path is shown in.
        path: The path to show, or None to leave the box empty - str(None) would spell out 'None'.
    """
    widget.path = '' if path is None else path


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

    # Emitted with True as a load starts on its worker thread and with False once it has finished
    # either way, so the steps that would run against the pipeline it is emptying can hold off.
    loading_changed = Signal(bool)

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
        self._structure_group = structure_group = QGroupBox("Metadata structure")
        structure_layout = QHBoxLayout()

        self._csv_radio = QRadioButton(".CSV file")
        self._csv_radio.setChecked(True)  # Default selection
        self._path_radio = QRadioButton("Path-based")

        structure_layout.addWidget(self._csv_radio)
        structure_layout.addWidget(self._path_radio)
        structure_group.setLayout(structure_layout)

        # Group for data loading params, holding one page per metadata
        # structure and showing the one matching the checked radio button.
        self._loading_group = QGroupBox("Data loading")
        self._loading_group.setObjectName("parentGroup")
        self._loading_stack = QStackedLayout()

        # .csv metadata widgets. Built empty: _on_structure_changed() fills every box from the
        # parameters below, before any of them is connected.
        self._image_dir_dialog = DirectoryWidget('Images Directory...', placeholder='Drag a folder here...')
        self._mask_dir_dialog = DirectoryWidget('Masks Directory...', placeholder='(Optional) Drag a folder here...')
        self._metadata_dir_dialog = FileWidget('Metadata Path...', file_filter='CSV Files (*.csv)', placeholder='(Optional) Drag a file here...')

        # path-based metadata widgets
        self._path_image_dir_dialog = DirectoryWidget('Images Directory...', placeholder='Drag a folder here...')
        self._path_mask_dir_dialog = DirectoryWidget('Masks Directory...', placeholder='(Optional) Drag a folder here...')
        self._path_template = create_widget(
            label='Template', annotation=str, widget_type='LineEdit', options={'value':self._phenome.params.paths.template}
        )
        self._path_template_label = QLabel('Template')

        self._path_template_label.text = 'Template'

        self._csv_load_data_btn = QPushButton('Load Data')
        self._path_load_data_btn = QPushButton('Load Data')

        self._build_loading_group()
        self._loading_group.setLayout(self._loading_stack)

        # What each page's load reads: the box, what it is called in a tooltip, what it has to be,
        # and whether it may be left empty. The images are the one thing a load needs. Defined
        # before the first call to _on_structure_changed(), which checks them. The template is a
        # pattern rather than a path, and is left unchecked.
        self._csv_paths = [
            (self._image_dir_dialog, 'images directory', check_directory_exists, True),
            (self._mask_dir_dialog, 'masks directory', check_directory_exists, False),
            (self._metadata_dir_dialog, 'metadata file', check_csv_file_exists, False),
        ]
        self._path_paths = [
            (self._path_image_dir_dialog, 'images directory', check_directory_exists, True),
            (self._path_mask_dir_dialog, 'masks directory', check_directory_exists, False),
        ]

        self._csv_radio.toggled.connect(self._on_structure_changed)
        self._path_radio.toggled.connect(self._on_structure_changed)
        self._on_structure_changed()

        # Connect signals to update parameters when the paths change
        for widget, *_ in self._csv_paths + self._path_paths:
            widget.path_changed.connect(self._on_directory_changed)

        # The template is a pattern rather than a path, so it is in neither table
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
        layout.addWidget(self._loading_group, 0)
        layout.addWidget(self._table_area, 1)

        self._on_directory_changed()
        self.setLayout(layout)

    def _build_loading_group(self) -> None:
        """ Builds the parameter widgets for the two possible data structures. """

        # .csv metadata structure
        csv_page = QWidget()
        csv_loading_layout = QVBoxLayout(csv_page)
        csv_loading_layout.addWidget(self._image_dir_dialog)
        csv_loading_layout.addWidget(self._mask_dir_dialog)
        csv_loading_layout.addWidget(self._metadata_dir_dialog)
        csv_loading_layout.addStretch(0)
        csv_loading_layout.addWidget(self._csv_load_data_btn)

        # path-based metadata structure
        path_page = QWidget()
        path_loading_layout = QVBoxLayout(path_page)
        path_loading_layout.addWidget(self._path_image_dir_dialog)
        path_loading_layout.addWidget(self._path_mask_dir_dialog)
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
        # Read before any box is written. Writing one emits its change signal, which saves every box
        # of the page back into the parameters - including the ones not shown yet, still holding
        # whatever they held before. Read as it goes, a path just typed on one page would be put
        # back by the other page's stale box the moment the page was switched.
        images = self._phenome.params.paths.images
        masks = self._phenome.params.paths.masks
        metadata = self._phenome.params.paths.metadata

        _show(self._image_dir_dialog, images)
        _show(self._mask_dir_dialog, masks)
        _show(self._path_image_dir_dialog, images)
        _show(self._path_mask_dir_dialog, masks)
        _show(self._metadata_dir_dialog, metadata)

        # Change page
        page = 0 if self._csv_radio.isChecked() else 1
        self._loading_stack.setCurrentIndex(page)

        self._refresh_load_buttons()

    def _on_directory_changed(self) -> None:
        """ Update the parameters when a directory is changed.

        A box left empty names nothing, which is what None says: PhenoMe finds the images without
        masks and without metadata, leaving out the columns each would have added.
        """
        if self._csv_radio.isChecked():
            self._phenome.params.paths.images = _chosen(self._image_dir_dialog)
            self._phenome.params.paths.masks = _chosen(self._mask_dir_dialog)
            self._phenome.params.paths.metadata = _chosen(self._metadata_dir_dialog)
        else:
            self._phenome.params.paths.images = _chosen(self._path_image_dir_dialog)
            self._phenome.params.paths.masks = _chosen(self._path_mask_dir_dialog)

            # A LineEdit reports its emptiness honestly, unlike the path boxes
            self._phenome.params.paths.template = self._path_template.value or None

        self._refresh_load_buttons()

    def _refresh_load_buttons(self) -> None:
        """ Outline every path box green or red, and offer each page's load only while its boxes are all good.

        The colour and the gate come from the one pass over the boxes, so a box cannot read green
        while the load it feeds stays shut, nor red while it opens. Each page is gated on its own
        boxes alone: the metadata file is the csv page's, and a bad one must not close the load of the
        path page, which never reads it.
        """
        for button, paths in (
            (self._csv_load_data_btn, self._csv_paths),
            (self._path_load_data_btn, self._path_paths),
        ):
            problems = self._check_page(paths)

            button.setEnabled(not problems)
            button.setToolTip(f"Please {' and '.join(problems)} first." if problems else '')

    def _check_page(self, paths: list) -> list[str]:
        """ Outline every box a page's load reads, and say what is wrong with them.

        Empty is good for a box that may be left empty, and is the one thing wrong with a required
        one.

        Args:
            paths: The page's boxes, each as (widget, label, check, required).
        Returns:
            What has to be put right before the page can load, in the order of the boxes. Empty when
            every box is good.
        """
        problems = []

        for widget, label, check, required in paths:
            path = _chosen(widget)

            if path is None:
                good = not required
                if required:
                    problems.append(f'choose an {label}')
            else:
                good = check(path)
                if not good:
                    problems.append(f'fix the {label}')

            widget.setStyleSheet(STYLESHEETS['STYLE_OK' if good else 'STYLE_ERROR'])

        return problems

    def _on_load_data_csv(self) -> None:
        """ Load the data using a .csv metadata file and show it in the table. """
        self._load_data(self._phenome.load_data_csv)

    def _on_load_data_path(self) -> None:
        """ Load the data using path-based metadata and show it in the table. """
        self._load_data(self._phenome.load_data_path)

    def _load_data(self, load: Callable[[], pd.DataFrame | None]) -> None:
        """ Ask before dropping what is there, then load the dataset on a worker thread.

        The two metadata structures differ only in the call they load through, so they share this: the
        confirmation, the waiting and the table are the same either way.

        Run off the GUI thread so the window keeps painting while the files are found: on it, nothing
        could be drawn until the load returned, the waiting outline included.

        Args:
            load: The interface method loading the dataset, which drops the previous results itself.
        """
        # Asked before anything changes colour: a load called off here leaves the tab as it was
        if not self._confirm_clearing():
            logger.info('Loading was called off, leaving the previous results as they were.')
            return

        self._set_loading(True)

        worker = self._load_data_worker(load)
        worker.returned.connect(self._on_load_finished)
        worker.errored.connect(self._on_load_failed)
        worker.start()

    @thread_worker
    def _load_data_worker(self, load: Callable[[], pd.DataFrame | None]) -> None:
        """ Create the worker loading the dataset on a parallel thread.

        Touches PhenoMe alone: the dataset found is stored on the interface, and everything shown of it
        is the finish handlers' to draw, on the GUI thread.

        Args:
            load: The interface method loading the dataset.
        """
        load()

    def _on_load_finished(self) -> None:
        """ Show the dataset just loaded, and outline the buttons by whether any file was found. """
        # The interface drops the previous results before it reads a single file, so whoever shows
        # them is told whether or not the load itself got as far as finding any.
        self.results_cleared.emit()

        found = self._update_metadata_table(self._phenome.metadata_df)
        self._set_loading(False, GROUP_STYLESHEETS['STYLE_OK' if found else 'STYLE_ERROR'])

    def _on_load_failed(self, error: Exception) -> None:
        """ Say why a load raised, and leave the tab ready for another.

        Without this the worker would swallow the error, leaving the tab locked and its buttons
        waiting for good.

        Args:
            error: The exception the load raised, as the worker caught it.
        """
        # Dropped before the load got as far as failing, so they are gone whatever came of it
        self.results_cleared.emit()

        logger.error('Could not load the data: %s', error)
        notifications.show_error(f'Could not load data: {error}')
        self._set_loading(False, STYLESHEETS['STYLE_ERROR'])

    def _set_loading(self, loading: bool, outcome: str = '') -> None:
        """ Lock the tab while a load runs, and outline the buttons by how it is going.

        The paths are read on the worker as the load starts, so the boxes and the structure are held
        for the length of it rather than changed underneath it. A locked button keeps its outline,
        which is what shows the load is still going.

        Args:
            loading: True as a load starts, False once it has finished.
            outcome: The outline the buttons are left with once it has finished. Unused while loading,
                which always outlines them as waiting.
        """
        sheet = GROUP_STYLESHEETS['STYLE_WAIT'] if loading else outcome

        self._loading_group.setStyleSheet(sheet)

        self._structure_group.setEnabled(not loading)
        self._loading_group.setEnabled(not loading)

        # Unlocked to whatever the boxes allow: a red box still closes its button
        if not loading:
            self._refresh_load_buttons()

        self.loading_changed.emit(loading)

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

    def _update_metadata_table(self, metadata_df: pd.DataFrame | None) -> bool:
        """ Update the metadata table in the GUI with the given metadata dataframe.

        Args:
            metadata_df: The metadata dataframe to display in the table.
        Returns:
            True if the table was updated successfully, False if the dataframe was None or empty.
        """
        if metadata_df is None or metadata_df.empty:
            self._metadata_table.value = pd.DataFrame()  # Clear the table if no data is loaded
            self.data_loaded.emit()
            notifications.show_error("Could not load data. Please check the provided paths and files.")
            return False
        else:
            self._metadata_table.value = metadata_df
            self.data_loaded.emit()
            return True

    def _on_table_selection_changed(self) -> None:
        """ Display corresponding image and mask when a row is selected from the table. """
        row = self._metadata_table.native.currentRow()
        if row < 0:
            return

        image_paths, mask_paths, row_data = self._phenome.get_row_display_data(row)
        self._image_viewer.display_images(image_paths, mask_paths, row_data, names=[f"Image {row}"])
