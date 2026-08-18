import logging

import pandas as pd
from qtpy.QtCore import Signal
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .plotting import IMAGE_COLUMN, INDEX_COLUMN, metadata_columns

logger = logging.getLogger("main_logger")


class SelectionWindow(QDialog):
    """ Dialog containing a table where information about selected points is displayed.

    One row per selected point, naming the image and what is known about it. The properties are left
    out: a projection can carry dozens of them, and a table that wide reads as nothing at all.

    Picking a row says which image it stands for and no more: the window holds no PhenoMe state and
    no viewer, so showing the image is for whoever handed the selection over.
    """

    # Emitted with the place PhenoMe holds the image of the row that was picked
    image_selected = Signal(int)

    def __init__(self, parent: QWidget) -> None:
        """ Build the window and its table, empty until a selection is shown in it.

        Args:
            parent (QWidget): The widget the dialog is parented to, so it closes with it.
        """
        super().__init__(parent)

        self.setWindowTitle("Selected Points")
        self.resize(500, 400)

        # The points currently listed, kept beside the table: the table holds their values as text,
        # where the row that was picked has to be resolved back to the image it stands for
        self._points_df = None

        layout = QVBoxLayout(self)

        # Metadata table subset
        self._points_table = QTableView(self)
        self._points_table.setModel(QStandardItemModel(self._points_table))
        self._points_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        # A row stands for one image, so picking any of its cells picks the image
        self._points_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._points_table.selectionModel().selectionChanged.connect(self._on_table_selection)

        layout.addWidget(self._points_table)

        # Export options
        export_group = QGroupBox("Export selection")
        export_layout = QHBoxLayout()
        self._export_idx_button = QPushButton("Export Indices...")
        self._export_csv_button = QPushButton("Export CSV...")
        export_layout.addWidget(self._export_idx_button)
        export_layout.addWidget(self._export_csv_button)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

    def show_selection(self, points_df: pd.DataFrame | None) -> None:
        """ List the selected points, replacing whatever the table held before.

        Args:
            points_df (pd.DataFrame): The rows of the projection that were selected. None or empty
                empties the table, which is what a cleared selection comes back as.
        """
        model = self._points_table.model()
        model.clear()
        self._points_df = points_df

        if points_df is None or points_df.empty:
            return

        # The image is named first and placed second, and the metadata follows: what a point is,
        # rather than where the projection put it or what was measured on it.
        columns = [
            column for column in (INDEX_COLUMN, IMAGE_COLUMN) if column in points_df.columns
        ] + metadata_columns(points_df)

        model.setHorizontalHeaderLabels(columns)
        for _, point in points_df.iterrows():
            model.appendRow([QStandardItem(str(point[column])) for column in columns])

        self._points_table.resizeColumnsToContents()

    def clear_selection(self) -> None:
        """ Empty the table, the selection it was listing having been let go of. """
        self._points_table.model().clear()
        self._points_df = None

    def _on_table_selection(self) -> None:
        """ Say which image the picked row stands for, for whoever can show it.

        Also runs while the table is being refilled, when there is no row picked and nothing to say.
        """
        row = self._points_table.currentIndex().row()
        if row < 0 or self._points_df is None or self._points_df.empty:
            return

        point = self._points_df.iloc[row]
        index = int(point[INDEX_COLUMN]) if INDEX_COLUMN in self._points_df.columns else row

        self.image_selected.emit(index)

    def _on_export_idx_clicked(self) -> None:
        """
        """

    def _on_export_csv_clicked(self) -> None:
        """
        """

