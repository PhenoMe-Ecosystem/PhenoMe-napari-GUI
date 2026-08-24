import logging

import pandas as pd
from qtpy.QtCore import Signal
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .plotting import INDEX_COLUMN, format_value

logger = logging.getLogger("main_logger")

# How many rows a column's width is measured from, the rest of them being taken on trust
SIZED_FROM_ROWS = 20


class SelectionWindow(QDialog):
    """ Dialog containing a table where information about selected points is displayed.

    One row per selected point, holding the projection's own columns: where the point was placed, the
    image it stands for, its metadata and every property measured on it. The same rows PhenoMe's own
    explorer writes when it exports a selection to a CSV, which is a wide table by design.

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

        # Sized from the first rows rather than from all of them: a projection can be dozens of
        # columns wide and a selection hundreds of rows long, which is tens of thousands of cells to
        # measure for a width the first few already give.
        self._points_table.horizontalHeader().setResizeContentsPrecision(SIZED_FROM_ROWS)
        self._points_table.selectionModel().selectionChanged.connect(self._on_table_selection)

        layout.addWidget(self._points_table)

        # Export options
        export_group = QGroupBox("Export selection")
        export_layout = QHBoxLayout()
        self._export_idx_button = QPushButton("Export Indices...")
        self._export_idx_button.clicked.connect(self._on_export_idx_clicked)
        self._export_idx_button.setEnabled(False)
        self._export_csv_button = QPushButton("Export CSV...")
        self._export_csv_button.clicked.connect(self._on_export_csv_clicked)
        self._export_csv_button.setEnabled(False)
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
            self._export_idx_button.setEnabled(False)
            self._export_csv_button.setEnabled(False)
            return
        self._export_idx_button.setEnabled(True)
        self._export_csv_button.setEnabled(True)

        # The projection's own columns, in its own order: the coordinates, the image, its place, the
        # metadata and every property. The same rows PhenoMe's explorer writes to a CSV.
        columns = list(points_df.columns)

        model.setHorizontalHeaderLabels(columns)
        for _, point in points_df.iterrows():
            model.appendRow([QStandardItem(format_value(point[column])) for column in columns])

        self._points_table.resizeColumnsToContents()

    def clear_selection(self) -> None:
        """ Empty the table, the selection it was listing having been let go of.

        Listing nothing is what a cleared selection is, so it goes the same way an empty one does -
        the exports closing with it, there being nothing left to write.
        """
        self.show_selection(None)

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
        """Opens a directory selection dialog to save the selected image indices as a .txt file."""
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save text File As",
            "",
            "Text file (*.txt)"
        )

        if file_path and self._points_df is not None:
            with open(file_path, mode="w") as txt_file:
                idx_column = self._points_df["Index"].astype('string')
                idx_column = idx_column.to_list()
                txt_str = ",".join(idx_column)
                txt_file.write(txt_str)

    def _on_export_csv_clicked(self) -> None:
        """Opens a directory selection dialog to save the selection dataframe as a .csv file."""
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save CSV File As",
            "",
            "CSV file (*.csv)"
        )

        if file_path and self._points_df is not None:
            self._points_df.to_csv(file_path, index=False)

