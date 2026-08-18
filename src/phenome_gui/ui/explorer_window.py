import logging
from pathlib import Path

import pandas as pd
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

# Double precision rather than the plain QRangeSlider, which steps in whole numbers: the properties
# worth highlighting on include the ones living between 0 and 1, which it would collapse to two
# positions. The labelled variant writes the chosen bounds out, a bare slider showing no numbers.
from superqt import QLabeledDoubleRangeSlider

from ..utils.image_viewer import ImageViewer

# Derived from the Literal types the parameters are typed with, so a value the model would refuse can
# never be offered here. The sources are shared with the analysis forms, the feature spaces being
# PhenoMe's rather than any one step's.
from ..utils.params import (
    DIMS_CHOICES,
    EMBEDDING_METHOD_CHOICES,
    SOURCE_CHOICES,
)
from ..utils.phenome_interface import PhenoMeInterface
from .criteria_window import ExcludeWindow, FilterWindow
from .plotting import (
    INDEX_COLUMN,
    ExplorerPlot,
    is_continuous,
    resolve_column,
)
from .selection_window import SelectionWindow

logger = logging.getLogger("main_logger")


class InteractiveExplorerWindow(QDialog):
    """ The interactive plot of the embedded images.

    Holds the canvas the projection is drawn on, and the controls deciding how it is computed and how it
    is shown. Meant to be shown non-modally, so the images a selection points at can be looked at in the
    napari viewer while it stays open.

    Unlike the parameter forms, it writes straight into params.explorer rather than into a copy handed
    back once confirmed, there being nothing to confirm: the window stays open, and its settings are
    read whenever the projection is computed.

    TODO: The highlight group picks a field and a value but does not mark anything with them yet, and
    the selection group is still to be written.
    """

    def __init__(
        self,
        phenome: PhenoMeInterface,
        image_viewer: ImageViewer,
        parent: QWidget | None = None
    ) -> None:
        """ Build the window and its widgets.

        Args:
            phenome (PhenoMeInterface): The interface every call to PhenoMe goes through.
            image_viewer (ImageViewer): The viewer the selected image is displayed in.
            parent (QWidget): The widget the window is parented to, so it closes with it.
        """
        super().__init__(parent)
        self.setWindowTitle("Interactive Explorer")
        self.resize(1000, 600)

        self._phenome = phenome
        self._image_viewer = image_viewer

        self._filter_dialog = None
        self._exclude_dialog = None
        self._view_selection_dialog = None

        # The images a drawn box picked out, as PhenoMe indexes its results
        self._selected_indices = []

        self._build_ui()

    def _build_ui(self) -> None:
        """ Build the explorer window and its widgets.

        The canvas takes the spare width, with a column of controls beside it: a group to configure and
        compute the projection, one to decide how the points are drawn, and two left empty for the
        highlight and selection steps.
        """
        layout = QHBoxLayout()

        # Canvas the projection is drawn on. Empty until a projection has been computed. It says
        # which point was clicked and no more, the images being the window's to fetch and show.
        self._canvas = ExplorerPlot()
        self._canvas.point_clicked.connect(self._on_point_clicked)
        self._canvas.points_selected.connect(self._on_points_selected)

        # Column the control groups are stacked in, beside the canvas
        controls_layout = QVBoxLayout()

        # Group for the projection
        embedding_group = QGroupBox("Embedding")
        embedding_layout = QVBoxLayout()
        embedding_group.setLayout(embedding_layout)

        # Dimensionality reduction algorithm combo box
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(EMBEDDING_METHOD_CHOICES)
        self._method_combo.setCurrentText(self._phenome.params.explorer.method)
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo)
        embedding_layout.addLayout(method_row)

        # Number of dimensions combo box
        dims_row = QHBoxLayout()
        dims_row.addWidget(QLabel("Dims:"))
        self._dims_combo = QComboBox()
        self._dims_combo.addItems(DIMS_CHOICES)
        self._dims_combo.setCurrentText(self._phenome.params.explorer.dims)
        self._dims_combo.currentTextChanged.connect(self._on_dims_changed)
        dims_row.addWidget(self._dims_combo)
        embedding_layout.addLayout(dims_row)

        # Feature space combo box, opened by refresh_choices() once the properties have been computed
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source:"))
        self._source_combo = QComboBox()
        self._source_combo.addItems(SOURCE_CHOICES)
        self._source_combo.setCurrentText(self._phenome.params.explorer.source)
        self._source_combo.currentTextChanged.connect(self._on_source_changed)
        source_row.addWidget(self._source_combo)
        embedding_layout.addLayout(source_row)

        # Buttons for filters and exclusions, each opening a form rebuilt on every use
        filter_row = QHBoxLayout()
        self._filter_button = QPushButton("Filter...")
        self._exclude_button = QPushButton("Exclude...")
        self._filter_button.clicked.connect(self._on_filter_clicked)
        self._exclude_button.clicked.connect(self._on_exclude_clicked)
        filter_row.addWidget(self._filter_button)
        filter_row.addWidget(self._exclude_button)
        embedding_layout.addLayout(filter_row)

        # Projection button, which is what recomputes the plot rather than any of the combo boxes
        self._compute_button = QPushButton("Compute")
        self._compute_button.clicked.connect(self._on_compute_clicked)
        embedding_layout.addWidget(self._compute_button)

        # Group for how the points are drawn
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QVBoxLayout()
        appearance_group.setLayout(appearance_layout)

        # Metadata column the points are coloured by. Left empty here, as the columns are only known
        # once the data has been loaded; refresh_choices() is what fills it in.
        color_by_row = QHBoxLayout()
        color_by_row.addWidget(QLabel("Color by: "))
        self._color_by_combo = QComboBox()
        self._color_by_combo.currentTextChanged.connect(self._on_color_by_changed)
        color_by_row.addWidget(self._color_by_combo)
        appearance_layout.addLayout(color_by_row)

        # Group for marking points on the plot based on metadata/property filters
        self._highlight_group = QGroupBox("Highlight")
        highlight_layout = QVBoxLayout()

        # Field the points are marked on, offering what the colouring offers. Left empty here, as
        # its values are the projection's; refresh_highlight() is what fills it in.
        highlight_field_row = QHBoxLayout()
        highlight_field_row.addWidget(QLabel("Field: "))
        self._highlight_field_combo = QComboBox()
        self._highlight_field_combo.currentTextChanged.connect(self._on_highlight_field_changed)
        highlight_field_row.addWidget(self._highlight_field_combo, 1)
        highlight_layout.addLayout(highlight_field_row)

        # One control per kind of field, both built once and switched between: a field of labels is
        # picked from a list of them, where a measurement is picked from the range it spans. Kept in
        # a stack so neither loses what was set on it while the other is on screen.
        self._highlight_stack = QStackedWidget()
        self._highlight_values_combo = QComboBox()
        self._highlight_range_slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self._highlight_stack.addWidget(self._highlight_values_combo)
        self._highlight_stack.addWidget(self._highlight_range_slider)
        highlight_layout.addWidget(self._highlight_stack)

        # Marks the points answering the field and value above, and leaves the plot as it is when
        # unchecked. The value and the range are watched too, so a pick made with it already on
        # takes effect where it was made rather than on the next toggle.
        self._highlight_checkbox = QCheckBox("Highlight")
        self._highlight_checkbox.toggled.connect(self._apply_highlight)
        self._highlight_values_combo.currentIndexChanged.connect(self._apply_highlight)
        self._highlight_range_slider.valueChanged.connect(self._apply_highlight)
        highlight_layout.addWidget(self._highlight_checkbox)

        self._highlight_group.setLayout(highlight_layout)

        # Group for selecting points and exporting their indices or metadata.
        # 3 selection options: box, lasso or selecting the highlighted points
        selection_group = QGroupBox("Selection")
        selection_layout = QVBoxLayout()

        # Selection options
        selection_row = QHBoxLayout()
        self._box_select_button = QPushButton("Box")
        self._lasso_select_button = QPushButton("Lasso")
        self._highlight_select_button = QPushButton("Highlight")
        self._clear_select_button = QPushButton("Clear")

        # Checkable, so the plot waiting for a box is something the window shows rather than
        # something only the cursor gives away. It unchecks itself once the box has been drawn.
        self._box_select_button.setCheckable(True)
        self._box_select_button.clicked.connect(self._on_box_select_clicked)
        self._clear_select_button.clicked.connect(self._on_clear_selection_clicked)
        selection_row.addWidget(self._box_select_button)
        selection_row.addWidget(self._lasso_select_button)
        selection_row.addWidget(self._highlight_select_button)
        selection_row.addWidget(self._clear_select_button)
        selection_layout.addLayout(selection_row)

        # View selection
        select_view_row = QHBoxLayout()
        self._num_select_label = QLabel("0 points selected.")
        self._view_select_button = QPushButton("View selection...")
        self._view_select_button.clicked.connect(self._on_view_selection_clicked)
        select_view_row.addWidget(self._num_select_label)
        select_view_row.addWidget(self._view_select_button)
        selection_layout.addLayout(select_view_row)

        selection_group.setLayout(selection_layout)

        controls_layout.addWidget(embedding_group)
        controls_layout.addWidget(appearance_group)
        controls_layout.addWidget(self._highlight_group)
        controls_layout.addWidget(selection_group)
        controls_layout.addStretch(1)

        layout.addWidget(self._canvas, 1)
        layout.addLayout(controls_layout, 0)
        self.setLayout(layout)

        # Called here rather than next to each combo box, as it needs both of them to exist
        self.refresh_choices()

    def refresh_choices(self) -> None:
        """ Fill in the colouring columns, and open the choices the pipeline can currently answer for.

        The window is built once and kept, where the parameter forms are rebuilt on every use, so what
        it offers has to be brought up to date by hand: when it is opened, and again whenever a step it
        waits on has finished. Both conditions are the ones the analysis forms are gated on, the
        colouring needing the loaded metadata and two of the sources reading the properties.

        Being refreshed by hand is also why the properties only join the colouring choices the next
        time the window is opened, the same wait the sources have had since they were written.
        """
        columns = self._phenome.get_color_by_columns()
        stored_column = self._phenome.params.explorer.color_by

        # Refilling emits currentTextChanged twice, once on the clear and once on the first column
        # added, either of which would overwrite the stored column before it can be restored.
        self._color_by_combo.blockSignals(True)
        self._color_by_combo.clear()
        self._color_by_combo.addItems(columns)
        if stored_column in columns:
            self._color_by_combo.setCurrentText(stored_column)
        self._color_by_combo.blockSignals(False)

        self._color_by_combo.setEnabled(bool(columns))
        self._color_by_combo.setToolTip('' if columns else 'Please load data first.')

        # The combo box may have landed on the first column rather than on the stored one, so the
        # parameter is squared with it before the user touches anything.
        self._on_color_by_changed()

        has_properties = self._phenome.has_properties
        self._source_combo.setEnabled(has_properties)
        self._source_combo.setToolTip('' if has_properties else 'Please calculate properties first.')

        self.refresh_highlight()

    def refresh_highlight(self) -> None:
        """ Fill in the fields the points can be marked on, and the values of the chosen one.

        Called when the choices are refreshed and again whenever a projection has been computed: the
        fields are the ones the colouring offers, but their values are read off the projection, that
        being the only frame carrying the metadata and the properties side by side and the only one
        answering for the points actually on screen. The group waits on it for the same reason.
        """
        columns = self._phenome.get_color_by_columns()
        stored_field = self._highlight_field_combo.currentText()

        # Refilling emits currentTextChanged twice, once on the clear and once on the first column
        # added, so the field is restored by hand rather than left wherever the refill landed.
        self._highlight_field_combo.blockSignals(True)
        self._highlight_field_combo.clear()
        self._highlight_field_combo.addItems(columns)
        if stored_field in columns:
            self._highlight_field_combo.setCurrentText(stored_field)
        self._highlight_field_combo.blockSignals(False)

        has_projection = self._phenome.projection_df is not None
        self._highlight_group.setEnabled(has_projection)
        self._highlight_group.setToolTip('' if has_projection else 'Please compute a projection first.')

        # The combo box may have landed on another field than the stored one, so its values are
        # brought up to date whether or not the refill moved it
        self._on_highlight_field_changed()

    def _on_highlight_field_changed(self) -> None:
        """ Show the control the chosen field is picked with, and fill it from the projection.

        A field of labels is picked from the list of them it holds, a field of measurements from the
        range it spans, which is the same reading of a column the canvas colours by.
        """
        projection_df = self._phenome.projection_df
        column = None if projection_df is None else resolve_column(
            projection_df, self._highlight_field_combo.currentText()
        )

        if column is None or projection_df.empty:
            self._highlight_values_combo.clear()
            self._highlight_stack.setCurrentWidget(self._highlight_values_combo)
        else:
            values = projection_df[column].dropna()

            if is_continuous(values, column):
                self._show_highlight_range(values)
            else:
                self._show_highlight_values(values)

        # The field decides what is marked as much as the value does, so the plot follows it
        self._apply_highlight()

    def _show_highlight_values(self, values: pd.Series) -> None:
        """ List the values of a field of labels, each of them once.

        Args:
            values (pd.Series): The column's values, of which only the distinct ones are listed.
        """
        distinct = values.unique()
        try:
            ordered = sorted(distinct)
        except TypeError:
            ordered = sorted(distinct, key=str)

        picked = self._highlight_values_combo.currentData()

        # Each row shows its value as text but carries the value itself, the canvas matching a point
        # by equality against a column that holds the cluster labels as numbers: a cluster picked as
        # '1' would go down as text and mark no point at all.
        #
        # Refilling emits the change signal on the clear and again on the first value added, either
        # of which would mark the plot with a value nobody picked, so the pick is put back by hand.
        self._highlight_values_combo.blockSignals(True)
        self._highlight_values_combo.clear()
        for value in ordered:
            self._highlight_values_combo.addItem(str(value), value)

        restored = self._highlight_values_combo.findData(picked)
        if restored >= 0:
            self._highlight_values_combo.setCurrentIndex(restored)
        self._highlight_values_combo.blockSignals(False)

        self._highlight_stack.setCurrentWidget(self._highlight_values_combo)

    def _show_highlight_range(self, values: pd.Series) -> None:
        """ Span the slider across a field of measurements, handles wide open.

        The handles start at the ends of the range so the control reads as picking every point until
        it is moved, rather than as picking whichever slice it happened to open on.

        Args:
            values (pd.Series): The column's values, whose smallest and largest are the two ends.
        """
        low, high = float(values.min()), float(values.max())

        # A column holding the one value throughout would leave the slider no room to move, its two
        # ends being the same place
        if low == high:
            high = low + 1.0

        # Set with its signals blocked, as the two calls would otherwise mark the plot twice over,
        # once against the new range and once against handles not yet moved onto it
        self._highlight_range_slider.blockSignals(True)
        self._highlight_range_slider.setRange(low, high)
        self._highlight_range_slider.setValue((low, high))
        self._highlight_range_slider.blockSignals(False)

        self._highlight_stack.setCurrentWidget(self._highlight_range_slider)

    def _highlight_criterion(self):
        """ What the highlight control currently holds, for the canvas to mark the points against.

        Returns:
            The value picked from the list of them, or the two ends of the picked range - whichever
            control the chosen field put on screen.
        """
        if self._highlight_stack.currentWidget() is self._highlight_range_slider:
            return tuple(self._highlight_range_slider.value())

        return self._highlight_values_combo.currentData()

    def _apply_highlight(self) -> None:
        """ Mark the points answering the highlight, or take the marking off when it is switched off.

        Called whenever anything it reads changes - the box, the field, the value or the range - and
        again after the plot has been redrawn, a new drawing knowing nothing of what was marked on
        the one before it.
        """
        if not self._highlight_checkbox.isChecked():
            self._canvas.highlight_points()
            return

        self._canvas.highlight_points(self._highlight_field_combo.currentText(), self._highlight_criterion())

    def _on_method_changed(self) -> None:
        """ Update the projection method parameter when the selection changes.

        Changing it does not redraw the plot, the projection being computed by the Compute button.
        """
        self._phenome.params.explorer.method = self._method_combo.currentText()

    def _on_dims_changed(self) -> None:
        """ Update the number of dimensions parameter when the selection changes.

        Changing it does not redraw the plot, the projection being computed by the Compute button.
        """
        self._phenome.params.explorer.dims = self._dims_combo.currentText()

    def _on_source_changed(self) -> None:
        """ Update the feature space parameter when the selection changes.

        Changing it does not redraw the plot, the projection being computed by the Compute button.
        """
        self._phenome.params.explorer.source = self._source_combo.currentText()

    def _on_filter_clicked(self) -> None:
        """ Open the form choosing which images the projection is computed from.

        The form is rebuilt on every use, as the values it lists are read when it is constructed, so
        it always offers what the metadata currently holds. What was picked is kept however the form
        is closed: it has no Cancel button to mean the other thing.

        Picking nothing puts the parameter back to None rather than to an empty filter, which is the
        same to PhenoMe but reads as what it is. The filter is applied when the projection is next
        computed, the plot on screen having been computed from whatever was set before.
        """
        self._filter_dialog = FilterWindow(
            self,
            self._phenome.get_metadata_columns(),
            self._phenome.metadata_df,
            self._phenome.params.explorer.filter
        )
        self._filter_dialog.exec_()
        self._phenome.params.explorer.filter = self._filter_dialog.get_criteria() or None

        # Stringified rather than passed as a mapping: logging reads a lone mapping argument as the
        # values to interpolate by name, and raises out of the handler rather than logging anything
        picked = self._phenome.params.explorer.filter
        logger.info("Filtering the projection on %s.", str(picked) if picked else 'nothing')

    def _on_exclude_clicked(self) -> None:
        """ Open the form choosing which images the projection leaves out.

        The twin of the filter form in every way but the answer it takes: the same values are picked,
        and the projection then drops the images holding them rather than keeping them.
        """
        self._exclude_dialog = ExcludeWindow(
            self,
            self._phenome.get_metadata_columns(),
            self._phenome.metadata_df,
            self._phenome.params.explorer.exclude
        )
        self._exclude_dialog.exec_()
        self._phenome.params.explorer.exclude = self._exclude_dialog.get_criteria() or None

        picked = self._phenome.params.explorer.exclude
        logger.info("Excluding %s from the projection.", str(picked) if picked else 'nothing')

    def _on_compute_clicked(self) -> None:
        """ Begin the projection task.

        Run on a worker thread, t-SNE and UMAP taking long enough on a real dataset to freeze the
        viewer otherwise. The button is closed for the duration, as a second run would project the
        same images again into a plot only the last one to finish would reach.
        """
        logger.info("Compute clicked, with parameters %s", self._phenome.params.explorer)

        self._compute_button.setEnabled(False)
        worker = self._projection_worker()
        worker.returned.connect(self._on_finish_projection)
        worker.errored.connect(self._on_projection_failed)
        worker.start()

    @thread_worker
    def _projection_worker(self) -> None:
        """ Create worker to project the images on a parallel thread. """
        self._phenome.compute_projection()

    def _on_finish_projection(self) -> None:
        """ Called when the projection is done. Reopens the button and draws what came back.

        An empty run is drawn too, as an empty canvas: the plot that was there was computed from
        other parameters and no longer answers for the ones now on screen.
        """
        self._compute_button.setEnabled(True)

        projection_df = self._phenome.projection_df
        logger.info("Projected %d images.", 0 if projection_df is None else len(projection_df))

        self._canvas.plot_projection(projection_df, self._phenome.params.explorer.color_by)

        # A selection was made on the points that were drawn before, which these are not
        self._selected_indices = []
        self._num_select_label.setText("0 points selected.")
        self._show_selection()

        # The highlight controls read their values off the projection, so a new one has to reach
        # them. Refreshing the field is also what marks the new drawing, the highlight going with
        # the drawing it was applied to.
        self.refresh_highlight()

    def _on_projection_failed(self, error: Exception) -> None:
        """ Reopen the button when the projection raised, so the run can be attempted again.

        PhenoMe raises rather than returning nothing for the failures it can name - t-SNE given fewer
        samples than its perplexity, UMAP without its package - and the worker swallows the exception
        into this signal, leaving the button shut for good if nobody listens.

        Args:
            error (Exception): The exception the projection raised, as the worker caught it.
        """
        self._compute_button.setEnabled(True)
        logger.error("Could not compute the projection: %s", error)

    def _on_point_clicked(self, row: int) -> None:
        """ Show the image a clicked point stands for in the napari viewer.

        The layers already there are replaced rather than added to, the plot being a way of looking
        through the images one at a time: the point clicked last is what is on screen, and a browse
        around the cloud leaves nothing piled up behind it.

        Args:
            row (int): Row of the drawn projection the clicked point stands for.
        """
        image_paths, mask_paths, metadata = self._phenome.get_projection_image_data(row)
        name = Path(image_paths[0]).name

        logger.info("Showing '%s', from row %d of the projection.", name, row)
        self._image_viewer.display_images(image_paths, mask_paths, metadata, names=[name])

    def _on_image_selected(self, index: int) -> None:
        """ Show the image a picked row of the selection table stands for.

        The same walk through the images as clicking the points themselves, from the list of what a
        box picked out rather than from the plot, and showing them the same way: one at a time, the
        viewer being cleared as each is shown.

        Args:
            index (int): Place of the image in the PhenoMe results, as the table's row carries it.
        """
        image_paths, mask_paths, metadata = self._phenome.get_image_data_at(index)
        name = Path(image_paths[0]).name

        logger.info("Showing '%s', picked from the selection.", name)
        self._image_viewer.display_images(image_paths, mask_paths, metadata, names=[name])

    def _on_color_by_changed(self) -> None:
        """ Update the colouring parameter, and recolour the drawn points to match.

        The combo box is empty until data has been loaded, where the parameter says None rather than an
        empty string, so an unpicked column reads the same way as it does for the analysis steps.

        Only the colouring is redone, the coordinates being the projection's rather than the column's.
        A canvas with nothing on it yet is left alone: this also runs while the choices are refreshed,
        which would otherwise wipe the plot every time the window is reopened.
        """
        self._phenome.params.explorer.color_by = self._color_by_combo.currentText() or None

        if self._phenome.projection_df is not None:
            self._canvas.plot_projection(self._phenome.projection_df, self._phenome.params.explorer.color_by)

            # Recolouring draws the points again, which knows nothing of what was marked on the
            # drawing before it
            self._apply_highlight()

    def _on_box_select_clicked(self) -> None:
        """ Wait for a box to be drawn over the plot, or stop waiting when the button is let go of. """
        if self._box_select_button.isChecked():
            self._canvas.begin_box_selection()
        else:
            self._canvas.cancel_box_selection()

    def _on_points_selected(self, indices: list[int]) -> None:
        """ Take note of the points a drawn box picked out, and show what they are.

        The canvas says which images were picked and washes them over; the count and the table are
        the window's to keep, the canvas holding no metadata to fill a table with.

        Args:
            indices (list[int]): The index PhenoMe holds each selected image at.
        """
        self._selected_indices = indices
        self._box_select_button.setChecked(False)

        self._num_select_label.setText(f"{len(indices)} points selected.")
        self._show_selection()

    def _on_clear_selection_clicked(self) -> None:
        """ Let go of the selected points, on the plot and in the table alike. """
        self._selected_indices = []
        self._canvas.clear_selection()

        self._num_select_label.setText("0 points selected.")
        self._show_selection()

    def _selected_points(self) -> "pd.DataFrame | None":
        """ The rows of the projection the selected images were drawn from.

        Returns:
            pd.DataFrame: One row per selected image, or None when nothing is selected or there is no
                projection to read them from.
        """
        projection_df = self._phenome.projection_df
        if projection_df is None or not self._selected_indices:
            return None

        if INDEX_COLUMN not in projection_df.columns:
            return projection_df.iloc[self._selected_indices]

        return projection_df[projection_df[INDEX_COLUMN].isin(self._selected_indices)]

    def _show_selection(self) -> None:
        """ Fill the selection table, when there is a window open holding one. """
        if self._view_selection_dialog is not None:
            self._view_selection_dialog.show_selection(self._selected_points())

    def _on_view_selection_clicked(self) -> None:
        """ Open the window listing the selected points, or raise it when it is already open.

        Filled before it is shown, so opening it after a selection lists that selection rather than
        whatever it was left holding.
        """
        if self._view_selection_dialog is None:
            self._view_selection_dialog = SelectionWindow(self)
            self._view_selection_dialog.image_selected.connect(self._on_image_selected)

        self._show_selection()

        self._view_selection_dialog.show()
        self._view_selection_dialog.raise_()
        self._view_selection_dialog.activateWindow()
