import logging
from pathlib import Path

from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

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
from .plotting import ExplorerPlot

logger = logging.getLogger("main_logger")


class InteractiveExplorerWindow(QDialog):
    """ The interactive plot of the embedded images.

    Holds the canvas the projection is drawn on, and the controls deciding how it is computed and how it
    is shown. Meant to be shown non-modally, so the images a selection points at can be looked at in the
    napari viewer while it stays open.

    Unlike the parameter forms, it writes straight into params.explorer rather than into a copy handed
    back once confirmed, there being nothing to confirm: the window stays open, and its settings are
    read whenever the projection is computed.

    TODO: The projection is computed but not drawn yet; drawing it, colouring it, and the highlight
    and selection groups are all still to be written.
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

        # Column the control groups are stacked in, beside the canvas
        controls_layout = QVBoxLayout()

        # Group for the projection
        embedding_group = QGroupBox("Embedding")
        embedding_layout = QVBoxLayout()
        embedding_group.setLayout(embedding_layout)

        # Dimensionality reduction algorithm combo box
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(EMBEDDING_METHOD_CHOICES)
        self._method_combo.setCurrentText(self._phenome.params.explorer.method)
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo)
        embedding_layout.addLayout(method_row)

        # Number of dimensions combo box
        dims_row = QHBoxLayout()
        dims_row.addWidget(QLabel("Dims"))
        self._dims_combo = QComboBox()
        self._dims_combo.addItems(DIMS_CHOICES)
        self._dims_combo.setCurrentText(self._phenome.params.explorer.dims)
        self._dims_combo.currentTextChanged.connect(self._on_dims_changed)
        dims_row.addWidget(self._dims_combo)
        embedding_layout.addLayout(dims_row)

        # Feature space combo box, opened by refresh_choices() once the properties have been computed
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source"))
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

        # Group for marking chosen points out on the plot, filled later
        highlight_group = QGroupBox("Highlight")
        highlight_layout = QVBoxLayout()
        highlight_group.setLayout(highlight_layout)

        # Group for what to do with the points picked on the plot, filled later
        selection_group = QGroupBox("Selection")
        selection_layout = QVBoxLayout()
        selection_group.setLayout(selection_layout)

        controls_layout.addWidget(embedding_group)
        controls_layout.addWidget(appearance_group)
        controls_layout.addWidget(highlight_group)
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
        """ Opens a window to set metadata filters for the projections.
        """

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
