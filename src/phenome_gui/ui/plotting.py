"""The canvas the interactive explorer draws its projection on.

Holds no PhenoMe state: it is handed the coordinates to draw and the name of the column to colour
them by, and knows nothing about where either came from.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters as exporters
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QFont, QPainterPath, QPolygonF
from qtpy.QtWidgets import QGraphicsPathItem, QToolTip, QWidget

logger = logging.getLogger("main_logger")

# Colours of the groups, in a fixed order and cycled once there are more groups than colours.
SERIES_COLOURS = pg.colormap.get('glasbey_light', 'colorcet')
N_COLOURS = len(SERIES_COLOURS.getColors())

# Ramp a measured column is coloured along, the one PhenoMe's own explorer opens on
CONTINUOUS_COLORMAP = 'viridis'

# Columns holding a label rather than a measurement, whatever their type. PhenoMe reserves the same
# names, and also calls a numeric column categorical when its values are whole numbers and there are
# no more than 256 of them - which turns 'area', a pixel count, into hundreds of one-image groups as
# soon as the dataset is small enough. Here the name is what makes a column discrete, so a property
# is coloured along a ramp whatever its size.
CATEGORICAL_COLUMNS = {'cluster', 'cluster_id', 'class'}

# Number of unique values from which a columns is considered continuous
MIN_CONTINUOUS = 8

# Background the points are drawn on, matching napari's dark theme. The colours above are stepped for
# this surface, so the two go together.
SURFACE = '#1a1a19'

# Axes and labels, kept recessive so the points are what is read first
AXIS_COLOUR = '#898781'

# Points drawn for a column that could not be resolved, and the group missing values are gathered in
UNGROUPED_COLOUR = SERIES_COLOURS[0]
MISSING_LABEL = 'Unknown'

# Points a measured column has no value for, left neutral rather than given a place on the ramp
MISSING_COLOUR = AXIS_COLOUR

# The coordinate columns PhenoMe names its components with
COMPONENT_PREFIX = 'Component '

# Column holding the name of each image, and the one holding its place in the PhenoMe results. The
# first is what a hovered point is called, the second is bookkeeping and stays out of the tooltip.
IMAGE_COLUMN = 'Image'
INDEX_COLUMN = 'Index'

# Diameter of a drawn point
MARKER_SIZE = 8

# The colorbar a measured column is read by, drawn over the plot rather than beside it: a bar given a
# column of its own takes its width from the points, and the plot would move every time the colouring
# changed from labels to a measurement. Floated, it sits where the legend does for a column of labels,
# and the axes stay where they are whatever the points are coloured by. Its lengths are in pixels,
# being drawn over the plot rather than in it, and are a good deal smaller than pyqtgraph's own.
COLORBAR_LENGTH = 160
COLORBAR_THICKNESS = 16
COLORBAR_HEIGHT = 64

# Inset from the top right corner of the plot area, matching the legend's own offset
COLORBAR_MARGIN = 10

# Point size of the column's name over the bar and of the values under it, both kept down to the size
# of the bar itself, and the length of the marks the values sit on. The marks are drawn back into the
# ramp, which is pyqtgraph's own way round and what the negative length asks for.
COLORBAR_LABEL_SIZE = 12
COLORBAR_TICK_SIZE = 10
COLORBAR_TICK_LENGTH = -5

# Ring drawn around the hovered point. White reads against every colour of the palette and the ramp
# alike, and the ring sits just outside the point so it marks it without hiding what colour it is.
HIGHLIGHT_COLOUR = '#ffffff'
HIGHLIGHT_WIDTH = 2
HIGHLIGHT_SIZE = MARKER_SIZE + 4

# Opacity of a point the highlight leaves out, out of 255. Low enough to read as background, high
# enough that the shape of the cloud it belongs to is still there to be read against.
NORMAL_ALPHA = 200
DIMMED_ALPHA = 33

# How near the pointer has to come to a point, in pixels, for it to count as hovered. Measured in
# pixels rather than in the projection's own units, those meaning something different from one
# projection to the next and changing again with every zoom.
HOVER_RADIUS = 12

# The two shapes a selection can be drawn as, and what the plot answers when asked which it is
# waiting for. Either reads as true while it waits, and None once it is not waiting for anything.
BOX_SELECTION = 'box'
LASSO_SELECTION = 'lasso'

# The lasso is drawn the way pyqtgraph draws its own rubber band - the same yellow line over the same
# wash - so the two shapes a selection can be drawn as look like one another.
LASSO_PEN_COLOUR = (255, 255, 100)
LASSO_BRUSH_COLOUR = (255, 255, 0, 100)

# Wash laid over a selected point, rather than a colour replacing its own: what the point was
# coloured by is still worth reading once it has been picked out.
SELECTION_COLOUR = "#EEFF00"
SELECTION_ALPHA = 110
SELECTION_SIZE = MARKER_SIZE + 2

# The file types the plot can be written as, each keyed by the extension it is written under. Named
# here rather than picked apart in the writing, so a format added to the canvas is added in one place.
EXPORT_FORMATS = {
    'svg': exporters.SVGExporter,
    'png': exporters.ImageExporter,
}


def is_continuous(values: pd.Series, name: str) -> bool:
    """ Whether a column is a measurement to draw along a ramp rather than labels to group by.

    Public, as the controls beside the canvas have to read a column the same way it does: a field
    offering a range where the plot draws a ramp, and a list of values where it draws groups.

    Args:
        values: The column's values, whose type is what decides in the general case.
        name: The column's name, which is what settles the numeric columns holding labels.
    Returns:
        True for a column to colour along a ramp, False for one to draw as groups.
    """
    if name.lower() in CATEGORICAL_COLUMNS:
        return False

    # A boolean is numeric to pandas, and two shades of a ramp for true and false say less than two
    # named groups do
    if pd.api.types.is_bool_dtype(values):
        return False

    # Check if unique number of values is sufficient
    if len(values.unique()) < MIN_CONTINUOUS:
        return False

    return pd.api.types.is_numeric_dtype(values)


def resolve_column(projection_df: pd.DataFrame, name: str | None) -> str | None:
    """ Find the projection's column holding the given metadata key or property.

    PhenoMe capitalises the metadata columns it copies into a projection, so the key the GUI offers
    ('drug_name') is not the column it ends up in ('Drug_name'), and the two are matched without
    their case. Public for the same reason as is_continuous: the controls beside the canvas have to
    find a column the same way it does.

    Args:
        projection_df: The projected coordinates and the columns alongside them.
        name: Key to look for. None or empty resolves to nothing, no key having been picked.
    Returns:
        The matching column of the projection, or None when it carries no such column.
    """
    if not name:
        return None

    for column in projection_df.columns:
        if column.lower() == name.lower():
            return column

    # A projection is computed from the images PhenoMe holds, which a column added to the metadata
    # afterwards has not reached yet
    logger.warning("Column '%s' is not in the projection.", name)
    return None


def metadata_columns(projection_df: pd.DataFrame) -> list[str]:
    """ The columns of a projection that describe an image rather than place or measure it.

    PhenoMe capitalises the metadata keys it copies into a projection and leaves the property keys as
    the plugins named them, which is what tells the two apart here. Public alongside is_continuous
    and resolve_column: the tooltip and the table of a selection both read an image the same way, and
    neither has room for the dozens of properties a projection can carry.

    Args:
        projection_df: The projected coordinates and the columns alongside them.
    Returns:
        The metadata columns, in the order the projection carries them.
    """
    placed = {IMAGE_COLUMN, INDEX_COLUMN}

    return [
        column for column in projection_df.columns
        if column not in placed and not column.startswith(COMPONENT_PREFIX) and column[:1].isupper()
    ]


def format_value(value) -> str:
    """ Write a column's value the way it should be read rather than parsed.

    Public alongside the other three: the tooltip and the table of a selection both show a projection
    to be read, and a coordinate or a property written to its last decimal reads as nothing.

    Args:
        value: The value held for one image, of whatever type its column has.
    Returns:
        The value as text, a measurement kept to a few figures and a missing one named.
    """
    if pd.isna(value):
        return MISSING_LABEL

    if isinstance(value, float | np.floating):
        return f'{value:.4g}'

    return str(value)


def group_order(value) -> tuple[int, float, str]:
    """ Sort key listing a label where a reader would look for it.

    Numbers read in numeric order, so the clusters run 1, 2, 10 rather than the 1, 10, 2 their names
    would give; text reads alphabetically, without its case, one group being no more important than
    another for being capitalised. The images nothing is known about are gathered under one label
    which is not a group of its own, and are kept at the end of the list whatever the rest of it says.

    Public alongside the other helpers: the field a highlight is picked from is picked against the
    legend, so the two have to list the same values in the same order.

    Args:
        value: The label of one group, of whatever type the column holding it has.
    Returns:
        Key ordering the numbers first, the rest of the labels after them, and the missing
        ones last.
    """
    # The canvas has already gathered the missing values under the label by the time it groups by
    # them, where the field a highlight is picked from reads the column as PhenoMe left it
    if pd.isna(value) or value == MISSING_LABEL:
        return (2, 0.0, '')

    # A cluster label arrives as a number, and a metadata column read out of a file can hold its
    # numbers as text, which should read as the numbers they are either way
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = None

    # A column can hold the word 'nan' without meaning a missing value, and it has no place among the
    # numbers: it compares false against every one of them, leaving the order to the sort's own hand
    if number is None or np.isnan(number):
        return (1, 0.0, str(value).casefold())

    return (0, number, str(value).casefold())


class _SelectionViewBox(pg.ViewBox):
    """ The plot's view box, which can be asked once for a shape drawn over it.

    pyqtgraph already draws a rubber band and works out what it covers - that is how its rectangle
    zoom works - so being asked for a box is that same drag with the zoom left off at the end. A
    lasso is the same drag again, gathering where the pointer went rather than only where it started
    and stopped, and drawn with the same yellow band pyqtgraph uses for the box.
    """

    # Emitted with the shape drawn, in the plot's own coordinates: a rectangle for the one, the path
    # the pointer took for the other
    region_selected = Signal(object)
    lasso_selected = Signal(object)

    def __init__(self, *args, **kwargs) -> None:
        """ Build the view box, panning and zooming until it is asked for a shape. """
        super().__init__(*args, **kwargs)

        # Which shape it is waiting for, or None while it is waiting for none
        self.selecting = None

        # Where the pointer has been through the lasso being drawn, and the band drawn along it
        self._lasso_points = []
        self._lasso_band = None

    @property
    def lasso_band(self):
        """ The band drawn along the lasso, built the first time one is drawn.

        Built the way pyqtgraph builds its own rubber band, down to the pen and the wash: over
        everything else, and left out of what the view sizes itself to, so drawing a lasso never
        moves the plot under it.
        """
        if self._lasso_band is None:
            band = QGraphicsPathItem()
            band.setPen(pg.mkPen(LASSO_PEN_COLOUR, width=1))
            band.setBrush(pg.mkBrush(*LASSO_BRUSH_COLOUR))
            band.setZValue(1e9)
            band.hide()

            self._lasso_band = band
            self.addItem(band, ignoreBounds=True)

        return self._lasso_band

    def mouseDragEvent(self, event, axis=None) -> None:  # noqa: N802 - the name pyqtgraph calls it by
        """ Draw the shape being dragged, and hand it over once it is let go of.

        Args:
            event: The drag, as the scene reports it.
            axis: The axis the drag was made on, for a drag on one of them rather than on the plot.
        """
        # Dragging along an axis is that axis being scaled, whatever the plot is being asked for
        if not self.selecting or axis is not None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseDragEvent(event, axis=axis)
            return

        event.accept()

        if self.selecting == LASSO_SELECTION:
            self._drag_lasso(event)
        else:
            self._drag_box(event)

    def _drag_box(self, event) -> None:
        """ Draw the box being dragged, and hand it over once it is let go of.

        Args:
            event: The drag, as the scene reports it.
        """
        if not event.isFinish():
            self.updateScaleBox(event.buttonDownPos(), event.pos())
            return

        self.rbScaleBox.hide()

        # Mapped the way the rectangle zoom maps it, which is what turns the band drawn on screen
        # into the coordinates the points are held in
        drawn = QRectF(pg.Point(event.buttonDownPos(event.button())), pg.Point(event.pos()))
        drawn = self.childGroup.mapRectFromParent(drawn)

        # One box per asking: the mode is over as soon as the box is let go of
        self.selecting = None
        self.region_selected.emit(drawn)

    def _drag_lasso(self, event) -> None:
        """ Draw the lasso being dragged, and hand the path over once it is let go of.

        Args:
            event: The drag, as the scene reports it.
        """
        # Gathered in the coordinates the points are held in rather than in the pixels the drag is
        # reported in: the band is drawn among the points, and what is handed over is read against
        # them. Mapped a point at a time, the way the box maps its two corners.
        if not self._lasso_points:
            self._lasso_points.append(self.childGroup.mapFromParent(pg.Point(event.buttonDownPos())))

        # Where the pointer is now, on the last move as much as on the ones before it: a drag ends
        # where it was let go of, and the loop is not closed until that has been drawn round to
        self._lasso_points.append(self.childGroup.mapFromParent(pg.Point(event.pos())))

        if not event.isFinish():
            self._draw_lasso_band()
            return

        self.lasso_band.hide()

        drawn = QPolygonF(self._lasso_points)
        self._lasso_points = []

        # One lasso per asking, as with a box
        self.selecting = None
        self.lasso_selected.emit(drawn)

    def _draw_lasso_band(self) -> None:
        """ Redraw the band along the path the pointer has taken so far.

        Closed as it is drawn rather than only at the end, so what the loop already holds is there to
        be read while it is still being drawn round.
        """
        path = QPainterPath(self._lasso_points[0])
        for point in self._lasso_points[1:]:
            path.lineTo(point)
        path.closeSubpath()

        self.lasso_band.setPath(path)
        self.lasso_band.show()


class ExplorerPlot(pg.GraphicsLayoutWidget):
    """ The scatter plot of a projected embedding.

    A column of labels is drawn as one series per group, named in the legend so a group can be picked
    out of the cloud by its colour. A column of measurements is drawn along a colour ramp instead,
    with a colorbar in place of the legend, there being nothing to name.

    Hovering a point rings it and reads out what is known about it, the drawn frame being kept for
    exactly that: the points are drawn grouped, so the plot items alone could not say which row of
    the projection the pointer is over.

    TODO: The highlighting of a chosen selection is still to be written.
    """

    # Emitted with the row of the drawn projection whose point was clicked. Which image that row
    # stands for is for whoever handed the frame over to say, the canvas holding no PhenoMe state.
    point_clicked = Signal(int)

    # Emitted with the image indices a drawn box covers, once the box has been let go of
    points_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """ Build the canvas, empty until a projection is drawn on it.

        Args:
            parent: The widget the canvas is nested in.
        """
        super().__init__(parent)
        self.setBackground(SURFACE)

        # The plot's own view box, which pans and zooms as pyqtgraph's does until it is asked for a
        # box, and hands the drawn rectangle over rather than zooming to it
        self._viewbox = _SelectionViewBox()
        self._viewbox.region_selected.connect(self._on_region_selected)
        self._viewbox.lasso_selected.connect(self._on_lasso_selected)

        self._plot = self.addPlot(row=0, col=0, viewBox=self._viewbox)
        self._legend = self._plot.addLegend(offset=(-10, 10))

        # Floated over the plot when a measured column is drawn, and taken off again when it is not
        self._colorbar = None

        # Connected to the canvas rather than to the bar, which is built and dropped with every
        # drawing: the canvas outlives all of them, where a bar left connected would be asked to
        # place itself after it had been taken off the scene
        self._plot.vb.sigResized.connect(lambda _: self._place_colorbar())

        # What is currently drawn, kept for the hovering: the projection itself, the two columns its
        # coordinates were taken from, and the column its colours were.
        self._points = None
        self._x_column = None
        self._y_column = None
        self._color_column = None

        # Row the tooltip is currently describing, or None when the pointer is over no point, and the
        # ring drawn around that point while it is
        self._hovered = None
        self._highlight = None

        # Which of the drawn rows the highlight leaves out, or None when nothing is highlighted
        self._dimmed = None

        # Which of them a box has picked out, and the wash drawn over those while it holds
        self._selected = None
        self._selection = None

        for axis in ('bottom', 'left'):
            self._plot.getAxis(axis).setPen(AXIS_COLOUR)
            self._plot.getAxis(axis).setTextPen(AXIS_COLOUR)

        # Mouse connections for interactivity
        self.scene().sigMouseMoved.connect(self._on_mouse_over)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def plot_projection(self, projection_df: pd.DataFrame | None, color_by: str | None = None) -> None:
        """ Draw a projection, replacing whatever was on the canvas before.

        Args:
            projection_df: The projected coordinates, one row per image, as PhenoMe
                returns them. None or empty leaves the canvas cleared, which is what an empty or
                failed run comes back as.
            color_by: Metadata column or property the points are coloured by. Labels are drawn
                as groups and measurements along a ramp. None, or a column the projection does not
                carry, draws them as a single unnamed group.
        """
        previous, selected = self._points, self._selected
        self._clear()

        if projection_df is None or projection_df.empty:
            return

        components = [column for column in projection_df.columns if column.startswith(COMPONENT_PREFIX)]

        # The canvas draws a flat cloud and the projection is always computed with two components, so
        # anything else is a frame this canvas did not ask for and cannot place: left undrawn rather
        # than shown as the slice of itself the first two columns would make.
        if len(components) != 2:
            logger.info("A %d-dimensional projection cannot be drawn on this canvas.", len(components))
            return

        x_column, y_column = components

        # Whether these are the points already on the canvas: a recolouring hands the same frame
        # back, where a projection that has just been computed is a frame of its own
        redrawn = previous is projection_df

        # A selection belongs to the points it was drawn over, so it is kept when the same frame is
        # drawn again - a recolouring is the same images under another colour - and let go of when
        # another one takes its place.
        # Kept whole rather than as coordinates alone, the tooltip reading the rest of the row and
        # the dimming drawing it again
        self._points = projection_df
        self._selected = selected if redrawn else None
        self._x_column, self._y_column = x_column, y_column
        self._color_column = resolve_column(projection_df, color_by)

        self._draw()

        # Another projection's components are on another scale entirely, and pyqtgraph only fits the
        # view to what it holds until the plot is first panned or zoomed, after which the points of
        # the next one land off the view or in a corner of it. Fitted after the drawing, the range
        # being read off the items just added, and only for points the canvas was not already
        # showing: a recolouring is the same cloud, and is left wherever it was being looked at.
        if not redrawn:
            self._plot.vb.autoRange()

    def _draw(self) -> None:
        """ Draw the frame currently held, coloured by the column currently held.

        Kept apart from plot_projection() so the dimming can draw the same frame again through the
        same two paths, rather than reaching into the items they left behind: what a point is
        coloured by decides which of them drew it, and the legend and the colorbar go with them.
        """
        self._plot.setLabel('bottom', self._x_column)
        self._plot.setLabel('left', self._y_column)

        column = self._color_column

        if column is None:
            self._add_series(
                self._points[self._x_column],
                self._points[self._y_column],
                UNGROUPED_COLOUR,
                self._dimmed_brushes(UNGROUPED_COLOUR)
            )
        elif is_continuous(self._points[column], column):
            self._draw_ramp(self._points, self._x_column, self._y_column, column)
        else:
            self._draw_groups(self._points, self._x_column, self._y_column, column)

        # Last, so the wash lands over the points rather than under the crowd around them
        self._draw_selection()

    def _draw_selection(self) -> None:
        """ Wash the selected points over, leaving the rest of the drawing as it is. """
        if self._selected is None or not self._selected.any():
            return

        wash = pg.mkBrush(SELECTION_COLOUR)
        colour = wash.color()
        colour.setAlpha(SELECTION_ALPHA)
        wash.setColor(colour)

        self._selection = pg.ScatterPlotItem(
            x=self._points.loc[self._selected, self._x_column].to_numpy(),
            y=self._points.loc[self._selected, self._y_column].to_numpy(),
            brush=wash,
            pen=None,
            size=SELECTION_SIZE
        )
        self._plot.addItem(self._selection)

    def begin_box_selection(self) -> None:
        """ Wait for a box to be drawn over the plot, and pick the points it covers out.

        The mode lasts one box: the view box goes back to panning and zooming as soon as the box is
        let go of, so the plot is not left in a state the pointer gives no sign of.
        """
        self._begin_selection(BOX_SELECTION)

    def begin_lasso_selection(self) -> None:
        """ Wait for a lasso to be drawn over the plot, and pick the points inside it out.

        The same one-shot waiting as a box, for the shape a box cannot make: a cluster that runs at
        an angle, or curls, is drawn round rather than boxed in.
        """
        self._begin_selection(LASSO_SELECTION)

    def _begin_selection(self, shape: str) -> None:
        """ Wait for one of the two shapes to be drawn over the plot.

        Args:
            shape: Which of them to wait for, BOX_SELECTION or LASSO_SELECTION.
        """
        self._viewbox.selecting = shape
        self._viewbox.setCursor(Qt.CursorShape.CrossCursor)

    def cancel_selection(self) -> None:
        """ Stop waiting for a shape, leaving whatever is already selected as it is. """
        self._viewbox.selecting = None
        self._viewbox.unsetCursor()

    def _on_region_selected(self, rect: QRectF) -> None:
        """ Pick out the points a drawn box covers, and announce which images they are.

        Args:
            rect: The box that was drawn, in the plot's own coordinates.
        """
        self._viewbox.unsetCursor()

        indices = self.select_in_box(rect)
        logger.info("Selected %d points.", len(indices))

        self.points_selected.emit(indices)

    def _on_lasso_selected(self, polygon: QPolygonF) -> None:
        """ Pick out the points a drawn lasso encloses, and announce which images they are.

        Args:
            polygon: The path the pointer took, in the plot's own coordinates.
        """
        self._viewbox.unsetCursor()

        indices = self.select_in_lasso(polygon)
        logger.info("Selected %d points.", len(indices))

        self.points_selected.emit(indices)

    def select_in_lasso(self, polygon: QPolygonF) -> list[int]:
        """ Pick out the points enclosed by a drawn path, washing them over on the plot.

        Args:
            polygon: The path the points are picked from, in the plot's own coordinates.
                Closed as it is read, so a loop left ajar still holds what it went round.
        Returns:
            The image each picked point stands for, as PhenoMe indexes its results. The
            rows of the projection stand in on a frame carrying no such column.
        """
        if self._points is None or self._points.empty:
            return []

        # Two points make a line and a line holds nothing, which is what a click or a straight drag
        # leaves behind
        if polygon.count() < 3:
            self._selected = None
        else:
            inside = [
                polygon.containsPoint(QPointF(x, y), Qt.FillRule.OddEvenFill)
                for x, y in zip(self._points[self._x_column], self._points[self._y_column], strict=True)
            ]
            self._selected = pd.Series(inside, index=self._points.index)

        self._clear_canvas()
        self._draw()

        return self.selected_indices()

    def select_in_box(self, rect: QRectF) -> list[int]:
        """ Pick out the points falling inside a box, washing them over on the plot.

        Args:
            rect: The box the points are picked from, in the plot's own coordinates.
        Returns:
            The image each picked point stands for, as PhenoMe indexes its results. The
            rows of the projection stand in on a frame carrying no such column.
        """
        if self._points is None or self._points.empty:
            return []

        normalised = rect.normalized()
        x_values = self._points[self._x_column]
        y_values = self._points[self._y_column]

        self._selected = (
            x_values.between(normalised.left(), normalised.right())
            & y_values.between(normalised.top(), normalised.bottom())
        )

        self._clear_canvas()
        self._draw()

        return self.selected_indices()

    def select_matching(self, field: str | None, criterion) -> list[int]:
        """ Pick out the points answering a criterion, washing them over on the plot.

        The other way to a selection: the points a box is drawn around, or the points a field and a
        value name. What comes of it is the same either way, this being the same mask the highlight
        fades the plot by, read the right way up.

        Args:
            field: Metadata column or property the criterion is read on. None, or a column the
                projection does not carry, picks nothing out.
            criterion: Either two numbers, read as the ends of a range the value has to fall between,
                or a single value the point has to hold. None picks nothing out.
        Returns:
            The image each picked point stands for, as PhenoMe indexes its results. The
            rows of the projection stand in on a frame carrying no such column.
        """
        if self._points is None or self._points.empty:
            return []

        self._selected = self._matching_rows(field, criterion)

        self._clear_canvas()
        self._draw()

        return self.selected_indices()

    def selected_indices(self) -> list[int]:
        """ The images the current selection stands for.

        Returns:
            The index PhenoMe holds each selected image at, or the rows of the projection
            when it carries no index of its own. Empty when nothing is selected.
        """
        if self._points is None or self._selected is None:
            return []

        picked = self._points[self._selected]
        if INDEX_COLUMN in picked.columns:
            return [int(index) for index in picked[INDEX_COLUMN]]

        return [int(row) for row in np.flatnonzero(self._selected.to_numpy())]

    def clear_selection(self) -> None:
        """ Let go of the selected points, taking the wash off the plot. """
        if self._points is None:
            self._selected = None
            return

        self._selected = None
        self._clear_canvas()
        self._draw()

    def highlight_points(self, field: str | None = None, criterion: str | tuple[float] | None=None) -> None:
        """ Mark out the points answering a criterion, fading back every other one.

        The points left out are drawn faded rather than taken off the canvas, so what was picked is
        read against the cloud it was picked from rather than on an empty plot.

        Args:
            field: Metadata column or property the criterion is read on. None, or a column the
                projection does not carry, leaves every point as it is.
            criterion: Either two numbers, read as the ends of a range the value has to fall between,
                or a single value the point has to hold. None leaves every point as it is.
        """
        if self._points is None:
            return

        # What is faded back is what the criterion does not answer for
        matching = self._matching_rows(field, criterion)
        self._dimmed = None if matching is None else ~matching

        self._clear_canvas()
        self._draw()

    def _matching_rows(self, field: str | None, criterion: str | tuple[float]) -> "pd.Series | None":
        """ Work out which of the drawn rows answer a criterion.

        Read one way by the highlighting, which fades back the rows that do not, and the other by the
        selecting, which picks out the rows that do: one criterion, one answer, two uses.

        Args:
            field: Column the criterion is read on.
            criterion: The range or the value the points are held to.
        Returns:
            True for each row answering the criterion, or None when there is no criterion
            to hold them to and nothing to say.
        """
        column = resolve_column(self._points, field)
        if column is None or criterion is None:
            return None

        values = self._points[column]

        # Two numbers are the ends of a range, anything else is a value to hold the points to. A row
        # with no value answers neither, an image PhenoMe knows nothing about being one the criterion
        # cannot speak for.
        if isinstance(criterion, tuple | list) and len(criterion) == 2:
            low, high = criterion
            matching = values.between(low, high)
        else:
            matching = values == criterion

        return matching.fillna(False).astype(bool)

    def _draw_groups(self, projection_df: pd.DataFrame, x_column: str, y_column: str, column: str) -> None:
        """ Draw one series per value of a column of labels, named in the legend.

        The points are drawn in one order and listed in another: the crowd has to go down before the
        groups that would be lost under it, where the legend is read by looking a label up and has to
        be in the order the labels themselves are in.

        Args:
            projection_df: The projected coordinates and the columns alongside them.
            x_column: Column holding the first component.
            y_column: Column holding the second component.
            column: Column the points are grouped by.
        """
        # PhenoMe leaves the metadata of an image it knows nothing about empty, and the clustering
        # leaves its noise points unlabelled, neither of which should read as a group called 'nan'.
        labels = projection_df[column].astype('object').where(projection_df[column].notna(), MISSING_LABEL)

        # Drawn largest group first, so the small ones land on top of the crowd rather than under it.
        # The count alone would leave same-sized groups in the order pandas happened to hash them in.
        groups = sorted(labels.unique(), key=lambda value: (-(labels == value).sum(), str(value)))

        # The order the legend is listed in, which the colours are taken in as well: a group then
        # keeps the colour it wore when a re-clustering or another projection changed how many points
        # it holds, where a colour taken by size moves to whichever group grew into it.
        listed = sorted(groups, key=group_order)
        places = {group: place for place, group in enumerate(listed)}

        drawn = {}
        for group in groups:
            rows = labels == group
            colour = SERIES_COLOURS[places[group] % N_COLOURS]
            drawn[group] = self._add_series(
                projection_df.loc[rows, x_column],
                projection_df.loc[rows, y_column],
                colour,
                self._dimmed_brushes(colour, rows),
                name=str(group)
            )

        # Each series adds its own entry as it is drawn, which leaves the legend in the order they
        # were drawn in. Filled again here in the order the labels read in, from the same items, so
        # every entry keeps the swatch its group was drawn with.
        self._legend.clear()
        for group in listed:
            self._legend.addItem(drawn[group], str(group))

    def _draw_ramp(self, projection_df: pd.DataFrame, x_column: str, y_column: str, column: str) -> None:
        """ Draw the points of a measured column along a colour ramp, with a colorbar to read it by.

        One series holding every point, each with its own colour: the values run continuously, so
        there is no group to name and nothing for a legend to say.

        Args:
            projection_df: The projected coordinates and the columns alongside them.
            x_column: Column holding the first component.
            y_column: Column holding the second component.
            column: Column the points are coloured by.
        """
        values = projection_df[column].astype(float)
        low, high = float(values.min()), float(values.max())

        # A column holding the one value throughout has no range to spread the points over, and would
        # divide the ramp by zero. Its points are drawn at the bottom of it instead.
        span = high - low
        positions = (values - low) / span if span else pd.Series(0.0, index=values.index)

        colormap = pg.colormap.get(CONTINUOUS_COLORMAP)
        dimmed = [False] * len(positions) if self._dimmed is None else list(self._dimmed)
        brushes = [
            self._brush(MISSING_COLOUR if pd.isna(position) else colormap.map(position, mode='qcolor'), dim)
            for position, dim in zip(positions, dimmed, strict=True)
        ]

        self._plot.addItem(
            pg.ScatterPlotItem(
                x=projection_df[x_column].to_numpy(),
                y=projection_df[y_column].to_numpy(),
                brush=brushes,
                pen=None,
                size=MARKER_SIZE
            )
        )

        self._colorbar = pg.ColorBarItem(
            values=(low, high),
            colorMap=colormap,
            label=column,
            interactive=False,
            orientation='horizontal',
            width=COLORBAR_THICKNESS
        )

        # The bar is framed by an axis on each of its four sides. Lying down it carries both the
        # values and the label along the bottom one and draws the other three as the frame, so all
        # four are toned down to match the plot's own axes.
        for axis in ('left', 'right', 'top', 'bottom'):
            self._colorbar.getAxis(axis).setPen(AXIS_COLOUR)
            self._colorbar.getAxis(axis).setTextPen(AXIS_COLOUR)

        # The values are read off a bar a fraction of pyqtgraph's own, and are written to match it:
        # in a smaller hand, and on marks short enough not to outweigh the ramp they point at
        ticks = QFont()
        ticks.setPointSize(COLORBAR_TICK_SIZE)
        self._colorbar.axis.setStyle(tickFont=ticks, tickLength=COLORBAR_TICK_LENGTH)

        # The column is named over the bar rather than under it, where the name would be left hanging
        # a line below the values by the room the axis keeps for it. Styled in full, a label taking
        # the colour it is written in from the same call that sizes it.
        self._colorbar.axis.setLabel(None)
        self._colorbar.getAxis('top').setLabel(
            column, color=AXIS_COLOUR, **{'font-size': f'{COLORBAR_LABEL_SIZE}pt'}
        )

        # Drawn over the points, so it has to let the pointer through to them: a point under the bar
        # is still hovered and still clicked
        for item in (self._colorbar, self._colorbar.bar):
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.setAcceptHoverEvents(False)

        # Parented to the view box rather than added to the canvas' layout, which is what keeps it
        # off the plot's width. Its place is in pixels from the corner, so it stays the same size and
        # in the same corner however the points underneath it are panned and zoomed.
        self._colorbar.setParentItem(self._plot.vb)
        self._place_colorbar()

    def _place_colorbar(self) -> None:
        """ Sit the colorbar in the top right of the plot area, and keep it there as the plot resizes.

        Called again whenever the view box is resized, the corner it is measured from moving with
        every resize of the window and with every axis that grows wide enough to push the plot along.
        """
        if self._colorbar is None:
            return

        # The view box's own rectangle starts at its top left corner, which is also where the bar's
        # place is measured from, so the corner is the width alone
        area = self._plot.vb.boundingRect()
        left = area.width() - COLORBAR_LENGTH - COLORBAR_MARGIN

        # Held in a layout an item is given its size; floated it has to be told what it is
        self._colorbar.setGeometry(QRectF(left, COLORBAR_MARGIN, COLORBAR_LENGTH, COLORBAR_HEIGHT))

    def _clear_canvas(self) -> None:
        """ Take everything drawn off the canvas, leaving the frame it was drawn from.

        The legend and the colorbar are cleared by hand: neither is one of the plot's data items, so
        clearing the plot leaves them behind and the next drawing would land on top of them. They
        also stand in for each other, so each has to undo what the other left.

        Kept apart from _clear() for the redrawing: dimming the points draws the same frame again,
        and would have nothing to draw if the frame went with the items.
        """
        self._legend.clear()
        self._plot.clear()

        # Taken off the scene it was floated onto, the canvas' layout never having held it
        if self._colorbar is not None:
            self.scene().removeItem(self._colorbar)
            self._colorbar = None

        # The ring and the selection wash went with the plot's other items, leaving only the
        # references to them to let go of
        self._hovered = None
        self._highlight = None
        self._selection = None

    def _clear(self) -> None:
        """ Take everything off the canvas, the frame it was drawn from included. """
        self._clear_canvas()

        # Dropped along with the points, so a hover over a cleared canvas answers for nothing, and
        # so a highlight does not outlive the projection it was picked against
        self._points = None
        self._x_column = None
        self._y_column = None
        self._color_column = None
        self._dimmed = None
        self._selected = None

    def _nearest_point(self, position: QPointF) -> int | None:
        """ Find the drawn point nearest a position, so long as one is near enough to have been meant.

        The distance is measured in pixels rather than in the projection's own units: how far apart
        two points look is what the pointer is aimed by, and the same gap means a different distance
        from one projection to the next and again at every zoom.

        Args:
            position: Where the pointer is, in the plot's own coordinates.
        Returns:
            Row of the drawn projection the position falls on, or None when the canvas is empty
            or nothing is within reach.
        """
        if self._points is None or self._points.empty:
            return None

        # Size of one pixel in plot coordinates, which is also what turns the reach into a distance.
        # Each axis has its own, the two rarely being drawn to the same scale.
        x_scale, y_scale = self._plot.vb.viewPixelSize()
        if not x_scale or not y_scale:
            return None

        x_offsets = (self._points[self._x_column].to_numpy() - position.x()) / x_scale
        y_offsets = (self._points[self._y_column].to_numpy() - position.y()) / y_scale
        distances = np.hypot(x_offsets, y_offsets)

        nearest = int(distances.argmin())
        return nearest if distances[nearest] <= HOVER_RADIUS else None

    def _on_mouse_clicked(self, event: MouseClickEvent) -> None:
        """ Announce the point that was clicked, for whoever can show the image it stands for.

        The point is looked for again rather than taken from the hovering: the two answer the same
        way for a pointer that moved onto a point before pressing, and a click arriving any other way
        would otherwise be answered with wherever the pointer was last seen.

        Args:
            event: The click as the canvas reports it, holding where it landed and
                which button was pressed. Only the left one picks a point out, the right one
                belonging to the view's own menu.
        """
        if event.button() != Qt.MouseButton.LeftButton:
            return

        row = self._nearest_point(self._plot.vb.mapSceneToView(event.scenePos()))
        if row is None:
            return

        # Marked as answered, so the click is not taken up again by the view underneath
        event.accept()

        logger.info("Clicked the point at row %d of the projection.", row)
        self.point_clicked.emit(row)

    def _on_mouse_over(self, position: QPointF) -> None:
        """ Show what is known about the point under the pointer, and take the tooltip away again.

        The canvas reports every move over the whole of itself, the axes and the margins included, so
        the ones landing outside the plot are turned away before any point is looked for.

        Args:
            position: Where the pointer is, in the scene's coordinates, as the canvas
                reports every move over itself.
        """
        row = None
        if self._plot.sceneBoundingRect().contains(position):
            row = self._nearest_point(self._plot.vb.mapSceneToView(position))

        # Only a move onto another point is worth redrawing the ring for, where the tooltip is shown
        # again on every move
        if row != self._hovered:
            self._highlight_point(row)
            self._hovered = row

        if row is None:
            QToolTip.hideText()
            return

        # Shown again on every move rather than only when the point changes, which is what keeps the
        # tooltip alive under a resting pointer instead of letting it time out.
        QToolTip.showText(self.mapToGlobal(self.mapFromScene(position)), self._describe_point(row))

    def _highlight_point(self, row: int | None) -> None:
        """ Ring the hovered point, and take the ring off the point that was hovered before.

        Drawn as an item of its own rather than by repainting the point: the points are drawn grouped
        into a handful of series, and a single one of them cannot be given a pen of its own without
        breaking its series up. Being added last also puts the ring above the cloud, where a point
        under a crowded patch would otherwise be ringed underneath it.

        Args:
            row: Row of the drawn projection to ring. None takes the ring off and rings nothing.
        """
        if self._highlight is not None:
            self._plot.removeItem(self._highlight)
            self._highlight = None

        if row is None or self._points is None:
            return

        point = self._points.iloc[row]
        self._highlight = pg.ScatterPlotItem(
            x=[float(point[self._x_column])],
            y=[float(point[self._y_column])],
            pen=pg.mkPen(HIGHLIGHT_COLOUR, width=HIGHLIGHT_WIDTH),
            brush=None,
            size=HIGHLIGHT_SIZE
        )
        self._plot.addItem(self._highlight)

    def _describe_point(self, row: int) -> str:
        """ Write out what the projection holds about one image, for the tooltip to show.

        The image is named first, then the value it is currently coloured by - the whole reason a
        point is where it is on screen - and then the rest of what is known about it. Only the
        metadata follows: the properties are many, and the one that matters is already at the top
        whenever it is what the colouring reads.

        Args:
            row: Row of the drawn projection to describe.
        Returns:
            The image's name and values, one per line.
        """
        point = self._points.iloc[row]
        lines = []

        if IMAGE_COLUMN in self._points.columns:
            lines.append(str(point[IMAGE_COLUMN]))

        if self._color_column is not None:
            lines.append(f'{self._color_column}: {format_value(point[self._color_column])}')

        lines += [
            f'{column}: {format_value(point[column])}'
            for column in self._metadata_columns()
            if column != self._color_column
        ]

        return '\n'.join(lines)

    def _metadata_columns(self) -> list[str]:
        """ The metadata columns of the drawn projection.

        Returns:
            The metadata columns, in the order the projection carries them.
        """
        return metadata_columns(self._points)

    def _brush(self, colour, dimmed: bool = False) -> "pg.QtGui.QBrush":
        """ The brush a point is drawn with, faded when the highlight leaves it out.

        Args:
            colour: The colour the point would be drawn in, as a name or as a QColor.
            dimmed: True to fade it back, False to leave it as it is.
        Returns:
            The colour, at full opacity or at the dimmed one.
        """
        brush = pg.mkBrush(colour)
        if dimmed:
            faded = brush.color()
            faded.setAlpha(DIMMED_ALPHA)
            brush.setColor(faded)
        else:
            faded = brush.color()
            faded.setAlpha(NORMAL_ALPHA)
            brush.setColor(faded)

        return brush

    def _dimmed_brushes(self, colour, rows=None) -> list | None:
        """ One brush per point, faded where the highlight leaves the point out.

        Args:
            colour: The colour the points would be drawn in.
            rows: Mask picking the rows being drawn out of the frame, for a set holding some of them.
                None for a set holding all of them.
        Returns:
            A brush for each point, or None while nothing is highlighted and they can share one.
        """
        if self._dimmed is None:
            return None

        dimmed = self._dimmed if rows is None else self._dimmed[rows]
        return [self._brush(colour, dim) for dim in dimmed]

    def _add_series(
        self, x: pd.Series, y: pd.Series, colour, brushes: list | None = None, name: str | None = None
    ) -> pg.ScatterPlotItem:
        """ Draw one group of points.

        Args:
            x: Coordinates along the first component.
            y: Coordinates along the second component.
            colour: Colour the group is drawn in, and the colour its legend entry is drawn in.
            brushes: A brush per point, for a group the highlight has split. None to leave
                every point on the group's own colour.
            name: Name of the group, which is what puts it in the legend. None to leave it out.
        Returns:
            The series drawn, which is what a legend entry is made from.
        """
        series = pg.ScatterPlotItem(
            x=x.to_numpy(),
            y=y.to_numpy(),
            brush=self._brush(colour),
            pen=None,
            size=MARKER_SIZE,
            name=name
        )

        # Set after the fact rather than passed in: the legend draws its swatch from the brush the
        # series was built with, and a list of them leaves it drawing something that is no colour at
        # all. Setting them here leaves that one where the group's colour is.
        if brushes is not None:
            series.setBrush(brushes)

        self._plot.addItem(series)
        return series

    def export_plot(self, path: str | Path, extension: str) -> None:
        """ Write the plot as it is currently drawn to a file.

        What is written is the plot item rather than the canvas, which is the same picture: the
        legend and the colorbar are parented to the view box, so both go down with the points, and
        the background they are drawn on goes with them.

        Args:
            path: Where the file is written, the caller having named it.
            extension: The file type to write, 'svg' or 'png'. Read without its case or a
                leading dot, so a name's own suffix can be handed straight over.
        Raises:
            ValueError: The extension names a file type the canvas cannot write.
        """
        key = extension.lower().lstrip('.')

        if key not in EXPORT_FORMATS:
            offered = ', '.join(f"'{name}'" for name in EXPORT_FORMATS)
            raise ValueError(f"Unrecognized file type '{extension}'; must be one of {offered}.")

        self.setBackground(None)
        EXPORT_FORMATS[key](self._plot).export(str(path))
        logger.info("Wrote the plot to '%s'.", path)
        self.setBackground(SURFACE)

