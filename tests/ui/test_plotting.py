"""Tests for the explorer canvas.

The canvas is handed a projection and the name of a column to colour it by, and knows nothing else,
so the frames here are written by hand rather than computed.
"""

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import QPointF, Qt
from qtpy.QtWidgets import QToolTip

from phenome_gui.ui.plotting import (
    DIMMED_ALPHA,
    HIGHLIGHT_COLOUR,
    HOVER_RADIUS,
    MISSING_COLOUR,
    SERIES_COLOURS,
    ExplorerPlot,
)


@pytest.fixture
def plot(qtbot) -> ExplorerPlot:
    """ An empty canvas. """
    canvas = ExplorerPlot()
    qtbot.addWidget(canvas)
    return canvas


def projection(groups: list[str] | None = None, n_components: int = 2, column: str = 'Drug_name') -> pd.DataFrame:
    """ A frame shaped like the one PhenoMe's reduction returns.

    Args:
        groups: Value of the colouring column for each row. None for a frame without one.
        n_components: Number of coordinate columns, as the number of dimensions asked for.
        column: Name of the colouring column, capitalised the way PhenoMe writes it.
    """
    length = len(groups) if groups is not None else 3
    frame = pd.DataFrame(
        {f'Component {i + 1}': np.arange(length, dtype=float) + i for i in range(n_components)}
    )
    frame['Image'] = [f'img_{i:03d}.tif' for i in range(length)]
    frame['Index'] = range(length)
    if groups is not None:
        frame[column] = groups

    # A second metadata column, so the tooltip has something to read out beyond the coloured one.
    # PhenoMe capitalises the metadata it copies in and hands the values back as text.
    if column != 'Time':
        frame['Time'] = [('0', '24')[row % 2] for row in range(length)]
    return frame


def series(plot: ExplorerPlot) -> dict[str | None, int]:
    """ The drawn groups, as a name to number of points mapping. """
    return {item.opts.get('name'): len(item.getData()[0]) for item in plot._plot.listDataItems()}


def colours(plot: ExplorerPlot) -> list[str]:
    """ The colour of each drawn group, in the order they were drawn. """
    return [item.opts['brush'].color().name() for item in plot._plot.listDataItems()]


def legend_labels(plot: ExplorerPlot) -> list[str]:
    """ The names currently listed in the legend. """
    return [label.text for _, label in plot._legend.items]


def point_colours(plot: ExplorerPlot) -> list[str]:
    """ The colour of each point of the one drawn series, in the order they were given. """
    (item,) = plot._plot.listDataItems()
    return [spot.brush().color().name() for spot in item.points()]


def colorbar_label(plot: ExplorerPlot) -> str:
    """ The name written alongside the colorbar, whichever of its two axes carries it. """
    labels = [plot._colorbar.getAxis(axis).labelText for axis in ('left', 'right')]
    return next((label for label in labels if label), '')


def measured(values: list, column: str = 'area') -> pd.DataFrame:
    """ A projection carrying a measured column, as the properties arrive alongside the coordinates. """
    frame = projection([''] * len(values))
    return frame.drop(columns='Drug_name').assign(**{column: values})


# ----------------------------------------------------------------------------
# Drawing the groups
# ----------------------------------------------------------------------------


def test_each_group_is_drawn_and_named(plot: ExplorerPlot) -> None:
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    assert series(plot) == {'ctrl': 2, 'drugA': 1}
    assert sorted(legend_labels(plot)) == ['ctrl', 'drugA']


def test_the_points_of_a_group_are_its_own(plot: ExplorerPlot) -> None:
    frame = projection(['ctrl', 'drugA', 'ctrl'])

    plot.plot_projection(frame, 'drug_name')

    drawn = {item.opts['name']: item.getData() for item in plot._plot.listDataItems()}
    assert list(drawn['drugA'][0]) == [1.0]
    assert list(drawn['drugA'][1]) == [2.0]
    assert sorted(drawn['ctrl'][0]) == [0.0, 2.0]


def test_the_largest_group_is_drawn_first(plot: ExplorerPlot) -> None:
    """Drawn in that order so the small groups land on top of the crowd rather than under it."""
    plot.plot_projection(projection(['drugA', 'ctrl', 'ctrl', 'ctrl']), 'drug_name')

    assert [item.opts['name'] for item in plot._plot.listDataItems()] == ['ctrl', 'drugA']


def test_the_colours_are_taken_in_order(plot: ExplorerPlot) -> None:
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    assert colours(plot) == SERIES_COLOURS[:2]


def test_the_colours_start_over_past_the_last_one(plot: ExplorerPlot) -> None:
    """A ninth group wears the first colour again, telling it apart only by where its points sit."""
    groups = [f'group_{i:02d}' for i in range(len(SERIES_COLOURS) + 1)]

    plot.plot_projection(projection(groups), 'drug_name')

    assert len(series(plot)) == len(SERIES_COLOURS) + 1
    assert colours(plot) == [*SERIES_COLOURS, SERIES_COLOURS[0]]


def test_groups_of_the_same_size_keep_a_settled_order(plot: ExplorerPlot) -> None:
    """Ordered by size alone, same-sized groups would swap colours from one redraw to the next."""
    plot.plot_projection(projection(['b', 'a', 'c']), 'drug_name')
    first = colours(plot)

    plot.plot_projection(projection(['b', 'a', 'c']), 'drug_name')

    assert [item.opts['name'] for item in plot._plot.listDataItems()] == ['a', 'b', 'c']
    assert colours(plot) == first


# ----------------------------------------------------------------------------
# Finding the column to colour by
# ----------------------------------------------------------------------------


def test_the_capitalised_column_is_found(plot: ExplorerPlot) -> None:
    """PhenoMe capitalises what it copies in, so 'drug_name' is asked for but 'Drug_name' is there."""
    plot.plot_projection(projection(['ctrl', 'drugA', 'drugA']), 'drug_name')

    assert series(plot) == {'ctrl': 1, 'drugA': 2}


@pytest.mark.parametrize('color_by', [None, '', 'cluster'], ids=['none', 'empty', 'not-in-frame'])
def test_without_a_column_the_points_are_one_unnamed_group(plot: ExplorerPlot, color_by) -> None:
    """A column added to the metadata after the projection ran is not in it, which must not raise."""
    plot.plot_projection(projection(['ctrl', 'drugA', 'drugA']), color_by)

    assert series(plot) == {None: 3}
    assert legend_labels(plot) == []


def test_missing_values_are_grouped_rather_than_named_nan(plot: ExplorerPlot) -> None:
    """The clustering leaves its noise points unlabelled, which would otherwise read as a group 'nan'."""
    frame = projection(['ctrl', 'ctrl', 'drugA'])
    frame['cluster'] = pd.array([0, pd.NA, pd.NA], dtype='Int64')

    plot.plot_projection(frame, 'cluster')

    assert series(plot) == {'Unknown': 2, '0': 1}


# ----------------------------------------------------------------------------
# Measured columns, drawn along a ramp
# ----------------------------------------------------------------------------


def test_a_measured_column_is_drawn_as_one_series_with_a_colorbar(plot: ExplorerPlot) -> None:
    plot.plot_projection(measured([10.5, 20.5, 30.5]), 'area')

    assert series(plot) == {None: 3}
    assert legend_labels(plot) == []
    assert plot._colorbar is not None
    assert plot._colorbar.values == (10.5, 30.5)


def test_the_colorbar_is_named_after_the_column(plot: ExplorerPlot) -> None:
    plot.plot_projection(measured([1.0, 2.0], column='eccentricity'), 'eccentricity')

    assert 'eccentricity' in colorbar_label(plot)


def test_the_points_run_along_the_ramp(plot: ExplorerPlot) -> None:
    """Every point takes its own colour from its value, rather than the whole series taking one."""
    plot.plot_projection(measured([0.0, 5.0, 10.0]), 'area')

    colours = point_colours(plot)
    assert len(set(colours)) == 3


def test_whole_numbers_are_still_a_measurement(plot: ExplorerPlot) -> None:
    """PhenoMe would call this one categorical: 'area' is a pixel count, whole and here barely varied.

    Drawn as groups it would be a legend of one-image entries, which is why the name rather than the
    values is what makes a column discrete here.
    """
    plot.plot_projection(measured([120.0, 340.0, 512.0]), 'area')

    assert series(plot) == {None: 3}
    assert plot._colorbar is not None


def test_a_numbered_label_column_is_still_a_group(plot: ExplorerPlot) -> None:
    """The cluster labels are numbers, and a ramp over them would read as a measurement they are not."""
    frame = measured([0, 1, 1], column='cluster')

    plot.plot_projection(frame, 'cluster')

    assert series(plot) == {'1': 2, '0': 1}
    assert plot._colorbar is None


def test_true_and_false_are_two_groups(plot: ExplorerPlot) -> None:
    """A boolean is numeric to pandas, but two named groups say more than two shades of a ramp."""
    plot.plot_projection(measured([True, False, False], column='is_control'), 'is_control')

    assert series(plot) == {'False': 2, 'True': 1}
    assert plot._colorbar is None


def test_a_point_without_a_value_is_left_neutral(plot: ExplorerPlot) -> None:
    """It has no place on the ramp, and dropping it would leave a hole in the cloud."""
    plot.plot_projection(measured([10.0, float('nan'), 30.0]), 'area')

    assert series(plot) == {None: 3}
    assert point_colours(plot)[1] == MISSING_COLOUR
    assert point_colours(plot)[0] != MISSING_COLOUR


def test_a_measured_column_holding_one_value_does_not_divide_by_zero(plot: ExplorerPlot) -> None:
    plot.plot_projection(measured([7.0, 7.0, 7.0]), 'area')

    assert series(plot) == {None: 3}
    assert plot._colorbar.values == (7.0, 7.0)


def test_the_two_kinds_of_colouring_undo_each_other(plot: ExplorerPlot) -> None:
    """Each leaves chrome of its own behind, so switching either way has to take the other's away."""
    plot.plot_projection(measured([1.0, 2.0, 3.0]), 'area')
    assert plot._colorbar is not None

    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    assert plot._colorbar is None
    assert sorted(legend_labels(plot)) == ['ctrl', 'drugA']

    plot.plot_projection(measured([1.0, 2.0, 3.0]), 'area')

    assert plot._colorbar is not None
    assert legend_labels(plot) == []


# ----------------------------------------------------------------------------
# What cannot be drawn
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('n_components', [1, 3], ids=['too-few', 'three-dimensional'])
def test_only_two_dimensions_are_drawn(plot: ExplorerPlot, n_components: int) -> None:
    """The canvas is flat, so a 3D projection is left undrawn rather than shown as a slice of itself."""
    plot.plot_projection(projection(['ctrl', 'drugA', 'drugA'], n_components=n_components), 'drug_name')

    assert series(plot) == {}


@pytest.mark.parametrize('frame', [None, pd.DataFrame()], ids=['none', 'empty'])
def test_an_empty_projection_clears_the_canvas(plot: ExplorerPlot, frame) -> None:
    plot.plot_projection(projection(['ctrl', 'drugA']), 'drug_name')

    plot.plot_projection(frame, 'drug_name')

    assert series(plot) == {}
    assert legend_labels(plot) == []


def test_redrawing_replaces_what_was_there(plot: ExplorerPlot) -> None:
    """The legend is not one of the plot's items, so its entries pile up unless cleared by hand."""
    plot.plot_projection(projection(['ctrl', 'drugA', 'drugA']), 'drug_name')

    plot.plot_projection(projection(['x', 'y', 'y'], column='Time'), 'time')

    assert series(plot) == {'y': 2, 'x': 1}
    assert sorted(legend_labels(plot)) == ['x', 'y']


# ----------------------------------------------------------------------------
# The axes
# ----------------------------------------------------------------------------


def test_the_axes_are_named_after_the_components(plot: ExplorerPlot) -> None:
    plot.plot_projection(projection(['ctrl', 'drugA']), 'drug_name')

    assert plot._plot.getAxis('bottom').labelText == 'Component 1'
    assert plot._plot.getAxis('left').labelText == 'Component 2'


# ----------------------------------------------------------------------------
# Marking out the points a criterion answers for
# ----------------------------------------------------------------------------


def by_row(plot: ExplorerPlot, read) -> list:
    """ Read something off every drawn point, in the order the frame holds them.

    The points are drawn a group at a time, so they are put back in frame order by their coordinates,
    which the projection helper numbers from zero.

    Args:
        plot: The canvas to read.
        read: What to read off each drawn point, given its brush colour.
    """
    values = {}
    for item in plot._plot.listDataItems():
        for spot in item.points():
            values[round(spot.pos().x())] = read(spot.brush().color())

    return [values[row] for row in sorted(values)]


def alphas(plot: ExplorerPlot) -> list[int]:
    """ The opacity of every drawn point, in frame order. """
    return by_row(plot, lambda colour: colour.alpha())


def hues(plot: ExplorerPlot) -> list[str]:
    """ The colour of every drawn point, in frame order, the opacity aside. """
    return by_row(plot, lambda colour: colour.name())


def test_a_value_leaves_its_points_alone_and_fades_the_rest(plot: ExplorerPlot) -> None:
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    plot.highlight_points('drug_name', 'drugA')

    assert alphas(plot) == [DIMMED_ALPHA, DIMMED_ALPHA, 255]


def test_the_groups_and_their_colours_survive_the_marking(plot: ExplorerPlot) -> None:
    """The plot still reads as what it was, with the picked points marked out on it."""
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')
    before = hues(plot)

    plot.highlight_points('drug_name', 'drugA')

    assert series(plot) == {'ctrl': 2, 'drugA': 1}
    assert sorted(legend_labels(plot)) == ['ctrl', 'drugA']
    assert hues(plot) == before


def test_the_legend_keeps_its_colours_while_the_points_are_marked(plot: ExplorerPlot) -> None:
    """The legend draws its swatch from the colour the series was built with, not from its points."""
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    plot.highlight_points('drug_name', 'drugA')

    assert colours(plot) == SERIES_COLOURS[:2]


def test_a_range_leaves_the_points_inside_it_alone(plot: ExplorerPlot) -> None:
    plot.plot_projection(measured([10.0, 20.0, 30.0]), 'area')

    plot.highlight_points('area', (15.0, 25.0))

    assert alphas(plot) == [DIMMED_ALPHA, 255, DIMMED_ALPHA]
    assert plot._colorbar is not None


def test_the_ends_of_a_range_are_inside_it(plot: ExplorerPlot) -> None:
    """The slider opens on the whole span, which has to mark every point rather than none."""
    plot.plot_projection(measured([10.0, 20.0, 30.0]), 'area')

    plot.highlight_points('area', (10.0, 30.0))

    assert alphas(plot) == [255, 255, 255]


def test_a_capitalised_column_is_found_from_its_gui_name(plot: ExplorerPlot) -> None:
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    plot.highlight_points('drug_name', 'ctrl')

    assert alphas(plot) == [255, 255, DIMMED_ALPHA]


def test_a_point_without_a_value_is_left_out(plot: ExplorerPlot) -> None:
    """A criterion cannot speak for an image PhenoMe knows nothing about."""
    plot.plot_projection(measured([10.0, float('nan'), 30.0]), 'area')

    plot.highlight_points('area', (5.0, 35.0))

    assert alphas(plot) == [255, DIMMED_ALPHA, 255]


@pytest.mark.parametrize(
    ('field', 'criterion'),
    [
        pytest.param(None, None, id='nothing-picked'),
        pytest.param('drug_name', None, id='no-criterion'),
        pytest.param('gone', 'ctrl', id='not-in-the-projection'),
    ],
)
def test_nothing_to_mark_leaves_every_point_alone(plot: ExplorerPlot, field, criterion) -> None:
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    plot.highlight_points(field, criterion)

    assert alphas(plot) == [255, 255, 255]


def test_the_marking_can_be_taken_off(plot: ExplorerPlot) -> None:
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')
    plot.highlight_points('drug_name', 'drugA')

    plot.highlight_points()

    assert alphas(plot) == [255, 255, 255]


def test_marking_an_empty_canvas_does_nothing(plot: ExplorerPlot) -> None:
    plot.highlight_points('drug_name', 'ctrl')

    assert series(plot) == {}


def test_a_new_projection_is_drawn_unmarked(plot: ExplorerPlot) -> None:
    """The marking was picked against the points it was applied to, not against these."""
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')
    plot.highlight_points('drug_name', 'drugA')

    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    assert alphas(plot) == [255, 255, 255]


def test_the_marking_holds_without_a_colouring(plot: ExplorerPlot) -> None:
    """The field marked on is picked apart from the one coloured by, so either can stand alone."""
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), None)

    plot.highlight_points('drug_name', 'drugA')

    assert alphas(plot) == [DIMMED_ALPHA, DIMMED_ALPHA, 255]


# ----------------------------------------------------------------------------
# Hovering a point
# ----------------------------------------------------------------------------

# The points the projection helper lays down, as (Component 1, Component 2)
POINTS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


@pytest.fixture
def hovered(plot: ExplorerPlot) -> ExplorerPlot:
    """ A canvas with three points drawn on it, at a settled scale.

    The range is pinned so a distance in pixels is a known distance in plot coordinates, the canvas
    never being shown and so never auto-ranging to what it holds.
    """
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')
    plot._plot.vb.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)
    return plot


def pixel_size(plot: ExplorerPlot) -> tuple[float, float]:
    """ How much of the plot's own coordinates one pixel covers, along each axis. """
    return plot._plot.vb.viewPixelSize()


def in_scene(plot: ExplorerPlot, x: float, y: float) -> QPointF:
    """ A position given in plot coordinates, as the scene reports it to the hover handler. """
    return plot._plot.vb.mapViewToScene(QPointF(x, y))


class FakeClick:
    """ Stands in for a pyqtgraph mouse click, which takes a real scene event to build. """

    def __init__(self, position: QPointF, button=Qt.MouseButton.LeftButton) -> None:
        self._position = position
        self._button = button
        self.accepted = False

    def button(self):
        """ The button that was pressed. """
        return self._button

    def scenePos(self) -> QPointF:  # noqa: N802 - the name pyqtgraph calls it by
        """ Where the click landed, in the scene's coordinates. """
        return self._position

    def accept(self) -> None:
        """ Take the click as answered, so the view underneath leaves it alone. """
        self.accepted = True


def ringed(plot: ExplorerPlot) -> tuple[float, float] | None:
    """ Where the highlight ring currently sits, or None when no point is ringed. """
    if plot._highlight is None:
        return None

    x, y = plot._highlight.getData()
    return float(x[0]), float(y[0])


def test_the_point_under_the_pointer_is_the_one_found(hovered: ExplorerPlot) -> None:
    assert hovered._nearest_point(QPointF(*POINTS[1])) == 1


def test_a_point_within_reach_is_still_found(hovered: ExplorerPlot) -> None:
    x_scale, _ = pixel_size(hovered)
    just_short = POINTS[0][0] + (HOVER_RADIUS - 4) * x_scale

    assert hovered._nearest_point(QPointF(just_short, POINTS[0][1])) == 0


def test_a_point_out_of_reach_is_not_found(hovered: ExplorerPlot) -> None:
    """The reach is what keeps an empty stretch of canvas from answering with the whole cloud."""
    x_scale, _ = pixel_size(hovered)
    too_far = POINTS[0][0] + (HOVER_RADIUS + 6) * x_scale

    assert hovered._nearest_point(QPointF(too_far, POINTS[0][1])) is None


def test_the_reach_is_measured_in_pixels_rather_than_in_the_projection(hovered: ExplorerPlot) -> None:
    """A projection spanning thousands and one spanning fractions have to hover the same way."""
    x_scale, _ = pixel_size(hovered)
    off_by_ten_pixels = QPointF(POINTS[1][0] + 10 * x_scale, POINTS[1][1])
    assert hovered._nearest_point(off_by_ten_pixels) == 1

    # The same gap in plot coordinates, now far more than ten pixels across
    hovered._plot.vb.setRange(xRange=(0, 0.5), yRange=(0, 10), padding=0)

    assert hovered._nearest_point(off_by_ten_pixels) is None


def test_the_nearer_of_two_points_is_the_one_found(hovered: ExplorerPlot) -> None:
    x_scale, y_scale = pixel_size(hovered)
    nearer_to_the_second = QPointF(POINTS[1][0] + 2 * x_scale, POINTS[1][1] + 2 * y_scale)

    assert hovered._nearest_point(nearer_to_the_second) == 1


def test_nothing_is_found_on_a_canvas_with_nothing_drawn(plot: ExplorerPlot) -> None:
    assert plot._nearest_point(QPointF(0.0, 0.0)) is None


def test_nothing_is_found_once_the_canvas_is_cleared(hovered: ExplorerPlot) -> None:
    """The points go with the drawing, so a hover afterwards answers for nothing."""
    hovered.plot_projection(None)

    assert hovered._nearest_point(QPointF(*POINTS[1])) is None


# ----------------------------------------------------------------------------
# What the tooltip says
# ----------------------------------------------------------------------------


def test_hovering_a_point_names_it_and_reads_it_out(hovered: ExplorerPlot) -> None:
    hovered._on_mouse_over(in_scene(hovered, *POINTS[2]))

    lines = QToolTip.text().splitlines()
    assert lines[0] == 'img_002.tif'
    assert lines[1] == 'Drug_name: drugA'
    assert 'Time: 0' in lines


def test_the_colouring_value_is_read_out_first(hovered: ExplorerPlot) -> None:
    """It is the reason the point is the colour it is, so it comes before the rest."""
    hovered.plot_projection(projection(['ctrl', 'ctrl', 'drugA'], column='Time'), 'time')
    hovered._plot.vb.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)

    hovered._on_mouse_over(in_scene(hovered, *POINTS[0]))

    assert QToolTip.text().splitlines()[1] == 'Time: ctrl'


def test_a_measured_column_is_read_out_to_a_few_figures(hovered: ExplorerPlot) -> None:
    hovered.plot_projection(measured([1234.5678, 2.0, 3.0]), 'area')
    hovered._plot.vb.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)

    hovered._on_mouse_over(in_scene(hovered, *POINTS[0]))

    assert 'area: 1235' in QToolTip.text()


def test_an_uncoloured_point_is_still_named_and_read_out(hovered: ExplorerPlot) -> None:
    hovered.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), None)
    hovered._plot.vb.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)

    hovered._on_mouse_over(in_scene(hovered, *POINTS[1]))

    lines = QToolTip.text().splitlines()
    assert lines[0] == 'img_001.tif'
    assert lines[1] == 'Drug_name: ctrl'
    assert not any(line.startswith('None') for line in lines)


def test_the_colouring_column_is_read_out_once(hovered: ExplorerPlot) -> None:
    """It is a metadata column like any other, and would otherwise be written twice."""
    hovered._on_mouse_over(in_scene(hovered, *POINTS[0]))

    assert QToolTip.text().count('Drug_name:') == 1


def test_hovering_nothing_takes_the_tooltip_away(hovered: ExplorerPlot) -> None:
    """Asserted on the canvas rather than on Qt's tooltip, which an offscreen run never takes down."""
    hovered._on_mouse_over(in_scene(hovered, *POINTS[0]))
    assert hovered._hovered == 0

    x_scale, _ = pixel_size(hovered)
    hovered._on_mouse_over(in_scene(hovered, POINTS[0][0] + 40 * x_scale, POINTS[0][1]))

    assert hovered._hovered is None


def test_hovering_off_the_plot_takes_the_tooltip_away(hovered: ExplorerPlot) -> None:
    """The canvas reports every move over itself, the axes and the margins included."""
    hovered._on_mouse_over(in_scene(hovered, *POINTS[0]))
    assert hovered._hovered == 0

    hovered._on_mouse_over(QPointF(-500.0, -500.0))

    assert hovered._hovered is None


def test_clearing_the_canvas_forgets_the_hovered_point(hovered: ExplorerPlot) -> None:
    hovered._on_mouse_over(in_scene(hovered, *POINTS[0]))

    hovered.plot_projection(None)

    assert hovered._hovered is None
    assert ringed(hovered) is None


# ----------------------------------------------------------------------------
# The ring around the hovered point
# ----------------------------------------------------------------------------


def test_the_hovered_point_is_ringed(hovered: ExplorerPlot) -> None:
    hovered._on_mouse_over(in_scene(hovered, *POINTS[1]))

    assert ringed(hovered) == POINTS[1]


def test_the_ring_is_a_white_outline(hovered: ExplorerPlot) -> None:
    """White so it reads against every colour of the palette and of the ramp alike."""
    hovered._on_mouse_over(in_scene(hovered, *POINTS[1]))

    (spot,) = hovered._highlight.points()
    assert spot.pen().color().name() == HIGHLIGHT_COLOUR
    assert spot.brush().style() == Qt.BrushStyle.NoBrush


def test_the_ring_is_drawn_over_the_cloud(hovered: ExplorerPlot) -> None:
    """A point in a crowded patch would otherwise be ringed underneath the points around it."""
    hovered._on_mouse_over(in_scene(hovered, *POINTS[1]))

    assert hovered._plot.listDataItems()[-1] is hovered._highlight


def test_the_ring_follows_the_pointer_to_another_point(hovered: ExplorerPlot) -> None:
    hovered._on_mouse_over(in_scene(hovered, *POINTS[0]))

    hovered._on_mouse_over(in_scene(hovered, *POINTS[2]))

    assert ringed(hovered) == POINTS[2]
    assert len([item for item in hovered._plot.listDataItems() if item is hovered._highlight]) == 1


def test_the_ring_is_taken_off_when_the_pointer_leaves(hovered: ExplorerPlot) -> None:
    hovered._on_mouse_over(in_scene(hovered, *POINTS[0]))
    assert ringed(hovered) == POINTS[0]

    x_scale, _ = pixel_size(hovered)
    hovered._on_mouse_over(in_scene(hovered, POINTS[0][0] + 40 * x_scale, POINTS[0][1]))

    assert ringed(hovered) is None


def test_resting_on_a_point_does_not_ring_it_again(hovered: ExplorerPlot) -> None:
    """The tooltip is shown again on every move; the ring is only redrawn when the point changes."""
    hovered._on_mouse_over(in_scene(hovered, *POINTS[1]))
    first = hovered._highlight

    x_scale, _ = pixel_size(hovered)
    hovered._on_mouse_over(in_scene(hovered, POINTS[1][0] + 2 * x_scale, POINTS[1][1]))

    assert hovered._highlight is first


def test_a_point_of_a_measured_colouring_is_ringed_too(hovered: ExplorerPlot) -> None:
    hovered.plot_projection(measured([1.0, 2.0, 3.0]), 'area')
    hovered._plot.vb.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)

    hovered._on_mouse_over(in_scene(hovered, *POINTS[2]))

    assert ringed(hovered) == POINTS[2]


# ----------------------------------------------------------------------------
# Clicking a point
# ----------------------------------------------------------------------------


def test_clicking_a_point_announces_its_row(hovered: ExplorerPlot, qtbot) -> None:
    with qtbot.waitSignal(hovered.point_clicked, timeout=1000) as announced:
        hovered._on_mouse_clicked(FakeClick(in_scene(hovered, *POINTS[2])))

    assert announced.args == [2]


def test_clicking_the_first_point_announces_it_too(hovered: ExplorerPlot, qtbot) -> None:
    """Row 0 is a row like any other, where a plain truth test would drop it."""
    with qtbot.waitSignal(hovered.point_clicked, timeout=1000) as announced:
        hovered._on_mouse_clicked(FakeClick(in_scene(hovered, *POINTS[0])))

    assert announced.args == [0]


def test_a_click_on_a_point_is_taken_as_answered(hovered: ExplorerPlot) -> None:
    """Left to fall through, the click would be taken up again by the view underneath."""
    click = FakeClick(in_scene(hovered, *POINTS[1]))

    hovered._on_mouse_clicked(click)

    assert click.accepted


def test_clicking_nothing_announces_nothing(hovered: ExplorerPlot, qtbot) -> None:
    x_scale, _ = pixel_size(hovered)
    empty = in_scene(hovered, POINTS[0][0] + 40 * x_scale, POINTS[0][1])

    with qtbot.assertNotEmitted(hovered.point_clicked):
        hovered._on_mouse_clicked(FakeClick(empty))


def test_the_right_button_belongs_to_the_view(hovered: ExplorerPlot) -> None:
    """It opens pyqtgraph's own menu, which picking a point out must not take over."""
    click = FakeClick(in_scene(hovered, *POINTS[1]), button=Qt.MouseButton.RightButton)

    hovered._on_mouse_clicked(click)

    assert not click.accepted


def test_clicking_a_cleared_canvas_announces_nothing(hovered: ExplorerPlot) -> None:
    position = in_scene(hovered, *POINTS[1])
    hovered.plot_projection(None)

    click = FakeClick(position)
    hovered._on_mouse_clicked(click)

    assert not click.accepted
