"""Tests for the explorer canvas.

The canvas is handed a projection and the name of a column to colour it by, and knows nothing else,
so the frames here are written by hand rather than computed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QPolygonF
from qtpy.QtWidgets import QToolTip

from phenome_gui.ui.plotting import (
    CATEGORICAL_COLUMNS,
    DIMMED_ALPHA,
    HOVER_RADIUS,
    MIN_CONTINUOUS,
    MISSING_COLOUR,
    MISSING_LABEL,
    NORMAL_ALPHA,
    SELECTION_ALPHA,
    SELECTION_COLOUR,
    SERIES_COLOURS,
    ExplorerPlot,
    group_order,
    is_continuous,
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
    """ The names currently listed in the legend, in the order it lists them. """
    return [label.text for _, label in plot._legend.items]


def legend_colours(plot: ExplorerPlot) -> list[str]:
    """ The colour of each legend entry's swatch, in the order the legend lists them.

    An entry is made from the series it names, and draws its swatch from the brush that series was
    built with, so the two are read the same way.
    """
    return [sample.item.opts['brush'].color().name() for sample, _ in plot._legend.items]


def point_colours(plot: ExplorerPlot) -> list[str]:
    """ The colour of each point of the one drawn series, in the order they were given. """
    (item,) = plot._plot.listDataItems()
    return [spot.brush().color().name() for spot in item.points()]


def colorbar_label(plot: ExplorerPlot) -> str:
    """ The name written alongside the colorbar, whichever of its four axes carries it. """
    labels = [plot._colorbar.getAxis(axis).labelText for axis in ('left', 'right', 'top', 'bottom')]
    return next((label for label in labels if label), '')


def indexed_projection(indices: list[int]) -> pd.DataFrame:
    """ A projection whose rows say where PhenoMe holds each of their images.

    Args:
        indices: The place in the PhenoMe results each row stands for, one per point.
    """
    frame = projection(['ctrl', 'ctrl', 'drugA'][: len(indices)])
    frame['Index'] = indices
    return frame


def measured(values: list, column: str = 'area') -> pd.DataFrame:
    """ A projection carrying a measured column, as the properties arrive alongside the coordinates. """
    frame = projection([''] * len(values))
    return frame.drop(columns='Drug_name').assign(**{column: values})


# Enough distinct values for the canvas to read a column as a measurement rather than as labels
MEASUREMENTS = [10.0 * step for step in range(1, MIN_CONTINUOUS + 1)]

# The group colours in the order the canvas takes them, written the way a drawn brush reads back
SERIES_HEX = [colour.name() for colour in SERIES_COLOURS.getColors('qcolor')]


# ----------------------------------------------------------------------------
# Drawing the groups
# ----------------------------------------------------------------------------


def test_each_group_is_drawn_and_named(plot: ExplorerPlot) -> None:
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    assert series(plot) == {'ctrl': 2, 'drugA': 1}
    assert legend_labels(plot) == ['ctrl', 'drugA']


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


def test_the_legend_lists_the_groups_in_their_own_order(plot: ExplorerPlot) -> None:
    """A group is looked up in the legend by its name, which is no help when its place is its size."""
    plot.plot_projection(projection(['zeta', 'zeta', 'zeta', 'alpha', 'mid']), 'drug_name')

    assert [item.opts['name'] for item in plot._plot.listDataItems()] == ['zeta', 'alpha', 'mid']
    assert legend_labels(plot) == ['alpha', 'mid', 'zeta']


def test_the_legend_reads_numbers_as_numbers(plot: ExplorerPlot) -> None:
    """The cluster labels are numbers, which listed by their names would read 1, 10, 2."""
    plot.plot_projection(projection([2, 10, 1], column='Cluster'), 'cluster')

    assert legend_labels(plot) == ['1', '2', '10']


def test_the_legend_leaves_the_case_of_a_name_out_of_it(plot: ExplorerPlot) -> None:
    """Sorted by their characters, the capitalised names would all come before the rest."""
    plot.plot_projection(projection(['beta', 'Alpha', 'gamma']), 'drug_name')

    assert legend_labels(plot) == ['Alpha', 'beta', 'gamma']


def test_the_images_nothing_is_known_about_are_listed_last(plot: ExplorerPlot) -> None:
    """They are what is left over rather than a group of their own, wherever their label would sort."""
    plot.plot_projection(projection(['zeta', None, 'alpha']), 'drug_name')

    assert legend_labels(plot) == ['alpha', 'zeta', MISSING_LABEL]


def test_the_colours_are_taken_in_the_order_the_legend_lists(plot: ExplorerPlot) -> None:
    """The legend then reads down the palette, rather than jumping about it by how large a group is."""
    plot.plot_projection(projection(['zeta', 'zeta', 'alpha']), 'drug_name')

    assert legend_colours(plot) == SERIES_HEX[:2]

    # 'zeta' is drawn first, being the larger of the two, and wears the second colour
    assert colours(plot) == [SERIES_HEX[1], SERIES_HEX[0]]


def test_a_group_keeps_its_colour_as_its_size_changes(plot: ExplorerPlot) -> None:
    """Taken by size, the colours would swap the moment a re-clustering moved a point between groups."""
    plot.plot_projection(projection(['zeta', 'zeta', 'alpha']), 'drug_name')
    first = dict(zip(legend_labels(plot), legend_colours(plot), strict=True))

    plot.plot_projection(projection(['zeta', 'alpha', 'alpha']), 'drug_name')

    assert dict(zip(legend_labels(plot), legend_colours(plot), strict=True)) == first


def test_the_colours_start_over_past_the_last_one(plot: ExplorerPlot) -> None:
    """The group past the last colour wears the first one again, telling it apart only by where its
    points sit. The groups are named so that sorting them by name puts them in that order.
    """
    groups = [f'group_{i:03d}' for i in range(len(SERIES_HEX) + 1)]

    plot.plot_projection(projection(groups), 'drug_name')

    assert len(series(plot)) == len(SERIES_HEX) + 1
    assert colours(plot) == [*SERIES_HEX, SERIES_HEX[0]]


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
# Telling a measurement from a set of labels
# ----------------------------------------------------------------------------


def test_a_column_of_few_values_is_read_as_labels() -> None:
    """Too few values to make a ramp of: a handful of numbers name their images rather than measure them."""
    assert not is_continuous(pd.Series([1.0, 2.0, 3.0]), 'area')


def test_the_number_of_values_is_what_counts_rather_than_the_number_of_rows() -> None:
    """A thousand images of three sizes are three groups, however many rows say so."""
    repeated = pd.Series([1.0, 2.0, 3.0] * 400)

    assert not is_continuous(repeated, 'area')


@pytest.mark.parametrize(
    ('distinct', 'continuous'),
    [
        pytest.param(MIN_CONTINUOUS - 1, False, id='one-short'),
        pytest.param(MIN_CONTINUOUS, True, id='just-enough'),
    ],
)
def test_the_threshold_is_where_it_says_it_is(distinct: int, continuous: bool) -> None:
    values = pd.Series(np.arange(distinct, dtype=float))

    assert is_continuous(values, 'area') is continuous


def test_a_column_of_one_value_is_read_as_labels() -> None:
    """A measurement that never varies has no range to spread points over, and reads as one group."""
    assert not is_continuous(pd.Series([7.0] * 20), 'area')


def test_a_measurement_with_gaps_is_still_a_measurement() -> None:
    """Images PhenoMe knows nothing about leave holes, which say nothing about the kind of column."""
    values = pd.Series([*np.arange(MIN_CONTINUOUS, dtype=float), float('nan'), float('nan')])

    assert is_continuous(values, 'area')


@pytest.mark.parametrize('name', sorted(CATEGORICAL_COLUMNS))
def test_a_column_named_for_labels_is_never_a_measurement(name: str) -> None:
    """The cluster labels are numbers, and plenty of them, but a ramp over them measures nothing."""
    values = pd.Series(np.arange(50))

    assert not is_continuous(values, name)


def test_a_column_of_words_is_never_a_measurement() -> None:
    assert not is_continuous(pd.Series([f'drug_{value}' for value in range(50)]), 'drug_name')


def test_true_and_false_are_never_a_measurement() -> None:
    """A boolean is numeric to pandas, but two shades of a ramp say less than two named groups."""
    assert not is_continuous(pd.Series([True, False] * 25), 'is_control')


# ----------------------------------------------------------------------------
# The order the labels are listed in
# ----------------------------------------------------------------------------


def test_numbers_are_ordered_as_numbers() -> None:
    assert sorted([10, 2, 1], key=group_order) == [1, 2, 10]


def test_numbers_written_as_text_are_ordered_as_numbers() -> None:
    """A metadata column read out of a file holds its numbers as text, and means them as numbers."""
    assert sorted(['10', '2', '1'], key=group_order) == ['1', '2', '10']


def test_the_numbers_come_before_the_words() -> None:
    """Nothing orders one against the other, so they are kept apart rather than interleaved."""
    assert sorted(['ctrl', 2, 1], key=group_order) == [1, 2, 'ctrl']


def test_words_are_ordered_without_their_case() -> None:
    assert sorted(['beta', 'Alpha'], key=group_order) == ['Alpha', 'beta']


def test_the_word_nan_is_a_word() -> None:
    """A column can hold it without meaning a missing value, and it has no place among the numbers."""
    assert sorted(['nan', 1], key=group_order) == [1, 'nan']


def test_the_missing_label_is_ordered_last() -> None:
    assert sorted([MISSING_LABEL, 'zeta', 'alpha'], key=group_order) == ['alpha', 'zeta', MISSING_LABEL]


@pytest.mark.parametrize('missing', [None, float('nan'), pd.NA], ids=['none', 'nan', 'na'])
def test_a_missing_value_is_ordered_last(missing) -> None:
    """The field a highlight is picked from reads a column as PhenoMe left it, holes and all."""
    assert sorted(['zeta', missing, 'alpha'], key=group_order)[-1] is missing


# ----------------------------------------------------------------------------
# Measured columns, drawn along a ramp
# ----------------------------------------------------------------------------


def test_a_measured_column_is_drawn_as_one_series_with_a_colorbar(plot: ExplorerPlot) -> None:
    plot.plot_projection(measured(MEASUREMENTS), 'area')

    assert series(plot) == {None: len(MEASUREMENTS)}
    assert legend_labels(plot) == []
    assert plot._colorbar is not None
    assert plot._colorbar.values == (MEASUREMENTS[0], MEASUREMENTS[-1])


def test_the_colorbar_is_named_after_the_column(plot: ExplorerPlot) -> None:
    plot.plot_projection(measured(MEASUREMENTS, column='eccentricity'), 'eccentricity')

    assert 'eccentricity' in colorbar_label(plot)


def test_the_points_run_along_the_ramp(plot: ExplorerPlot) -> None:
    """Every point takes its own colour from its value, rather than the whole series taking one."""
    plot.plot_projection(measured(MEASUREMENTS), 'area')

    colours = point_colours(plot)
    assert len(set(colours)) == len(MEASUREMENTS)


def test_whole_numbers_are_still_a_measurement(plot: ExplorerPlot) -> None:
    """PhenoMe would call this one categorical: 'area' is a pixel count, and whole numbers of them.

    It calls a numeric column discrete up to 256 distinct values, which a pixel count on any small
    dataset falls under. Here it is the name that reserves a column for labels, and a handful of
    values that turns one into them, so a spread of whole numbers is still measured.
    """
    plot.plot_projection(measured([120.0, 340.0, 512.0, 600.0, 780.0, 900.0, 1100.0, 1500.0]), 'area')

    assert series(plot) == {None: 8}
    assert plot._colorbar is not None


def test_a_numbered_label_column_is_still_a_group(plot: ExplorerPlot) -> None:
    """The cluster labels are numbers, and enough of them to make a ramp of, which they are not."""
    frame = measured(list(range(MIN_CONTINUOUS + 2)), column='cluster')

    plot.plot_projection(frame, 'cluster')

    assert len(series(plot)) == MIN_CONTINUOUS + 2
    assert plot._colorbar is None


def test_true_and_false_are_two_groups(plot: ExplorerPlot) -> None:
    """A boolean is numeric to pandas, but two named groups say more than two shades of a ramp."""
    plot.plot_projection(measured([True, False, False], column='is_control'), 'is_control')

    assert series(plot) == {'False': 2, 'True': 1}
    assert plot._colorbar is None


def test_a_point_without_a_value_is_left_neutral(plot: ExplorerPlot) -> None:
    """It has no place on the ramp, and dropping it would leave a hole in the cloud."""
    plot.plot_projection(measured([float('nan'), *MEASUREMENTS]), 'area')

    assert series(plot) == {None: len(MEASUREMENTS) + 1}
    assert point_colours(plot)[0] == MISSING_COLOUR
    assert point_colours(plot)[1] != MISSING_COLOUR


def test_a_column_of_one_value_is_drawn_as_one_group(plot: ExplorerPlot) -> None:
    """A measurement that never varies has no range to spread the points over, and reads as a label."""
    plot.plot_projection(measured([7.0] * 10), 'area')

    assert series(plot) == {'7.0': 10}
    assert plot._colorbar is None


def test_the_two_kinds_of_colouring_undo_each_other(plot: ExplorerPlot) -> None:
    """Each leaves chrome of its own behind, so switching either way has to take the other's away."""
    plot.plot_projection(measured(MEASUREMENTS), 'area')
    assert plot._colorbar is not None

    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    assert plot._colorbar is None
    assert legend_labels(plot) == ['ctrl', 'drugA']

    plot.plot_projection(measured(MEASUREMENTS), 'area')

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
    assert legend_labels(plot) == ['x', 'y']


# ----------------------------------------------------------------------------
# The axes
# ----------------------------------------------------------------------------


def test_the_axes_are_named_after_the_components(plot: ExplorerPlot) -> None:
    plot.plot_projection(projection(['ctrl', 'drugA']), 'drug_name')

    assert plot._plot.getAxis('bottom').labelText == 'Component 1'
    assert plot._plot.getAxis('left').labelText == 'Component 2'


# ----------------------------------------------------------------------------
# The scale the points are drawn at
# ----------------------------------------------------------------------------

# The view pinned somewhere of its own, standing in for a plot that has been zoomed into: pyqtgraph
# fits the view to what it holds only until the plot is first panned or zoomed.
PINNED_RANGE = {'xRange': (0.0, 1.0), 'yRange': (0.0, 1.0), 'padding': 0}


def far_projection(scale: float = 500.0) -> pd.DataFrame:
    """ A projection whose coordinates are on a wholly different scale, as another method's are.

    Args:
        scale: What the coordinates the projection helper lays down are stretched by.
    """
    frame = projection(['ctrl', 'ctrl', 'drugA'])
    for column in ('Component 1', 'Component 2'):
        frame[column] *= scale
    return frame


def test_a_new_projection_is_fitted_to_its_own_points(plot: ExplorerPlot) -> None:
    """One method's components are on another's scale, and would be a dot in a corner otherwise."""
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')
    plot._plot.vb.setRange(**PINNED_RANGE)

    plot.plot_projection(far_projection(), 'drug_name')

    # Fitted with a padding of pyqtgraph's own choosing, so the view covers the points rather than
    # ending exactly on them
    (left, right), (bottom, top) = plot._plot.vb.viewRange()
    assert left <= 0.0 and right >= 1000.0
    assert bottom <= 500.0 and top >= 1500.0


def test_recolouring_leaves_the_view_where_it_was(plot: ExplorerPlot) -> None:
    """The same cloud under another colour, which should stay where it was being looked at."""
    frame = projection(['ctrl', 'ctrl', 'drugA'])
    plot.plot_projection(frame, 'drug_name')
    plot._plot.vb.setRange(**PINNED_RANGE)

    plot.plot_projection(frame, 'time')

    assert plot._plot.vb.viewRange() == [[0.0, 1.0], [0.0, 1.0]]


def test_highlighting_leaves_the_view_where_it_was(plot: ExplorerPlot) -> None:
    """Marking the points draws them again, which is no reason to move the plot under them."""
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')
    plot._plot.vb.setRange(**PINNED_RANGE)

    plot.highlight_points('drug_name', 'ctrl')

    assert plot._plot.vb.viewRange() == [[0.0, 1.0], [0.0, 1.0]]


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

    assert alphas(plot) == [DIMMED_ALPHA, DIMMED_ALPHA, NORMAL_ALPHA]


def test_the_groups_and_their_colours_survive_the_marking(plot: ExplorerPlot) -> None:
    """The plot still reads as what it was, with the picked points marked out on it."""
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')
    before = hues(plot)

    plot.highlight_points('drug_name', 'drugA')

    assert series(plot) == {'ctrl': 2, 'drugA': 1}
    assert legend_labels(plot) == ['ctrl', 'drugA']
    assert hues(plot) == before


def test_the_legend_keeps_its_colours_while_the_points_are_marked(plot: ExplorerPlot) -> None:
    """The legend draws its swatch from the colour the series was built with, not from its points."""
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    plot.highlight_points('drug_name', 'drugA')

    assert colours(plot) == SERIES_HEX[:2]


def test_a_range_leaves_the_points_inside_it_alone(plot: ExplorerPlot) -> None:
    plot.plot_projection(measured(MEASUREMENTS), 'area')

    # The values run 10, 20, 30 and up, so the range holds the second, third and fourth of them
    plot.highlight_points('area', (15.0, 45.0))

    assert alphas(plot)[:5] == [DIMMED_ALPHA, NORMAL_ALPHA, NORMAL_ALPHA, NORMAL_ALPHA, DIMMED_ALPHA]
    assert set(alphas(plot)[4:]) == {DIMMED_ALPHA}
    assert plot._colorbar is not None


def test_the_ends_of_a_range_are_inside_it(plot: ExplorerPlot) -> None:
    """The slider opens on the whole span, which has to mark every point rather than none."""
    plot.plot_projection(measured([10.0, 20.0, 30.0]), 'area')

    plot.highlight_points('area', (10.0, 30.0))

    assert alphas(plot) == [NORMAL_ALPHA, NORMAL_ALPHA, NORMAL_ALPHA]


def test_a_capitalised_column_is_found_from_its_gui_name(plot: ExplorerPlot) -> None:
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    plot.highlight_points('drug_name', 'ctrl')

    assert alphas(plot) == [NORMAL_ALPHA, NORMAL_ALPHA, DIMMED_ALPHA]


def test_a_point_without_a_value_is_left_out(plot: ExplorerPlot) -> None:
    """A criterion cannot speak for an image PhenoMe knows nothing about."""
    plot.plot_projection(measured([10.0, float('nan'), 30.0]), 'area')

    plot.highlight_points('area', (5.0, 35.0))

    assert alphas(plot) == [NORMAL_ALPHA, DIMMED_ALPHA, NORMAL_ALPHA]


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

    assert alphas(plot) == [NORMAL_ALPHA, NORMAL_ALPHA, NORMAL_ALPHA]


def test_the_marking_can_be_taken_off(plot: ExplorerPlot) -> None:
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')
    plot.highlight_points('drug_name', 'drugA')

    plot.highlight_points()

    assert alphas(plot) == [NORMAL_ALPHA, NORMAL_ALPHA, NORMAL_ALPHA]


def test_marking_an_empty_canvas_does_nothing(plot: ExplorerPlot) -> None:
    plot.highlight_points('drug_name', 'ctrl')

    assert series(plot) == {}


def test_a_new_projection_is_drawn_unmarked(plot: ExplorerPlot) -> None:
    """The marking was picked against the points it was applied to, not against these."""
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')
    plot.highlight_points('drug_name', 'drugA')

    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), 'drug_name')

    assert alphas(plot) == [NORMAL_ALPHA, NORMAL_ALPHA, NORMAL_ALPHA]


def test_the_marking_holds_without_a_colouring(plot: ExplorerPlot) -> None:
    """The field marked on is picked apart from the one coloured by, so either can stand alone."""
    plot.plot_projection(projection(['ctrl', 'ctrl', 'drugA']), None)

    plot.highlight_points('drug_name', 'drugA')

    assert alphas(plot) == [DIMMED_ALPHA, DIMMED_ALPHA, NORMAL_ALPHA]


# ----------------------------------------------------------------------------
# Picking points out with a box
# ----------------------------------------------------------------------------

# The points the projection helper lays down, as (Component 1, Component 2)
LAID_OUT = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


def box(left: float, top: float, right: float, bottom: float) -> QRectF:
    """ A box over the plot, in the coordinates the points are held in. """
    return QRectF(QPointF(left, top), QPointF(right, bottom))


def selection(plot: ExplorerPlot):
    """ The item washing the selected points over, or None when nothing is selected. """
    return plot._selection


def test_a_box_picks_out_the_points_it_covers(plot: ExplorerPlot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    picked = plot.select_in_box(box(0.5, 1.5, 2.5, 3.5))

    assert picked == [20, 30]


def test_the_edges_of_a_box_are_inside_it(plot: ExplorerPlot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    picked = plot.select_in_box(box(0.0, 1.0, 1.0, 2.0))

    assert picked == [10, 20]


def test_a_box_drawn_the_other_way_round_covers_the_same_points(plot: ExplorerPlot) -> None:
    """A box is dragged from any of its corners, and reads the same whichever one it started at."""
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    assert plot.select_in_box(box(2.5, 3.5, 0.5, 1.5)) == [20, 30]


def test_a_box_over_nothing_picks_out_nothing(plot: ExplorerPlot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    assert plot.select_in_box(box(8.0, 8.0, 9.0, 9.0)) == []
    assert selection(plot) is None


def test_a_box_on_an_empty_canvas_picks_out_nothing(plot: ExplorerPlot) -> None:
    assert plot.select_in_box(box(0.0, 0.0, 5.0, 5.0)) == []


def test_the_rows_stand_in_for_an_index_the_projection_has_not_got(plot: ExplorerPlot) -> None:
    """The synthetic frames of a test carry no Index, where PhenoMe's projections always do."""
    frame = projection(['ctrl', 'ctrl', 'drugA']).drop(columns='Index')

    plot.plot_projection(frame, 'drug_name')

    assert plot.select_in_box(box(0.5, 1.5, 2.5, 3.5)) == [1, 2]


def test_the_picked_points_are_washed_over(plot: ExplorerPlot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    plot.select_in_box(box(0.5, 1.5, 2.5, 3.5))

    (spot_x, _) = selection(plot).getData()
    assert list(spot_x) == [1.0, 2.0]

    # Compared without their case, a colour being the same colour however it is written down
    assert selection(plot).opts['brush'].color().name().lower() == SELECTION_COLOUR.lower()
    assert selection(plot).opts['brush'].color().alpha() == SELECTION_ALPHA


def test_the_wash_is_drawn_over_the_cloud(plot: ExplorerPlot) -> None:
    """A picked point in a crowded patch would otherwise be washed underneath its neighbours."""
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    plot.select_in_box(box(0.5, 1.5, 2.5, 3.5))

    assert plot._plot.listDataItems()[-1] is selection(plot)


def test_clearing_takes_the_wash_off(plot: ExplorerPlot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')
    plot.select_in_box(box(0.5, 1.5, 2.5, 3.5))

    plot.clear_selection()

    assert selection(plot) is None
    assert plot.selected_indices() == []


def test_a_selection_survives_a_recolouring(plot: ExplorerPlot) -> None:
    """The points are the same points, whatever they are now coloured by."""
    frame = indexed_projection([10, 20, 30])
    plot.plot_projection(frame, 'drug_name')
    plot.select_in_box(box(0.5, 1.5, 2.5, 3.5))

    plot.plot_projection(frame, 'time')

    assert plot.selected_indices() == [20, 30]
    assert selection(plot) is not None


def test_a_selection_goes_with_the_points_it_was_drawn_over(plot: ExplorerPlot) -> None:
    """Another projection is other points, which a box drawn over these has nothing to say about."""
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')
    plot.select_in_box(box(0.5, 1.5, 2.5, 3.5))

    plot.plot_projection(indexed_projection([40, 50, 60]), 'drug_name')

    assert plot.selected_indices() == []
    assert selection(plot) is None


def test_a_selection_survives_the_points_being_marked(plot: ExplorerPlot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')
    plot.select_in_box(box(0.5, 1.5, 2.5, 3.5))

    plot.highlight_points('drug_name', 'ctrl')

    assert plot.selected_indices() == [20, 30]
    assert selection(plot) is not None


class FakeDrag:
    """ Stands in for a drag over the plot, which takes a real scene event to build. """

    def __init__(self, start: QPointF, end: QPointF, finish: bool = True, button=Qt.MouseButton.LeftButton):
        # pyqtgraph's own point type, which its panning does arithmetic on where a QPointF raises
        self._start = pg.Point(start)
        self._end = pg.Point(end)
        self._finish = finish
        self._button = button
        self.accepted = False

    def button(self):
        """ The button held down through the drag. """
        return self._button

    def isFinish(self) -> bool:  # noqa: N802 - the name pyqtgraph calls it by
        """ Whether the button has been let go of. """
        return self._finish

    def buttonDownPos(self, button=None):  # noqa: N802 - the name pyqtgraph calls it by
        """ Where the drag started. """
        return self._start

    def pos(self):
        """ Where the pointer is now. """
        return self._end

    def lastPos(self):  # noqa: N802 - the name pyqtgraph calls it by
        """ Where the pointer was on the move before. """
        return self._start

    def accept(self) -> None:
        """ Take the drag as answered. """
        self.accepted = True


def test_a_box_is_asked_for_once(plot: ExplorerPlot, qtbot) -> None:
    """The mode is over as soon as the box is let go of, rather than staying on until told."""
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')
    plot.begin_box_selection()
    assert plot._viewbox.selecting

    with qtbot.waitSignal(plot._viewbox.region_selected, timeout=1000):
        plot._viewbox.mouseDragEvent(FakeDrag(QPointF(0.0, 0.0), QPointF(50.0, 50.0)))

    assert not plot._viewbox.selecting


def test_a_drag_that_is_not_over_leaves_the_mode_on(plot: ExplorerPlot) -> None:
    """The box is still being drawn, and the points under it are not settled until it is let go of."""
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')
    plot.begin_box_selection()

    plot._viewbox.mouseDragEvent(FakeDrag(QPointF(0.0, 0.0), QPointF(20.0, 20.0), finish=False))

    assert plot._viewbox.selecting


def test_a_drag_pans_the_plot_when_no_box_was_asked_for(plot: ExplorerPlot, qtbot) -> None:
    """Panning and zooming are what the plot does the rest of the time, and are left alone."""
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    with qtbot.assertNotEmitted(plot._viewbox.region_selected):
        plot._viewbox.mouseDragEvent(FakeDrag(QPointF(0.0, 0.0), QPointF(50.0, 50.0)))


def test_the_picked_images_are_announced(plot: ExplorerPlot, qtbot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    with qtbot.waitSignal(plot.points_selected, timeout=1000) as announced:
        plot._on_region_selected(box(0.5, 1.5, 2.5, 3.5))

    assert announced.args == [[20, 30]]


@pytest.mark.parametrize('begin', ['begin_box_selection', 'begin_lasso_selection'], ids=['box', 'lasso'])
def test_waiting_for_a_shape_can_be_called_off(plot: ExplorerPlot, begin: str) -> None:
    getattr(plot, begin)()

    plot.cancel_selection()

    assert not plot._viewbox.selecting


# ----------------------------------------------------------------------------
# Picking points out with a lasso
# ----------------------------------------------------------------------------


def lasso(*corners: tuple[float, float]) -> QPolygonF:
    """ A path drawn over the plot, in the coordinates the points are held in. """
    return QPolygonF([QPointF(x, y) for x, y in corners])


def test_a_lasso_picks_out_the_points_inside_it(plot: ExplorerPlot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    # A loop around the last two of (0, 1), (1, 2), (2, 3)
    picked = plot.select_in_lasso(lasso((0.5, 1.5), (2.5, 1.5), (2.5, 3.5), (0.5, 3.5)))

    assert picked == [20, 30]


def test_a_lasso_holds_only_what_it_went_round(plot: ExplorerPlot) -> None:
    """The reason to draw one: a shape a box would have to take its neighbours in to cover."""
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    # A rectangle over all three points, with a bay cut into it from the right that the middle one
    # sits in: everything but x from 0.5 to 3 at y from 1.5 to 2.5
    picked = plot.select_in_lasso(
        lasso(
            (-1.0, 0.0), (3.0, 0.0), (3.0, 1.5), (0.5, 1.5),
            (0.5, 2.5), (3.0, 2.5), (3.0, 4.0), (-1.0, 4.0),
        )
    )

    # The points sit at (0, 1), (1, 2) and (2, 3), the middle one inside the bay
    assert picked == [10, 30]


def test_a_lasso_left_ajar_still_holds(plot: ExplorerPlot) -> None:
    """The path is closed as it is read, a drawn loop rarely ending where it started."""
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    picked = plot.select_in_lasso(lasso((0.5, 1.5), (2.5, 1.5), (2.5, 3.5), (0.6, 3.4)))

    assert picked == [20, 30]


@pytest.mark.parametrize(
    'corners',
    [
        pytest.param(((1.0, 2.0),), id='one-point'),
        pytest.param(((0.0, 1.0), (2.0, 3.0)), id='a-line'),
    ],
)
def test_a_path_that_encloses_nothing_picks_out_nothing(plot: ExplorerPlot, corners) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    assert plot.select_in_lasso(lasso(*corners)) == []
    assert selection(plot) is None


def test_a_lasso_on_an_empty_canvas_picks_out_nothing(plot: ExplorerPlot) -> None:
    assert plot.select_in_lasso(lasso((0.0, 0.0), (5.0, 0.0), (5.0, 5.0))) == []


def test_the_points_a_lasso_picks_out_are_washed_over(plot: ExplorerPlot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    plot.select_in_lasso(lasso((0.5, 1.5), (2.5, 1.5), (2.5, 3.5), (0.5, 3.5)))

    (spot_x, _) = selection(plot).getData()
    assert list(spot_x) == [1.0, 2.0]


def test_a_drag_draws_the_lasso_and_hands_it_over(plot: ExplorerPlot, qtbot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')
    plot.begin_lasso_selection()

    plot._viewbox.mouseDragEvent(FakeDrag(QPointF(0.0, 0.0), QPointF(20.0, 0.0), finish=False))
    plot._viewbox.mouseDragEvent(FakeDrag(QPointF(0.0, 0.0), QPointF(20.0, 20.0), finish=False))
    assert plot._viewbox.lasso_band.isVisible()

    with qtbot.waitSignal(plot._viewbox.lasso_selected, timeout=1000) as announced:
        plot._viewbox.mouseDragEvent(FakeDrag(QPointF(0.0, 0.0), QPointF(0.0, 20.0)))

    # Where the drag began, and each place it was reported at since
    assert announced.args[0].count() == 4
    assert not plot._viewbox.lasso_band.isVisible()
    assert not plot._viewbox.selecting


def test_a_drag_pans_the_plot_when_no_lasso_was_asked_for(plot: ExplorerPlot, qtbot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    with qtbot.assertNotEmitted(plot._viewbox.lasso_selected):
        plot._viewbox.mouseDragEvent(FakeDrag(QPointF(0.0, 0.0), QPointF(50.0, 50.0)))


def test_the_band_is_drawn_among_the_points(plot: ExplorerPlot) -> None:
    """It hangs with them rather than in the pixels the drag was reported in, or it is drawn off-screen."""
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')
    plot._plot.vb.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)
    plot.begin_lasso_selection()

    corner = plot._viewbox.childGroup.mapToParent(QPointF(2.0, 3.0))
    plot._viewbox.mouseDragEvent(FakeDrag(corner, corner, finish=False))

    drawn = plot._viewbox.lasso_band.path().boundingRect()
    assert drawn.left() == pytest.approx(2.0, abs=0.1)
    assert drawn.top() == pytest.approx(3.0, abs=0.1)


# ----------------------------------------------------------------------------
# Picking points out by what they hold
# ----------------------------------------------------------------------------


def test_a_value_picks_out_the_points_holding_it(plot: ExplorerPlot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    picked = plot.select_matching('drug_name', 'ctrl')

    assert picked == [10, 20]


def test_a_range_picks_out_the_points_inside_it(plot: ExplorerPlot) -> None:
    plot.plot_projection(measured([10.0, 20.0, 30.0]), 'area')

    picked = plot.select_matching('area', (15.0, 25.0))

    assert picked == [1]


def test_the_points_picked_by_a_value_are_washed_over(plot: ExplorerPlot) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    plot.select_matching('drug_name', 'drugA')

    (spot_x, _) = selection(plot).getData()
    assert list(spot_x) == [2.0]


@pytest.mark.parametrize(
    ('field', 'criterion'),
    [
        pytest.param(None, None, id='nothing-picked'),
        pytest.param('drug_name', None, id='no-criterion'),
        pytest.param('gone', 'ctrl', id='not-in-the-projection'),
    ],
)
def test_no_criterion_picks_out_nothing(plot: ExplorerPlot, field, criterion) -> None:
    plot.plot_projection(indexed_projection([10, 20, 30]), 'drug_name')

    assert plot.select_matching(field, criterion) == []
    assert selection(plot) is None


def test_a_criterion_on_an_empty_canvas_picks_out_nothing(plot: ExplorerPlot) -> None:
    assert plot.select_matching('drug_name', 'ctrl') == []


def test_what_a_highlight_fades_is_what_a_criterion_leaves_out(plot: ExplorerPlot) -> None:
    """One mask read two ways: the same field and value fade the plot and pick the points out."""
    frame = indexed_projection([10, 20, 30])

    plot.plot_projection(frame, 'drug_name')
    plot.highlight_points('drug_name', 'ctrl')
    dimmed = list(plot._dimmed)

    plot.plot_projection(frame, 'drug_name')
    plot.select_matching('drug_name', 'ctrl')

    assert list(~plot._selected) == dimmed


# ----------------------------------------------------------------------------
# Hovering a point
# ----------------------------------------------------------------------------

# The points the projection helper lays down, as (Component 1, Component 2)
POINTS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


@pytest.fixture
def hovered(plot: ExplorerPlot) -> ExplorerPlot:
    """ A canvas with three points drawn on it, at a settled scale.

    The range is pinned so a distance in pixels is a known distance in plot coordinates, rather than
    left at the padded fit the canvas opens a projection on.
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


# ----------------------------------------------------------------------------
# Writing the plot out as a picture
# ----------------------------------------------------------------------------

# The first bytes of a PNG file, which is what says the image exporter wrote it
PNG_MAGIC = b'\x89PNG'


@pytest.fixture
def drawn(plot: ExplorerPlot) -> ExplorerPlot:
    """ A canvas with a projection drawn on it, there being something to write out. """
    plot.plot_projection(projection(['ctrl', 'drugA', 'drugA']), 'drug_name')
    return plot


def test_the_plot_is_written_as_an_svg(drawn: ExplorerPlot, tmp_path: Path) -> None:
    written = tmp_path / 'plot.svg'

    drawn.export_plot(written, 'svg')

    assert '<svg' in written.read_text()


def test_the_plot_is_written_as_a_png(drawn: ExplorerPlot, tmp_path: Path) -> None:
    written = tmp_path / 'plot.png'

    drawn.export_plot(written, 'png')

    assert written.read_bytes().startswith(PNG_MAGIC)


def test_the_groups_reach_the_written_plot(drawn: ExplorerPlot, tmp_path: Path) -> None:
    """The legend is parented to the view box, so it goes down with the points rather than being lost."""
    written = tmp_path / 'plot.svg'

    drawn.export_plot(written, 'svg')

    assert 'ctrl' in written.read_text()


@pytest.mark.parametrize('extension', ['SVG', '.svg', '.SVG'], ids=['upper', 'dotted', 'both'])
def test_the_file_type_is_read_however_it_is_spelled(
    drawn: ExplorerPlot, tmp_path: Path, extension: str
) -> None:
    """A name's own suffix can be handed over as it was read off it, dot and all."""
    written = tmp_path / 'plot.svg'

    drawn.export_plot(written, extension)

    assert '<svg' in written.read_text()


def test_a_path_is_taken_as_text_or_as_a_path(drawn: ExplorerPlot, tmp_path: Path) -> None:
    written = tmp_path / 'plot.png'

    drawn.export_plot(str(written), 'png')

    assert written.read_bytes().startswith(PNG_MAGIC)


def test_a_file_type_the_canvas_cannot_write_is_refused(drawn: ExplorerPlot, tmp_path: Path) -> None:
    written = tmp_path / 'plot.jpg'

    with pytest.raises(ValueError, match='jpg'):
        drawn.export_plot(written, 'jpg')

    assert not written.exists()


def test_an_empty_canvas_is_still_written(plot: ExplorerPlot, tmp_path: Path) -> None:
    """Nothing gates the canvas itself: the window is what waits for a projection to be drawn."""
    written = tmp_path / 'plot.png'

    plot.export_plot(written, 'png')

    assert written.read_bytes().startswith(PNG_MAGIC)
