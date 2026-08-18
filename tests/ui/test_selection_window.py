"""Tests for the window listing the points a selection picked out.

The window is handed the rows of the projection that were selected, and knows nothing else, so the
frames here are written by hand.
"""

import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget

from phenome_gui.ui.selection_window import SelectionWindow


@pytest.fixture
def parent(qtbot) -> QWidget:
    """ A bare widget to parent the dialog to, so it is destroyed with the test. """
    widget = QWidget()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def window(parent: QWidget) -> SelectionWindow:
    """ A selection window with nothing listed in it yet. """
    return SelectionWindow(parent)


@pytest.fixture
def points() -> pd.DataFrame:
    """ Two rows of a projection, carrying what PhenoMe puts alongside the coordinates. """
    return pd.DataFrame(
        {
            'Component 1': [0.0, 1.0],
            'Component 2': [1.0, 2.0],
            'Image': ['img_000.tif', 'img_001.tif'],
            'Index': [10, 20],
            'Drug_name': ['ctrl', 'drugA'],
            'area': [120.0, 340.0],
        }
    )


def headers(window: SelectionWindow) -> list[str]:
    """ The columns the table is currently showing. """
    model = window._points_table.model()
    return [
        model.headerData(column, Qt.Orientation.Horizontal)
        for column in range(model.columnCount())
    ]


def cells(window: SelectionWindow, row: int) -> list[str]:
    """ The values listed on one row of the table. """
    model = window._points_table.model()
    return [model.item(row, column).text() for column in range(model.columnCount())]


def test_the_table_starts_empty(window: SelectionWindow) -> None:
    assert window._points_table.model().rowCount() == 0


def test_every_selected_point_is_listed(window: SelectionWindow, points: pd.DataFrame) -> None:
    window.show_selection(points)

    assert window._points_table.model().rowCount() == 2


def test_the_image_is_named_and_placed_first(window: SelectionWindow, points: pd.DataFrame) -> None:
    window.show_selection(points)

    assert headers(window)[:2] == ['Index', 'Image']
    assert cells(window, 0)[:2] == ['10', 'img_000.tif']


def test_the_metadata_follows_the_image(window: SelectionWindow, points: pd.DataFrame) -> None:
    window.show_selection(points)

    assert headers(window) == ['Index', 'Image', 'Drug_name']
    assert cells(window, 1) == ['20', 'img_001.tif', 'drugA']


def test_the_coordinates_and_the_properties_are_left_out(
    window: SelectionWindow, points: pd.DataFrame
) -> None:
    """A projection can carry dozens of properties, and a table that wide reads as nothing at all."""
    assert 'area' not in headers(window) if window._points_table.model().columnCount() else True

    window.show_selection(points)

    assert 'area' not in headers(window)
    assert not any(header.startswith('Component') for header in headers(window))


def test_listing_a_selection_replaces_the_one_before(window: SelectionWindow, points: pd.DataFrame) -> None:
    window.show_selection(points)

    window.show_selection(points.head(1))

    assert window._points_table.model().rowCount() == 1


@pytest.mark.parametrize('selection', [None, pd.DataFrame()], ids=['none', 'empty'])
def test_an_empty_selection_empties_the_table(window: SelectionWindow, points: pd.DataFrame, selection) -> None:
    window.show_selection(points)

    window.show_selection(selection)

    assert window._points_table.model().rowCount() == 0


def test_clearing_empties_the_table(window: SelectionWindow, points: pd.DataFrame) -> None:
    window.show_selection(points)

    window.clear_selection()

    assert window._points_table.model().rowCount() == 0


# ----------------------------------------------------------------------------
# Picking a row
# ----------------------------------------------------------------------------


def pick(window: SelectionWindow, row: int, column: int = 0) -> None:
    """ Pick a cell of the table, as clicking it would. """
    window._points_table.setCurrentIndex(window._points_table.model().index(row, column))


def test_picking_a_row_says_which_image_it_stands_for(
    window: SelectionWindow, points: pd.DataFrame, qtbot
) -> None:
    window.show_selection(points)

    with qtbot.waitSignal(window.image_selected, timeout=1000) as announced:
        pick(window, 1)

    assert announced.args == [20]


def test_the_row_picked_is_the_one_announced(window: SelectionWindow, points: pd.DataFrame, qtbot) -> None:
    """Read off the frame the table was filled from, rather than off whichever row came first."""
    window.show_selection(points)

    with qtbot.waitSignal(window.image_selected, timeout=1000) as announced:
        pick(window, 0)

    assert announced.args == [10]


def test_picking_a_cell_picks_its_whole_row(window: SelectionWindow, points: pd.DataFrame, qtbot) -> None:
    """A row stands for one image, so any of its cells is that image."""
    window.show_selection(points)

    with qtbot.waitSignal(window.image_selected, timeout=1000) as announced:
        pick(window, 1, column=2)

    assert announced.args == [20]
    assert len(window._points_table.selectionModel().selectedRows()) == 1


def test_a_frame_without_an_index_falls_back_to_the_row(window: SelectionWindow, qtbot) -> None:
    """The canvas answers with row positions on such a frame too, so the two agree."""
    window.show_selection(
        pd.DataFrame({'Image': ['a.tif', 'b.tif'], 'Drug_name': ['ctrl', 'drugA']})
    )

    with qtbot.waitSignal(window.image_selected, timeout=1000) as announced:
        pick(window, 1)

    assert announced.args == [1]


def test_refilling_the_table_announces_nothing(window: SelectionWindow, points: pd.DataFrame, qtbot) -> None:
    """Clearing the model drops the picked row, which is no pick of anyone's."""
    window.show_selection(points)
    pick(window, 1)

    with qtbot.assertNotEmitted(window.image_selected):
        window.show_selection(points.head(1))


def test_clearing_the_table_announces_nothing(window: SelectionWindow, points: pd.DataFrame, qtbot) -> None:
    window.show_selection(points)
    pick(window, 1)

    with qtbot.assertNotEmitted(window.image_selected):
        window.clear_selection()
