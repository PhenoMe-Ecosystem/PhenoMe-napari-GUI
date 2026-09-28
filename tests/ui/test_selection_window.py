"""Tests for the window listing the points a selection picked out.

The window is handed the rows of the projection that were selected, and knows nothing else, so the
frames here are written by hand.
"""

from pathlib import Path

import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QWidget

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


@pytest.fixture
def saving_to(monkeypatch: pytest.MonkeyPatch):
    """ Return a hook answering the save dialog with a path, without showing it.

    Returns:
        Called as saving_to(tmp_path / 'picked.csv'), or with '' for
        the dialog being closed without a name.
    """
    def _answer(file_path) -> None:
        monkeypatch.setattr(
            QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (str(file_path), '')
        )

    return _answer


def test_the_table_starts_empty(window: SelectionWindow) -> None:
    assert window._points_table.model().rowCount() == 0


def test_every_selected_point_is_listed(window: SelectionWindow, points: pd.DataFrame) -> None:
    window.show_selection(points)

    assert window._points_table.model().rowCount() == 2


def test_the_columns_are_the_projection_s_own(window: SelectionWindow, points: pd.DataFrame) -> None:
    """The same rows PhenoMe's explorer writes to a CSV, which is every column of the frame."""
    window.show_selection(points)

    assert headers(window) == list(points.columns)


def test_the_coordinates_and_the_properties_are_listed_too(
    window: SelectionWindow, points: pd.DataFrame
) -> None:
    window.show_selection(points)

    assert 'Component 1' in headers(window)
    assert 'area' in headers(window)


def test_a_row_reads_across_the_frame(window: SelectionWindow, points: pd.DataFrame) -> None:
    window.show_selection(points)

    assert cells(window, 1) == ['1', '2', 'img_001.tif', '20', 'drugA', '340']


def test_a_measurement_is_written_to_be_read(window: SelectionWindow) -> None:
    """A property to its last decimal fills the column and says no more than four figures do."""
    window.show_selection(
        pd.DataFrame({'Image': ['a.tif'], 'Index': [1], 'area': [0.12345678901234]})
    )

    assert cells(window, 0) == ['a.tif', '1', '0.1235']


def test_a_projection_is_listed_column_for_column(window: SelectionWindow) -> None:
    """Shaped as prepare_embedding_dataframe builds it: coordinates, image, place, metadata, properties.

    Listing it whole is what keeps the table and a CSV of the same selection saying the same thing.
    """
    projection = pd.DataFrame(
        {
            'Component 1': [0.5, 1.5],
            'Component 2': [1.5, 2.5],
            'Image': ['img_000.tif', 'img_001.tif'],
            'Index': [10, 20],
            'Drug_name': ['ctrl', 'drugA'],
            'Time': ['60_min', '60_min'],
            'area': [120.0, 340.0],
            'eccentricity': [0.34, 0.99],
        }
    )

    window.show_selection(projection)

    assert headers(window) == list(projection.columns)
    assert window._points_table.model().rowCount() == 2


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


# ----------------------------------------------------------------------------
# What can be exported, and when
# ----------------------------------------------------------------------------


def exports(window: SelectionWindow) -> tuple[bool, bool]:
    """ Whether each of the two export buttons can be pressed. """
    return window._export_idx_button.isEnabled(), window._export_csv_button.isEnabled()


def test_there_is_nothing_to_export_until_something_is_selected(window: SelectionWindow) -> None:
    assert exports(window) == (False, False)


def test_a_selection_opens_the_exports(window: SelectionWindow, points: pd.DataFrame) -> None:
    window.show_selection(points)

    assert exports(window) == (True, True)


@pytest.mark.parametrize('selection', [None, pd.DataFrame()], ids=['none', 'empty'])
def test_an_empty_selection_closes_the_exports(window: SelectionWindow, points: pd.DataFrame, selection) -> None:
    window.show_selection(points)

    window.show_selection(selection)

    assert exports(window) == (False, False)


def test_clearing_closes_the_exports(window: SelectionWindow, points: pd.DataFrame) -> None:
    """There is nothing left to write once the selection has been let go of."""
    window.show_selection(points)

    window.clear_selection()

    assert exports(window) == (False, False)


# ----------------------------------------------------------------------------
# Exporting the indices
# ----------------------------------------------------------------------------


def test_the_indices_are_written_out(
    window: SelectionWindow, points: pd.DataFrame, saving_to, tmp_path: Path
) -> None:
    """The places PhenoMe holds the images at, which is what an index file is for."""
    written = tmp_path / 'indices.txt'
    window.show_selection(points)
    saving_to(written)

    window._export_idx_button.click()

    assert written.read_text() == '10,20'


def test_closing_the_dialog_writes_no_indices(
    window: SelectionWindow, points: pd.DataFrame, saving_to, tmp_path: Path
) -> None:
    window.show_selection(points)
    saving_to('')

    window._export_idx_button.click()

    assert list(tmp_path.iterdir()) == []


def test_the_indices_of_nothing_are_not_written(
    window: SelectionWindow, saving_to, tmp_path: Path
) -> None:
    """The button is closed with no selection, which the writing does not lean on."""
    written = tmp_path / 'indices.txt'
    saving_to(written)

    window._on_export_idx_clicked()

    assert not written.exists()


# ----------------------------------------------------------------------------
# Exporting the table
# ----------------------------------------------------------------------------


def test_the_selection_is_written_out_as_a_csv(
    window: SelectionWindow, points: pd.DataFrame, saving_to, tmp_path: Path
) -> None:
    written = tmp_path / 'selection.csv'
    window.show_selection(points)
    saving_to(written)

    window._export_csv_button.click()

    pd.testing.assert_frame_equal(pd.read_csv(written), points)


def test_the_csv_holds_the_values_rather_than_what_the_table_shows(
    window: SelectionWindow, saving_to, tmp_path: Path
) -> None:
    """The table rounds a measurement to be read; a file is written to be worked on."""
    written = tmp_path / 'selection.csv'
    window.show_selection(pd.DataFrame({'Image': ['a.tif'], 'Index': [1], 'area': [0.12345678901234]}))
    saving_to(written)

    window._export_csv_button.click()

    assert cells(window, 0)[-1] == '0.1235'
    assert '0.12345678901234' in written.read_text()


def test_the_csv_leaves_out_the_row_numbers(
    window: SelectionWindow, points: pd.DataFrame, saving_to, tmp_path: Path
) -> None:
    """Written without the frame's own index, which stands for nothing outside this table."""
    written = tmp_path / 'selection.csv'
    window.show_selection(points)
    saving_to(written)

    window._export_csv_button.click()

    assert written.read_text().splitlines()[0] == ','.join(points.columns)


def test_closing_the_dialog_writes_no_csv(
    window: SelectionWindow, points: pd.DataFrame, saving_to, tmp_path: Path
) -> None:
    window.show_selection(points)
    saving_to('')

    window._export_csv_button.click()

    assert list(tmp_path.iterdir()) == []
