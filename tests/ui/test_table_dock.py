"""Tests for the area the metadata table is docked in, and the dock that comes back to it.

Everything here is built under a QMainWindow, which napari puts every plugin widget inside of and
which is what makes the way back hard: Qt refuses to redock a dock it finds in no dock area of the
main window it belongs to, so a dock dropped straight into the layout of a panel pops out and
stays out. A test on a bare parent passes either way and would say nothing.

The widgets are never shown, the fixtures only building them: `isVisible()` is False for a widget
whose window was never shown, so what a close did is read off `isHidden()`, which only an explicit
hide sets. Floating is asked for rather than dragged, the drag itself being Qt's own.
"""

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from phenome_gui.ui.table_dock import TableDockArea

CLOSABLE = QDockWidget.DockWidgetFeature.DockWidgetClosable
TOP_AREA = Qt.DockWidgetArea.TopDockWidgetArea


@pytest.fixture
def window(qtbot) -> QMainWindow:
    """ A panel holding a dock area, inside the main window that stands in for napari's. """
    panel = QWidget()
    layout = QVBoxLayout(panel)

    area = TableDockArea('Metadata table', panel)
    area.setWidget(QLabel('table'))
    layout.addWidget(area, 1)

    window = QMainWindow()
    window.setCentralWidget(panel)
    qtbot.addWidget(window)
    return window


@pytest.fixture
def area(window: QMainWindow) -> TableDockArea:
    """ The dock area of the built window. """
    return window.findChild(TableDockArea)


# ----------------------------------------------------------------------------
# Popping the table out
# ----------------------------------------------------------------------------


def test_the_table_floats_out_of_the_panel(area: TableDockArea) -> None:
    """Nothing may pull it back in on its own: it stays out until it is closed."""
    area.dock.setFloating(True)

    assert area.dock.isFloating()
    assert not area.dock.isHidden()


def test_only_the_floating_table_offers_a_close_button(area: TableDockArea) -> None:
    assert not area.dock.features() & CLOSABLE

    area.dock.setFloating(True)
    assert area.dock.features() & CLOSABLE

    area.dock.setFloating(False)
    assert not area.dock.features() & CLOSABLE


# ----------------------------------------------------------------------------
# Putting it back
# ----------------------------------------------------------------------------


def test_closing_the_floating_table_puts_it_back(area: TableDockArea, qtbot) -> None:
    """The way back is taken a turn of the event loop later, once Qt is done hiding the dock."""
    area.dock.setFloating(True)

    area.dock.close()

    qtbot.waitUntil(lambda: not area.dock.isFloating())
    assert area.dockWidgetArea(area.dock) == TOP_AREA
    assert not area.dock.isHidden()


def test_the_table_still_holds_its_widget_afterwards(area: TableDockArea, qtbot) -> None:
    table = area.dock.widget()

    area.dock.setFloating(True)
    area.dock.close()

    qtbot.waitUntil(lambda: not area.dock.isFloating())
    assert area.dock.widget() is table


def test_the_table_can_be_popped_out_again_after_a_close(area: TableDockArea, qtbot) -> None:
    area.dock.setFloating(True)
    area.dock.close()
    qtbot.waitUntil(lambda: not area.dock.isFloating())

    area.dock.setFloating(True)

    assert area.dock.isFloating()
    assert not area.dock.isHidden()


def test_closing_the_table_does_not_stop_the_window_from_closing(
    window: QMainWindow, area: TableDockArea
) -> None:
    """A refused close stops the round of closes a quit is made of, leaving napari open."""
    window.show()
    area.dock.setFloating(True)

    QApplication.closeAllWindows()

    assert not window.isVisible()
