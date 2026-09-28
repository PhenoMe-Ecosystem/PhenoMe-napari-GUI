"""The dock a table is shown in, which pops out of its panel and back into it.

Used by the metadata table of the loading panel and the retrieved images table of the analysis
panel. A table is popped out for a bigger view rather than dismissed, so closing it puts it back
where it was instead of hiding it: nothing in the panel would bring a hidden one back.
"""

import logging

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QDockWidget, QMainWindow, QWidget

logger = logging.getLogger("main_logger")

# What the dock may be done with, docked and floating. The close button is only offered once the
# dock is out on its own, closing it being how it is put back rather than a way to hide it.
_DOCKED_FEATURES = (QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                    QDockWidget.DockWidgetFeature.DockWidgetMovable)
_FLOATING_FEATURES = _DOCKED_FEATURES | QDockWidget.DockWidgetFeature.DockWidgetClosable


class TableDock(QDockWidget):
    """ A dock whose close button puts it back in its area instead of leaving it hidden.

    Only ever built by the TableDockArea it goes back to, a dock knowing no way back to an area
    it was never given.
    """

    def __init__(self, title: str, parent: QMainWindow) -> None:
        """ Build the dock, docked and without a close button.

        Args:
            title: The name shown on the title bar, and on the window once it floats. It
                tells the two docked tables apart, so it says which table this one holds.
            parent: The area the dock is added to, and returns to when closed.
        """
        super().__init__(title, parent)
        self.setFeatures(_DOCKED_FEATURES)
        self.topLevelChanged.connect(self._on_top_level_changed)

    def _on_top_level_changed(self, top_level: bool) -> None:
        """ Offer the close button only while the dock is out on its own.

        Args:
            top_level: Whether the dock has just become a window of its own.
        """
        self.setFeatures(_FLOATING_FEATURES if top_level else _DOCKED_FEATURES)

    def closeEvent(self, event: QCloseEvent) -> None:
        """ Take a close asked of the floating dock as a way back into its area.

        The close is let through rather than refused: a refusal stops the round of closes an
        application quit is made of, which would leave napari open on a table popped out and
        closed. Qt hides the dock once this returns, so the way back is taken on the next turn
        of the event loop, where the hide cannot undo it.

        Args:
            event: The close being asked for, which is always let through.
        """
        was_floating = self.isFloating()
        super().closeEvent(event)

        if was_floating and event.isAccepted():
            QTimer.singleShot(0, self._dock_back)

    def _dock_back(self) -> None:
        """ Put the dock back in its area, and show it where the close left it hidden. """
        self.setFloating(False)
        self.show()
        logger.debug('The %s dock was closed and put back in its panel.', self.windowTitle())


class TableDockArea(QMainWindow):
    """ The area a table is docked in, standing in a panel as a plain widget.

    A dock only knows the way back to an area it was given, and only a QMainWindow hands one
    out: dropped straight into the layout of a panel, a dock can be popped out and never put
    back, Qt refusing to redock one it finds in no area at all. The area is what a panel holds
    onto, therefore, the dock living in it rather than in the panel itself.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """ Build the area and the dock it holds.

        Args:
            title: The name shown on the title bar of the dock, and on its window.
            parent: The widget the area is nested in, whose layout holds its slot.
        """
        super().__init__(parent)

        # A widget among others in the panel rather than a window of its own, which is what a
        # QMainWindow otherwise is. No central widget is set: the dock is all the area shows.
        self.setWindowFlags(Qt.WindowType.Widget)

        self.dock = TableDock(title, self)
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.dock)

    def setWidget(self, widget: QWidget) -> None:
        """ Show the given widget in the dock.

        Args:
            widget: The widget the dock is to hold, a table so far.
        """
        self.dock.setWidget(widget)
