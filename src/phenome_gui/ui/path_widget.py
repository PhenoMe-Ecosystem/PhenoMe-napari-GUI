from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)


class _PathLineEdit(QLineEdit):
    """ A line edit that takes a dropped file or folder as the whole of its text.

    QLineEdit already accepts drops, but a file dragged from a file browser arrives as a list of
    URLs and the default inserts their text at the cursor: a `file:///` URL on some platforms, a path
    spliced into whatever was in the box on others. Here a drop means "use this path" instead.
    """

    def __init__(self, text: str = '', parent: QWidget | None = None, placeholder: str = "") -> None:
        """ Build the line edit, ready to take drops.

        Args:
            text: The text the box opens with.
            parent: The widget the line edit is nested in.
            placeholder: The placeholder text for the line edit.
        """
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText(placeholder)

    @staticmethod
    def _local_path(event: QDragEnterEvent | QDragMoveEvent | QDropEvent) -> str | None:
        """ The local path a drag of URLs carries, or None when it carries none.

        Args:
            event: The drag or the drop being asked about, which carries URLs.
        Returns:
            The path of the first item dragged, when it is a local file or folder. A drag of several
            items gives the first of them; one whose first item is a remote URL gives None.
        """
        urls = event.mimeData().urls()
        if not urls or not urls[0].isLocalFile():
            return None

        return urls[0].toLocalFile()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """ Accept a drag carrying a local path, refuse one carrying only remote URLs, and leave text alone.

        Args:
            event: The drag entering the box.
        """
        self._check_drag_valid(event, super().dragEnterEvent)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """ Keep judging the drag as it moves over the box.

        Judged again here rather than only on entering: QLineEdit's own handling would otherwise
        judge the drag afresh at every move, as text to be placed at the cursor.

        Args:
            event: The drag moving over the box.
        """
        self._check_drag_valid(event, super().dragMoveEvent)

    def _check_drag_valid(self, event: QDragEnterEvent | QDragMoveEvent, default) -> None:
        """ Take a drag of a local path, refuse a drag of other URLs, and leave plain text to QLineEdit.

        A drag of URLs is the path's business throughout: one that is not local is refused rather than
        handed on, the line edit's own handling being to splice the URL's text in at the cursor.

        Args:
            event: The drag being judged.
            default: The line edit's own handling, for a drag of plain text.
        """
        if not event.mimeData().hasUrls():
            default(event)
        elif self._local_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """ Replace the whole text with the path dropped, refuse other URLs, and leave text alone.

        Args:
            event: The drop onto the box.
        """
        if not event.mimeData().hasUrls():
            super().dropEvent(event)
            return

        path = self._local_path(event)
        if path is None:
            event.ignore()
            return

        self.setText(path)
        event.acceptProposedAction()

class PathWidget(QWidget):
    """ A box holding a path, with a button opening a dialog to pick one and room to drop one onto.

    Reports the path exactly as it reads, an empty box being '' - turning that into None, or deciding
    whether an empty box is allowed, is the caller's business. Whether a file or a folder is wanted is
    the two subclasses' to say: DirectoryWidget and FileWidget.
    """

    # Emitted with the new text whenever it changes, however it was changed: typed, picked in the
    # dialog, dropped, or set through the path property.
    path_changed = Signal(str)

    def __init__(self, label: str, initial_path: str = "", placeholder: str = "", parent: QWidget | None = None) -> None:
        """ Build the box and its button.

        Args:
            label: The text of the browse button. Also the dialog's title, without a trailing '...'.
            initial_path: The path the box opens with. Empty for none.
            parent: The widget this one is nested in.
        """
        super().__init__(parent)

        self._caption = label.rstrip('.').strip()

        self._line_edit = _PathLineEdit(initial_path, placeholder=placeholder)
        self._line_edit.setReadOnly(False)
        self._line_edit.textChanged.connect(self.path_changed.emit)

        self._browse_button = QPushButton(label)
        self._browse_button.clicked.connect(self._on_browse_clicked)

        layout = QHBoxLayout()
        # Flush with its contents, as magicgui's FileEdit was: the box sits inside the page's own
        # layout, which already spaces the boxes apart, and a margin here would add to that on every side.
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._line_edit)
        layout.addWidget(self._browse_button)
        self.setLayout(layout)

    def _on_browse_clicked(self) -> None:
        """ Open the dialog where the box currently points, and take what is picked in it.

        A dialog closed without a pick answers '', which leaves the box as it was rather than
        emptying it: closing the dialog is not asking for no path.
        """
        chosen = self._choose_path(self.path)

        if chosen:
            self.path = chosen

    def _choose_path(self, start: str) -> str:
        """ Ask for a path through the dialog the kind of box calls for.

        Args:
            start: Where the dialog opens, the box's current path.
        Returns:
            The path picked, or '' when the dialog was closed without one.
        Raises:
            NotImplementedError: Always: a PathWidget does not know whether it wants a file or a
                folder. Use DirectoryWidget or FileWidget.
        """
        raise NotImplementedError(
            'A PathWidget does not know whether it asks for a file or a folder: '
            'use DirectoryWidget or FileWidget.'
        )

    def setStyleSheet(self, style: str) -> None:
        """ Set the style of the line edit, leaving the browse button as it is.

        Args:
            style: The stylesheet to apply, as held in STYLESHEETS.
        """
        self._line_edit.setStyleSheet(style)

    def styleSheet(self) -> str:
        """ The style of the line edit, which is where setStyleSheet() puts it.

        Read from the same place it is written: QWidget's own would answer for the widget around the
        box, which is never styled, and report every box as unstyled.

        Returns:
            The stylesheet the line edit was given, '' for none.
        """
        return self._line_edit.styleSheet()

    @property
    def path(self) -> str:
        """ The path in the box, exactly as it reads. '' for an empty box. """
        return self._line_edit.text()

    @path.setter
    def path(self, value: str | Path) -> None:
        """ Put a path in the box, which emits path_changed when it differs from what was there.

        Args:
            value: The path to show, as text or as a Path. '' to empty the box.
        """
        self._line_edit.setText(str(value))


class DirectoryWidget(PathWidget):
    """ A path box asking for a folder, for the images and the masks. """

    def _choose_path(self, start: str) -> str:
        """ Ask for a folder.

        Args:
            start: Where the dialog opens, the box's current path.
        Returns:
            The folder picked, or '' when the dialog was closed without one.
        """
        return QFileDialog.getExistingDirectory(self, self._caption, start)


class FileWidget(PathWidget):
    """ A path box asking for an existing file, such as the metadata .csv. """

    def __init__(
        self,
        label: str,
        initial_path: str = "",
        file_filter: str = "",
        placeholder: str = "",
        parent: QWidget | None = None,
    ) -> None:
        """ Build the box and its button.

        Args:
            label: The text of the browse button. Also the dialog's title, without a trailing '...'.
            initial_path: The path the box opens with. Empty for none.
            file_filter: The files the dialog offers, in Qt's own form, e.g. 'CSV Files (*.csv)'.
                Empty to offer every file.
            placeholder: The text to display when the box is empty.
            parent: The widget this one is nested in.
        """
        super().__init__(label, initial_path, placeholder, parent)
        self._file_filter = file_filter

    def _choose_path(self, start: str) -> str:
        """ Ask for an existing file.

        Args:
            start: Where the dialog opens, the box's current path.
        Returns:
            The file picked, or '' when the dialog was closed without one.
        """
        path, _ = QFileDialog.getOpenFileName(self, self._caption, start, self._file_filter)
        return path
