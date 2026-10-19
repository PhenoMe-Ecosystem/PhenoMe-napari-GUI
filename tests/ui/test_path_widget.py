"""Tests for the path boxes: the browse dialog, dropping a path onto the box, and the change signal.

The dialogs are stood in for, being modal. Drops are made by handing the line edit real drag and drop
events built around a QMimeData, which needs no display.
"""

from pathlib import Path

import pytest
from qtpy.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from qtpy.QtGui import QDragEnterEvent, QDropEvent
from qtpy.QtWidgets import QFileDialog

from phenome_gui.ui.path_widget import DirectoryWidget, FileWidget, PathWidget

CSV_FILTER = 'CSV Files (*.csv)'


@pytest.fixture
def directory(qtbot) -> DirectoryWidget:
    """ A folder box opening on a path of its own. """
    widget = DirectoryWidget('Images Directory...', '/data/start')
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def file(qtbot) -> FileWidget:
    """ A file box for a metadata .csv, opening on a path of its own. """
    widget = FileWidget('Metadata Path...', '/data/meta.csv', file_filter=CSV_FILTER)
    qtbot.addWidget(widget)
    return widget


class Dialog:
    """ Stands in for QFileDialog, answering as a test asks and noting how it was opened. """

    def __init__(self, answer: str) -> None:
        """ Build a dialog that answers with a given path.

        Args:
            answer: What the dialog comes back with. '' for it being closed without a pick.
        """
        self.answer = answer
        self.opened_with = None

    def directory(self, parent, caption, start, *args, **kwargs) -> str:
        """ Stand in for QFileDialog.getExistingDirectory. """
        self.opened_with = {'caption': caption, 'start': start}
        return self.answer

    def file(self, parent, caption, start, file_filter='', *args, **kwargs) -> tuple[str, str]:
        """ Stand in for QFileDialog.getOpenFileName, which answers with a filter beside the path. """
        self.opened_with = {'caption': caption, 'start': start, 'filter': file_filter}
        return self.answer, file_filter


@pytest.fixture
def answering(monkeypatch: pytest.MonkeyPatch):
    """ Return a hook making both dialogs answer with a path.

    Returns:
        Called as answering('/picked'), or with '' for the dialog being closed. Returns the stand-in,
        which says how the dialog was opened.
    """
    def _answer(path: str) -> Dialog:
        dialog = Dialog(path)
        monkeypatch.setattr(QFileDialog, 'getExistingDirectory', dialog.directory)
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', dialog.file)
        return dialog

    return _answer


def dragged(*urls: QUrl) -> QMimeData:
    """ What a drag from a file browser carries: a list of URLs. """
    mime_data = QMimeData()
    mime_data.setUrls(list(urls))
    return mime_data


def drop(widget: PathWidget, mime_data: QMimeData) -> QDropEvent:
    """ Drop something onto a path box, the way a release over it would.

    Args:
        widget: The box dropped onto.
        mime_data: What was dragged. Kept alive by the caller, the event only borrowing it.
    Returns:
        The drop event, to ask whether it was taken.
    """
    event = QDropEvent(
        QPointF(5, 5), Qt.DropAction.CopyAction, mime_data,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    widget._line_edit.dropEvent(event)
    return event


def enter(widget: PathWidget, mime_data: QMimeData) -> QDragEnterEvent:
    """ Drag something over a path box, the way entering it would.

    Args:
        widget: The box dragged over.
        mime_data: What is being dragged.
    Returns:
        The drag event, to ask whether the box would take it.
    """
    event = QDragEnterEvent(
        QPoint(5, 5), Qt.DropAction.CopyAction, mime_data,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    widget._line_edit.dragEnterEvent(event)
    return event


# ----------------------------------------------------------------------------
# What the box is built as
# ----------------------------------------------------------------------------


def test_the_box_opens_on_its_initial_path(directory: DirectoryWidget) -> None:
    assert directory.path == '/data/start'


# ----------------------------------------------------------------------------
# Browsing, through the dialog the kind of box calls for
# ----------------------------------------------------------------------------


def test_a_directory_box_takes_the_folder_picked(directory: DirectoryWidget, answering) -> None:
    answering('/data/images')

    directory._browse_button.click()

    assert directory.path == '/data/images'


def test_a_file_box_takes_the_file_picked(file: FileWidget, answering) -> None:
    answering('/data/other.csv')

    file._browse_button.click()

    assert file.path == '/data/other.csv'


@pytest.mark.parametrize('box', ['directory', 'file'])
def test_closing_the_dialog_keeps_the_path(box: str, answering, request) -> None:
    """Closing the dialog is not asking for no path."""
    widget = request.getfixturevalue(box)
    before = widget.path
    answering('')

    widget._browse_button.click()

    assert widget.path == before


@pytest.mark.parametrize('box', ['directory', 'file'])
def test_the_dialog_opens_where_the_box_points(box: str, answering, request) -> None:
    widget = request.getfixturevalue(box)
    dialog = answering('')

    widget._browse_button.click()

    assert dialog.opened_with['start'] == widget.path


def test_the_dialog_is_titled_after_the_button(directory: DirectoryWidget, answering) -> None:
    """The button's trailing '...' says a dialog follows; the dialog itself has no need of it."""
    dialog = answering('')

    directory._browse_button.click()

    assert dialog.opened_with['caption'] == 'Images Directory'


def test_a_file_box_offers_only_its_files(file: FileWidget, answering) -> None:
    dialog = answering('')

    file._browse_button.click()

    assert dialog.opened_with['filter'] == CSV_FILTER


def test_a_bare_path_box_does_not_know_what_to_ask_for(qtbot) -> None:
    """Called directly rather than clicked: raised from a slot, it would end the test elsewhere."""
    widget = PathWidget('Browse...')
    qtbot.addWidget(widget)

    with pytest.raises(NotImplementedError, match='DirectoryWidget or FileWidget'):
        widget._on_browse_clicked()


# ----------------------------------------------------------------------------
# The change signal, however the path came to change
# ----------------------------------------------------------------------------


def test_typing_announces_the_path(directory: DirectoryWidget, qtbot) -> None:
    with qtbot.waitSignal(directory.path_changed, timeout=100) as signal:
        directory._line_edit.setText('/typed')

    assert signal.args == ['/typed']


def test_setting_the_path_announces_it(directory: DirectoryWidget, qtbot) -> None:
    with qtbot.waitSignal(directory.path_changed, timeout=100) as signal:
        directory.path = '/set'

    assert signal.args == ['/set']


def test_browsing_announces_the_path(directory: DirectoryWidget, answering, qtbot) -> None:
    answering('/picked')

    with qtbot.waitSignal(directory.path_changed, timeout=100) as signal:
        directory._browse_button.click()

    assert signal.args == ['/picked']


def test_closing_the_dialog_announces_nothing(directory: DirectoryWidget, answering, qtbot) -> None:
    answering('')

    with qtbot.assertNotEmitted(directory.path_changed):
        directory._browse_button.click()


def test_emptying_the_box_announces_an_empty_path(directory: DirectoryWidget, qtbot) -> None:
    """Reported as it reads: turning '' into None is the caller's business."""
    with qtbot.waitSignal(directory.path_changed, timeout=100) as signal:
        directory.path = ''

    assert signal.args == ['']


# ----------------------------------------------------------------------------
# Dropping a path onto the box
# ----------------------------------------------------------------------------


def test_a_dropped_folder_becomes_the_path(directory: DirectoryWidget, tmp_path: Path) -> None:
    mime_data = dragged(QUrl.fromLocalFile(str(tmp_path)))

    drop(directory, mime_data)

    assert directory.path == str(tmp_path)


def test_a_dropped_file_becomes_the_path(file: FileWidget, tmp_path: Path) -> None:
    path = tmp_path / 'meta.csv'
    path.write_text('filename\n')
    mime_data = dragged(QUrl.fromLocalFile(str(path)))

    drop(file, mime_data)

    assert file.path == str(path)


def test_a_drop_replaces_the_whole_path(directory: DirectoryWidget, tmp_path: Path) -> None:
    """Rather than splicing the path in at the cursor, which is what the line edit would do."""
    directory._line_edit.setCursorPosition(3)
    mime_data = dragged(QUrl.fromLocalFile(str(tmp_path)))

    drop(directory, mime_data)

    assert directory.path == str(tmp_path)


def test_any_local_path_is_taken(directory: DirectoryWidget, tmp_path: Path) -> None:
    """A file dropped on a folder box is taken too: the wrong kind is flagged where a typed one is."""
    path = tmp_path / 'image.tif'
    path.write_bytes(b'')
    mime_data = dragged(QUrl.fromLocalFile(str(path)))

    drop(directory, mime_data)

    assert directory.path == str(path)


def test_dropping_several_takes_the_first(directory: DirectoryWidget, tmp_path: Path) -> None:
    first, second = tmp_path / 'first', tmp_path / 'second'
    mime_data = dragged(QUrl.fromLocalFile(str(first)), QUrl.fromLocalFile(str(second)))

    drop(directory, mime_data)

    assert directory.path == str(first)


def test_a_drop_announces_the_path(directory: DirectoryWidget, tmp_path: Path, qtbot) -> None:
    mime_data = dragged(QUrl.fromLocalFile(str(tmp_path)))

    with qtbot.waitSignal(directory.path_changed, timeout=100) as signal:
        drop(directory, mime_data)

    assert signal.args == [str(tmp_path)]


def test_a_local_path_is_welcomed_as_it_is_dragged_in(directory: DirectoryWidget, tmp_path: Path) -> None:
    """Accepted on entering, so the cursor says a drop will be taken before the button is let go."""
    mime_data = dragged(QUrl.fromLocalFile(str(tmp_path)))

    event = enter(directory, mime_data)

    assert event.isAccepted()


def test_a_remote_url_is_left_alone(directory: DirectoryWidget) -> None:
    """Not a path on this machine, so there is nothing to write into the box."""
    before = directory.path
    mime_data = dragged(QUrl('https://example.com/data'))

    drop(directory, mime_data)

    assert directory.path == before


def test_a_remote_url_is_refused_as_it_is_dragged_in(directory: DirectoryWidget) -> None:
    """Refused outright rather than handed to the line edit, which would take it as text."""
    mime_data = dragged(QUrl('https://example.com/data'))

    event = enter(directory, mime_data)

    assert not event.isAccepted()


def test_plain_text_is_still_dropped_as_text(directory: DirectoryWidget) -> None:
    """Only a drag of URLs is the path's business: text is left to the line edit's own handling."""
    directory.path = ''
    mime_data = QMimeData()
    mime_data.setText('/typed/elsewhere')

    drop(directory, mime_data)

    assert directory.path == '/typed/elsewhere'


def test_a_path_can_be_set_as_a_path(directory: DirectoryWidget, tmp_path: Path) -> None:
    """The natural thing to hand a path box, which QLineEdit would refuse as anything but text."""
    directory.path = tmp_path

    assert directory.path == str(tmp_path)
