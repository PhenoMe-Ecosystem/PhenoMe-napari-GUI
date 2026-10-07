from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QListWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# How the window opens: wide enough for a list and a description side by side, the description being
# the half worth reading.
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 400

# How the width is shared between the two, as stretch factors: the list holds names of a few words,
# the pane holds paragraphs.
LIST_STRETCH = 1
DETAILS_STRETCH = 2


class OptionWindow(QDialog):
    """ A window one option is picked from, listing what is offered and describing what is highlighted.

    Picks an option and no more: nothing is acted on here, the caller reading what was picked through
    selected_option() once the window has been accepted. What each option is described as is the
    subclass's to say, through _describe().
    """

    def __init__(
        self,
        parent: QWidget | None,
        options: list[str],
        current: str | None,
        title: str,
        accept_text: str,
    ) -> None:
        """ Build the window and open it on an option.

        Args:
            parent: The widget the dialog is parented to, so it closes with it.
            options: The options offered, in the order they are listed.
            current: The option to highlight when it opens. None, or one not offered, opens on the
                first of them: a window with nothing picked would offer an accept button meaning nothing.
            title: The window's title.
            accept_text: What the accept button reads, saying what pressing it does.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self._options = list(options)

        layout = QVBoxLayout(self)

        # The options offered, and what is known about the one highlighted, side by side
        panes = QHBoxLayout()

        self._option_list = QListWidget()
        self._option_list.addItems(self._options)
        panes.addWidget(self._option_list, LIST_STRETCH)

        # Read-only and rendering HTML, so a description can carry a heading, a list and a link.
        # Links open in a browser rather than in the pane, which has no way back.
        self._details = QTextBrowser()
        self._details.setOpenExternalLinks(True)
        panes.addWidget(self._details, DETAILS_STRETCH)

        layout.addLayout(panes)

        # Named rather than left as Qt's Ok, the button saying what pressing it does
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.addButton(accept_text, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connected before the opening selection is made, so the pane is filled by it rather than
        # left empty until something is clicked
        self._option_list.currentTextChanged.connect(self._on_option_selected)
        self._select(current)

    def _select(self, option: str | None) -> None:
        """ Highlight an option, falling back to the first offered.

        The rows are the options in their own order, so an option's place in the list is its place
        among them and no searching of the widget is needed.

        Args:
            option: The option to highlight. None or one not offered falls back.
        """
        row = self._options.index(option) if option in self._options else 0

        self._option_list.setCurrentRow(row)

    def _on_option_selected(self, option: str) -> None:
        """ Show what is known about the option just highlighted.

        Args:
            option: The highlighted option.
        """
        self._details.setHtml(self._describe(option))

    def _describe(self, option: str) -> str:
        """ Say what an option is, for the pane beside the list.

        Args:
            option: The option to describe, as listed.
        Returns:
            An HTML fragment describing it, empty for one with nothing to say.
        Raises:
            NotImplementedError: Always: every window describes its options its own way.
        """
        raise NotImplementedError(f'{type(self).__name__} does not say how its options are described.')

    def selected_option(self) -> str:
        """ The option currently highlighted, for the caller to act on once the window is accepted.

        Returns:
            The option, as listed.
        """
        return self._option_list.currentItem().text()
