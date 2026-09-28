from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QListWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..registries.model_details import get_model_details
from ..registries.models import MODEL_OPTIONS

# How the window opens: wide enough for a list and a description side by side, the description being
# the half worth reading.
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 400

# How the width is shared between the two, as stretch factors: the list holds names of a few words,
# the pane holds paragraphs.
LIST_STRETCH = 1
DETAILS_STRETCH = 2


class ModelWindow(QDialog):
    """ The window a model is picked from, listing what is offered and describing what is highlighted.

    Picks a model and no more: nothing is loaded here and no PhenoMe state is held, the loading being
    a download of some hundreds of megabytes that belongs on a worker thread. The caller reads what
    was picked through selected_model() once the window has been accepted.

    The list shows the names the GUI offers, where the descriptions are registered under the names
    the registries use; MODEL_OPTIONS is the mapping between the two.
    """

    def __init__(self, parent: QWidget, model: str | None = None) -> None:
        """ Build the window and open it on a model.

        Args:
            parent: The widget the dialog is parented to, so it closes with it.
            model: The model to highlight when it opens, as the GUI names it. None, or one no
                longer offered, opens on the first of them: a window with nothing picked would offer
                a Load Model button meaning nothing.
        """
        super().__init__(parent)
        self.setWindowTitle("Select Model")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        layout = QVBoxLayout(self)

        # The models offered, and what is known about the one highlighted, side by side
        panes = QHBoxLayout()

        self._model_list = QListWidget()
        self._model_list.addItems(MODEL_OPTIONS.keys())
        panes.addWidget(self._model_list, LIST_STRETCH)

        # Read-only and rendering HTML, so a description can carry a heading, a list of what the
        # model was trained on and a link to its paper. Links open in a browser rather than in the
        # pane, which has no way back.
        self._details = QTextBrowser()
        self._details.setOpenExternalLinks(True)
        panes.addWidget(self._details, DETAILS_STRETCH)

        layout.addLayout(panes)

        # Named rather than left as Qt's Ok, the button saying what pressing it does: it closes the
        # window and starts a download.
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.addButton('Load Model', QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connected before the opening selection is made, so the pane is filled by it rather than
        # left empty until something is clicked
        self._model_list.currentTextChanged.connect(self._on_model_selected)
        self._select(model)

    def _select(self, model: str | None) -> None:
        """ Highlight a model, falling back to the first offered.

        The rows are MODEL_OPTIONS in its own order, so a model's place in the list is its place in
        the mapping and no searching of the widget is needed.

        Args:
            model: The model to highlight, as the GUI names it. None or unknown falls back.
        """
        offered = list(MODEL_OPTIONS)
        row = offered.index(model) if model in offered else 0

        self._model_list.setCurrentRow(row)

    def _on_model_selected(self, model: str) -> None:
        """ Show what is known about the model just highlighted.

        Args:
            model: The highlighted model, as the GUI names it.
        """
        self._details.setHtml(get_model_details(MODEL_OPTIONS.get(model, '')))

    def selected_model(self) -> str:
        """ The model currently highlighted, for the caller to load once the window is accepted.

        Returns:
            The model as the GUI names it, which is what the process parameters hold.
        """
        return self._model_list.currentItem().text()
