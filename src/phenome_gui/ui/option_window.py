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
from ..registries.properties_descriptions import PRESET_INFO
from ..utils.params import PRESET_CHOICES

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
    selected_option() once the window has been accepted. A subclass names the window and its accept
    button through `title` and `accept_text`, and says what each option is through _describe().
    """

    # The window's title, and what its accept button reads - named rather than left as Qt's Ok, the
    # button saying what pressing it does
    title = 'Select'
    accept_text = 'Select'

    def __init__(self, parent: QWidget | None, options: list[str], current: str | None = None) -> None:
        """ Build the window and open it on an option.

        Args:
            parent: The widget the dialog is parented to, so it closes with it.
            options: The options offered, in the order they are listed.
            current: The option to highlight when it opens. None, or one not offered, opens on the
                first of them: a window with nothing picked would offer an accept button meaning nothing.
        """
        super().__init__(parent)
        self.setWindowTitle(self.title)
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

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.addButton(self.accept_text, QDialogButtonBox.ButtonRole.AcceptRole)
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


class ModelWindow(OptionWindow):
    """ The window a model is picked from.

    Picks a model and no more: nothing is loaded here and no PhenoMe state is held, the loading being
    a download of some hundreds of megabytes left to the embedding extraction.

    The list shows the names the GUI offers, where the descriptions are registered under the names
    the registries use; MODEL_OPTIONS is the mapping between the two.
    """

    title = 'Select Model'
    accept_text = 'Select Model'

    def __init__(self, parent: QWidget | None, model: str | None = None) -> None:
        """ Build the window and open it on a model.

        Args:
            parent: The widget the dialog is parented to, so it closes with it.
            model: The model to highlight when it opens, as the GUI names it. None, or one no
                longer offered, opens on the first of them.
        """
        super().__init__(parent, list(MODEL_OPTIONS), model)

    def _describe(self, model: str) -> str:
        """ Look up what is known about a model, under the name the registries know it by.

        Args:
            model: The model, as the GUI names it.
        Returns:
            An HTML fragment describing the model, empty for one with no description yet.
        """
        return get_model_details(MODEL_OPTIONS.get(model, ''))

    def selected_model(self) -> str:
        """ The model currently highlighted, for the caller to select once the window is accepted.

        Returns:
            The model as the GUI names it, which is what the process parameters hold.
        """
        return self.selected_option()


class PropertyWindow(OptionWindow):
    """ The window a properties preset is picked from.

    Picks a preset and no more: nothing is computed here.

    The list shows the presets as the parameters hold them, capitalised, where the descriptions are
    registered under PhenoMe's lower case names. Only the presets the parameter accepts are offered:
    turning the properties off is the checkbox's to do, not a preset's.
    """

    title = 'Select Properties Preset'
    accept_text = 'Select Preset'

    def __init__(self, parent: QWidget | None, preset: str | None = None) -> None:
        """ Build the window and open it on a preset.

        Args:
            parent: The widget the dialog is parented to, so it closes with it.
            preset: The preset to highlight when it opens, as the parameters hold it. None, or one not
                offered, opens on the first of them.
        """
        super().__init__(parent, PRESET_CHOICES, preset)

    def _describe(self, preset: str) -> str:
        """ Look up what a preset computes, under the lower case name PhenoMe knows it by.

        Args:
            preset: The preset, as the parameters hold it.
        Returns:
            An HTML fragment describing the preset, empty for one with no description.
        """
        return PRESET_INFO.get(preset.lower(), {}).get('description', '')

    def selected_preset(self) -> str:
        """ The preset currently highlighted, for the caller to select once the window is accepted.

        Returns:
            The preset as the parameters hold it.
        """
        return self.selected_option()
