from qtpy.QtWidgets import QWidget

from ..registries.model_details import get_model_details
from ..registries.models import MODEL_OPTIONS
from .option_window import OptionWindow


class ModelWindow(OptionWindow):
    """ The window a model is picked from, listing what is offered and describing what is highlighted.

    Picks a model and no more: nothing is loaded here and no PhenoMe state is held, the loading being
    a download of some hundreds of megabytes left to the embedding extraction. The caller reads what
    was picked through selected_model() once the window has been accepted.

    The list shows the names the GUI offers, where the descriptions are registered under the names
    the registries use; MODEL_OPTIONS is the mapping between the two.
    """

    def __init__(self, parent: QWidget | None, model: str | None = None) -> None:
        """ Build the window and open it on a model.

        Args:
            parent: The widget the dialog is parented to, so it closes with it.
            model: The model to highlight when it opens, as the GUI names it. None, or one no
                longer offered, opens on the first of them.
        """
        # Named 'Select Model' rather than Ok: it closes the window and selects the model, without loading it
        super().__init__(parent, list(MODEL_OPTIONS), model, 'Select Model', 'Select Model')

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
