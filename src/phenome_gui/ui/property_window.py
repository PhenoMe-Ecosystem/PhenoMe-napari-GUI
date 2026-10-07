from qtpy.QtWidgets import QWidget

from ..registries.properties_descriptions import PRESET_INFO
from ..utils.params import PRESET_CHOICES
from .option_window import OptionWindow


class PropertyWindow(OptionWindow):
    """ The window a properties preset is picked from, listing the presets and describing what is highlighted.

    Picks a preset and no more: nothing is computed here. The caller reads what was picked through
    selected_preset() once the window has been accepted.

    The list shows the presets as the parameters hold them, capitalised, where the descriptions are
    registered under PhenoMe's lower case names. Only the presets the parameter accepts are offered:
    turning the properties off is the checkbox's to do, not a preset's.
    """

    def __init__(self, parent: QWidget | None, preset: str | None = None) -> None:
        """ Build the window and open it on a preset.

        Args:
            parent: The widget the dialog is parented to, so it closes with it.
            preset: The preset to highlight when it opens, as the parameters hold it. None, or one not
                offered, opens on the first of them.
        """
        super().__init__(parent, PRESET_CHOICES, preset, 'Select Properties Preset', 'Select Preset')

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
