"""Tests for the window a properties preset is picked from, and the option window it shares with the model's."""

import pytest

from phenome_gui.registries.properties_descriptions import PRESET_INFO
from phenome_gui.ui.option_window import OptionWindow
from phenome_gui.ui.property_window import PropertyWindow
from phenome_gui.utils.params import PRESET_CHOICES

# A preset other than the first, so opening on it cannot be confused with the fallback
OTHER_PRESET = PRESET_CHOICES[2]


@pytest.fixture
def window(qtbot):
    """ Return a hook building a window on a given preset.

    Returns:
        Called as window('Complete'), or with nothing for the window opening on whatever it falls back to.
    """
    def _build(preset: str | None = None) -> PropertyWindow:
        built = PropertyWindow(None, preset)
        qtbot.addWidget(built)
        return built

    return _build


def listed(built: PropertyWindow) -> list[str]:
    """ The presets the window currently offers, in the order it offers them. """
    return [built._option_list.item(row).text() for row in range(built._option_list.count())]


def description(preset: str) -> str:
    """ What the registry says about a preset, as the pane renders it to text. """
    return PRESET_INFO[preset.lower()]['description'].replace('<b>', '').replace('</b>', '')


# ----------------------------------------------------------------------------
# The list
# ----------------------------------------------------------------------------


def test_every_preset_the_parameter_accepts_is_listed(window) -> None:
    """And only those: a preset picked here has to be one the parameter can hold."""
    assert listed(window()) == PRESET_CHOICES


def test_the_window_opens_on_the_preset_it_was_given(window) -> None:
    assert window(OTHER_PRESET).selected_preset() == OTHER_PRESET


@pytest.mark.parametrize('preset', [None, 'Some Retired Preset'])
def test_the_window_falls_back_to_the_first_preset(window, preset) -> None:
    assert window(preset).selected_preset() == PRESET_CHOICES[0]


def test_the_highlighted_preset_is_the_one_picked(window) -> None:
    built = window()

    built._option_list.setCurrentRow(PRESET_CHOICES.index(OTHER_PRESET))

    assert built.selected_preset() == OTHER_PRESET


# ----------------------------------------------------------------------------
# The details pane
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('preset', PRESET_CHOICES)
def test_every_preset_offered_is_described(preset: str) -> None:
    """A preset the parameter accepts but the registry does not describe would open on an empty pane."""
    assert PRESET_INFO.get(preset.lower(), {}).get('description')


def test_the_pane_describes_the_preset_opened_on(window) -> None:
    """Filled as the window is built, under the lower case name the registry uses."""
    assert window(OTHER_PRESET)._details.toPlainText() == description(OTHER_PRESET)


def test_picking_a_preset_describes_it(window) -> None:
    built = window()

    built._option_list.setCurrentRow(PRESET_CHOICES.index(OTHER_PRESET))

    assert built._details.toPlainText() == description(OTHER_PRESET)


# ----------------------------------------------------------------------------
# The option window both pickers are built on
# ----------------------------------------------------------------------------


def test_an_option_window_must_say_how_it_describes_its_options(window) -> None:
    """The base has no descriptions of its own: a subclass has to give them."""
    with pytest.raises(NotImplementedError):
        OptionWindow._describe(window(), PRESET_CHOICES[0])
