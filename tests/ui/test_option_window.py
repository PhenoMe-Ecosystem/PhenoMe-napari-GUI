"""Tests for the windows a model and a properties preset are picked from, and the option window they share.

The windows pick and no more: nothing here loads a model or computes properties. The model
descriptions are stubbed wherever a test needs text to read, the real ones being unwritten; the preset
descriptions are read from their registry as they are.
"""

import pytest
from qtpy.QtGui import QTextDocument

from phenome_gui.registries.models import MODEL_OPTIONS
from phenome_gui.registries.properties_descriptions import PRESET_INFO
from phenome_gui.ui import option_window as option_window_module
from phenome_gui.ui.option_window import (
    ModelWindow,
    OptionWindow,
    PropertyWindow,
)
from phenome_gui.utils.params import PRESET_CHOICES

# A model other than the first, so opening on it cannot be confused with the fallback. Read off the
# registry rather than spelled out, which is what let these tests drift from it when it was renamed.
OTHER_MODEL = list(MODEL_OPTIONS)[2]


@pytest.fixture
def model_window(qtbot):
    """ Return a hook building a window on a given model.

    Returns:
        Called as model_window('DINOv2 - Large'), or with nothing for
        the window opening on whatever it falls back to.
    """
    def _build(model: str | None = None) -> ModelWindow:
        built = ModelWindow(None, model)
        qtbot.addWidget(built)
        return built

    return _build


@pytest.fixture
def described(monkeypatch: pytest.MonkeyPatch):
    """ Return a hook giving the models descriptions, the real ones being unwritten.

    Stubbed at the window rather than in the registry, so what is checked is the window asking for
    the right model's description rather than the registry's own lookup.

    Returns:
        Called as described({'dinov2_vitl14_reg': '<p>Large</p>'}), keyed the
        way the registries are.
    """
    def _describe(details: dict) -> None:
        monkeypatch.setattr(
            option_window_module, 'get_model_details', lambda key: details.get(key, '')
        )

    return _describe


def listed(built: ModelWindow) -> list[str]:
    """ The models the window currently offers, in the order it offers them. """
    return [built._option_list.item(row).text() for row in range(built._option_list.count())]


# ----------------------------------------------------------------------------
# The list, which is what the combo box used to offer
# ----------------------------------------------------------------------------


def test_every_model_the_gui_offers_is_listed(model_window) -> None:
    """Offered here and nowhere else now, so a model missing from the list is one nobody can load."""
    assert listed(model_window()) == list(MODEL_OPTIONS)


def test_the_window_opens_on_the_model_it_was_given(model_window) -> None:
    built = model_window(OTHER_MODEL)

    assert built.selected_model() == OTHER_MODEL


def test_the_window_opens_on_something_when_given_nothing(model_window) -> None:
    """A window with nothing picked would offer a Select Model button meaning nothing."""
    assert model_window().selected_model() == next(iter(MODEL_OPTIONS))


def test_a_model_no_longer_offered_falls_back(model_window) -> None:
    """The parameter outlives the registry, so what it names may since have been taken out."""
    assert model_window('Some - Retired Model').selected_model() == next(iter(MODEL_OPTIONS))


def test_the_highlighted_model_is_the_one_picked(model_window) -> None:
    built = model_window()

    built._option_list.setCurrentRow(list(MODEL_OPTIONS).index(OTHER_MODEL))

    assert built.selected_model() == OTHER_MODEL


# ----------------------------------------------------------------------------
# The details pane, which says what a name cannot
# ----------------------------------------------------------------------------


def test_the_pane_describes_the_model_opened_on(model_window, described) -> None:
    """Filled as the window is built, rather than staying empty until something is clicked."""
    described({MODEL_OPTIONS[OTHER_MODEL]: '<p>The large one.</p>'})

    assert 'The large one.' in model_window(OTHER_MODEL)._details.toPlainText()


def test_picking_a_model_describes_it(model_window, described) -> None:
    described({MODEL_OPTIONS[OTHER_MODEL]: '<p>The large one.</p>'})
    built = model_window()

    built._option_list.setCurrentRow(list(MODEL_OPTIONS).index(OTHER_MODEL))

    assert 'The large one.' in built._details.toPlainText()


def test_the_pane_is_keyed_the_way_the_registries_are(model_window, described) -> None:
    """The list shows the names the GUI offers, where a description is registered under PhenoMe's."""
    described({OTHER_MODEL: '<p>Keyed by the wrong name.</p>'})

    assert model_window(OTHER_MODEL)._details.toPlainText() == ''


def test_a_model_with_no_description_leaves_the_pane_empty(model_window, described) -> None:
    """Which is every model for now: the pane is scaffolding until the descriptions are written."""
    described({})

    assert model_window()._details.toPlainText() == ''


def test_moving_off_a_described_model_clears_the_pane(model_window, described) -> None:
    """The pane answers for what is highlighted, not for whatever was last worth saying."""
    described({MODEL_OPTIONS[OTHER_MODEL]: '<p>The large one.</p>'})
    built = model_window(OTHER_MODEL)

    built._option_list.setCurrentRow(0)

    assert built._details.toPlainText() == ''


# ============================================================================
# The preset window
# ============================================================================

# A preset other than the first, so opening on it cannot be confused with the fallback
OTHER_PRESET = PRESET_CHOICES[2]


@pytest.fixture
def preset_window(qtbot):
    """ Return a hook building a window on a given preset.

    Returns:
        Called as preset_window('Complete'), or with nothing for the window opening on whatever it falls back to.
    """
    def _build(preset: str | None = None) -> PropertyWindow:
        built = PropertyWindow(None, preset)
        qtbot.addWidget(built)
        return built

    return _build


def listed_presets(built: PropertyWindow) -> list[str]:
    """ The presets the window currently offers, in the order it offers them. """
    return [built._option_list.item(row).text() for row in range(built._option_list.count())]


def description(preset: str) -> str:
    """ What the registry says about a preset, rendered to text the way the pane renders it. """
    document = QTextDocument()
    document.setHtml(PRESET_INFO[preset.lower()]['description'])
    return document.toPlainText()


# ----------------------------------------------------------------------------
# The preset list
# ----------------------------------------------------------------------------


def test_every_preset_the_parameter_accepts_is_listed(preset_window) -> None:
    """And only those: a preset picked here has to be one the parameter can hold."""
    assert listed_presets(preset_window()) == PRESET_CHOICES


def test_the_window_opens_on_the_preset_it_was_given(preset_window) -> None:
    assert preset_window(OTHER_PRESET).selected_preset() == OTHER_PRESET


@pytest.mark.parametrize('preset', [None, 'Some Retired Preset'])
def test_the_window_falls_back_to_the_first_preset(preset_window, preset) -> None:
    assert preset_window(preset).selected_preset() == PRESET_CHOICES[0]


def test_the_highlighted_preset_is_the_one_picked(preset_window) -> None:
    built = preset_window()

    built._option_list.setCurrentRow(PRESET_CHOICES.index(OTHER_PRESET))

    assert built.selected_preset() == OTHER_PRESET


# ----------------------------------------------------------------------------
# The preset details pane
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('preset', PRESET_CHOICES)
def test_every_preset_offered_is_described(preset: str) -> None:
    """A preset the parameter accepts but the registry does not describe would open on an empty pane."""
    assert PRESET_INFO.get(preset.lower(), {}).get('description')


def test_the_pane_describes_the_preset_opened_on(preset_window) -> None:
    """Filled as the window is built, under the lower case name the registry uses."""
    assert preset_window(OTHER_PRESET)._details.toPlainText() == description(OTHER_PRESET)


def test_picking_a_preset_describes_it(preset_window) -> None:
    built = preset_window()

    built._option_list.setCurrentRow(PRESET_CHOICES.index(OTHER_PRESET))

    assert built._details.toPlainText() == description(OTHER_PRESET)


# ----------------------------------------------------------------------------
# The option window both pickers are built on
# ----------------------------------------------------------------------------


def test_an_option_window_must_say_how_it_describes_its_options(preset_window) -> None:
    """The base has no descriptions of its own: a subclass has to give them."""
    with pytest.raises(NotImplementedError):
        OptionWindow._describe(preset_window(), PRESET_CHOICES[0])
