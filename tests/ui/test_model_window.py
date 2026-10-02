"""Tests for the window a model is picked from.

The window picks a model and no more: nothing here loads one, and the descriptions it shows are
stubbed wherever a test needs text to read, the real ones being deliberately unwritten.
"""

import pytest

from phenome_gui.registries.models import MODEL_OPTIONS
from phenome_gui.ui import model_window as model_window_module
from phenome_gui.ui.model_window import ModelWindow

# A model other than the first, so opening on it cannot be confused with the fallback. Read off the
# registry rather than spelled out, which is what let these tests drift from it when it was renamed.
OTHER_MODEL = list(MODEL_OPTIONS)[2]


@pytest.fixture
def window(qtbot):
    """ Return a hook building a window on a given model.

    Returns:
        Called as window('DINOv2 - Large'), or with nothing for
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
            model_window_module, 'get_model_details', lambda key: details.get(key, '')
        )

    return _describe


def listed(built: ModelWindow) -> list[str]:
    """ The models the window currently offers, in the order it offers them. """
    return [built._model_list.item(row).text() for row in range(built._model_list.count())]


# ----------------------------------------------------------------------------
# The list, which is what the combo box used to offer
# ----------------------------------------------------------------------------


def test_every_model_the_gui_offers_is_listed(window) -> None:
    """Offered here and nowhere else now, so a model missing from the list is one nobody can load."""
    assert listed(window()) == list(MODEL_OPTIONS)


def test_the_window_opens_on_the_model_it_was_given(window) -> None:
    built = window(OTHER_MODEL)

    assert built.selected_model() == OTHER_MODEL


def test_the_window_opens_on_something_when_given_nothing(window) -> None:
    """A window with nothing picked would offer a Select Model button meaning nothing."""
    assert window().selected_model() == next(iter(MODEL_OPTIONS))


def test_a_model_no_longer_offered_falls_back(window) -> None:
    """The parameter outlives the registry, so what it names may since have been taken out."""
    assert window('Some - Retired Model').selected_model() == next(iter(MODEL_OPTIONS))


def test_the_highlighted_model_is_the_one_picked(window) -> None:
    built = window()

    built._model_list.setCurrentRow(list(MODEL_OPTIONS).index(OTHER_MODEL))

    assert built.selected_model() == OTHER_MODEL


# ----------------------------------------------------------------------------
# The details pane, which says what a name cannot
# ----------------------------------------------------------------------------


def test_the_pane_describes_the_model_opened_on(window, described) -> None:
    """Filled as the window is built, rather than staying empty until something is clicked."""
    described({MODEL_OPTIONS[OTHER_MODEL]: '<p>The large one.</p>'})

    assert 'The large one.' in window(OTHER_MODEL)._details.toPlainText()


def test_picking_a_model_describes_it(window, described) -> None:
    described({MODEL_OPTIONS[OTHER_MODEL]: '<p>The large one.</p>'})
    built = window()

    built._model_list.setCurrentRow(list(MODEL_OPTIONS).index(OTHER_MODEL))

    assert 'The large one.' in built._details.toPlainText()


def test_the_pane_is_keyed_the_way_the_registries_are(window, described) -> None:
    """The list shows the names the GUI offers, where a description is registered under PhenoMe's."""
    described({OTHER_MODEL: '<p>Keyed by the wrong name.</p>'})

    assert window(OTHER_MODEL)._details.toPlainText() == ''


def test_a_model_with_no_description_leaves_the_pane_empty(window, described) -> None:
    """Which is every model for now: the pane is scaffolding until the descriptions are written."""
    described({})

    assert window()._details.toPlainText() == ''


def test_moving_off_a_described_model_clears_the_pane(window, described) -> None:
    """The pane answers for what is highlighted, not for whatever was last worth saying."""
    described({MODEL_OPTIONS[OTHER_MODEL]: '<p>The large one.</p>'})
    built = window(OTHER_MODEL)

    built._model_list.setCurrentRow(0)

    assert built._details.toPlainText() == ''


