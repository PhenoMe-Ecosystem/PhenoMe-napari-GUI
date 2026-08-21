"""Tests for the interactive explorer window.

Unlike the parameter forms, the window writes straight into params.explorer as the combo boxes are
touched: it is non-modal, so there is nothing to confirm and no copy to hand back. It is also built
once and kept, which is why refresh_choices() has to bring its choices up to date by hand.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest
from qtpy.QtCore import Qt

from phenome_gui._widget import PhenoMeWidget
from phenome_gui.ui.criteria_window import ExcludeWindow, FilterWindow
from phenome_gui.ui.explorer_window import InteractiveExplorerWindow
from phenome_gui.ui.plotting import BOX_SELECTION, LASSO_SELECTION
from phenome_gui.utils.params import (
    DIMS_CHOICES,
    EMBEDDING_METHOD_CHOICES,
    SOURCE_CHOICES,
)

# The columns of the test dataset's .csv, in the order they are loaded. PhenoMe adds an 'id' column of
# its own on top, so these are what the metadata holds rather than all of it.
COLUMNS = ['filename', 'drug_name', 'time']


@pytest.fixture
def make_explorer(widget: PhenoMeWidget, qtbot):
    """ Return a hook building an explorer window on the widget's interface and viewer.

    A factory rather than a plain fixture, as the window reads the parameters when it is constructed:
    a test asking what it opens on has to set them first.

    Returns:
        Callable[[], InteractiveExplorerWindow]: Called as make_explorer().
    """
    def _build() -> InteractiveExplorerWindow:
        window = InteractiveExplorerWindow(widget._phenome, widget._image_viewer, widget)
        qtbot.addWidget(window)
        return window

    return _build


@pytest.fixture
def explorer(make_explorer) -> InteractiveExplorerWindow:
    """ An explorer window on a widget with nothing loaded or processed. """
    return make_explorer()


@pytest.fixture
def loaded_explorer(widget: PhenoMeWidget, make_explorer) -> InteractiveExplorerWindow:
    """ An explorer window built once the test dataset has been loaded. """
    widget.loading_panel._on_load_data_csv()
    return make_explorer()


def items(combo) -> list[str]:
    """ The choices currently offered by a Qt combo box. """
    return [combo.itemText(i) for i in range(combo.count())]


def explorer_params(window: InteractiveExplorerWindow):
    """ The explorer section of the parameters the window writes into. """
    return window._phenome.params.explorer


# ----------------------------------------------------------------------------
# The choices, which come from the parameter model
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('combo', 'choices'),
    [
        pytest.param('_method_combo', EMBEDDING_METHOD_CHOICES, id='method'),
        pytest.param('_dims_combo', DIMS_CHOICES, id='dims'),
        pytest.param('_source_combo', SOURCE_CHOICES, id='source'),
    ],
)
def test_the_combos_offer_the_values_the_model_accepts(
    explorer: InteractiveExplorerWindow, combo: str, choices: list[str]
) -> None:
    """Offering anything else would raise on assignment, the fields being typed as literals."""
    assert items(getattr(explorer, combo)) == choices


def test_the_combos_open_on_the_stored_parameters(widget: PhenoMeWidget, make_explorer) -> None:
    widget._phenome.params.explorer.method = 'PCA'
    widget._phenome.params.explorer.dims = '3D'
    widget._phenome.params.explorer.source = 'properties'

    explorer = make_explorer()

    assert explorer._method_combo.currentText() == 'PCA'
    assert explorer._dims_combo.currentText() == '3D'
    assert explorer._source_combo.currentText() == 'properties'


# ----------------------------------------------------------------------------
# Writing straight into the shared parameters
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('combo', 'field', 'value'),
    [
        pytest.param('_method_combo', 'method', 'UMAP', id='method'),
        pytest.param('_dims_combo', 'dims', '3D', id='dims'),
        pytest.param('_source_combo', 'source', 'combined', id='source'),
    ],
)
def test_changing_a_combo_writes_its_parameter(
    explorer: InteractiveExplorerWindow, combo: str, field: str, value: str
) -> None:
    getattr(explorer, combo).setCurrentText(value)

    assert getattr(explorer_params(explorer), field) == value


def test_the_window_edits_the_shared_parameters_rather_than_a_copy(
    explorer: InteractiveExplorerWindow, widget: PhenoMeWidget
) -> None:
    """The window stays open and is read whenever a projection is computed, so there is nothing to confirm."""
    explorer._method_combo.setCurrentText('PCA')

    assert widget._phenome.params.explorer.method == 'PCA'


# ----------------------------------------------------------------------------
# What the window waits on
# ----------------------------------------------------------------------------


def test_the_source_waits_on_the_properties(explorer: InteractiveExplorerWindow) -> None:
    assert not explorer._source_combo.isEnabled()
    assert explorer._source_combo.toolTip() == 'Please calculate properties first.'


def test_the_source_opens_once_the_properties_are_computed(explorer: InteractiveExplorerWindow) -> None:
    explorer._phenome.properties_df = pd.DataFrame({'image_path': ['a.tif'], 'area': [1.0]})

    explorer.refresh_choices()

    assert explorer._source_combo.isEnabled()
    assert explorer._source_combo.toolTip() == ''


def test_the_colouring_waits_on_the_data(explorer: InteractiveExplorerWindow) -> None:
    assert items(explorer._color_by_combo) == []
    assert not explorer._color_by_combo.isEnabled()
    assert explorer._color_by_combo.toolTip() == 'Please load data first.'
    # An unpicked column reads as None, the way the analysis steps say they have no grouping key
    assert explorer_params(explorer).color_by is None


# ----------------------------------------------------------------------------
# The colouring column, refilled from the metadata
# ----------------------------------------------------------------------------


def test_loading_the_data_opens_the_colouring(loaded_explorer: InteractiveExplorerWindow) -> None:
    offered = items(loaded_explorer._color_by_combo)

    assert offered == loaded_explorer._phenome.get_metadata_columns()
    assert set(COLUMNS) <= set(offered)
    assert loaded_explorer._color_by_combo.isEnabled()
    assert loaded_explorer._color_by_combo.toolTip() == ''


def test_the_colouring_says_which_column_it_landed_on(loaded_explorer: InteractiveExplorerWindow) -> None:
    """color_by starts as None, so the combo lands on the first column and the parameter has to follow."""
    assert explorer_params(loaded_explorer).color_by == COLUMNS[0]


def test_a_stored_column_survives_a_refresh(widget: PhenoMeWidget, make_explorer) -> None:
    """Refilling the combo emits its change signal twice, which would overwrite the stored column."""
    widget.loading_panel._on_load_data_csv()
    widget._phenome.params.explorer.color_by = 'time'

    explorer = make_explorer()

    assert explorer._color_by_combo.currentText() == 'time'
    assert explorer_params(explorer).color_by == 'time'


def test_a_column_that_is_gone_falls_back_to_the_first(widget: PhenoMeWidget, make_explorer) -> None:
    """A column can vanish under the window, the data being loadable again while it stays open."""
    widget.loading_panel._on_load_data_csv()
    widget._phenome.params.explorer.color_by = 'vanished'

    explorer = make_explorer()

    assert explorer._color_by_combo.currentText() == COLUMNS[0]
    assert explorer_params(explorer).color_by == COLUMNS[0]


def test_changing_the_colouring_writes_its_parameter(loaded_explorer: InteractiveExplorerWindow) -> None:
    loaded_explorer._color_by_combo.setCurrentText('drug_name')

    assert explorer_params(loaded_explorer).color_by == 'drug_name'


def test_a_refresh_picks_up_a_column_added_elsewhere(loaded_explorer: InteractiveExplorerWindow) -> None:
    """This is how the cluster labels become a colouring option, the clustering never touching the window."""
    loaded_explorer._phenome.metadata_df['cluster'] = [0, 1] * 3

    loaded_explorer.refresh_choices()

    assert 'cluster' in items(loaded_explorer._color_by_combo)


def test_the_properties_join_the_colouring_choices(loaded_explorer: InteractiveExplorerWindow, processed) -> None:
    """A property is as worth colouring by as a metadata column, once it has been computed."""
    assert 'area' not in items(loaded_explorer._color_by_combo)
    results = processed(loaded_explorer._phenome, 6)
    results.properties = [{'area': 12.0}] * 6

    loaded_explorer.refresh_choices()

    assert 'area' in items(loaded_explorer._color_by_combo)


def test_losing_the_data_puts_the_colouring_back_to_none(loaded_explorer: InteractiveExplorerWindow) -> None:
    """The emptied combo answers with an empty string, which would land in the parameter as one."""
    loaded_explorer._phenome.metadata_df = None

    loaded_explorer.refresh_choices()

    assert items(loaded_explorer._color_by_combo) == []
    assert explorer_params(loaded_explorer).color_by is None


# ----------------------------------------------------------------------------
# Computing the projection
# ----------------------------------------------------------------------------


@pytest.fixture
def computing(loaded_explorer: InteractiveExplorerWindow, reduction, sync_worker):
    """ Return a hook stubbing the projection and running it inline.

    The window starts the projection on a worker thread, which would make the assertions racy, and no
    test may run a real reduction.

    Returns:
        Callable[[pd.DataFrame | None], ReductionSpy]: Called as computing(frame).
    """
    def _stub(frame: pd.DataFrame | None = None):
        stand_in = reduction(coordinates() if frame is None else frame)
        sync_worker(loaded_explorer, '_projection_worker', loaded_explorer._phenome.compute_projection)
        return stand_in

    return _stub


def coordinates(n_components: int = 2, groups: list[str] | None = None) -> pd.DataFrame:
    """ A frame shaped like the one the reduction returns, metadata columns included.

    Args:
        n_components: Number of coordinate columns, as the number of dimensions asked for.
        groups: Value of the capitalised metadata columns for each row. None for the dataset's own.
    """
    groups = groups or ['ctrl', 'ctrl', 'drugA']
    frame = pd.DataFrame(
        {f'Component {i + 1}': [float(row) for row in range(len(groups))] for i in range(n_components)}
    )

    # PhenoMe capitalises the metadata columns it copies into the projection, and hands their values
    # back as text, which is what keeps a numeric one like the time drawn as groups. The first three
    # rows are the ones most of these tests read; a longer frame goes on at a third time.
    frame['Drug_name'] = groups
    frame['Time'] = [('0', '24', '24')[row] if row < 3 else '48' for row in range(len(groups))]
    return frame


def drawn(window: InteractiveExplorerWindow) -> dict[str | None, int]:
    """ The groups currently on the canvas, as a name to number of points mapping. """
    return {item.opts.get('name'): len(item.getData()[0]) for item in window._canvas._plot.listDataItems()}


def test_computing_reports_the_parameters_it_runs_with(
    loaded_explorer: InteractiveExplorerWindow, computing, caplog: pytest.LogCaptureFixture
) -> None:
    computing()
    loaded_explorer._method_combo.setCurrentText('UMAP')

    with caplog.at_level(logging.INFO, logger='main_logger'):
        loaded_explorer._compute_button.click()

    assert 'Compute clicked' in caplog.text
    assert "method='UMAP'" in caplog.text


def test_computing_projects_with_the_chosen_method_and_dims(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    stand_in = computing(coordinates(3))
    loaded_explorer._method_combo.setCurrentText('UMAP')
    loaded_explorer._dims_combo.setCurrentText('3D')

    loaded_explorer._compute_button.click()

    assert stand_in.last == {
        'method': 'umap',
        'n_components': 3,
        'source': 'embeddings',
        'filters': None,
        'exclude': None,
    }


def test_computing_keeps_the_projection_to_the_chosen_images(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    """Set in the forms, applied by the next run: the plot on screen was computed before them."""
    stand_in = computing()
    explorer_params(loaded_explorer).filter = {'drug_name': ['ctrl']}
    explorer_params(loaded_explorer).exclude = {'time': [24]}

    loaded_explorer._compute_button.click()

    assert stand_in.last['filters'] == {'drug_name': ['ctrl']}
    assert stand_in.last['exclude'] == {'time': [24]}


def test_computing_keeps_the_coordinates(loaded_explorer: InteractiveExplorerWindow, computing) -> None:
    frame = coordinates()
    computing(frame)

    loaded_explorer._compute_button.click()

    assert loaded_explorer._phenome.projection_df is frame


def test_the_button_is_closed_while_the_projection_runs(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    """A second run would project the same images into a plot only the last one to finish would reach."""
    stand_in = computing()
    while_running = []
    stand_in.on_call = lambda: while_running.append(loaded_explorer._compute_button.isEnabled())

    loaded_explorer._compute_button.click()

    assert while_running == [False]
    assert loaded_explorer._compute_button.isEnabled()


def test_the_button_reopens_when_there_was_nothing_to_project(
    loaded_explorer: InteractiveExplorerWindow, computing, reduction
) -> None:
    """An empty answer is a normal outcome, and must not leave the window unable to try again."""
    computing()
    reduction(None)

    loaded_explorer._compute_button.click()

    assert loaded_explorer._phenome.projection_df is None
    assert loaded_explorer._compute_button.isEnabled()


def test_a_failed_projection_reopens_the_button(
    loaded_explorer: InteractiveExplorerWindow, computing, caplog: pytest.LogCaptureFixture
) -> None:
    """PhenoMe raises for the failures it can name, such as t-SNE on fewer samples than its perplexity."""
    stand_in = computing()

    def _raise() -> None:
        raise ValueError('perplexity must be less than n_samples')

    stand_in.on_call = _raise

    with caplog.at_level(logging.ERROR, logger='main_logger'):
        loaded_explorer._compute_button.click()

    assert 'Could not compute the projection' in caplog.text
    assert loaded_explorer._compute_button.isEnabled()


def test_computing_changes_no_parameter(loaded_explorer: InteractiveExplorerWindow, computing) -> None:
    """The combos are what set the parameters; the button only reads them."""
    computing()
    before = explorer_params(loaded_explorer).model_copy(deep=True)

    loaded_explorer._compute_button.click()

    assert explorer_params(loaded_explorer) == before


# ----------------------------------------------------------------------------
# Drawing what came back
# ----------------------------------------------------------------------------


def test_a_finished_projection_is_drawn(loaded_explorer: InteractiveExplorerWindow, computing) -> None:
    computing()
    loaded_explorer._color_by_combo.setCurrentText('drug_name')

    loaded_explorer._compute_button.click()

    assert drawn(loaded_explorer) == {'ctrl': 2, 'drugA': 1}


def test_an_empty_projection_clears_the_canvas(loaded_explorer: InteractiveExplorerWindow, computing, reduction) -> None:
    computing()
    loaded_explorer._compute_button.click()
    assert drawn(loaded_explorer)

    reduction(None)
    loaded_explorer._compute_button.click()

    assert drawn(loaded_explorer) == {}


def test_choosing_another_column_recolours_without_recomputing(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    """The coordinates are the projection's, so only the grouping is redone."""
    stand_in = computing()
    loaded_explorer._color_by_combo.setCurrentText('drug_name')
    loaded_explorer._compute_button.click()
    runs = len(stand_in.calls)

    loaded_explorer._color_by_combo.setCurrentText('time')

    assert drawn(loaded_explorer) == {'24': 2, '0': 1}
    assert len(stand_in.calls) == runs


def test_colouring_by_a_property_draws_a_ramp(loaded_explorer: InteractiveExplorerWindow, computing) -> None:
    """A measured column is drawn along a colorbar, where a metadata one keeps its legend."""
    frame = measured_projection(AREAS)
    computing(frame)
    loaded_explorer._phenome.params.explorer.color_by = 'area'

    loaded_explorer._compute_button.click()

    assert drawn(loaded_explorer) == {None: len(AREAS)}
    assert loaded_explorer._canvas._colorbar is not None
    assert loaded_explorer._canvas._legend.items == []


# ----------------------------------------------------------------------------
# Choosing what the projection is computed from
# ----------------------------------------------------------------------------


@pytest.fixture
def criteria_form(monkeypatch: pytest.MonkeyPatch):
    """ Return a hook driving one of the two picking forms without showing it.

    Patches its exec_ so it chooses a field, picks the given values on it and then closes, which is
    what a user would leave behind.

    Returns:
        Callable[..., list]: Called as criteria_form(FilterWindow, 'drug_name', 'ctrl'), returning
            the forms opened so far.
    """
    opened = []

    def _drive(window_class: type, field: str | None = None, *picks: str) -> list:
        def _exec(dialog) -> int:
            opened.append(dialog)
            if field:
                dialog._field_combo.setCurrentText(field)
            for text in picks:
                for item in dialog._values_list.findItems(text, Qt.MatchFlag.MatchExactly):
                    item.setSelected(True)
            if picks:
                dialog._add_button.click()
            return 0

        monkeypatch.setattr(window_class, 'exec_', _exec)
        return opened

    return _drive


# The two buttons, each with the form it opens and the parameter it writes into
CRITERIA_BUTTONS = [
    pytest.param('_filter_button', FilterWindow, 'filter', id='filter'),
    pytest.param('_exclude_button', ExcludeWindow, 'exclude', id='exclude'),
]


@pytest.mark.parametrize(('button', 'form', 'field'), CRITERIA_BUTTONS)
class TestCriteriaForms:
    """Both buttons open the same form under a different name and write a parameter of their own."""

    def test_the_button_opens_the_form(
        self, loaded_explorer: InteractiveExplorerWindow, criteria_form, button, form, field
    ) -> None:
        opened = criteria_form(form)

        getattr(loaded_explorer, button).click()

        assert len(opened) == 1
        assert isinstance(opened[0], form)

    def test_the_form_offers_the_metadata_fields_and_their_values(
        self, loaded_explorer: InteractiveExplorerWindow, criteria_form, button, form, field
    ) -> None:
        """Metadata only: a value list is no way to pick on a measured property."""
        opened = criteria_form(form, 'drug_name')

        getattr(loaded_explorer, button).click()

        fields = [opened[0]._field_combo.itemText(i) for i in range(opened[0]._field_combo.count())]
        assert fields == loaded_explorer._phenome.get_metadata_columns()
        assert 'ctrl' in [opened[0]._values_list.item(i).text() for i in range(opened[0]._values_list.count())]

    def test_what_the_form_picked_lands_in_the_parameters(
        self, loaded_explorer: InteractiveExplorerWindow, criteria_form, button, form, field
    ) -> None:
        """The form has no Cancel, so closing it is not a way of taking the picks back."""
        criteria_form(form, 'drug_name', 'ctrl')

        getattr(loaded_explorer, button).click()

        assert getattr(explorer_params(loaded_explorer), field) == {'drug_name': ['ctrl']}

    def test_picking_nothing_leaves_the_parameter_unset(
        self, loaded_explorer: InteractiveExplorerWindow, criteria_form, button, form, field
    ) -> None:
        """None and an empty criterion are the same to PhenoMe, and None reads as what it is."""
        criteria_form(form, 'drug_name')

        getattr(loaded_explorer, button).click()

        assert getattr(explorer_params(loaded_explorer), field) is None

    def test_the_form_opens_again_on_what_was_picked_before(
        self, loaded_explorer: InteractiveExplorerWindow, criteria_form, button, form, field
    ) -> None:
        criteria_form(form, 'drug_name', 'ctrl')
        getattr(loaded_explorer, button).click()

        opened = criteria_form(form, 'drug_name')
        getattr(loaded_explorer, button).click()

        picked = opened[-1]._picked_list
        assert [picked.item(i).text() for i in range(picked.count())] == ['ctrl']

    def test_the_form_is_rebuilt_on_every_use(
        self, loaded_explorer: InteractiveExplorerWindow, criteria_form, button, form, field
    ) -> None:
        """Its values are read when it is built, so a reload has to reach it."""
        opened = criteria_form(form)

        getattr(loaded_explorer, button).click()
        getattr(loaded_explorer, button).click()

        assert opened[0] is not opened[1]


@pytest.mark.parametrize(('button', 'form', 'field'), CRITERIA_BUTTONS)
def test_criteria_on_two_fields_are_reported_without_raising(
    loaded_explorer: InteractiveExplorerWindow, criteria_form, button: str, form: type, field: str
) -> None:
    """A mapping handed to the logger on its own is read as values to interpolate, and raises."""
    setattr(explorer_params(loaded_explorer), field, {'drug_name': ['ctrl'], 'time': [24]})
    criteria_form(form)

    getattr(loaded_explorer, button).click()

    assert getattr(explorer_params(loaded_explorer), field) == {'drug_name': ['ctrl'], 'time': [24]}


def test_the_two_forms_write_their_own_parameter(
    loaded_explorer: InteractiveExplorerWindow, criteria_form
) -> None:
    """Filtering and excluding are set apart and applied together, not one instead of the other."""
    criteria_form(FilterWindow, 'drug_name', 'ctrl')
    loaded_explorer._filter_button.click()

    criteria_form(ExcludeWindow, 'time', '24')
    loaded_explorer._exclude_button.click()

    assert explorer_params(loaded_explorer).filter == {'drug_name': ['ctrl']}
    assert explorer_params(loaded_explorer).exclude == {'time': [24]}


# ----------------------------------------------------------------------------
# The highlight controls, read off the drawn projection
# ----------------------------------------------------------------------------


def measured_projection(areas: list[float]) -> pd.DataFrame:
    """ A projection carrying a measured property alongside its metadata.

    As many rows as there are values: a column is only read as a measurement once it holds enough
    distinct values to spread a ramp over.
    """
    frame = coordinates(groups=[('ctrl', 'drugA')[row % 2] for row in range(len(areas))])
    frame['area'] = areas
    return frame


# Enough distinct values for a column to be read as a measurement rather than as labels
AREAS = [120.0, 512.0, 340.0, 220.0, 780.0, 410.0, 615.0, 905.0]


def highlight_page(window: InteractiveExplorerWindow):
    """ The control currently shown for the chosen field. """
    return window._highlight_stack.currentWidget()


@pytest.fixture
def measured(loaded_explorer: InteractiveExplorerWindow, computing, processed):
    """ Return a hook drawing a projection carrying a measured property.

    The property has to be one the pipeline holds as well as one the projection carries: the fields
    on offer are the colouring's, which reads the computed property keys rather than the frame.

    Returns:
        Callable[[list[float]], None]: Called as measured([120.0, 512.0]).
    """
    def _draw(areas: list[float]) -> None:
        results = processed(loaded_explorer._phenome, len(areas))
        results.properties = [{'area': area} for area in areas]

        computing(measured_projection(areas))
        loaded_explorer._compute_button.click()

    return _draw


def test_the_highlight_field_offers_what_the_colouring_offers(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    """Both read the same columns, so the plot and the marking can be told to answer for the same one."""
    computing()
    loaded_explorer._compute_button.click()

    assert items(loaded_explorer._highlight_field_combo) == items(loaded_explorer._color_by_combo)


def test_the_highlight_group_waits_on_a_projection(loaded_explorer: InteractiveExplorerWindow) -> None:
    """Its values are the drawn points', so there is nothing for it to offer until they exist."""
    assert not loaded_explorer._highlight_group.isEnabled()
    assert loaded_explorer._highlight_group.toolTip() == 'Please compute a projection first.'


def test_the_highlight_group_opens_once_a_projection_is_drawn(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing()

    loaded_explorer._compute_button.click()

    assert loaded_explorer._highlight_group.isEnabled()
    assert loaded_explorer._highlight_group.toolTip() == ''


def test_a_field_of_labels_is_picked_from_its_values(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing()
    loaded_explorer._compute_button.click()

    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')

    assert highlight_page(loaded_explorer) is loaded_explorer._highlight_values_combo
    assert items(loaded_explorer._highlight_values_combo) == ['ctrl', 'drugA']


def test_a_field_of_measurements_is_picked_from_its_range(
    loaded_explorer: InteractiveExplorerWindow, measured
) -> None:
    measured(AREAS)

    loaded_explorer._highlight_field_combo.setCurrentText('area')

    assert highlight_page(loaded_explorer) is loaded_explorer._highlight_range_slider
    assert loaded_explorer._highlight_range_slider.minimum() == min(AREAS)
    assert loaded_explorer._highlight_range_slider.maximum() == max(AREAS)


def test_the_range_opens_on_the_whole_of_it(loaded_explorer: InteractiveExplorerWindow, measured) -> None:
    """Wide open reads as picking every point, where a slice would pick one nobody asked for."""
    measured(AREAS)

    loaded_explorer._highlight_field_combo.setCurrentText('area')

    assert tuple(loaded_explorer._highlight_range_slider.value()) == (min(AREAS), max(AREAS))


def test_the_control_changes_with_the_kind_of_field(
    loaded_explorer: InteractiveExplorerWindow, measured
) -> None:
    measured(AREAS)

    loaded_explorer._highlight_field_combo.setCurrentText('area')
    assert highlight_page(loaded_explorer) is loaded_explorer._highlight_range_slider

    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')

    assert highlight_page(loaded_explorer) is loaded_explorer._highlight_values_combo


def test_a_column_of_one_value_is_picked_from_a_list_of_it(
    loaded_explorer: InteractiveExplorerWindow, measured
) -> None:
    """A measurement that never varies is read as a label, so it is picked the way labels are."""
    measured([7.0] * len(AREAS))

    loaded_explorer._highlight_field_combo.setCurrentText('area')

    assert highlight_page(loaded_explorer) is loaded_explorer._highlight_values_combo
    assert items(loaded_explorer._highlight_values_combo) == ['7.0']


def test_a_field_the_projection_does_not_carry_offers_nothing(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    """A column added to the metadata after the run is not in it, which must not raise."""
    computing()
    loaded_explorer._compute_button.click()
    loaded_explorer._phenome.metadata_df['cluster'] = [0, 1] * 3
    loaded_explorer.refresh_highlight()

    loaded_explorer._highlight_field_combo.setCurrentText('cluster')

    assert highlight_page(loaded_explorer) is loaded_explorer._highlight_values_combo
    assert items(loaded_explorer._highlight_values_combo) == []


def test_computing_again_brings_the_values_up_to_date(
    loaded_explorer: InteractiveExplorerWindow, computing, reduction
) -> None:
    computing()
    loaded_explorer._compute_button.click()
    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')
    assert items(loaded_explorer._highlight_values_combo) == ['ctrl', 'drugA']

    reduction(coordinates(groups=['drugB', 'drugB', 'drugC']))
    loaded_explorer._compute_button.click()

    assert items(loaded_explorer._highlight_values_combo) == ['drugB', 'drugC']


def test_the_highlight_checkbox_changes_no_parameter(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    """It marks what is drawn, and says nothing about how the projection is computed."""
    computing()
    loaded_explorer._compute_button.click()
    before = explorer_params(loaded_explorer).model_copy(deep=True)

    loaded_explorer._highlight_checkbox.setChecked(True)

    assert loaded_explorer._highlight_checkbox.text() == 'Highlight'
    assert explorer_params(loaded_explorer) == before


# ----------------------------------------------------------------------------
# Marking the drawn points from the highlight controls
# ----------------------------------------------------------------------------


def dimmed_rows(window: InteractiveExplorerWindow) -> list[bool]:
    """ Which of the drawn points the canvas is currently fading back. """
    dimmed = window._canvas._dimmed
    return [] if dimmed is None else list(dimmed)


def test_checking_the_box_marks_the_picked_points(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing()
    loaded_explorer._compute_button.click()
    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')
    loaded_explorer._highlight_values_combo.setCurrentText('drugA')

    loaded_explorer._highlight_checkbox.setChecked(True)

    # The projection holds ctrl, ctrl, drugA
    assert dimmed_rows(loaded_explorer) == [True, True, False]


def test_unchecking_the_box_takes_the_marking_off(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing()
    loaded_explorer._compute_button.click()
    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')
    loaded_explorer._highlight_checkbox.setChecked(True)

    loaded_explorer._highlight_checkbox.setChecked(False)

    assert dimmed_rows(loaded_explorer) == []


def test_picking_another_value_marks_it_straight_away(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    """A pick made with the box already checked takes effect where it is made."""
    computing()
    loaded_explorer._compute_button.click()
    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')
    loaded_explorer._highlight_values_combo.setCurrentText('ctrl')
    loaded_explorer._highlight_checkbox.setChecked(True)

    loaded_explorer._highlight_values_combo.setCurrentText('drugA')

    assert dimmed_rows(loaded_explorer) == [True, True, False]


def test_moving_the_range_marks_what_it_now_holds(
    loaded_explorer: InteractiveExplorerWindow, measured
) -> None:
    measured(AREAS)
    loaded_explorer._highlight_field_combo.setCurrentText('area')
    loaded_explorer._highlight_checkbox.setChecked(True)

    loaded_explorer._highlight_range_slider.setValue((300.0, 600.0))

    assert dimmed_rows(loaded_explorer) == [not 300.0 <= area <= 600.0 for area in AREAS]


def test_a_pick_made_with_the_box_unchecked_marks_nothing(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing()
    loaded_explorer._compute_button.click()

    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')
    loaded_explorer._highlight_values_combo.setCurrentText('drugA')

    assert dimmed_rows(loaded_explorer) == []


def test_the_picked_value_keeps_its_type(loaded_explorer: InteractiveExplorerWindow, computing) -> None:
    """The combo shows the time as text where the column holds it as one, which would match nothing."""
    frame = coordinates()
    frame['Cluster'] = [0, 1, 1]
    computing(frame)
    loaded_explorer._compute_button.click()
    loaded_explorer._phenome.metadata_df['cluster'] = [0, 1] * 3
    loaded_explorer.refresh_highlight()

    loaded_explorer._highlight_field_combo.setCurrentText('cluster')
    loaded_explorer._highlight_values_combo.setCurrentText('1')
    loaded_explorer._highlight_checkbox.setChecked(True)

    assert loaded_explorer._highlight_values_combo.currentData() == 1
    assert dimmed_rows(loaded_explorer) == [True, False, False]


def test_recolouring_keeps_the_marking(loaded_explorer: InteractiveExplorerWindow, computing) -> None:
    """Recolouring draws the points again, which would otherwise come back unmarked."""
    computing()
    loaded_explorer._compute_button.click()
    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')
    loaded_explorer._highlight_values_combo.setCurrentText('drugA')
    loaded_explorer._highlight_checkbox.setChecked(True)

    loaded_explorer._color_by_combo.setCurrentText('time')

    assert dimmed_rows(loaded_explorer) == [True, True, False]


def test_computing_again_marks_the_new_points(
    loaded_explorer: InteractiveExplorerWindow, computing, reduction
) -> None:
    computing()
    loaded_explorer._compute_button.click()
    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')
    loaded_explorer._highlight_values_combo.setCurrentText('drugA')
    loaded_explorer._highlight_checkbox.setChecked(True)

    reduction(coordinates(groups=['drugA', 'ctrl', 'ctrl']))
    loaded_explorer._compute_button.click()

    assert dimmed_rows(loaded_explorer) == [False, True, True]


# ----------------------------------------------------------------------------
# Picking points out with a box
# ----------------------------------------------------------------------------


def indexed(indices: list[int]) -> pd.DataFrame:
    """ A projection whose rows say where PhenoMe holds each of their images. """
    frame = coordinates()
    frame['Index'] = indices
    return frame


def table_rows(window: InteractiveExplorerWindow) -> int:
    """ How many points the selection window is currently listing. """
    return window._view_selection_dialog._points_table.model().rowCount()


def test_the_box_button_asks_the_canvas_for_one(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing()
    loaded_explorer._compute_button.click()

    loaded_explorer._box_select_button.click()

    assert loaded_explorer._canvas._viewbox.selecting


def test_pressing_the_box_button_again_calls_it_off(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing()
    loaded_explorer._compute_button.click()
    loaded_explorer._box_select_button.click()

    loaded_explorer._box_select_button.click()

    assert not loaded_explorer._canvas._viewbox.selecting


def test_the_lasso_button_asks_the_canvas_for_one(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing()
    loaded_explorer._compute_button.click()

    loaded_explorer._lasso_select_button.click()

    assert loaded_explorer._canvas._viewbox.selecting == LASSO_SELECTION


@pytest.mark.parametrize(
    ('pressed', 'released', 'waiting'),
    [
        pytest.param('_lasso_select_button', '_box_select_button', LASSO_SELECTION, id='lasso-over-box'),
        pytest.param('_box_select_button', '_lasso_select_button', BOX_SELECTION, id='box-over-lasso'),
    ],
)
def test_the_two_shapes_are_a_choice_of_one(
    loaded_explorer: InteractiveExplorerWindow, computing, pressed: str, released: str, waiting: str
) -> None:
    """Checking one lets the other go: the plot waits for whichever was asked for last."""
    computing()
    loaded_explorer._compute_button.click()
    getattr(loaded_explorer, released).click()

    getattr(loaded_explorer, pressed).click()

    assert getattr(loaded_explorer, pressed).isChecked()
    assert not getattr(loaded_explorer, released).isChecked()
    assert loaded_explorer._canvas._viewbox.selecting == waiting


def test_a_drawn_lasso_counts_the_points_it_holds(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing(indexed([10, 20, 30]))
    loaded_explorer._compute_button.click()
    loaded_explorer._lasso_select_button.click()

    loaded_explorer._canvas.points_selected.emit([20, 30])

    assert loaded_explorer._num_select_label.text() == '2 points selected.'
    assert not loaded_explorer._lasso_select_button.isChecked()


def test_a_drawn_box_counts_the_points_it_covers(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing(indexed([10, 20, 30]))
    loaded_explorer._compute_button.click()

    loaded_explorer._canvas.points_selected.emit([20, 30])

    assert loaded_explorer._num_select_label.text() == '2 points selected.'
    assert loaded_explorer._selected_indices == [20, 30]


def test_a_drawn_box_lets_the_button_go(loaded_explorer: InteractiveExplorerWindow, computing) -> None:
    """The mode is over once the box is drawn, and the button is what says so."""
    computing(indexed([10, 20, 30]))
    loaded_explorer._compute_button.click()
    loaded_explorer._box_select_button.click()

    loaded_explorer._canvas.points_selected.emit([20])

    assert not loaded_explorer._box_select_button.isChecked()


def test_the_selection_fills_the_table(loaded_explorer: InteractiveExplorerWindow, computing) -> None:
    computing(indexed([10, 20, 30]))
    loaded_explorer._compute_button.click()
    loaded_explorer._canvas.points_selected.emit([20, 30])

    loaded_explorer._on_view_selection_clicked()

    assert table_rows(loaded_explorer) == 2


def test_a_selection_made_with_the_window_open_reaches_it(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing(indexed([10, 20, 30]))
    loaded_explorer._compute_button.click()
    loaded_explorer._on_view_selection_clicked()

    loaded_explorer._canvas.points_selected.emit([10])

    assert table_rows(loaded_explorer) == 1


def test_clearing_lets_go_of_the_selection(loaded_explorer: InteractiveExplorerWindow, computing) -> None:
    computing(indexed([10, 20, 30]))
    loaded_explorer._compute_button.click()
    loaded_explorer._canvas.points_selected.emit([20, 30])
    loaded_explorer._on_view_selection_clicked()

    loaded_explorer._clear_select_button.click()

    assert loaded_explorer._selected_indices == []
    assert loaded_explorer._num_select_label.text() == '0 points selected.'
    assert table_rows(loaded_explorer) == 0
    assert loaded_explorer._canvas.selected_indices() == []


def test_the_highlight_button_selects_the_points_it_answers_for(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing(indexed([10, 20, 30]))
    loaded_explorer._compute_button.click()
    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')
    loaded_explorer._highlight_values_combo.setCurrentText('ctrl')

    loaded_explorer._highlight_select_button.click()

    # The projection holds ctrl, ctrl, drugA
    assert loaded_explorer._selected_indices == [10, 20]
    assert loaded_explorer._num_select_label.text() == '2 points selected.'


def test_the_highlight_button_pays_no_mind_to_the_checkbox(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    """The box fades the plot; the field and the value are picked either way."""
    computing(indexed([10, 20, 30]))
    loaded_explorer._compute_button.click()
    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')
    loaded_explorer._highlight_values_combo.setCurrentText('drugA')
    assert not loaded_explorer._highlight_checkbox.isChecked()

    loaded_explorer._highlight_select_button.click()
    unchecked = list(loaded_explorer._selected_indices)

    loaded_explorer._highlight_checkbox.setChecked(True)
    loaded_explorer._highlight_select_button.click()

    assert unchecked == [30]
    assert loaded_explorer._selected_indices == unchecked


def test_the_highlight_button_selects_on_a_range_too(
    loaded_explorer: InteractiveExplorerWindow, measured
) -> None:
    measured(AREAS)
    loaded_explorer._highlight_field_combo.setCurrentText('area')
    loaded_explorer._highlight_range_slider.setValue((300.0, 600.0))

    loaded_explorer._highlight_select_button.click()

    inside = sum(300.0 <= area <= 600.0 for area in AREAS)
    assert loaded_explorer._num_select_label.text() == f'{inside} points selected.'


def test_the_highlight_button_fills_the_table(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    computing(indexed([10, 20, 30]))
    loaded_explorer._compute_button.click()
    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')
    loaded_explorer._highlight_values_combo.setCurrentText('ctrl')
    loaded_explorer._on_view_selection_clicked()

    loaded_explorer._highlight_select_button.click()

    assert table_rows(loaded_explorer) == 2


def test_the_highlight_button_calls_off_a_waiting_box(
    loaded_explorer: InteractiveExplorerWindow, computing
) -> None:
    """The box was waiting to say which points; the button has just said it another way."""
    computing(indexed([10, 20, 30]))
    loaded_explorer._compute_button.click()
    loaded_explorer._box_select_button.click()

    loaded_explorer._highlight_select_button.click()

    assert not loaded_explorer._canvas._viewbox.selecting
    assert not loaded_explorer._box_select_button.isChecked()


def test_picking_a_row_of_the_selection_shows_its_image(
    loaded_explorer: InteractiveExplorerWindow, clickable, viewer
) -> None:
    """The table walks through the picked images the way the metadata table walks through them all."""
    paths = clickable([4, 0, 2])
    loaded_explorer._canvas.points_selected.emit([4, 0])
    loaded_explorer._on_view_selection_clicked()
    table = loaded_explorer._view_selection_dialog._points_table

    table.setCurrentIndex(table.model().index(0, 0))

    assert viewer.names == [Path(paths[4]).name, f'{Path(paths[4]).name} - Mask']
    assert viewer.added[0]['metadata']['drug_name'] == loaded_explorer._phenome.metadata_df.iloc[4]['drug_name']


def test_picking_another_row_replaces_the_layers(
    loaded_explorer: InteractiveExplorerWindow, clickable, viewer
) -> None:
    paths = clickable([4, 0, 2])
    loaded_explorer._canvas.points_selected.emit([4, 0])
    loaded_explorer._on_view_selection_clicked()
    table = loaded_explorer._view_selection_dialog._points_table

    table.setCurrentIndex(table.model().index(0, 0))
    table.setCurrentIndex(table.model().index(1, 0))

    assert viewer.layers == []
    assert viewer.names[-2:] == [Path(paths[0]).name, f'{Path(paths[0]).name} - Mask']


def test_computing_again_lets_go_of_the_selection(
    loaded_explorer: InteractiveExplorerWindow, computing, reduction
) -> None:
    """It was drawn over the points that were there before, which these are not."""
    computing(indexed([10, 20, 30]))
    loaded_explorer._compute_button.click()
    loaded_explorer._canvas.points_selected.emit([20, 30])
    loaded_explorer._on_view_selection_clicked()

    reduction(indexed([40, 50, 60]))
    loaded_explorer._compute_button.click()

    assert loaded_explorer._selected_indices == []
    assert loaded_explorer._num_select_label.text() == '0 points selected.'
    assert table_rows(loaded_explorer) == 0


# ----------------------------------------------------------------------------
# Clicking a point through to the viewer
# ----------------------------------------------------------------------------


@pytest.fixture
def clickable(loaded_explorer: InteractiveExplorerWindow, computing, processed):
    """ Return a hook drawing a projection whose points can be resolved back to real images.

    The projection carries where PhenoMe holds each image rather than its path, so the pipeline has
    to hold the paths for a clicked point to lead anywhere.

    Returns:
        Callable[[list[int]], list[str]]: Called as clickable([4, 0, 2]), returning the image paths.
    """
    def _draw(indices: list[int]) -> list[str]:
        paths = list(loaded_explorer._phenome.metadata_df['file_path'])
        processed(loaded_explorer._phenome, len(paths)).img_path = paths

        frame = coordinates()
        frame['Index'] = indices
        computing(frame)
        loaded_explorer._compute_button.click()
        return paths

    return _draw


def test_clicking_a_point_shows_its_image(
    loaded_explorer: InteractiveExplorerWindow, clickable, viewer
) -> None:
    paths = clickable([4, 0, 2])

    loaded_explorer._canvas.point_clicked.emit(1)

    assert viewer.names == [Path(paths[0]).name, f'{Path(paths[0]).name} - Mask']
    assert [layer['kind'] for layer in viewer.added] == ['image', 'labels']


def test_the_shown_image_carries_its_metadata(
    loaded_explorer: InteractiveExplorerWindow, clickable, viewer
) -> None:
    clickable([4, 0, 2])

    loaded_explorer._canvas.point_clicked.emit(0)

    expected = loaded_explorer._phenome.metadata_df.iloc[4]['drug_name']
    assert viewer.added[0]['metadata']['drug_name'] == expected


def test_clicking_another_point_replaces_the_layers(
    loaded_explorer: InteractiveExplorerWindow, clickable, viewer
) -> None:
    """The plot is a way of looking through the images, not of piling them up in the viewer."""
    paths = clickable([4, 0, 2])

    loaded_explorer._canvas.point_clicked.emit(0)
    loaded_explorer._canvas.point_clicked.emit(2)

    assert viewer.layers == []
    assert viewer.names[-2:] == [Path(paths[2]).name, f'{Path(paths[2]).name} - Mask']


def test_refreshing_the_choices_leaves_an_empty_canvas_alone(
    loaded_explorer: InteractiveExplorerWindow
) -> None:
    """Refreshing runs the colouring handler, which must not draw before anything has been computed."""
    loaded_explorer.refresh_choices()

    assert drawn(loaded_explorer) == {}


def test_reopening_the_window_keeps_the_plot(loaded_explorer: InteractiveExplorerWindow, computing) -> None:
    """The window is built once and kept, so a refresh must not wipe what is on the canvas."""
    computing()
    loaded_explorer._color_by_combo.setCurrentText('drug_name')
    loaded_explorer._compute_button.click()

    loaded_explorer.refresh_choices()

    assert drawn(loaded_explorer) == {'ctrl': 2, 'drugA': 1}


# ----------------------------------------------------------------------------
# Letting go of everything when another dataset is loaded
# ----------------------------------------------------------------------------


@pytest.fixture
def explored(loaded_explorer: InteractiveExplorerWindow, computing) -> InteractiveExplorerWindow:
    """ An explorer holding a drawn projection, a highlight applied to it and a listed selection. """
    computing(indexed([10, 20, 30]))
    loaded_explorer._compute_button.click()

    loaded_explorer._highlight_field_combo.setCurrentText('drug_name')
    loaded_explorer._highlight_checkbox.setChecked(True)

    loaded_explorer._canvas.points_selected.emit([20, 30])
    loaded_explorer._on_view_selection_clicked()

    return loaded_explorer


def reloaded(window: InteractiveExplorerWindow) -> InteractiveExplorerWindow:
    """ Load a dataset the way the loading panel does, then tell the window as the widget does.

    The interface drops its results and finds the new files before the window hears about it, so the
    window is cleared against the metadata it is about to offer rather than against nothing at all.
    """
    window._phenome.load_data_csv()
    window.clear_results()
    return window


def test_clearing_takes_the_projection_off_the_canvas(explored: InteractiveExplorerWindow) -> None:
    reloaded(explored)

    assert drawn(explored) == {}
    assert explored._canvas._points is None


def test_a_load_lets_go_of_the_selection(explored: InteractiveExplorerWindow) -> None:
    """It points at places in results that are no longer there."""
    reloaded(explored)

    assert explored._selected_indices == []
    assert explored._num_select_label.text() == '0 points selected.'
    assert explored._canvas.selected_indices() == []
    assert table_rows(explored) == 0


def test_clearing_closes_the_highlight_back_down(explored: InteractiveExplorerWindow) -> None:
    assert explored._highlight_group.isEnabled()

    reloaded(explored)

    assert not explored._highlight_checkbox.isChecked()
    assert not explored._highlight_group.isEnabled()
    assert explored._highlight_group.toolTip() == 'Please compute a projection first.'


def test_clearing_stops_waiting_for_a_shape(explored: InteractiveExplorerWindow) -> None:
    explored._box_select_button.click()

    reloaded(explored)

    assert not explored._box_select_button.isChecked()
    assert not explored._lasso_select_button.isChecked()
    assert explored._canvas._viewbox.selecting is None


def test_clearing_refills_the_colouring_from_the_metadata_now_loaded(
    explored: InteractiveExplorerWindow
) -> None:
    explored._color_by_combo.setCurrentText('drug_name')

    reloaded(explored)

    assert items(explored._color_by_combo) == explored._phenome.get_color_by_columns()


def test_clearing_after_a_failed_load_leaves_nothing_offered(explored: InteractiveExplorerWindow) -> None:
    """A load that found no file leaves no metadata to colour by, and the window has to read as much."""
    explored._phenome.clear_results()

    explored.clear_results()

    assert items(explored._color_by_combo) == []
    assert not explored._color_by_combo.isEnabled()
    assert explorer_params(explored).color_by is None


def test_clearing_an_untouched_window_raises_nothing(explorer: InteractiveExplorerWindow) -> None:
    """The window is cleared on every load, including the ones made before anything was computed."""
    explorer.clear_results()

    assert drawn(explorer) == {}
    assert explorer._selected_indices == []
