"""Tests for the forms picking which images a projection is computed from.

The filter and the exclusion are one form under two names, so everything but the two names is tested
on both. The form is handed the fields and the metadata to read their values from, and knows nothing
else, so the frames here are written by hand.
"""

import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QWidget

from phenome_gui.ui.criteria_window import ExcludeWindow, FilterWindow

COLUMNS = ['drug_name', 'time', 'cluster']


@pytest.fixture(params=[FilterWindow, ExcludeWindow], ids=['filter', 'exclude'])
def window_class(request) -> type:
    """ Each of the two forms in turn, which behave alike in everything below. """
    return request.param


@pytest.fixture
def parent(qtbot) -> QWidget:
    """ A bare widget to parent the dialog to, so it is destroyed with the test. """
    widget = QWidget()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def metadata() -> pd.DataFrame:
    """ A metadata frame with a repeated value, a numeric field and the cluster labels. """
    return pd.DataFrame(
        {
            'drug_name': ['ctrl', 'ctrl', 'drugA', 'drugB'],
            'time': [0, 24, 24, 0],
            'cluster': pd.array([1, 10, 2, pd.NA], dtype='Int64'),
        }
    )


@pytest.fixture
def form(window_class: type, parent: QWidget, metadata: pd.DataFrame):
    """ A form on the test metadata, with nothing picked yet. """
    return window_class(parent, COLUMNS, metadata)


def items(values) -> list[str]:
    """ The values currently listed by a list widget. """
    return [values.item(row).text() for row in range(values.count())]


def choices(combo) -> list[str]:
    """ The fields currently offered by a combo box. """
    return [combo.itemText(index) for index in range(combo.count())]


def labels(form) -> list[str]:
    """ The text of every label on the form, in the order they were built. """
    return [label.text() for label in form.findChildren(QLabel)]


def select(values, *texts: str) -> None:
    """ Select the given values in a list widget, as clicking them would. """
    for text in texts:
        for item in values.findItems(text, Qt.MatchFlag.MatchExactly):
            item.setSelected(True)


# ----------------------------------------------------------------------------
# What tells the two forms apart
# ----------------------------------------------------------------------------


def test_each_form_goes_under_its_own_name(parent: QWidget, metadata: pd.DataFrame) -> None:
    assert FilterWindow(parent, COLUMNS, metadata).windowTitle() == 'Filter Images'
    assert ExcludeWindow(parent, COLUMNS, metadata).windowTitle() == 'Exclude Images'


def test_each_form_heads_its_picks_its_own_way(parent: QWidget, metadata: pd.DataFrame) -> None:
    """The values are the same values; what differs is what the projection then does with them."""
    assert labels(FilterWindow(parent, COLUMNS, metadata)) == ['Field: ', 'Values', 'Filtered Values']
    assert labels(ExcludeWindow(parent, COLUMNS, metadata)) == ['Field: ', 'Values', 'Excluded Values']


# ----------------------------------------------------------------------------
# The field and its values
# ----------------------------------------------------------------------------


def test_the_fields_are_the_ones_it_was_given(form) -> None:
    """The cluster is a metadata column like any other once the clustering has written it."""
    assert choices(form._field_combo) == COLUMNS


def test_the_values_of_the_first_field_are_listed(form) -> None:
    assert items(form._values_list) == ['ctrl', 'drugA', 'drugB']
    assert items(form._picked_list) == []


def test_the_values_are_listed_once_each(form) -> None:
    """Two images of the same drug are one value to pick, not two."""
    assert items(form._values_list).count('ctrl') == 1


def test_numbers_are_ordered_as_numbers(form) -> None:
    """Ordered as text, cluster 10 would come between 1 and 2."""
    form._field_combo.setCurrentText('cluster')

    assert items(form._values_list) == ['1', '2', '10']


def test_a_value_the_frame_has_none_of_is_not_listed(form) -> None:
    """The clustering leaves its noise points unlabelled, which is not a value to pick."""
    form._field_combo.setCurrentText('cluster')

    assert len(items(form._values_list)) == 3


def test_choosing_another_field_lists_its_values(form) -> None:
    form._field_combo.setCurrentText('time')

    assert items(form._values_list) == ['0', '24']


def test_the_field_waits_on_the_data(window_class: type, parent: QWidget) -> None:
    form = window_class(parent, [], None)

    assert choices(form._field_combo) == []
    assert not form._field_combo.isEnabled()
    assert form._field_combo.toolTip() == 'Please load data first.'
    assert items(form._values_list) == []


def test_a_field_the_metadata_does_not_carry_lists_nothing(
    window_class: type, parent: QWidget, metadata: pd.DataFrame
) -> None:
    form = window_class(parent, ['gone'], metadata)

    assert items(form._values_list) == []


# ----------------------------------------------------------------------------
# Moving values between the lists
# ----------------------------------------------------------------------------


def test_adding_a_value_picks_it(form) -> None:
    select(form._values_list, 'drugA')

    form._add_button.click()

    assert items(form._picked_list) == ['drugA']
    assert form.get_criteria() == {'drug_name': ['drugA']}


def test_a_picked_value_leaves_the_values_it_was_picked_from(form) -> None:
    """A value sits in one list or the other, so the left one reads as what is left to pick."""
    select(form._values_list, 'drugA')

    form._add_button.click()

    assert 'drugA' not in items(form._values_list)


def test_several_values_are_added_at_once(form) -> None:
    select(form._values_list, 'ctrl', 'drugB')

    form._add_button.click()

    assert items(form._picked_list) == ['ctrl', 'drugB']


def test_removing_a_value_puts_it_back(form) -> None:
    select(form._values_list, 'ctrl', 'drugA')
    form._add_button.click()

    select(form._picked_list, 'ctrl')
    form._remove_button.click()

    assert items(form._picked_list) == ['drugA']
    assert 'ctrl' in items(form._values_list)


def test_adding_nothing_changes_nothing(form) -> None:
    form._add_button.click()

    assert items(form._picked_list) == []
    assert form.get_criteria() == {}


def test_removing_nothing_changes_nothing(form) -> None:
    select(form._values_list, 'ctrl')
    form._add_button.click()

    form._remove_button.click()

    assert items(form._picked_list) == ['ctrl']


# ----------------------------------------------------------------------------
# Letting go of everything at once
# ----------------------------------------------------------------------------


def test_clearing_lets_go_of_every_field(form) -> None:
    """The fields not on screen are the ones there is otherwise no way to take back."""
    select(form._values_list, 'ctrl')
    form._add_button.click()
    form._field_combo.setCurrentText('cluster')
    select(form._values_list, '10')
    form._add_button.click()

    form._clear_button.click()

    assert form.get_criteria() == {}


def test_clearing_puts_the_values_back_in_the_list(form) -> None:
    select(form._values_list, 'ctrl', 'drugA')
    form._add_button.click()

    form._clear_button.click()

    assert items(form._picked_list) == []
    assert items(form._values_list) == ['ctrl', 'drugA', 'drugB']


def test_clearing_reaches_the_field_that_is_not_shown(form) -> None:
    select(form._values_list, 'ctrl')
    form._add_button.click()
    form._field_combo.setCurrentText('time')

    form._clear_button.click()
    form._field_combo.setCurrentText('drug_name')

    assert items(form._picked_list) == []
    assert 'ctrl' in items(form._values_list)


def test_the_clear_button_waits_until_there_is_something_to_clear(form) -> None:
    """It is also the only word the form gives on whether the fields it is not showing hold picks."""
    assert not form._clear_button.isEnabled()

    select(form._values_list, 'ctrl')
    form._add_button.click()

    assert form._clear_button.isEnabled()


def test_the_clear_button_closes_again_once_everything_is_let_go_of(form) -> None:
    select(form._values_list, 'ctrl')
    form._add_button.click()

    form._clear_button.click()

    assert not form._clear_button.isEnabled()


def test_the_clear_button_stays_open_while_another_field_holds_picks(form) -> None:
    """What is picked elsewhere is still there to be cleared, whatever the field on screen shows."""
    select(form._values_list, 'ctrl')
    form._add_button.click()

    form._field_combo.setCurrentText('time')

    assert items(form._picked_list) == []
    assert form._clear_button.isEnabled()


def test_a_form_opening_on_picks_can_clear_them(window_class: type, parent: QWidget, metadata) -> None:
    criteria = {'drug_name': ['drugB']}
    form = window_class(parent, COLUMNS, metadata, criteria)
    assert form._clear_button.isEnabled()

    form._clear_button.click()

    assert form.get_criteria() == {}
    assert criteria == {'drug_name': ['drugB']}


# ----------------------------------------------------------------------------
# One set of picks per field
# ----------------------------------------------------------------------------


def test_each_field_keeps_its_own_picks(form) -> None:
    select(form._values_list, 'ctrl')
    form._add_button.click()

    form._field_combo.setCurrentText('time')

    assert items(form._picked_list) == []
    assert items(form._values_list) == ['0', '24']


def test_coming_back_to_a_field_shows_what_was_picked_there(form) -> None:
    select(form._values_list, 'ctrl')
    form._add_button.click()

    form._field_combo.setCurrentText('time')
    form._field_combo.setCurrentText('drug_name')

    assert items(form._picked_list) == ['ctrl']
    assert 'ctrl' not in items(form._values_list)


def test_the_picks_of_every_field_are_handed_back_together(form) -> None:
    select(form._values_list, 'ctrl')
    form._add_button.click()
    form._field_combo.setCurrentText('cluster')
    select(form._values_list, '10')
    form._add_button.click()

    assert form.get_criteria() == {'drug_name': ['ctrl'], 'cluster': [10]}


def test_a_value_is_handed_back_as_the_metadata_holds_it(form) -> None:
    """PhenoMe matches a criterion by equality, and holds the cluster labels it writes as numbers: a
    cluster handed back as the text it is shown as would match no image at all."""
    form._field_combo.setCurrentText('cluster')
    select(form._values_list, '10')

    form._add_button.click()

    (picked,) = form.get_criteria()['cluster']
    assert picked == 10
    assert not isinstance(picked, str)


def test_a_field_left_alone_is_not_handed_back(form) -> None:
    """An empty list of values would read as a criterion no image can answer, rather than as none."""
    select(form._values_list, 'ctrl')
    form._add_button.click()
    select(form._picked_list, 'ctrl')
    form._remove_button.click()

    assert form.get_criteria() == {}


# ----------------------------------------------------------------------------
# Opening on picks made before
# ----------------------------------------------------------------------------


def test_the_form_opens_on_the_picks_it_was_given(
    window_class: type, parent: QWidget, metadata: pd.DataFrame
) -> None:
    form = window_class(parent, COLUMNS, metadata, {'drug_name': ['drugB']})

    assert items(form._picked_list) == ['drugB']
    assert items(form._values_list) == ['ctrl', 'drugA']


def test_editing_the_form_leaves_the_caller_untouched(
    window_class: type, parent: QWidget, metadata: pd.DataFrame
) -> None:
    criteria = {'drug_name': ['drugB']}
    form = window_class(parent, COLUMNS, metadata, criteria)

    select(form._picked_list, 'drugB')
    form._remove_button.click()

    assert criteria == {'drug_name': ['drugB']}
    assert form.get_criteria() == {}
