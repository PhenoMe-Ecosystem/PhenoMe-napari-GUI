"""Tests for the three parameter forms.

All three edit a copy of the parameters and only hand it back through get_params(), so the caller can
apply a confirmed edit and discard a cancelled one. The prototype and outlier forms also share a group-by
combo and a source combo, which are parametrised over both below.
"""

import pytest
from qtpy.QtWidgets import QWidget

from phenome_gui.ui.clustering_params import ClusterParamsWindow
from phenome_gui.ui.prototypes_outliers_params import (
    OutliersParamsWindow,
    PrototypesParamsWindow,
)
from phenome_gui.utils.params import (
    METHOD_CHOICES,
    SOURCE_CHOICES,
    ClusteringParams,
    OutliersParams,
    PrototypesParams,
)

COLUMNS = ['filename', 'drug_name', 'time']

# The two forms built around a group-by combo, with the parameter model each one edits
GROUPING_FORMS = [
    pytest.param(PrototypesParamsWindow, PrototypesParams, id='prototypes'),
    pytest.param(OutliersParamsWindow, OutliersParams, id='outliers'),
]


@pytest.fixture
def parent(qtbot) -> QWidget:
    """ A bare widget to parent the dialogs to, so they are destroyed with the test. """
    widget = QWidget()
    qtbot.addWidget(widget)
    return widget


def items(dropdown) -> list[str]:
    """ The choices currently offered by a magicgui combo box. """
    native = dropdown.native
    return [native.itemText(i) for i in range(native.count())]


# ----------------------------------------------------------------------------
# The group-by combo, shared by the prototype and outlier forms
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(('form', 'model'), GROUPING_FORMS)
def test_group_by_is_closed_before_any_data_is_loaded(parent: QWidget, form: type, model: type) -> None:
    dialog = form(parent, model(), [])

    assert items(dialog._group_by_dropdown) == []
    assert not dialog._group_by_dropdown.native.isEnabled()
    assert dialog._group_by_dropdown.native.toolTip() == 'Please load data first.'


@pytest.mark.parametrize(('form', 'model'), GROUPING_FORMS)
def test_group_by_survives_the_unset_default(parent: QWidget, form: type, model: type) -> None:
    """group_by starts as None, which is not one of the columns; passing it as the value would raise."""
    dialog = form(parent, model(), COLUMNS)

    assert items(dialog._group_by_dropdown) == COLUMNS
    # Having landed on the first column, the copy has to say so rather than still claim None
    assert dialog.get_params().group_by == COLUMNS[0]


@pytest.mark.parametrize(('form', 'model'), GROUPING_FORMS)
def test_group_by_keeps_a_column_that_still_exists(parent: QWidget, form: type, model: type) -> None:
    dialog = form(parent, model(group_by='time'), COLUMNS)

    assert dialog.get_params().group_by == 'time'


# ----------------------------------------------------------------------------
# The source combo, shared by all three forms
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('form', 'model', 'positional'),
    [
        pytest.param(ClusterParamsWindow, ClusteringParams, (), id='clustering'),
        pytest.param(PrototypesParamsWindow, PrototypesParams, (COLUMNS,), id='prototypes'),
        pytest.param(OutliersParamsWindow, OutliersParams, (COLUMNS,), id='outliers'),
    ],
)
@pytest.mark.parametrize('properties_available', [False, True])
def test_source_is_gated_on_the_properties(
    parent: QWidget, form: type, model: type, positional: tuple, properties_available: bool
) -> None:
    dialog = form(parent, model(), *positional, properties_available)

    assert items(dialog._source_dropdown) == SOURCE_CHOICES
    assert dialog._source_dropdown.native.isEnabled() is properties_available


# ----------------------------------------------------------------------------
# Editing the copy, never the caller's parameters
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('form', 'model', 'positional', 'field', 'value'),
    [
        pytest.param(ClusterParamsWindow, ClusteringParams, (), '_n_clusters_spin', 9, id='clustering'),
        pytest.param(PrototypesParamsWindow, PrototypesParams, (COLUMNS,), '_n_prototypes_spin', 9, id='prototypes'),
        pytest.param(OutliersParamsWindow, OutliersParams, (COLUMNS,), '_threshold_spin', 9.0, id='outliers'),
    ],
)
def test_editing_the_form_leaves_the_caller_untouched(
    parent: QWidget, form: type, model: type, positional: tuple, field: str, value
) -> None:
    params = model()
    before = params.model_copy(deep=True)
    dialog = form(parent, params, *positional)

    getattr(dialog, field).value = value

    assert params == before
    assert dialog.get_params() != before


# ----------------------------------------------------------------------------
# Form-specific behaviour
# ----------------------------------------------------------------------------


def test_outlier_methods_are_the_two_phenome_knows(parent: QWidget) -> None:
    dialog = OutliersParamsWindow(parent, OutliersParams(), COLUMNS)

    assert items(dialog._method_dropdown) == METHOD_CHOICES == ['z-score', 'iqr']


def test_outlier_form_tracks_all_four_values(parent: QWidget) -> None:
    dialog = OutliersParamsWindow(parent, OutliersParams(), COLUMNS, True)

    dialog._group_by_dropdown.value = 'time'
    dialog._method_dropdown.value = 'iqr'
    dialog._threshold_spin.value = 1.5
    dialog._source_dropdown.value = 'combined'

    assert dialog.get_params() == OutliersParams(
        group_by='time', method='iqr', threshold=1.5, source='combined'
    )


def test_prototype_form_tracks_all_three_values(parent: QWidget) -> None:
    dialog = PrototypesParamsWindow(parent, PrototypesParams(), COLUMNS, True)

    dialog._group_by_dropdown.value = 'drug_name'
    dialog._n_prototypes_spin.value = 8
    dialog._source_dropdown.value = 'properties'

    assert dialog.get_params() == PrototypesParams(
        group_by='drug_name', n_prototypes=8, source='properties'
    )


@pytest.mark.parametrize(
    ('method', 'n_clusters_on', 'dbscan_on'),
    [('kmeans', True, False), ('gmm', True, False), ('dbscan', False, True)],
)
def test_clustering_form_shows_the_fields_of_the_chosen_method(
    parent: QWidget, method: str, n_clusters_on: bool, dbscan_on: bool
) -> None:
    dialog = ClusterParamsWindow(parent, ClusteringParams())

    dialog._method_dropdown.value = method

    assert dialog._n_clusters_spin.native.isEnabled() is n_clusters_on
    assert dialog._dbscan_eps_spin.native.isEnabled() is dbscan_on
    assert dialog._min_samples_spin.native.isEnabled() is dbscan_on
    # The enabling has to follow the change that caused it, not the one after
    assert dialog.get_params().clustering_method == method


def test_clustering_form_seeds_min_samples_from_its_own_parameter(parent: QWidget) -> None:
    params = ClusteringParams(dbscan_eps=0.5, dbscan_min_samples=42)

    dialog = ClusterParamsWindow(parent, params)

    assert dialog._min_samples_spin.value == 42


@pytest.mark.parametrize(
    ('form', 'model', 'positional'),
    [
        pytest.param(ClusterParamsWindow, ClusteringParams, (), id='clustering'),
        pytest.param(PrototypesParamsWindow, PrototypesParams, (COLUMNS,), id='prototypes'),
        pytest.param(OutliersParamsWindow, OutliersParams, (COLUMNS,), id='outliers'),
    ],
)
def test_the_buttons_settle_the_dialog(parent: QWidget, form: type, model: type, positional: tuple) -> None:
    accepted = form(parent, model(), *positional)
    accepted._accept()
    assert accepted.result() == accepted.Accepted

    rejected = form(parent, model(), *positional)
    rejected._cancel()
    assert rejected.result() == rejected.Rejected
