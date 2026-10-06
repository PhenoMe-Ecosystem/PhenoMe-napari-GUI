"""Tests for the parameter forms.

All of them edit a copy of the parameters and only hand it back through get_params(), so the caller can
apply a confirmed edit and discard a cancelled one. The prototype and outlier forms also share a group-by
combo and a source combo, which are parametrised over both below. The advanced processing form is built
from plain Qt widgets rather than magicgui ones, and is tested in a section of its own.
"""

import pytest
from qtpy.QtWidgets import QDialogButtonBox, QWidget

from phenome_gui.ui.advanced_processing_params import AdvancedProcessingParams
from phenome_gui.ui.clustering_params import ClusterParamsWindow
from phenome_gui.ui.prototypes_outliers_params import (
    OutliersParamsWindow,
    PrototypesParamsWindow,
)
from phenome_gui.utils.params import (
    CHANNEL_MODE_CHOICES,
    MAX_WORKERS,
    SOURCE_CHOICES,
    ClusteringParams,
    OutliersParams,
    ProcessParams,
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


# ----------------------------------------------------------------------------
# The advanced processing form
# ----------------------------------------------------------------------------


# Processing parameters other than the defaults in every field the form shows, and in some it does not
EDITED = ProcessParams(
    channel_mode='combined', batch_size=64, num_workers=2, device='cuda', lazy_checkpoint=True
)


def shown(dialog: AdvancedProcessingParams) -> tuple[str, int, int]:
    """ The channel mode, batch size and worker count the form is currently showing. """
    return (
        dialog._channel_mode_combo.currentText(),
        dialog._batch_size_spinbox.value(),
        dialog._num_workers_spinbox.value(),
    )


def test_the_advanced_form_opens_on_the_values_it_was_given(parent: QWidget) -> None:
    dialog = AdvancedProcessingParams(parent, EDITED)

    assert shown(dialog) == ('combined', 64, 2)


def test_the_advanced_form_hands_back_what_was_edited(parent: QWidget) -> None:
    dialog = AdvancedProcessingParams(parent, ProcessParams())

    dialog._channel_mode_combo.setCurrentText('combined')
    dialog._batch_size_spinbox.setValue(128)
    dialog._num_workers_spinbox.setValue(4)

    params = dialog.get_params()
    assert (params.channel_mode, params.batch_size, params.num_workers) == ('combined', 128, 4)


def test_the_advanced_form_keeps_the_fields_it_does_not_show(parent: QWidget) -> None:
    """The model, the device and the checkpointing are picked elsewhere, and handed back as they came."""
    dialog = AdvancedProcessingParams(parent, EDITED)

    dialog._batch_size_spinbox.setValue(512)

    assert dialog.get_params() == EDITED.model_copy(update={'batch_size': 512})


def test_editing_the_advanced_form_leaves_the_caller_untouched(parent: QWidget) -> None:
    params = ProcessParams()
    before = params.model_copy(deep=True)
    dialog = AdvancedProcessingParams(parent, params)

    dialog._batch_size_spinbox.setValue(512)
    edited = dialog.get_params()

    assert params == before
    assert edited != before


def test_the_worker_field_offers_no_more_than_the_model_takes(parent: QWidget) -> None:
    """A count the form can offer and the model rejects would raise on a confirmed edit."""
    dialog = AdvancedProcessingParams(parent, ProcessParams())

    assert dialog._num_workers_spinbox.minimum() == 0
    assert dialog._num_workers_spinbox.maximum() == MAX_WORKERS


def test_the_channel_mode_offers_the_values_the_model_accepts(parent: QWidget) -> None:
    """Offering anything else would raise on assignment, the field being typed as a literal."""
    combo = AdvancedProcessingParams(parent, ProcessParams())._channel_mode_combo

    assert [combo.itemText(row) for row in range(combo.count())] == CHANNEL_MODE_CHOICES


@pytest.mark.parametrize(
    ('button', 'result'),
    [
        pytest.param(QDialogButtonBox.StandardButton.Ok, 'Accepted', id='ok'),
        pytest.param(QDialogButtonBox.StandardButton.Cancel, 'Rejected', id='cancel'),
    ],
)
def test_the_advanced_form_buttons_settle_it(parent: QWidget, button, result: str) -> None:
    dialog = AdvancedProcessingParams(parent, ProcessParams())

    dialog.findChild(QDialogButtonBox).button(button).click()

    assert dialog.result() == getattr(dialog, result)


def sizes(dialog: AdvancedProcessingParams) -> dict[str, tuple[bool, int, bool]]:
    """ Each optional size as shown: whether its box is ticked, its value, and whether it can be edited. """
    return {
        'resize': (
            dialog._resize_checkbox.isChecked(),
            dialog._resize_spinbox.value(),
            dialog._resize_spinbox.isEnabled(),
        ),
        'pad': (
            dialog._pad_checkbox.isChecked(),
            dialog._pad_spinbox.value(),
            dialog._pad_spinbox.isEnabled(),
        ),
    }


def test_the_form_opens_resizing_but_not_padding(parent: QWidget) -> None:
    """Off, a size is still shown, and greyed out, so ticking its box offers one worth running with."""
    dialog = AdvancedProcessingParams(parent, ProcessParams())

    assert sizes(dialog) == {'resize': (True, 224, True), 'pad': (False, 224, False)}
    assert not dialog._l2_checkbox.isChecked()


def test_the_form_opens_on_the_sizes_it_was_given(parent: QWidget) -> None:
    dialog = AdvancedProcessingParams(
        parent, ProcessParams(resize_size=None, pad_size=300, l2_normalize_channels=True)
    )

    assert sizes(dialog) == {'resize': (False, 224, False), 'pad': (True, 300, True)}
    assert dialog._l2_checkbox.isChecked()


def test_unticking_a_size_hands_back_none(parent: QWidget) -> None:
    """None is what PhenoMe reads as skipping the step, where False would be taken for a size."""
    dialog = AdvancedProcessingParams(parent, ProcessParams())

    dialog._resize_checkbox.setChecked(False)

    assert dialog.get_params().resize_size is None


def test_ticking_a_size_hands_back_its_value(parent: QWidget) -> None:
    dialog = AdvancedProcessingParams(parent, ProcessParams())

    dialog._pad_checkbox.setChecked(True)
    dialog._pad_spinbox.setValue(300)

    assert dialog.get_params().pad_size == 300


def test_a_size_can_only_be_edited_while_its_box_is_ticked(parent: QWidget) -> None:
    dialog = AdvancedProcessingParams(parent, ProcessParams())

    dialog._pad_checkbox.setChecked(True)
    dialog._resize_checkbox.setChecked(False)

    assert dialog._pad_spinbox.isEnabled()
    assert not dialog._resize_spinbox.isEnabled()


def test_a_size_ticked_off_and_on_again_keeps_its_value(parent: QWidget) -> None:
    dialog = AdvancedProcessingParams(parent, ProcessParams(resize_size=512))

    dialog._resize_checkbox.setChecked(False)
    dialog._resize_checkbox.setChecked(True)

    assert dialog.get_params().resize_size == 512


def test_ticking_the_channel_normalisation_hands_it_back(parent: QWidget) -> None:
    dialog = AdvancedProcessingParams(parent, ProcessParams())

    dialog._l2_checkbox.setChecked(True)

    assert dialog.get_params().l2_normalize_channels is True
