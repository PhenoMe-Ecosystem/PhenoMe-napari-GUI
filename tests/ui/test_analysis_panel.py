"""Tests for the analysis panel: clustering, prototype and outlier retrieval, and the images table."""

import pandas as pd
import pytest

from phenome_gui.ui.analysis_panel import AnalysisPanel
from phenome_gui.ui.clustering_params import ClusterParamsWindow
from phenome_gui.ui.prototypes_outliers_params import (
    OutliersParamsWindow,
    PrototypesParamsWindow,
)


@pytest.fixture
def panel(widget, processed) -> AnalysisPanel:
    """ The analysis panel of a built widget, with data loaded and the images looking processed. """
    widget.loading_panel._on_load_data_csv()
    processed(widget._phenome, 6)
    widget.analysis_panel.set_task_running(False)  # re-runs the refresh now the pipeline has embeddings
    return widget.analysis_panel


@pytest.fixture
def fresh(widget) -> AnalysisPanel:
    """ The analysis panel of a widget with nothing loaded or processed. """
    return widget.analysis_panel


def buttons(panel: AnalysisPanel) -> tuple:
    """ The three buttons that all wait on the embeddings. """
    return (panel._run_clustering_button, panel._prototype_button, panel._outlier_button)


def shown_columns(table) -> list[str]:
    """ The column names currently displayed by a magicgui table. """
    return list(table.value['columns'])


def prototype_frame(paths: list[str]) -> pd.DataFrame:
    """ A frame shaped like what find_prototypes returns with a resolved grouping key. """
    return pd.DataFrame(
        {
            'image_index': range(len(paths)),
            'image_path': paths,
            'image_name': [f'name_{i}' for i in range(len(paths))],
            'rank_in_group': [1, 2, 1][: len(paths)],
            'drug_name': ['ctrl', 'ctrl', 'drugA'][: len(paths)],
        }
    )


# ----------------------------------------------------------------------------
# Gating
# ----------------------------------------------------------------------------


def test_everything_waits_on_the_embeddings(fresh: AnalysisPanel) -> None:
    for button in buttons(fresh):
        assert not button.isEnabled(), button.text()
        assert button.toolTip() == 'Please extract embeddings first.'


def test_everything_opens_once_the_images_are_processed(panel: AnalysisPanel) -> None:
    for button in buttons(panel):
        assert button.isEnabled(), button.text()
        assert button.toolTip() == ''


def test_the_options_button_is_never_gated(fresh: AnalysisPanel) -> None:
    """It only edits parameters, so there is nothing for it to wait on."""
    assert fresh._cluster_options_button.isEnabled()


@pytest.mark.parametrize('running', [True, False])
def test_a_running_task_locks_everything(panel: AnalysisPanel, running: bool) -> None:
    panel.set_task_running(running)

    for button in buttons(panel):
        assert button.isEnabled() is not running
    if running:
        assert panel._run_clustering_button.toolTip() == 'Please wait for the running task to finish.'


# ----------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------


def test_the_clustering_form_applies_on_accept(panel: AnalysisPanel, dialog_result) -> None:
    dialog_result(ClusterParamsWindow, accepted=True, _method_dropdown='gmm', _n_clusters_spin=8)

    panel._on_cluster_options_clicked()

    assert panel._phenome.get_param('clustering', 'clustering_method') == 'gmm'
    assert panel._phenome.get_param('clustering', 'n_clusters') == 8


def test_the_clustering_form_discards_on_cancel(panel: AnalysisPanel, dialog_result) -> None:
    before = dict(panel._phenome.params['clustering'])
    dialog_result(ClusterParamsWindow, accepted=False, _method_dropdown='gmm', _n_clusters_spin=8)

    panel._on_cluster_options_clicked()

    assert panel._phenome.params['clustering'] == before


def test_finishing_the_clustering_is_announced(panel: AnalysisPanel, qtbot) -> None:
    with qtbot.waitSignal(panel.clustering_finished, timeout=1000):
        panel._on_finish_clustering()


def test_the_clustering_button_runs_and_announces(panel: AnalysisPanel, sync_worker, spy, qtbot) -> None:
    paths = list(panel._phenome.metadata_df['file_path'])
    pheno_spy = spy(pd.DataFrame({'image_path': paths, 'cluster': [0, 1] * 3}))
    panel._phenome.pheno.compute_clustering = pheno_spy
    sync_worker(panel, '_compute_clustering_worker', panel._phenome.compute_clustering)

    with qtbot.waitSignal(panel.clustering_finished, timeout=1000):
        panel._on_run_clustering_clicked()

    assert pheno_spy.called
    assert 'cluster' in panel._phenome.metadata_df.columns


# ----------------------------------------------------------------------------
# Prototypes and outliers: the form gates the run
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('form', 'handler', 'worker', 'call', 'pheno_method'),
    [
        pytest.param(
            PrototypesParamsWindow,
            '_on_prototype_clicked',
            '_prototypes_worker',
            'get_prototype_images',
            'find_prototypes',
            id='prototypes',
        ),
        pytest.param(
            OutliersParamsWindow,
            '_on_outlier_clicked',
            '_outliers_worker',
            'get_outlier_images',
            'detect_outliers',
            id='outliers',
        ),
    ],
)
class TestRetrievalRuns:
    """The retrieval buttons open a form, and confirming it is what runs the step."""

    def test_cancelling_runs_nothing(
        self, panel, dialog_result, sync_worker, spy, form, handler, worker, call, pheno_method
    ) -> None:
        pheno_spy = spy(pd.DataFrame())
        setattr(panel._phenome.pheno, pheno_method, pheno_spy)
        sync_worker(panel, worker, getattr(panel._phenome, call))
        dialog_result(form, accepted=False, _group_by_dropdown='drug_name')

        getattr(panel, handler)()

        assert not pheno_spy.called

    def test_cancelling_changes_no_parameter(
        self, panel, dialog_result, sync_worker, spy, form, handler, worker, call, pheno_method
    ) -> None:
        section = 'prototypes' if pheno_method == 'find_prototypes' else 'outliers'
        before = dict(panel._phenome.params[section])
        setattr(panel._phenome.pheno, pheno_method, spy(pd.DataFrame()))
        sync_worker(panel, worker, getattr(panel._phenome, call))
        dialog_result(form, accepted=False, _group_by_dropdown='drug_name')

        getattr(panel, handler)()

        assert panel._phenome.params[section] == before

    def test_confirming_runs_the_step(
        self, panel, dialog_result, sync_worker, spy, form, handler, worker, call, pheno_method
    ) -> None:
        pheno_spy = spy(pd.DataFrame())
        setattr(panel._phenome.pheno, pheno_method, pheno_spy)
        sync_worker(panel, worker, getattr(panel._phenome, call))
        dialog_result(form, accepted=True, _group_by_dropdown='drug_name')

        getattr(panel, handler)()

        assert pheno_spy.called
        section = 'prototypes' if pheno_method == 'find_prototypes' else 'outliers'
        assert panel._phenome.get_param(section, 'group_by') == 'drug_name'


def test_a_confirmed_search_fills_the_table(panel: AnalysisPanel, dialog_result, sync_worker, spy) -> None:
    paths = list(panel._phenome.metadata_df['file_path'])[:3]
    panel._phenome.pheno.find_prototypes = spy(prototype_frame(paths))
    sync_worker(panel, '_prototypes_worker', panel._phenome.get_prototype_images)
    dialog_result(PrototypesParamsWindow, accepted=True, _group_by_dropdown='drug_name')

    panel._on_prototype_clicked()

    assert panel._images_table.native.rowCount() == 3
    assert panel._images_group_column == 'drug_name'


# ----------------------------------------------------------------------------
# The images table
# ----------------------------------------------------------------------------


def test_the_table_hides_the_columns_it_only_needs_behind_the_scenes(panel: AnalysisPanel) -> None:
    frame = prototype_frame(['a.tif', 'b.tif', 'c.tif'])

    panel._update_images_table(frame, 'drug_name')

    assert shown_columns(panel._images_table) == ['image_name', 'rank_in_group', 'drug_name']
    # The path is still needed to load the image, so the full frame is kept
    assert list(panel._images_df.columns) == list(frame.columns)


@pytest.mark.parametrize(
    ('columns', 'group_by', 'expected'),
    [
        pytest.param({'drug_name': ['ctrl']}, 'drug_name', 'drug_name', id='resolved-key'),
        pytest.param({'group': ['All']}, '', 'group', id='prototype-fallback'),
        pytest.param({'threshold': [3.0]}, '', None, id='outlier-omits-it'),
    ],
)
def test_the_group_column_is_resolved_three_ways(
    panel: AnalysisPanel, columns: dict, group_by: str, expected: str | None
) -> None:
    """find_prototypes falls back to a literal 'group' column; detect_outliers leaves it out entirely."""
    frame = pd.DataFrame({'image_path': ['a.tif'], 'image_name': ['a'], **columns})

    panel._update_images_table(frame, group_by)

    assert panel._images_group_column == expected


@pytest.mark.parametrize('frame', [None, pd.DataFrame()], ids=['none', 'empty'])
def test_an_empty_result_clears_the_table(panel: AnalysisPanel, frame) -> None:
    """Finding no outliers is a normal outcome, not an error."""
    panel._update_images_table(prototype_frame(['a.tif']), 'drug_name')

    panel._update_images_table(frame, 'drug_name')

    assert panel._images_table.native.rowCount() == 0
    assert panel._images_group_column is None


def test_selecting_a_row_on_a_cleared_table_does_nothing(panel: AnalysisPanel, viewer) -> None:
    panel._update_images_table(pd.DataFrame(), 'drug_name')

    panel._on_image_table_selection()

    assert viewer.added == []


# ----------------------------------------------------------------------------
# Selecting a retrieved image
# ----------------------------------------------------------------------------


def test_the_layer_is_named_after_the_group(panel: AnalysisPanel, viewer) -> None:
    paths = list(panel._phenome.metadata_df['file_path'])[:3]
    panel._update_images_table(prototype_frame(paths), 'drug_name')

    panel._images_table.native.setCurrentCell(2, 0)

    assert viewer.names[0] == 'drugA - name_2'


def test_the_mask_is_looked_up_from_the_metadata(panel: AnalysisPanel, viewer) -> None:
    """The retrieval frames carry the image path but no mask, so it has to come from the metadata."""
    paths = list(panel._phenome.metadata_df['file_path'])[:3]
    panel._update_images_table(prototype_frame(paths), 'drug_name')

    panel._images_table.native.setCurrentCell(1, 0)

    assert [layer['kind'] for layer in viewer.added] == ['image', 'labels']
    assert viewer.names == ['ctrl - name_1', 'ctrl - name_1 - Mask']


def test_an_ungrouped_run_names_the_layer_after_the_image_alone(panel: AnalysisPanel, viewer) -> None:
    path = panel._phenome.metadata_df.iloc[0]['file_path']
    frame = pd.DataFrame({'image_path': [path], 'image_name': ['solo'], 'threshold': [3.0]})
    panel._update_images_table(frame, '')

    panel._images_table.native.setCurrentCell(0, 0)

    assert viewer.names[0] == 'solo'
