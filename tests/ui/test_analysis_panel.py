"""Tests for the analysis panel: clustering, prototype and outlier retrieval, and the images table."""

import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMessageBox

from phenome_gui.ui.analysis_panel import AnalysisPanel
from phenome_gui.ui.clustering_params import ClusterParamsWindow
from phenome_gui.ui.loading_panel import LoadingPanel
from phenome_gui.ui.prototypes_outliers_params import (
    OutliersParamsWindow,
    PrototypesParamsWindow,
)
from phenome_gui.utils.image_viewer import ImageViewer


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
    """ The four buttons that all wait on the embeddings. """
    return (
        panel._run_clustering_button,
        panel._prototype_button,
        panel._outlier_button,
        panel._explorer_button,
    )


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

    assert panel._phenome.params.clustering.clustering_method == 'gmm'
    assert panel._phenome.params.clustering.n_clusters == 8


def test_the_clustering_form_discards_on_cancel(panel: AnalysisPanel, dialog_result) -> None:
    before = panel._phenome.params.clustering.model_copy(deep=True)
    dialog_result(ClusterParamsWindow, accepted=False, _method_dropdown='gmm', _n_clusters_spin=8)

    panel._on_cluster_options_clicked()

    assert panel._phenome.params.clustering == before


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
        before = getattr(panel._phenome.params, section).model_copy(deep=True)
        setattr(panel._phenome.pheno, pheno_method, spy(pd.DataFrame()))
        sync_worker(panel, worker, getattr(panel._phenome, call))
        dialog_result(form, accepted=False, _group_by_dropdown='drug_name')

        getattr(panel, handler)()

        assert getattr(panel._phenome.params, section) == before

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
        assert getattr(panel._phenome.params, section).group_by == 'drug_name'


def test_a_confirmed_search_fills_the_table(panel: AnalysisPanel, dialog_result, sync_worker, spy) -> None:
    paths = list(panel._phenome.metadata_df['file_path'])[:3]
    panel._phenome.pheno.find_prototypes = spy(prototype_frame(paths))
    sync_worker(panel, '_prototypes_worker', panel._phenome.get_prototype_images)
    dialog_result(PrototypesParamsWindow, accepted=True, _group_by_dropdown='drug_name')

    panel._on_prototype_clicked()

    assert panel._images_table.native.rowCount() == 3
    assert panel._images_group_column == 'drug_name'


# ----------------------------------------------------------------------------
# The explorer, which the panel only asks for
# ----------------------------------------------------------------------------


def test_the_explorer_button_asks_rather_than_opens(panel: AnalysisPanel, qtbot) -> None:
    """The window is shown alongside the widget, so the panel announces the request and nothing more."""
    with qtbot.waitSignal(panel.explorer_requested, timeout=1000):
        panel._explorer_button.click()


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


# ----------------------------------------------------------------------------
# Letting go of the retrieved images when another dataset is loaded
# ----------------------------------------------------------------------------


def test_clearing_empties_the_images_table(panel: AnalysisPanel) -> None:
    panel._update_images_table(prototype_frame(['a.tif', 'b.tif']), 'drug_name')
    assert panel._images_table.native.rowCount() == 2

    panel.clear_results()

    assert panel._images_table.native.rowCount() == 0
    assert panel._images_df is None
    assert panel._images_group_column is None


def test_clearing_closes_the_analysis_buttons_back_down(panel: AnalysisPanel, widget) -> None:
    """The embeddings the four steps run on went with the dataset they were extracted from."""
    assert all(button.isEnabled() for button in buttons(panel))

    widget._phenome.clear_results()
    panel.clear_results()

    assert not any(button.isEnabled() for button in buttons(panel))
    assert all(button.toolTip() == 'Please extract embeddings first.' for button in buttons(panel))


def test_clearing_an_empty_panel_raises_nothing(fresh: AnalysisPanel) -> None:
    """Every load clears the panel, including the ones made before any image was retrieved."""
    fresh.clear_results()

    assert fresh._images_table.native.rowCount() == 0


# ----------------------------------------------------------------------------
# A step that raises, which the worker swallows into a signal
# ----------------------------------------------------------------------------


class Reported:
    """ Stands in for the message shown when a step raised, which is modal and would hang a test. """

    def __init__(self) -> None:
        self.shown = 0
        self.last = ''

    def __call__(self, parent, title, text, *args, **kwargs) -> None:
        """ Take note of a message, and of what it said.

        Args:
            parent: The widget the message would be shown over.
            title: Its title bar.
            text: What it says, which is what the tests read.
            *args: Whatever buttons the caller passed, which are of no interest here.
            **kwargs: As above.
        """
        self.shown += 1
        self.last = text


@pytest.fixture
def reported(monkeypatch: pytest.MonkeyPatch) -> Reported:
    """ Catch the message a failed step shows, rather than opening a window nobody can close.

    Returns:
        The stand-in, holding what was said.
    """
    stand_in = Reported()
    monkeypatch.setattr(QMessageBox, 'critical', stand_in)
    return stand_in


FAILURES = [
    pytest.param('_on_clustering_failed', 'cluster the images', id='clustering'),
    pytest.param('_on_prototypes_failed', 'find the prototype images', id='prototypes'),
    pytest.param('_on_outliers_failed', 'detect the outlier images', id='outliers'),
]


@pytest.mark.parametrize(('handler', 'attempted'), FAILURES)
def test_a_failed_step_is_reported(panel: AnalysisPanel, reported, handler: str, attempted: str) -> None:
    """Nothing else marks the failure: the buttons stay open and the table keeps its last contents."""
    getattr(panel, handler)(ValueError('n_samples=3 while n_clusters=8'))

    assert reported.shown == 1
    assert attempted in reported.last


@pytest.mark.parametrize(('handler', 'attempted'), FAILURES)
def test_a_failed_step_says_what_went_wrong(panel: AnalysisPanel, reported, handler: str, attempted: str) -> None:
    getattr(panel, handler)(ValueError('n_samples=3 while n_clusters=8'))

    assert 'n_samples=3 while n_clusters=8' in reported.last


@pytest.mark.parametrize(('handler', 'attempted'), FAILURES)
def test_a_failed_step_leaves_the_buttons_open(panel: AnalysisPanel, reported, handler: str, attempted: str) -> None:
    """These steps never close their buttons, so a failure has nothing to reopen."""
    getattr(panel, handler)(ValueError('nope'))

    assert all(button.isEnabled() for button in buttons(panel))


# ----------------------------------------------------------------------------
# The dock area the images table is shown in
# ----------------------------------------------------------------------------


@pytest.fixture
def docked_panel(phenome, viewer, in_main_window) -> AnalysisPanel:
    """ A panel of its own, inside a main window, which is what the way back needs.

    Built here rather than taken off the widget, which the `panel` fixture hands over already
    owned by a widget of its own.
    """
    return in_main_window(lambda parent: AnalysisPanel(phenome, ImageViewer(viewer), parent))


def test_the_images_table_is_shown_in_a_dock_area(docked_panel: AnalysisPanel) -> None:
    area = docked_panel._images_area

    assert area.dock.widget() is docked_panel._images_table.native
    assert area.dockWidgetArea(area.dock) == Qt.DockWidgetArea.TopDockWidgetArea


def test_popping_the_images_table_out_and_closing_it_brings_it_back(
    docked_panel: AnalysisPanel, qtbot
) -> None:
    """The dock is what puts it back; this is the group box holding up its end, slot and table."""
    area = docked_panel._images_area
    group_layout = area.parentWidget().layout()
    slot = group_layout.indexOf(area)

    area.dock.setFloating(True)
    area.dock.close()

    qtbot.waitUntil(lambda: not area.dock.isFloating())
    assert area.dockWidgetArea(area.dock) == Qt.DockWidgetArea.TopDockWidgetArea
    assert group_layout.indexOf(area) == slot
    assert area.dock.widget() is docked_panel._images_table.native
    assert not area.dock.isHidden()


def test_the_two_tables_go_back_to_their_own_panels(docked_panel: AnalysisPanel, phenome, viewer,
                                                    in_main_window, qtbot) -> None:
    """One table popped out must not land in the other's slot, each area owning its own dock."""
    loading_panel = in_main_window(lambda parent: LoadingPanel(phenome, ImageViewer(viewer), parent))
    images, metadata = docked_panel._images_area, loading_panel._table_area

    images.dock.setFloating(True)
    metadata.dock.setFloating(True)
    images.dock.close()
    metadata.dock.close()

    qtbot.waitUntil(lambda: not images.dock.isFloating() and not metadata.dock.isFloating())
    assert images.dock.parentWidget() is images
    assert metadata.dock.parentWidget() is metadata
    assert images.dock.widget() is docked_panel._images_table.native
    assert metadata.dock.widget() is loading_panel._metadata_table.native
