"""Tests for PhenoMeWidget, which owns the shared objects and wires the panels to each other."""

import os
import threading

import pandas as pd
import pytest

from phenome_gui._widget import PhenoMeWidget
from phenome_gui.ui.clustering_params import ClusterParamsWindow
from phenome_gui.ui.explorer_window import InteractiveExplorerWindow

from .conftest import CHECKPOINT_PROPERTIES

PANEL_ATTRIBUTES = ['loading_panel', 'processing_panel', 'analysis_panel']


# ----------------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('attribute', PANEL_ATTRIBUTES)
def test_every_panel_shares_the_one_interface(widget: PhenoMeWidget, attribute: str) -> None:
    """A second interface would give a panel its own parameters and its own view of the pipeline."""
    assert getattr(widget, attribute)._phenome is widget._phenome


def test_the_two_displaying_panels_share_the_one_viewer(widget: PhenoMeWidget) -> None:
    assert widget.loading_panel._image_viewer is widget.analysis_panel._image_viewer


# ----------------------------------------------------------------------------
# The three signals, each proven by acting on one panel and asserting on another
# ----------------------------------------------------------------------------


def test_loading_data_opens_up_the_processing_panel(widget: PhenoMeWidget) -> None:
    extract = widget.processing_panel._process_button
    assert not extract.isEnabled()

    widget.loading_panel._on_load_data_csv()

    assert extract.isEnabled()


def test_a_running_task_locks_the_analysis_panel(widget: PhenoMeWidget, processed) -> None:
    processed(widget._phenome, 6)
    widget.processing_panel.refresh_buttons()

    widget.processing_panel._set_processing_task_running(True)

    assert widget.analysis_panel._task_running is True
    for button in (
        widget.analysis_panel._run_clustering_button,
        widget.analysis_panel._prototype_button,
        widget.analysis_panel._outlier_button,
        widget.analysis_panel._explorer_button,
    ):
        assert not button.isEnabled(), button.text()


def test_finishing_a_task_releases_the_analysis_panel(widget: PhenoMeWidget, processed) -> None:
    processed(widget._phenome, 6)
    widget.processing_panel._set_processing_task_running(True)

    widget.processing_panel._set_processing_task_running(False)

    assert widget.analysis_panel._task_running is False
    assert widget.analysis_panel._run_clustering_button.isEnabled()


def test_finishing_the_clustering_refreshes_the_metadata_table(widget: PhenoMeWidget, spy) -> None:
    widget.loading_panel._on_load_data_csv()
    paths = list(widget._phenome.metadata_df['file_path'])
    widget._phenome.pheno.compute_clustering = spy(
        pd.DataFrame({'image_path': paths, 'cluster': [0, 1, 0, 1, 0, 1]})
    )
    widget._phenome.compute_clustering()

    # The labels are in the metadata but the table has not been told yet
    assert 'cluster' not in list(widget.loading_panel._metadata_table.value['columns'])

    widget.analysis_panel._on_finish_clustering()

    assert 'cluster' in list(widget.loading_panel._metadata_table.value['columns'])


def test_a_dataset_of_images_alone_reaches_the_processing_panel(widget: PhenoMeWidget) -> None:
    """End to end: the masks and the metadata are PhenoMe's to do without, and the steps after do too."""
    widget._phenome.params.paths.masks = None
    widget._phenome.params.paths.metadata = None

    widget.loading_panel._csv_load_data_btn.click()

    assert len(widget._phenome.metadata_df) == 6
    assert widget.processing_panel._process_button.isEnabled()


@pytest.mark.real_load_worker
def test_a_load_holds_off_the_other_steps(widget: PhenoMeWidget, qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the real worker thread: the pipeline is emptied there, under steps that would run against it."""
    release = threading.Event()
    real_load = widget._phenome.load_data_csv

    def _held():
        release.wait(timeout=5)
        return real_load()

    monkeypatch.setattr(widget._phenome, 'load_data_csv', _held)
    widget._on_explorer_clicked()

    widget.loading_panel._csv_load_data_btn.click()

    try:
        assert not widget.processing_panel.isEnabled()
        assert not widget.analysis_panel.isEnabled()
        assert not widget._explorer_window.isEnabled()
    finally:
        release.set()

    qtbot.waitUntil(lambda: widget.processing_panel.isEnabled(), timeout=5000)
    assert widget.analysis_panel.isEnabled()
    assert widget._explorer_window.isEnabled()


def test_a_load_gives_the_steps_back_as_they_were(widget: PhenoMeWidget) -> None:
    """Locked as panels, so each button keeps its own state underneath: Save still waits on embeddings."""
    widget.loading_panel._csv_load_data_btn.click()

    assert widget.processing_panel.isEnabled()
    assert widget.processing_panel._process_button.isEnabled()
    assert not widget.processing_panel._save_checkpoint_button.isEnabled()


# ----------------------------------------------------------------------------
# A loaded checkpoint, which opens the analysis steps without a run being made here
# ----------------------------------------------------------------------------

ANALYSIS_BUTTONS = ('_run_clustering_button', '_prototype_button', '_outlier_button', '_explorer_button')


@pytest.fixture
def with_checkpoint(widget: PhenoMeWidget, written_checkpoint):
    """ Return a hook loading a real checkpoint through the processing panel.

    Written and read rather than stubbed: what is being checked is a whole chain of panels answering
    to one another off what PhenoMe actually holds after a load, which a stand-in would decide.

    Returns:
        Called as with_checkpoint(properties=False), returning its path.
    """
    def _load(properties: bool = True):
        widget.loading_panel._on_load_data_csv()
        path = written_checkpoint(widget._phenome, properties=properties)
        widget.processing_panel._load_checkpoint(path)
        return path

    return _load


def test_a_loaded_checkpoint_opens_every_analysis_step(widget: PhenoMeWidget, with_checkpoint) -> None:
    """The embeddings it holds are the ones those steps run on, however they came to be there."""
    for name in ANALYSIS_BUTTONS:
        assert not getattr(widget.analysis_panel, name).isEnabled()

    with_checkpoint()

    for name in ANALYSIS_BUTTONS:
        button = getattr(widget.analysis_panel, name)
        assert button.isEnabled(), name
        assert button.toolTip() == ''


@pytest.fixture
def cluster_form(monkeypatch: pytest.MonkeyPatch):
    """ Open the clustering form without showing it, and hand back what it offers.

    Its exec_ is modal and would hang the test, so it is answered as a cancel: the form is built
    either way, and what it was built offering is the whole of what these tests read.

    Returns:
        Called as cluster_form(widget).
    """
    monkeypatch.setattr(ClusterParamsWindow, 'exec_', lambda self: 0)

    def _open(widget: PhenoMeWidget) -> ClusterParamsWindow:
        widget.analysis_panel._on_cluster_options_clicked()
        return widget.analysis_panel._clustering_params_dialog

    return _open


def test_a_checkpoint_s_properties_open_the_feature_spaces(
    widget: PhenoMeWidget, with_checkpoint, cluster_form
) -> None:
    """Read from a file rather than just computed, which the forms must not be able to tell apart."""
    with_checkpoint(properties=True)

    form = cluster_form(widget)

    assert form._source_dropdown.native.isEnabled()


def test_a_checkpoint_without_properties_leaves_them_closed(
    widget: PhenoMeWidget, with_checkpoint, cluster_form
) -> None:
    """The half-done run has to read as half done: there is nothing yet to cluster on."""
    with_checkpoint(properties=False)

    form = cluster_form(widget)

    assert not form._source_dropdown.native.isEnabled()


def test_a_checkpoint_s_properties_join_the_colouring(widget: PhenoMeWidget, with_checkpoint) -> None:
    """What the explorer offers to colour by is the metadata and the properties together."""
    with_checkpoint(properties=True)

    assert set(CHECKPOINT_PROPERTIES) <= set(widget._phenome.get_color_by_columns())


def test_an_open_explorer_hears_about_a_loaded_checkpoint(
    widget: PhenoMeWidget, with_checkpoint
) -> None:
    """It is shown alongside the widget rather than reopened, so nothing else would bring it up to date."""
    widget.loading_panel._on_load_data_csv()
    widget._on_explorer_clicked()
    assert not widget._explorer_window._source_combo.isEnabled()

    with_checkpoint(properties=True)

    explorer = widget._explorer_window
    assert explorer._source_combo.isEnabled()

    offered = [explorer._color_by_combo.itemText(i) for i in range(explorer._color_by_combo.count())]
    assert set(CHECKPOINT_PROPERTIES) <= set(offered)


# ----------------------------------------------------------------------------
# The explorer window, owned here rather than by the panel that asks for it
# ----------------------------------------------------------------------------


def test_the_explorer_button_opens_the_window(widget: PhenoMeWidget, processed) -> None:
    processed(widget._phenome, 6)
    widget.analysis_panel.set_task_running(False)  # re-runs the refresh now the pipeline has embeddings
    assert widget._explorer_window is None

    widget.analysis_panel._explorer_button.click()

    assert isinstance(widget._explorer_window, InteractiveExplorerWindow)
    assert widget._explorer_window.isVisible()


def test_the_explorer_is_built_once_and_kept(widget: PhenoMeWidget) -> None:
    """Rebuilding it on every click would pile up windows, and dropping it would have it collected."""
    widget._on_explorer_clicked()
    first = widget._explorer_window

    widget._on_explorer_clicked()

    assert widget._explorer_window is first


def test_reopening_the_explorer_refreshes_its_choices(widget: PhenoMeWidget) -> None:
    """The kept window would otherwise still offer what the pipeline could answer for when it was built."""
    widget._on_explorer_clicked()
    assert widget._explorer_window._color_by_combo.count() == 0

    widget.loading_panel._on_load_data_csv()
    widget._on_explorer_clicked()

    combo = widget._explorer_window._color_by_combo
    assert [combo.itemText(i) for i in range(combo.count())] == widget._phenome.get_metadata_columns()


def test_the_explorer_shares_the_one_interface_and_viewer(widget: PhenoMeWidget) -> None:
    widget._on_explorer_clicked()

    assert widget._explorer_window._phenome is widget._phenome
    assert widget._explorer_window._image_viewer is widget._image_viewer


# ----------------------------------------------------------------------------
# Loading a second dataset, which nothing computed from the first may outlive
# ----------------------------------------------------------------------------


@pytest.fixture
def analysed(widget: PhenoMeWidget, processed, reduction, sync_worker) -> PhenoMeWidget:
    """ A widget holding a full session: data loaded, images processed, images retrieved and projected.

    The embeddings and the reduction are stubbed, neither being something a test may run.
    """
    widget.loading_panel._on_load_data_csv()
    processed(widget._phenome, 6)
    widget.processing_panel.refresh_buttons()
    widget.analysis_panel.set_task_running(False)

    widget._phenome.properties_df = pd.DataFrame({'area': [1.0] * 6})
    widget.analysis_panel._update_images_table(
        pd.DataFrame({
            'drug_name': ['ctrl', 'drugA'],
            'image_name': ['img_000.tif', 'img_003.tif'],
            'image_path': ['img_000.tif', 'img_003.tif'],
        }),
        'drug_name',
    )

    # A drawn projection with a selection on it, in the window the explorer button opens
    reduction(pd.DataFrame({
        'Component 1': [0.0, 1.0, 2.0],
        'Component 2': [0.0, 1.0, 2.0],
        'Index': [0, 1, 2],
        'drug_name': ['ctrl', 'ctrl', 'drugA'],
    }))
    widget.analysis_panel._explorer_button.click()
    sync_worker(widget._explorer_window, '_projection_worker', widget._phenome.compute_projection)
    widget._explorer_window._compute_button.click()
    widget._explorer_window._canvas.points_selected.emit([1, 2])

    # An image on the napari canvas, as browsing the metadata table leaves behind
    widget.loading_panel._metadata_table.native.setCurrentCell(2, 0)

    return widget


def test_a_second_load_empties_the_images_table(analysed: PhenoMeWidget, path_dataset) -> None:
    load_paths(analysed, path_dataset)

    assert analysed.analysis_panel._images_table.native.rowCount() == 0
    assert analysed.analysis_panel._images_df is None


def test_a_second_load_wipes_the_explorer(analysed: PhenoMeWidget, path_dataset) -> None:
    load_paths(analysed, path_dataset)

    explorer = analysed._explorer_window
    assert explorer._canvas._points is None
    assert explorer._canvas._plot.listDataItems() == []
    assert explorer._selected_indices == []
    assert explorer._num_select_label.text() == '0 points selected.'
    assert not explorer._highlight_group.isEnabled()


def test_a_second_load_takes_the_image_off_the_viewer(analysed: PhenoMeWidget, path_dataset, viewer) -> None:
    """What is on the canvas is an image of the previous dataset, which no longer stands for anything."""
    cleared = viewer.layers.cleared

    load_paths(analysed, path_dataset)

    assert viewer.layers.cleared == cleared + 1
    assert viewer.layers == []


def test_a_second_load_shows_the_new_dataset_alone(analysed: PhenoMeWidget, path_dataset) -> None:
    load_paths(analysed, path_dataset)

    assert analysed.loading_panel._metadata_table.native.rowCount() == 6
    assert analysed._phenome.has_results is False
    assert analysed._phenome.properties_df is None


def test_answering_no_to_the_second_load_keeps_everything(
    analysed: PhenoMeWidget, path_dataset, confirmation
) -> None:
    confirmation.accepted = False
    drawn = list(analysed._explorer_window._canvas._points['Index'])

    load_paths(analysed, path_dataset)

    assert analysed.analysis_panel._images_table.native.rowCount() == 2
    assert analysed._explorer_window._selected_indices == [1, 2]
    assert list(analysed._explorer_window._canvas._points['Index']) == drawn


def load_paths(widget: PhenoMeWidget, path_dataset) -> None:
    """ Load the second dataset through the loading panel, the way a user would. """
    widget.loading_panel._path_radio.setChecked(True)
    widget.loading_panel._path_image_dir_dialog.path = path_dataset['images']
    widget.loading_panel._path_mask_dir_dialog.path = path_dataset['masks']
    widget.loading_panel._path_template.value = path_dataset['template']
    widget.loading_panel._path_load_data_btn.click()


# ----------------------------------------------------------------------------
# Processing the images again, which nothing computed from the old embeddings may outlive
# ----------------------------------------------------------------------------


@pytest.fixture
def clustered(analysed: PhenoMeWidget) -> PhenoMeWidget:
    """ The full session, clustered on top, its labels showing in the metadata table. """
    analysed._phenome.metadata_df['cluster'] = pd.array([0, 1] * 3, dtype='Int64')
    analysed.loading_panel.refresh_table()
    analysed._explorer_window.refresh_choices()
    return analysed


def process_again(widget: PhenoMeWidget, sync_worker) -> None:
    """ Press Process Images over the existing results, the run itself doing nothing. """
    sync_worker(widget.processing_panel, '_process_images_worker', lambda *args: None)
    widget.processing_panel._process_button.click()


def metadata_columns(widget: PhenoMeWidget) -> list[str]:
    """ The columns the metadata table is currently showing. """
    return list(widget.loading_panel._metadata_table.value['columns'])


def test_processing_again_empties_the_images_table(clustered: PhenoMeWidget, sync_worker) -> None:
    process_again(clustered, sync_worker)

    assert clustered.analysis_panel._images_table.native.rowCount() == 0


def test_processing_again_takes_the_clusters_off_the_metadata_table(
    clustered: PhenoMeWidget, sync_worker
) -> None:
    """The labels go, the dataset they were put on stays."""
    assert 'cluster' in metadata_columns(clustered)

    process_again(clustered, sync_worker)

    assert 'cluster' not in metadata_columns(clustered)
    assert clustered.loading_panel._metadata_table.native.rowCount() == 6


def test_processing_again_wipes_the_explorer(clustered: PhenoMeWidget, sync_worker) -> None:
    process_again(clustered, sync_worker)

    explorer = clustered._explorer_window
    assert explorer._canvas._points is None
    assert explorer._selected_indices == []
    assert 'cluster' not in [explorer._color_by_combo.itemText(i) for i in range(explorer._color_by_combo.count())]


def test_refusing_to_process_again_keeps_everything(
    clustered: PhenoMeWidget, sync_worker, confirmation
) -> None:
    confirmation.accepted = False
    drawn = list(clustered._explorer_window._canvas._points['Index'])

    process_again(clustered, sync_worker)

    assert clustered.analysis_panel._images_table.native.rowCount() == 2
    assert 'cluster' in metadata_columns(clustered)
    assert clustered._explorer_window._selected_indices == [1, 2]
    assert list(clustered._explorer_window._canvas._points['Index']) == drawn


@pytest.fixture
def checkpoint_loaded(clustered: PhenoMeWidget, written_checkpoint) -> PhenoMeWidget:
    """ The clustered session, a checkpoint of the same dataset then loaded over it.

    Written from the metadata without the cluster labels, as a checkpoint written before the clustering
    would be: PhenoMe matches a checkpoint to the files by their metadata, which the labels are not.
    """
    labelled = clustered._phenome.metadata_df
    clustered._phenome.metadata_df = labelled.drop(columns='cluster')
    path = written_checkpoint(clustered._phenome, properties=False)
    clustered._phenome.metadata_df = labelled

    clustered.processing_panel._load_checkpoint(path)
    return clustered


def test_loading_a_checkpoint_wipes_the_explorer(checkpoint_loaded: PhenoMeWidget) -> None:
    """Drawn from the embeddings the checkpoint replaced, so nothing on it answers for the ones now held."""
    explorer = checkpoint_loaded._explorer_window
    assert explorer._canvas._points is None
    assert explorer._canvas._plot.listDataItems() == []
    assert explorer._selected_indices == []
    assert explorer._num_select_label.text() == '0 points selected.'
    assert not explorer._highlight_group.isEnabled()


def test_loading_a_checkpoint_empties_the_images_table(checkpoint_loaded: PhenoMeWidget) -> None:
    assert checkpoint_loaded.analysis_panel._images_table.native.rowCount() == 0
    assert checkpoint_loaded.analysis_panel._images_df is None


def test_loading_a_checkpoint_takes_the_clusters_off(checkpoint_loaded: PhenoMeWidget) -> None:
    """Off the metadata table and out of the explorer's colouring, the dataset itself staying."""
    explorer = checkpoint_loaded._explorer_window

    assert 'cluster' not in metadata_columns(checkpoint_loaded)
    assert checkpoint_loaded.loading_panel._metadata_table.native.rowCount() == 6
    assert 'cluster' not in [explorer._color_by_combo.itemText(i) for i in range(explorer._color_by_combo.count())]


def test_loading_a_checkpoint_leaves_the_analysis_ready_to_run_again(checkpoint_loaded: PhenoMeWidget) -> None:
    for name in ANALYSIS_BUTTONS:
        assert getattr(checkpoint_loaded.analysis_panel, name).isEnabled(), name


# ----------------------------------------------------------------------------
# The real plugin, which nothing else exercises
# ----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get('QT_QPA_PLATFORM') == 'offscreen',
    reason='napari needs a real display: its OpenGL canvas crashes the interpreter under offscreen Qt',
)
def test_the_plugin_docks_into_napari(make_napari_viewer) -> None:
    """Guards the napari.yaml entry point, which the rest of the suite bypasses by importing directly."""
    viewer = make_napari_viewer()

    _, plugin_widget = viewer.window.add_plugin_dock_widget('phenome-gui', 'PhenoMe')

    assert isinstance(plugin_widget, PhenoMeWidget)
    assert plugin_widget._tabs.count() == 3
