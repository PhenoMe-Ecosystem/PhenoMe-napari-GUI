"""Tests for PhenoMeWidget, which owns the shared objects and wires the panels to each other."""

import os

import pandas as pd
import pytest

from phenome_gui._widget import PhenoMeWidget
from phenome_gui.ui.analysis_panel import AnalysisPanel
from phenome_gui.ui.explorer_window import InteractiveExplorerWindow
from phenome_gui.ui.loading_panel import LoadingPanel
from phenome_gui.ui.processing_panel import ProcessingPanel

PANEL_ATTRIBUTES = ['loading_panel', 'processing_panel', 'analysis_panel']


# ----------------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------------


def test_the_three_panels_are_the_three_tabs(widget: PhenoMeWidget) -> None:
    assert widget._tabs.count() == 3
    assert [widget._tabs.tabText(i) for i in range(3)] == ['Data loading', 'Processing', 'Analysis']
    assert widget._tabs.widget(0) is widget.loading_panel
    assert widget._tabs.widget(1) is widget.processing_panel
    assert widget._tabs.widget(2) is widget.analysis_panel


def test_the_panels_are_of_their_own_types(widget: PhenoMeWidget) -> None:
    assert isinstance(widget.loading_panel, LoadingPanel)
    assert isinstance(widget.processing_panel, ProcessingPanel)
    assert isinstance(widget.analysis_panel, AnalysisPanel)


@pytest.mark.parametrize('attribute', PANEL_ATTRIBUTES)
def test_every_panel_shares_the_one_interface(widget: PhenoMeWidget, attribute: str) -> None:
    """A second interface would give a panel its own parameters and its own view of the pipeline."""
    assert getattr(widget, attribute)._phenome is widget._phenome


def test_the_two_displaying_panels_share_the_one_viewer(widget: PhenoMeWidget) -> None:
    assert widget.loading_panel._image_viewer is widget.analysis_panel._image_viewer


@pytest.mark.parametrize('attribute', PANEL_ATTRIBUTES)
def test_no_panel_reaches_for_a_sibling(widget: PhenoMeWidget, attribute: str) -> None:
    """The panels talk through signals; holding a sibling would put the wiring back inside them."""
    panel = getattr(widget, attribute)

    for sibling in PANEL_ATTRIBUTES:
        assert not hasattr(panel, sibling)


def panel_holding(native, widget: PhenoMeWidget):
    """ Walk up the parent chain until one of the panels is reached. """
    panels = {getattr(widget, attribute) for attribute in PANEL_ATTRIBUTES}
    parent = native.parentWidget()
    while parent is not None:
        if parent in panels:
            return parent
        parent = parent.parentWidget()
    return None


def test_each_table_sits_in_its_own_tab(widget: PhenoMeWidget) -> None:
    """A Qt widget has one parent, so adding it to another layout silently moves it out of its tab."""
    assert panel_holding(widget.loading_panel._metadata_table.native, widget) is widget.loading_panel
    assert panel_holding(widget.analysis_panel._images_table.native, widget) is widget.analysis_panel


# ----------------------------------------------------------------------------
# The three signals, each proven by acting on one panel and asserting on another
# ----------------------------------------------------------------------------


def test_loading_data_opens_up_the_processing_panel(widget: PhenoMeWidget) -> None:
    extract = widget.processing_panel._process_button
    assert extract.toolTip() == 'Please load data and load a model first.'

    widget.loading_panel._on_load_data_csv()

    assert extract.toolTip() == 'Please load a model first.'


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
