"""Tests for the data loading panel."""

from pathlib import Path

import pytest
from qtpy.QtCore import Qt

from phenome_gui.ui.loading_panel import LoadingPanel
from phenome_gui.utils.image_viewer import ImageViewer


@pytest.fixture
def panel(widget) -> LoadingPanel:
    """ The loading panel of a built widget, with its paths pointed at the test dataset. """
    return widget.loading_panel


def shown_columns(table) -> list[str]:
    """ The column names currently displayed by a magicgui table. """
    return list(table.value['columns'])


# ----------------------------------------------------------------------------
# The two metadata structures
# ----------------------------------------------------------------------------


def test_the_csv_structure_is_the_default(panel: LoadingPanel) -> None:
    assert panel._csv_radio.isChecked()
    assert panel._loading_stack.currentIndex() == 0


def test_choosing_the_path_structure_swaps_the_page(panel: LoadingPanel) -> None:
    panel._path_radio.setChecked(True)

    assert panel._loading_stack.currentIndex() == 1


def test_the_csv_page_writes_its_three_paths(panel: LoadingPanel, tmp_path: Path) -> None:
    panel._csv_radio.setChecked(True)
    panel._image_dir_dialog.value = tmp_path / 'imgs'
    panel._mask_dir_dialog.value = tmp_path / 'msks'
    panel._metadata_dir_dialog.value = tmp_path / 'meta.csv'

    assert panel._phenome.params.paths.images == tmp_path / 'imgs'
    assert panel._phenome.params.paths.masks == tmp_path / 'msks'
    assert panel._phenome.params.paths.metadata == tmp_path / 'meta.csv'


def test_the_path_page_writes_the_template_instead(panel: LoadingPanel, tmp_path: Path) -> None:
    """The template stays a string: PhenoMe calls str.replace on it, which a Path does not have."""
    panel._path_radio.setChecked(True)
    panel._path_image_dir_dialog.value = tmp_path / 'imgs'
    panel._path_template.value = '.../(drug)/(time)/(filename).*'

    assert panel._phenome.params.paths.images == tmp_path / 'imgs'
    assert panel._phenome.params.paths.template == '.../(drug)/(time)/(filename).*'
    assert not isinstance(panel._phenome.params.paths.template, Path)


def test_loading_from_the_paths_reads_the_metadata_off_them(panel: LoadingPanel, path_dataset) -> None:
    panel._path_radio.setChecked(True)
    panel._path_image_dir_dialog.value = path_dataset['images']
    panel._path_mask_dir_dialog.value = path_dataset['masks']
    panel._path_template.value = path_dataset['template']

    panel._on_load_data_path()

    assert panel._metadata_table.native.rowCount() == 6
    assert {'drug_name', 'time'} <= set(shown_columns(panel._metadata_table))
    assert set(panel._phenome.metadata_df['drug_name']) == {'ctrl', 'drugA', 'drugB'}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def test_loading_fills_the_table(panel: LoadingPanel) -> None:
    panel._on_load_data_csv()

    assert panel._metadata_table.native.rowCount() == 6
    assert 'drug_name' in shown_columns(panel._metadata_table)


def test_loading_announces_itself(panel: LoadingPanel, qtbot) -> None:
    with qtbot.waitSignal(panel.data_loaded, timeout=1000):
        panel._on_load_data_csv()


def test_a_failed_load_clears_the_table_and_raises(panel: LoadingPanel, tmp_path: Path, qtbot) -> None:
    """The raise is what surfaces the failure to the user, so it has to survive."""
    panel._on_load_data_csv()
    assert panel._metadata_table.native.rowCount() == 6

    panel._phenome.params.paths.images = tmp_path / 'empty'
    (tmp_path / 'empty').mkdir()

    with qtbot.waitSignal(panel.data_loaded, timeout=1000), pytest.raises(ValueError, match='Could not load data'):
        panel._on_load_data_csv()

    assert panel._metadata_table.native.rowCount() == 0


# ----------------------------------------------------------------------------
# Letting go of the previous results, and the question asked before it
# ----------------------------------------------------------------------------


def test_a_first_load_asks_nothing(panel: LoadingPanel, confirmation) -> None:
    """There is nothing to lose before a step has run, so the usual first load goes straight through."""
    panel._on_load_data_csv()

    assert confirmation.asked == 0
    assert panel._metadata_table.native.rowCount() == 6


def test_loading_over_results_asks_first(panel: LoadingPanel, processed, confirmation) -> None:
    processed(panel._phenome, 6)

    panel._on_load_data_csv()

    assert confirmation.asked == 1


def test_answering_no_loads_nothing(panel: LoadingPanel, processed, confirmation) -> None:
    processed(panel._phenome, 6)
    confirmation.accepted = False

    panel._on_load_data_csv()

    assert panel._phenome.has_results is True
    assert panel._phenome.metadata_df is None
    assert panel._metadata_table.native.rowCount() == 0


def test_answering_no_announces_nothing(panel: LoadingPanel, processed, confirmation, qtbot) -> None:
    processed(panel._phenome, 6)
    confirmation.accepted = False

    with qtbot.assertNotEmitted(panel.results_cleared), qtbot.assertNotEmitted(panel.data_loaded):
        panel._on_load_data_csv()


def test_a_load_announces_that_the_results_are_gone(panel: LoadingPanel, qtbot) -> None:
    with qtbot.waitSignal(panel.results_cleared, timeout=1000):
        panel._on_load_data_csv()


def test_loading_from_the_paths_announces_it_too(panel: LoadingPanel, path_dataset, qtbot) -> None:
    panel._path_radio.setChecked(True)
    panel._path_image_dir_dialog.value = path_dataset['images']
    panel._path_template.value = path_dataset['template']

    with qtbot.waitSignal(panel.results_cleared, timeout=1000):
        panel._on_load_data_path()


def test_a_failed_load_still_announces_that_the_results_are_gone(
    panel: LoadingPanel, tmp_path: Path, qtbot
) -> None:
    """They are dropped before a single file is read, so finding none does not bring them back."""
    panel._phenome.params.paths.images = tmp_path / 'empty'
    (tmp_path / 'empty').mkdir()

    with qtbot.waitSignal(panel.results_cleared, timeout=1000), pytest.raises(ValueError):
        panel._on_load_data_csv()


def test_a_load_that_raises_before_the_table_still_announces_it(panel: LoadingPanel, qtbot) -> None:
    """pd.read_csv raises on a missing metadata file, past the point where the results were dropped."""
    panel._phenome.params.paths.metadata = Path('no_such_file.csv')

    with qtbot.waitSignal(panel.results_cleared, timeout=1000), pytest.raises(FileNotFoundError):
        panel._on_load_data_csv()


def test_refresh_table_asks_nothing_and_announces_nothing(
    panel: LoadingPanel, processed, confirmation, qtbot
) -> None:
    """Redrawing the table is not a load: it drops nothing, and must not be taken for one."""
    panel._on_load_data_csv()
    processed(panel._phenome, 6)

    with qtbot.assertNotEmitted(panel.results_cleared):
        panel.refresh_table()

    assert confirmation.asked == 0
    assert panel._phenome.has_results is True


def test_refresh_table_picks_up_a_column_added_elsewhere(panel: LoadingPanel) -> None:
    """This is how the clustering labels reach the table, the analysis panel never touching it directly."""
    panel._on_load_data_csv()
    panel._phenome.metadata_df['cluster'] = 0

    assert 'cluster' not in shown_columns(panel._metadata_table)

    panel.refresh_table()

    assert 'cluster' in shown_columns(panel._metadata_table)


# ----------------------------------------------------------------------------
# Selecting a row
# ----------------------------------------------------------------------------


def test_selecting_a_row_displays_its_image_and_mask(panel: LoadingPanel, viewer) -> None:
    panel._on_load_data_csv()

    panel._metadata_table.native.setCurrentCell(2, 0)

    assert viewer.names == ['Image 2', 'Image 2 - Mask']
    assert [layer['kind'] for layer in viewer.added] == ['image', 'labels']
    assert viewer.added[0]['metadata']['drug_name'] == panel._phenome.metadata_df.iloc[2]['drug_name']


def test_selecting_another_row_replaces_the_layers(panel: LoadingPanel, viewer) -> None:
    panel._on_load_data_csv()

    panel._metadata_table.native.setCurrentCell(0, 0)
    panel._metadata_table.native.setCurrentCell(4, 0)

    # display_images clears the viewer each time, so only the latest pair should be on screen
    assert viewer.names[-2:] == ['Image 4', 'Image 4 - Mask']


def test_no_selection_displays_nothing(panel: LoadingPanel, viewer) -> None:
    """The handler also fires while the table is being repopulated, when there is no current row."""
    panel._on_table_selection_changed()

    assert viewer.added == []


# ----------------------------------------------------------------------------
# The dock area the table is shown in
# ----------------------------------------------------------------------------


@pytest.fixture
def docked_panel(phenome, viewer, in_main_window) -> LoadingPanel:
    """ A panel of its own, inside a main window, which is what the way back needs.

    Built here rather than taken off the widget, which the `panel` fixture hands over already
    owned by a widget of its own.
    """
    return in_main_window(lambda parent: LoadingPanel(phenome, ImageViewer(viewer), parent))


def test_the_table_is_shown_in_a_dock_area(docked_panel: LoadingPanel) -> None:
    area = docked_panel._table_area

    assert area.dock is docked_panel._table_dock
    assert area.dock.widget() is docked_panel._metadata_table.native
    assert area.dockWidgetArea(area.dock) == Qt.DockWidgetArea.TopDockWidgetArea


def test_popping_the_table_out_and_closing_it_brings_it_back(docked_panel: LoadingPanel, qtbot) -> None:
    """The dock is what puts it back; this is the panel holding up its end, its slot and its table."""
    area = docked_panel._table_area
    slot = docked_panel.layout().indexOf(area)

    area.dock.setFloating(True)
    area.dock.close()

    qtbot.waitUntil(lambda: not area.dock.isFloating())
    assert area.dockWidgetArea(area.dock) == Qt.DockWidgetArea.TopDockWidgetArea
    assert docked_panel.layout().indexOf(area) == slot
    assert area.dock.widget() is docked_panel._metadata_table.native
    assert not area.dock.isHidden()


# ----------------------------------------------------------------------------
# An empty box, which names nothing rather than naming the working directory
# ----------------------------------------------------------------------------


def clear(widget) -> None:
    """ Empty a directory box the way a user does, through its text.

    Setting the value instead would land it on the working directory, which is what the panel reads
    around and what a test must not read around with it.

    Args:
        widget: The FileEdit to empty.
    """
    widget.line_edit.value = ''


CSV_BOXES = [
    pytest.param('_image_dir_dialog', 'images', id='images'),
    pytest.param('_mask_dir_dialog', 'masks', id='masks'),
    pytest.param('_metadata_dir_dialog', 'metadata', id='metadata'),
]

PATH_BOXES = [
    pytest.param('_path_image_dir_dialog', 'images', id='images'),
    pytest.param('_path_mask_dir_dialog', 'masks', id='masks'),
]


@pytest.mark.parametrize(('box', 'field'), CSV_BOXES)
def test_emptying_a_csv_box_names_nothing(panel: LoadingPanel, box: str, field: str) -> None:
    panel._csv_radio.setChecked(True)

    clear(getattr(panel, box))

    assert getattr(panel._phenome.params.paths, field) is None


@pytest.mark.parametrize(('box', 'field'), PATH_BOXES)
def test_emptying_a_path_box_names_nothing(panel: LoadingPanel, box: str, field: str) -> None:
    panel._path_radio.setChecked(True)

    clear(getattr(panel, box))

    assert getattr(panel._phenome.params.paths, field) is None


def test_emptying_the_template_names_nothing(panel: LoadingPanel) -> None:
    """A pattern rather than a path, but empty means the same: no metadata read off the paths."""
    panel._path_radio.setChecked(True)

    panel._path_template.value = ''

    assert panel._phenome.params.paths.template is None


def test_a_filled_box_still_names_its_path(panel: LoadingPanel, tmp_path: Path) -> None:
    """The reading goes through the text now, which must not cost the ordinary case."""
    panel._csv_radio.setChecked(True)

    panel._image_dir_dialog.value = tmp_path / 'imgs'

    assert panel._phenome.params.paths.images == tmp_path / 'imgs'


def test_a_cleared_box_stays_cleared_across_the_pages(panel: LoadingPanel) -> None:
    """Shown through the text too: str(None) would spell the word 'None' into the box."""
    panel._csv_radio.setChecked(True)
    clear(panel._mask_dir_dialog)

    panel._path_radio.setChecked(True)

    assert str(panel._path_mask_dir_dialog.line_edit.value) == ''


# ----------------------------------------------------------------------------
# The load, which waits on the images and on nothing else
# ----------------------------------------------------------------------------


def load_buttons(panel: LoadingPanel) -> tuple:
    """ Both Load Data buttons, the two pages offering one each. """
    return panel._csv_load_data_btn, panel._path_load_data_btn


def test_the_load_waits_on_an_images_directory(panel: LoadingPanel) -> None:
    clear(panel._image_dir_dialog)

    for button in load_buttons(panel):
        assert not button.isEnabled()
        assert button.toolTip() == 'Please choose an images directory first.'


def test_naming_the_images_again_opens_the_load(panel: LoadingPanel, dataset) -> None:
    clear(panel._image_dir_dialog)

    panel._image_dir_dialog.value = dataset['images']

    for button in load_buttons(panel):
        assert button.isEnabled()
        assert button.toolTip() == ''


def test_the_load_does_not_wait_on_the_masks_or_the_metadata(panel: LoadingPanel) -> None:
    """Both are PhenoMe's to do without, so neither may close the load."""
    clear(panel._mask_dir_dialog)
    clear(panel._metadata_dir_dialog)

    for button in load_buttons(panel):
        assert button.isEnabled()


def test_loading_without_masks_finds_the_images(panel: LoadingPanel, dataset) -> None:
    """PhenoMe keeps the mask_path column and leaves it empty, which the image viewer reads as no mask."""
    panel._csv_radio.setChecked(True)
    panel._image_dir_dialog.value = dataset['images']
    panel._metadata_dir_dialog.value = dataset['metadata']
    clear(panel._mask_dir_dialog)

    panel._csv_load_data_btn.click()

    assert panel._metadata_table.native.rowCount() == 6
    assert panel._phenome.metadata_df['mask_path'].isna().all()


def test_loading_without_metadata_finds_the_images(panel: LoadingPanel, dataset) -> None:
    """The .csv is what the metadata columns come from, and the images are found without it."""
    panel._csv_radio.setChecked(True)
    panel._image_dir_dialog.value = dataset['images']
    panel._mask_dir_dialog.value = dataset['masks']
    clear(panel._metadata_dir_dialog)

    panel._csv_load_data_btn.click()

    assert panel._metadata_table.native.rowCount() == 6
    assert {'drug_name', 'time'}.isdisjoint(shown_columns(panel._metadata_table))


def test_loading_with_the_images_alone_finds_them(panel: LoadingPanel, dataset) -> None:
    """The whole of the point: only the images directory is needed."""
    panel._csv_radio.setChecked(True)
    panel._image_dir_dialog.value = dataset['images']
    clear(panel._mask_dir_dialog)
    clear(panel._metadata_dir_dialog)

    panel._csv_load_data_btn.click()

    assert panel._metadata_table.native.rowCount() == 6


def test_loading_from_the_paths_without_a_template_finds_them(panel: LoadingPanel, path_dataset) -> None:
    """The template is what the metadata is read off the paths with, and the images do without it."""
    panel._path_radio.setChecked(True)
    panel._path_image_dir_dialog.value = path_dataset['images']
    clear(panel._path_mask_dir_dialog)
    panel._path_template.value = ''

    panel._path_load_data_btn.click()

    assert panel._metadata_table.native.rowCount() == 6
