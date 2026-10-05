"""Tests for the data loading panel."""

import threading
from pathlib import Path

import pandas as pd
import pytest
from qtpy.QtCore import QMimeData, QPointF, Qt, QUrl
from qtpy.QtGui import QDropEvent
from qtpy.QtWidgets import QFileDialog

from phenome_gui.ui import loading_panel as loading_panel_module
from phenome_gui.ui.loading_panel import LoadingPanel
from phenome_gui.ui.styles import GROUP_STYLESHEETS, STYLESHEETS
from phenome_gui.utils.image_viewer import ImageViewer


@pytest.fixture
def panel(widget) -> LoadingPanel:
    """ The loading panel of a built widget, with its paths pointed at the test dataset. """
    return widget.loading_panel


@pytest.fixture
def shown_errors(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """ Catch the errors the panel shows through napari, rather than raising them as the tests run.

    Returns:
        Filled with each message shown, in order.
    """
    shown = []
    monkeypatch.setattr(loading_panel_module.notifications, 'show_error', shown.append)
    return shown


def shown_columns(table) -> list[str]:
    """ The column names currently displayed by a magicgui table. """
    return list(table.value['columns'])


# ----------------------------------------------------------------------------
# The two metadata structures
# ----------------------------------------------------------------------------


def test_choosing_the_path_structure_swaps_the_page(panel: LoadingPanel) -> None:
    panel._path_radio.setChecked(True)

    assert panel._loading_stack.currentIndex() == 1


def test_the_csv_page_writes_its_three_paths(panel: LoadingPanel, tmp_path: Path) -> None:
    panel._csv_radio.setChecked(True)
    panel._image_dir_dialog.path = tmp_path / 'imgs'
    panel._mask_dir_dialog.path = tmp_path / 'msks'
    panel._metadata_dir_dialog.path = tmp_path / 'meta.csv'

    assert panel._phenome.params.paths.images == tmp_path / 'imgs'
    assert panel._phenome.params.paths.masks == tmp_path / 'msks'
    assert panel._phenome.params.paths.metadata == tmp_path / 'meta.csv'


def test_the_path_page_writes_the_template_instead(panel: LoadingPanel, tmp_path: Path) -> None:
    """The template stays a string: PhenoMe calls str.replace on it, which a Path does not have."""
    panel._path_radio.setChecked(True)
    panel._path_image_dir_dialog.path = tmp_path / 'imgs'
    panel._path_template.value = '.../(drug)/(time)/(filename).*'

    assert panel._phenome.params.paths.images == tmp_path / 'imgs'
    assert panel._phenome.params.paths.template == '.../(drug)/(time)/(filename).*'
    assert not isinstance(panel._phenome.params.paths.template, Path)


def test_loading_from_the_paths_reads_the_metadata_off_them(panel: LoadingPanel, path_dataset) -> None:
    panel._path_radio.setChecked(True)
    panel._path_image_dir_dialog.path = path_dataset['images']
    panel._path_mask_dir_dialog.path = path_dataset['masks']
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


def test_a_load_finding_nothing_clears_the_table_and_says_so(
    panel: LoadingPanel, tmp_path: Path, shown_errors: list, qtbot
) -> None:
    """Shown as an error rather than raised: the load runs on a worker, where a raise reaches no one."""
    panel._on_load_data_csv()
    assert panel._metadata_table.native.rowCount() == 6

    panel._phenome.params.paths.images = tmp_path / 'empty'
    (tmp_path / 'empty').mkdir()

    with qtbot.waitSignal(panel.data_loaded, timeout=1000):
        panel._on_load_data_csv()

    assert panel._metadata_table.native.rowCount() == 0
    assert any('Could not load data' in message for message in shown_errors)


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
    panel._path_image_dir_dialog.path = path_dataset['images']
    panel._path_template.value = path_dataset['template']

    with qtbot.waitSignal(panel.results_cleared, timeout=1000):
        panel._on_load_data_path()


def test_a_failed_load_still_announces_that_the_results_are_gone(
    panel: LoadingPanel, tmp_path: Path, qtbot
) -> None:
    """They are dropped before a single file is read, so finding none does not bring them back."""
    panel._phenome.params.paths.images = tmp_path / 'empty'
    (tmp_path / 'empty').mkdir()

    with qtbot.waitSignal(panel.results_cleared, timeout=1000):
        panel._on_load_data_csv()


def test_a_load_that_raises_before_the_table_still_announces_it(
    panel: LoadingPanel, shown_errors: list, qtbot
) -> None:
    """pd.read_csv raises on a missing metadata file, past the point where the results were dropped.

    Caught by the worker and reported, rather than raised out of the click, which is where it went
    while the load ran on the GUI thread.
    """
    panel._phenome.params.paths.metadata = Path('no_such_file.csv')

    with qtbot.waitSignal(panel.results_cleared, timeout=1000):
        panel._on_load_data_csv()

    assert any('no_such_file.csv' in message for message in shown_errors)


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
    """ Empty a path box, the way deleting its text does.

    Args:
        widget: The path box to empty.
    """
    widget.path = ''


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

    panel._image_dir_dialog.path = tmp_path / 'imgs'

    assert panel._phenome.params.paths.images == tmp_path / 'imgs'


def test_a_cleared_box_stays_cleared_across_the_pages(panel: LoadingPanel) -> None:
    """Shown through the text too: str(None) would spell the word 'None' into the box."""
    panel._csv_radio.setChecked(True)
    clear(panel._mask_dir_dialog)

    panel._path_radio.setChecked(True)

    assert str(panel._path_mask_dir_dialog.path) == ''


# ----------------------------------------------------------------------------
# The load, which waits on the images and on nothing else
# ----------------------------------------------------------------------------


def load_buttons(panel: LoadingPanel) -> tuple:
    """ Both Load Data buttons, the two pages offering one each. """
    return panel._csv_load_data_btn, panel._path_load_data_btn


def test_the_load_waits_on_an_images_directory(panel: LoadingPanel) -> None:
    panel._csv_radio.setChecked(True)

    clear(panel._image_dir_dialog)

    assert not panel._csv_load_data_btn.isEnabled()
    assert panel._csv_load_data_btn.toolTip() == 'Please choose an images directory first.'


def test_the_other_page_waits_once_it_is_shown(panel: LoadingPanel) -> None:
    """Each page is gated on its own boxes, and a page's boxes are filled from the parameters when it
    is shown - so an images directory cleared on one page closes the other's load as it comes up."""
    panel._csv_radio.setChecked(True)
    clear(panel._image_dir_dialog)

    panel._path_radio.setChecked(True)

    assert not panel._path_load_data_btn.isEnabled()
    assert panel._path_load_data_btn.toolTip() == 'Please choose an images directory first.'


def test_naming_the_images_again_opens_the_load(panel: LoadingPanel, dataset) -> None:
    clear(panel._image_dir_dialog)

    panel._image_dir_dialog.path = dataset['images']

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
    panel._image_dir_dialog.path = dataset['images']
    panel._metadata_dir_dialog.path = dataset['metadata']
    clear(panel._mask_dir_dialog)

    panel._csv_load_data_btn.click()

    assert panel._metadata_table.native.rowCount() == 6
    assert panel._phenome.metadata_df['mask_path'].isna().all()


def test_loading_without_metadata_finds_the_images(panel: LoadingPanel, dataset) -> None:
    """The .csv is what the metadata columns come from, and the images are found without it."""
    panel._csv_radio.setChecked(True)
    panel._image_dir_dialog.path = dataset['images']
    panel._mask_dir_dialog.path = dataset['masks']
    clear(panel._metadata_dir_dialog)

    panel._csv_load_data_btn.click()

    assert panel._metadata_table.native.rowCount() == 6
    assert {'drug_name', 'time'}.isdisjoint(shown_columns(panel._metadata_table))


def test_loading_with_the_images_alone_finds_them(panel: LoadingPanel, dataset) -> None:
    """The whole of the point: only the images directory is needed."""
    panel._csv_radio.setChecked(True)
    panel._image_dir_dialog.path = dataset['images']
    clear(panel._mask_dir_dialog)
    clear(panel._metadata_dir_dialog)

    panel._csv_load_data_btn.click()

    assert panel._metadata_table.native.rowCount() == 6


def test_loading_from_the_paths_without_a_template_finds_them(panel: LoadingPanel, path_dataset) -> None:
    """The template is what the metadata is read off the paths with, and the images do without it."""
    panel._path_radio.setChecked(True)
    panel._path_image_dir_dialog.path = path_dataset['images']
    clear(panel._path_mask_dir_dialog)
    panel._path_template.value = ''

    panel._path_load_data_btn.click()

    assert panel._metadata_table.native.rowCount() == 6


# ----------------------------------------------------------------------------
# The outlines, which say whether each path names what its box asks for
# ----------------------------------------------------------------------------


def outlined(widget) -> str:
    """ Which outline a Qt widget is drawn with: 'ok', 'bad', or 'none' for one left unstyled. """
    sheet = widget.styleSheet()
    if sheet == STYLESHEETS['STYLE_OK']:
        return 'ok'
    if sheet == STYLESHEETS['STYLE_ERROR']:
        return 'bad'
    return 'none'


CSV_PATH_BOXES = ['_image_dir_dialog', '_mask_dir_dialog', '_metadata_dir_dialog']


@pytest.mark.parametrize('box', CSV_PATH_BOXES)
def test_a_box_naming_what_it_asks_for_is_green(panel: LoadingPanel, box: str) -> None:
    """The fixture points every box at the test dataset, which exists wherever the suite runs."""
    assert outlined(getattr(panel, box)) == 'ok'


@pytest.mark.parametrize('box', CSV_PATH_BOXES)
def test_a_path_to_nothing_is_red(panel: LoadingPanel, box: str, tmp_path: Path) -> None:
    getattr(panel, box).path = tmp_path / 'missing'

    assert outlined(getattr(panel, box)) == 'bad'


@pytest.mark.parametrize('box', ['_image_dir_dialog', '_mask_dir_dialog'])
def test_a_file_in_a_directory_box_is_red(panel: LoadingPanel, box: str, tmp_path: Path) -> None:
    """It exists, which is all the first check asked, and PhenoMe still cannot search it."""
    path = tmp_path / 'image.tif'
    path.write_bytes(b'')

    getattr(panel, box).path = path

    assert outlined(getattr(panel, box)) == 'bad'


def test_a_folder_in_the_metadata_box_is_red(panel: LoadingPanel, tmp_path: Path) -> None:
    panel._metadata_dir_dialog.path = tmp_path

    assert outlined(panel._metadata_dir_dialog) == 'bad'


def test_a_file_of_another_kind_in_the_metadata_box_is_red(panel: LoadingPanel, tmp_path: Path) -> None:
    path = tmp_path / 'metadata.txt'
    path.write_text('filename\n')

    panel._metadata_dir_dialog.path = path

    assert outlined(panel._metadata_dir_dialog) == 'bad'


def test_an_empty_images_box_is_red(panel: LoadingPanel) -> None:
    """The one box a load cannot do without."""
    clear(panel._image_dir_dialog)

    assert outlined(panel._image_dir_dialog) == 'bad'


@pytest.mark.parametrize('box', ['_mask_dir_dialog', '_metadata_dir_dialog'])
def test_an_empty_optional_box_is_green(panel: LoadingPanel, box: str) -> None:
    """Empty is an answer the load accepts for these, and green says the load will go."""
    clear(getattr(panel, box))

    assert outlined(getattr(panel, box)) == 'ok'


# ----------------------------------------------------------------------------
# The gate, which a red box closes
# ----------------------------------------------------------------------------


def test_a_red_box_closes_the_load(panel: LoadingPanel, tmp_path: Path) -> None:
    panel._csv_radio.setChecked(True)

    panel._mask_dir_dialog.path = tmp_path / 'missing'

    assert not panel._csv_load_data_btn.isEnabled()
    assert panel._csv_load_data_btn.toolTip() == 'Please fix the masks directory first.'


def test_every_red_box_is_named(panel: LoadingPanel, tmp_path: Path) -> None:
    """In the order of the boxes, so the tooltip reads down the page."""
    panel._csv_radio.setChecked(True)
    clear(panel._image_dir_dialog)
    panel._metadata_dir_dialog.path = tmp_path / 'missing.csv'

    assert panel._csv_load_data_btn.toolTip() == (
        'Please choose an images directory and fix the metadata file first.'
    )


def test_putting_a_box_right_opens_the_load_again(panel: LoadingPanel, dataset, tmp_path: Path) -> None:
    panel._csv_radio.setChecked(True)
    panel._mask_dir_dialog.path = tmp_path / 'missing'

    panel._mask_dir_dialog.path = dataset['masks']

    assert outlined(panel._mask_dir_dialog) == 'ok'
    assert panel._csv_load_data_btn.isEnabled()
    assert panel._csv_load_data_btn.toolTip() == ''


def test_a_bad_metadata_file_leaves_the_path_page_open(panel: LoadingPanel, tmp_path: Path) -> None:
    """The metadata file is the csv page's alone, and the path page's load never reads it."""
    panel._csv_radio.setChecked(True)
    panel._metadata_dir_dialog.path = tmp_path / 'missing.csv'

    panel._path_radio.setChecked(True)

    assert panel._path_load_data_btn.isEnabled()


def test_switching_pages_keeps_the_paths_just_typed(panel: LoadingPanel, tmp_path: Path) -> None:
    """Showing a page writes its boxes, and each write that changes one saves the whole page back.

    Both are changed, which is what it takes: writing the images box onto the path page saves that
    page back while its masks box still holds the old path, and read as it went, that old path would
    have been put back over the one just typed. A box written with what it already holds emits
    nothing, so changing the masks alone would never have shown it.
    """
    panel._csv_radio.setChecked(True)
    images = tmp_path / 'images'
    masks = tmp_path / 'masks'
    images.mkdir()
    masks.mkdir()
    panel._image_dir_dialog.path = images
    panel._mask_dir_dialog.path = masks

    panel._path_radio.setChecked(True)

    assert panel._phenome.params.paths.images == images
    assert panel._phenome.params.paths.masks == masks
    assert Path(str(panel._path_mask_dir_dialog.path)) == masks


# ----------------------------------------------------------------------------
# Browsing and dropping, which the path boxes bring with them
# ----------------------------------------------------------------------------


def test_browsing_for_the_images_writes_them(panel: LoadingPanel, dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    """A folder picked in the dialog reaches the parameters and is checked, as one typed would be."""
    panel._csv_radio.setChecked(True)
    clear(panel._image_dir_dialog)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args, **kwargs: str(dataset['images']))

    panel._image_dir_dialog._browse_button.click()

    assert panel._phenome.params.paths.images == dataset['images']
    assert outlined(panel._image_dir_dialog) == 'ok'
    assert panel._csv_load_data_btn.isEnabled()


def test_a_folder_dropped_on_the_masks_writes_them(panel: LoadingPanel, tmp_path: Path) -> None:
    """Dropped from a file browser, the path reaches the parameters the way a typed one does."""
    panel._csv_radio.setChecked(True)
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(str(tmp_path))])
    event = QDropEvent(
        QPointF(5, 5), Qt.DropAction.CopyAction, mime_data,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )

    panel._mask_dir_dialog._line_edit.dropEvent(event)

    assert panel._phenome.params.paths.masks == tmp_path
    assert outlined(panel._mask_dir_dialog) == 'ok'


# ----------------------------------------------------------------------------
# The load, run on a worker so the window can show that it is waiting
# ----------------------------------------------------------------------------


def outline(panel: LoadingPanel) -> str:
    """ How the data loading group is outlined: 'ok', 'bad', 'wait', or 'none'. """
    names = {
        GROUP_STYLESHEETS['STYLE_OK']: 'ok',
        GROUP_STYLESHEETS['STYLE_ERROR']: 'bad',
        GROUP_STYLESHEETS['STYLE_WAIT']: 'wait',
    }
    return names.get(panel._loading_group.styleSheet(), 'none')


@pytest.fixture
def held_load(panel: LoadingPanel, monkeypatch: pytest.MonkeyPatch):
    """ Hold the next load on its worker until the test lets it go.

    The real load is still made once it is let go, so what follows it is what follows a real one.

    Returns:
        The event that lets the load go on. Set it, or the worker waits out its timeout.
    """
    release = threading.Event()
    real_load = panel._phenome.load_data_csv

    def _held() -> pd.DataFrame:
        release.wait(timeout=5)
        return real_load()

    monkeypatch.setattr(panel._phenome, 'load_data_csv', _held)
    yield release
    release.set()


@pytest.mark.real_load_worker
def test_the_group_waits_while_the_load_runs(panel: LoadingPanel, held_load, qtbot) -> None:
    """On the real worker thread: the waiting outline is drawn while the files are still being found.

    On the GUI thread it never could be - nothing is painted until the load returns, by which time
    the outline has already been changed to how it ended.
    """
    panel._csv_radio.setChecked(True)

    panel._csv_load_data_btn.click()

    try:
        assert outline(panel) == 'wait'
        assert not panel._loading_group.isEnabled()
        assert not panel._structure_group.isEnabled()
    finally:
        held_load.set()

    qtbot.waitUntil(lambda: outline(panel) == 'ok', timeout=5000)
    assert panel._loading_group.isEnabled()
    assert panel._csv_load_data_btn.isEnabled()
    assert panel._metadata_table.native.rowCount() == 6


def test_calling_the_load_off_leaves_the_outline_as_it_was(panel: LoadingPanel, confirmation, processed) -> None:
    """Asked before anything changes colour, so a load called off is not left waiting for good."""
    panel._on_load_data_csv()
    processed(panel._phenome, 6)
    confirmation.accepted = False

    panel._on_load_data_csv()

    assert outline(panel) == 'ok'
    assert panel._loading_group.isEnabled()


def test_a_load_that_finds_files_ends_green(panel: LoadingPanel) -> None:
    panel._on_load_data_csv()

    assert outline(panel) == 'ok'


def test_a_load_that_finds_nothing_ends_red(panel: LoadingPanel, tmp_path: Path, shown_errors: list) -> None:
    (tmp_path / 'empty').mkdir()
    panel._phenome.params.paths.images = tmp_path / 'empty'

    panel._on_load_data_csv()

    assert outline(panel) == 'bad'


def test_a_load_that_raises_ends_red_and_unlocked(panel: LoadingPanel, shown_errors: list) -> None:
    """Without the failure handler the tab would stay locked, its outline waiting for good."""
    panel._phenome.params.paths.metadata = Path('no_such_file.csv')

    panel._on_load_data_csv()

    assert outline(panel) == 'bad'
    assert panel._loading_group.isEnabled()
    assert panel._structure_group.isEnabled()


def test_unlocking_keeps_a_red_box_closing_its_load(panel: LoadingPanel, tmp_path: Path, shown_errors: list) -> None:
    """Unlocked to what the boxes allow, rather than every button opened back up."""
    panel._csv_radio.setChecked(True)
    panel._on_load_data_csv()
    panel._mask_dir_dialog.path = tmp_path / 'missing'

    panel._set_loading(True)
    panel._set_loading(False, GROUP_STYLESHEETS['STYLE_OK'])

    assert not panel._csv_load_data_btn.isEnabled()
