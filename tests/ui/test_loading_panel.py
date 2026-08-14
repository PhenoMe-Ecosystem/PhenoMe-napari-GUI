"""Tests for the data loading panel."""

from pathlib import Path

import pytest

from phenome_gui.ui.loading_panel import LoadingPanel


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
