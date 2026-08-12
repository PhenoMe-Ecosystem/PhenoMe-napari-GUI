"""Tests for PhenoMeInterface, the single entry point the GUI calls PhenoMe through."""

from pathlib import Path

import pandas as pd
import pytest

from phenome_gui.utils.phenome_interface import (
    DEFAULT_PARAMS,
    PhenoMeInterface,
)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------


def test_parameters_start_from_the_defaults(phenome: PhenoMeInterface) -> None:
    for section in ('process', 'properties', 'clustering', 'prototypes', 'outliers'):
        assert phenome.params[section] == DEFAULT_PARAMS[section]


def test_parameters_are_copied_not_shared(phenome: PhenoMeInterface) -> None:
    phenome.set_param('clustering', 'n_clusters', 99)
    assert DEFAULT_PARAMS['clustering']['n_clusters'] == 5
    assert PhenoMeInterface(seed=42).get_param('clustering', 'n_clusters') == 5


def test_get_param_round_trips(phenome: PhenoMeInterface) -> None:
    phenome.set_param('prototypes', 'n_prototypes', 7)
    assert phenome.get_param('prototypes', 'n_prototypes') == 7


def test_get_param_returns_empty_for_an_unknown_key(phenome: PhenoMeInterface) -> None:
    assert phenome.get_param('clustering', 'nonexistent') == ''
    assert phenome.get_param('nonexistent', 'whatever') == ''


def test_set_path_coerces_to_a_path(phenome: PhenoMeInterface) -> None:
    phenome.set_path('checkpoint', '/tmp/somewhere.hdf5')
    assert phenome.get_path('checkpoint') == Path('/tmp/somewhere.hdf5')


# ----------------------------------------------------------------------------
# Data loading and metadata access
# ----------------------------------------------------------------------------


def test_load_data_csv_finds_every_image(loaded: PhenoMeInterface) -> None:
    assert len(loaded.metadata_df) == 6
    assert {'file_path', 'mask_path', 'drug_name', 'time'} <= set(loaded.metadata_df.columns)


def test_metadata_columns_are_empty_before_loading(phenome: PhenoMeInterface) -> None:
    assert phenome.get_metadata_columns() == []


def test_metadata_columns_leave_out_the_path_columns(loaded: PhenoMeInterface) -> None:
    columns = loaded.get_metadata_columns()
    assert 'file_path' not in columns
    assert 'mask_path' not in columns
    assert {'drug_name', 'time'} <= set(columns)


def test_row_display_data_returns_the_image_and_its_mask(loaded: PhenoMeInterface) -> None:
    image_paths, mask_paths, row_data = loaded.get_row_display_data(2)

    assert image_paths == [loaded.metadata_df.iloc[2]['file_path']]
    assert mask_paths == [loaded.metadata_df.iloc[2]['mask_path']]
    assert row_data[0]['drug_name'] == loaded.metadata_df.iloc[2]['drug_name']


def test_image_display_data_matches_on_the_path(loaded: PhenoMeInterface) -> None:
    # Deliberately not row 0, so a wrong lookup returning the first row would be caught
    wanted = loaded.metadata_df.iloc[3]
    image_paths, mask_paths, metadata = loaded.get_image_display_data(wanted['file_path'])

    assert image_paths == [wanted['file_path']]
    assert mask_paths == [wanted['mask_path']]
    assert metadata[0]['drug_name'] == wanted['drug_name']


def test_image_display_data_survives_an_unknown_path(loaded: PhenoMeInterface) -> None:
    # PhenoMe returns whatever its results hold, which can outlive a reload of the metadata
    image_paths, mask_paths, metadata = loaded.get_image_display_data('/gone/img.tif')

    assert image_paths == ['/gone/img.tif']
    assert mask_paths is None
    assert metadata == [{}]


def test_image_display_data_without_masks(loaded: PhenoMeInterface) -> None:
    loaded.metadata_df = loaded.metadata_df.drop(columns=['mask_path'])
    wanted = loaded.metadata_df.iloc[0]

    _, mask_paths, _ = loaded.get_image_display_data(wanted['file_path'])

    assert mask_paths is None


def test_image_display_data_before_loading(phenome: PhenoMeInterface) -> None:
    assert phenome.get_image_display_data('/anything.tif') == (['/anything.tif'], None, [{}])


# ----------------------------------------------------------------------------
# Pipeline state
# ----------------------------------------------------------------------------


def test_has_processed_images_is_a_property(phenome: PhenoMeInterface, processed) -> None:
    assert phenome.has_processed_images is False
    processed(phenome, 6)
    assert phenome.has_processed_images is True


def test_has_properties_is_a_property(phenome: PhenoMeInterface) -> None:
    assert phenome.has_properties is False
    phenome.properties_df = pd.DataFrame({'area': [1.0]})
    assert phenome.has_properties is True


# ----------------------------------------------------------------------------
# Clustering, and the labels it writes back into the metadata
# ----------------------------------------------------------------------------


def test_compute_clustering_forwards_its_parameters(loaded: PhenoMeInterface, spy) -> None:
    loaded.pheno.compute_clustering = spy(pd.DataFrame({'image_path': [], 'cluster': []}))
    loaded.set_param('clustering', 'clustering_method', 'dbscan')
    loaded.set_param('clustering', 'source', 'combined')
    loaded.set_param('clustering', 'n_clusters', 7)
    loaded.set_param('clustering', 'dbscan_eps', 0.25)
    loaded.set_param('clustering', 'dbscan_min_samples', 11)

    loaded.compute_clustering()

    assert loaded.pheno.compute_clustering.last == {
        'source': 'combined',
        'n_clusters': 7,
        'clustering_method': 'dbscan',
        'dbscan_eps': 0.25,
        'dbscan_min_samples': 11,
    }


def test_clusters_are_matched_by_path_not_by_row_order(loaded: PhenoMeInterface, spy) -> None:
    """PhenoMe skips images it fails to load, so a positional join would shift every later label."""
    paths = list(loaded.metadata_df['file_path'])
    # Shuffled, and missing rows 1 and 5, the way a skipped batch leaves them out
    returned = [paths[2], paths[0], paths[3], paths[4]]
    loaded.pheno.compute_clustering = spy(pd.DataFrame({'image_path': returned, 'cluster': [2, 0, 1, 1]}))

    loaded.compute_clustering()

    labels = loaded.metadata_df['cluster']
    assert labels.iloc[0] == 0
    assert labels.iloc[2] == 2
    assert labels.iloc[3] == 1
    assert labels.iloc[4] == 1
    assert pd.isna(labels.iloc[1])
    assert pd.isna(labels.iloc[5])


def test_cluster_labels_stay_whole_numbers(loaded: PhenoMeInterface, spy) -> None:
    """A plain int column would turn to floats as soon as one label is missing, showing '0.0' in the table."""
    paths = list(loaded.metadata_df['file_path'])
    loaded.pheno.compute_clustering = spy(pd.DataFrame({'image_path': paths[:2], 'cluster': [0, 1]}))

    loaded.compute_clustering()

    assert loaded.metadata_df['cluster'].dtype == 'Int64'


def test_rerunning_the_clustering_overwrites_the_column(loaded: PhenoMeInterface, spy) -> None:
    paths = list(loaded.metadata_df['file_path'])
    loaded.pheno.compute_clustering = spy(pd.DataFrame({'image_path': paths, 'cluster': [0] * 6}))
    loaded.compute_clustering()

    loaded.pheno.compute_clustering = spy(pd.DataFrame({'image_path': paths, 'cluster': [3] * 6}))
    loaded.compute_clustering()

    assert list(loaded.metadata_df.columns).count('cluster') == 1
    assert loaded.metadata_df['cluster'].unique().tolist() == [3]


def test_reloading_the_data_drops_the_cluster_column(loaded: PhenoMeInterface, spy) -> None:
    paths = list(loaded.metadata_df['file_path'])
    loaded.pheno.compute_clustering = spy(pd.DataFrame({'image_path': paths, 'cluster': [0] * 6}))
    loaded.compute_clustering()
    assert 'cluster' in loaded.metadata_df.columns

    loaded.load_data_csv()

    assert 'cluster' not in loaded.metadata_df.columns


def test_cluster_becomes_a_grouping_option(loaded: PhenoMeInterface, spy) -> None:
    paths = list(loaded.metadata_df['file_path'])
    loaded.pheno.compute_clustering = spy(pd.DataFrame({'image_path': paths, 'cluster': [0] * 6}))

    loaded.compute_clustering()

    assert 'cluster' in loaded.get_metadata_columns()


def test_an_empty_clustering_result_leaves_the_metadata_alone(loaded: PhenoMeInterface, spy) -> None:
    loaded.pheno.compute_clustering = spy(pd.DataFrame())

    loaded.compute_clustering()

    assert 'cluster' not in loaded.metadata_df.columns


# ----------------------------------------------------------------------------
# Prototypes and outliers
# ----------------------------------------------------------------------------


def test_prototypes_forward_group_by_as_cluster_col(loaded: PhenoMeInterface, spy) -> None:
    """The GUI names the grouping key group_by for both steps; PhenoMe calls it cluster_col here."""
    loaded.pheno.find_prototypes = spy(pd.DataFrame())
    loaded.set_param('prototypes', 'group_by', 'drug_name')
    loaded.set_param('prototypes', 'n_prototypes', 4)
    loaded.set_param('prototypes', 'source', 'properties')

    loaded.get_prototype_images()

    assert loaded.pheno.find_prototypes.last == {
        'cluster_col': 'drug_name',
        'n_prototypes': 4,
        'source': 'properties',
        'plot': False,
    }


def test_outliers_forward_their_parameters(loaded: PhenoMeInterface, spy) -> None:
    loaded.pheno.detect_outliers = spy(pd.DataFrame())
    loaded.set_param('outliers', 'group_by', 'time')
    loaded.set_param('outliers', 'method', 'iqr')
    loaded.set_param('outliers', 'threshold', 1.5)
    loaded.set_param('outliers', 'source', 'combined')

    loaded.get_outlier_images()

    assert loaded.pheno.detect_outliers.last == {
        'group_by': 'time',
        'method': 'iqr',
        'threshold': 1.5,
        'source': 'combined',
        'plot': False,
    }


@pytest.mark.parametrize(
    ('call', 'pheno_method', 'key'),
    [
        ('get_prototype_images', 'find_prototypes', 'cluster_col'),
        ('get_outlier_images', 'detect_outliers', 'group_by'),
    ],
)
def test_an_unset_grouping_key_means_one_group(
    loaded: PhenoMeInterface, spy, call: str, pheno_method: str, key: str
) -> None:
    """PhenoMe reads None as 'treat every image as one group'; '' would resolve to nothing and warn."""
    setattr(loaded.pheno, pheno_method, spy(pd.DataFrame()))

    getattr(loaded, call)()

    assert getattr(loaded.pheno, pheno_method).last[key] is None


@pytest.mark.parametrize(
    ('call', 'pheno_method', 'attribute'),
    [
        ('compute_clustering', 'compute_clustering', 'clustering_df'),
        ('get_prototype_images', 'find_prototypes', 'prototypes_df'),
        ('get_outlier_images', 'detect_outliers', 'outliers_df'),
    ],
)
def test_results_are_stored_and_returned(
    loaded: PhenoMeInterface, spy, call: str, pheno_method: str, attribute: str
) -> None:
    frame = pd.DataFrame({'image_path': ['a.tif'], 'cluster': [0]})
    setattr(loaded.pheno, pheno_method, spy(frame))

    returned = getattr(loaded, call)()

    assert returned is getattr(loaded, attribute)
    assert returned is frame
