"""Tests for PhenoMeInterface, the single entry point the GUI calls PhenoMe through."""

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from phenome_gui.utils.params import (
    DEFAULT_CHECKPOINT,
    DIMS_CHOICES,
    EMBEDDING_METHOD_CHOICES,
    MAX_WORKERS,
    Params,
)
from phenome_gui.utils.phenome_interface import (
    DIMS_COMPONENTS,
    DR_METHODS,
    PhenoMeInterface,
)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------


def test_parameters_start_from_the_defaults(phenome: PhenoMeInterface) -> None:
    """The paths are left out: the fixture points them at the test dataset."""
    defaults = Params()
    for section in ('process', 'properties', 'clustering', 'prototypes', 'outliers'):
        assert getattr(phenome.params, section) == getattr(defaults, section)


def test_parameters_are_copied_not_shared(phenome: PhenoMeInterface) -> None:
    """Each interface gets its own models, so one run's edits cannot leak into the next one's."""
    phenome.params.clustering.n_clusters = 99

    assert Params().clustering.n_clusters == 5
    assert PhenoMeInterface(seed=42).params.clustering.n_clusters == 5


def test_a_parameter_round_trips(phenome: PhenoMeInterface) -> None:
    phenome.params.prototypes.n_prototypes = 7
    assert phenome.params.prototypes.n_prototypes == 7


def test_the_criteria_start_unset(phenome: PhenoMeInterface) -> None:
    """Unset rather than empty, an empty criterion reading as one no image can answer."""
    assert phenome.params.explorer.filter is None
    assert phenome.params.explorer.exclude is None


def test_the_filter_holds_its_values_as_they_are(phenome: PhenoMeInterface) -> None:
    """PhenoMe matches them by equality, so a number picked on the cluster has to stay a number."""
    phenome.params.explorer.filter = {'drug_name': ['ctrl'], 'cluster': [0, 1]}

    assert phenome.params.explorer.filter == {'drug_name': ['ctrl'], 'cluster': [0, 1]}


def test_an_unknown_parameter_raises(phenome: PhenoMeInterface) -> None:
    """A mistyped name has to fail loudly rather than land in a slot nobody reads."""
    with pytest.raises(ValueError):
        phenome.params.clustering.nonexistent = 5


def test_an_out_of_range_parameter_raises(phenome: PhenoMeInterface) -> None:
    with pytest.raises(ValidationError):
        phenome.params.clustering.n_clusters = 0


def test_an_unknown_choice_raises(phenome: PhenoMeInterface) -> None:
    with pytest.raises(ValidationError):
        phenome.params.clustering.clustering_method = 'not-a-method'


def test_assigning_a_path_coerces_to_a_path(phenome: PhenoMeInterface) -> None:
    phenome.params.paths.checkpoint = '/tmp/somewhere.hdf5'
    assert phenome.params.paths.checkpoint == Path('/tmp/somewhere.hdf5')


def test_the_images_are_read_in_no_worker_by_default(phenome: PhenoMeInterface) -> None:
    """Every worker is a Python started from nothing, waited on before the first batch."""
    assert phenome.params.process.num_workers == 0


@pytest.mark.parametrize('count', [-1, MAX_WORKERS + 1])
def test_an_impossible_worker_count_raises(phenome: PhenoMeInterface, count: int) -> None:
    """Fewer than none is meaningless, and more than the form offers no machine would gain by."""
    with pytest.raises(ValidationError):
        phenome.params.process.num_workers = count


@pytest.mark.parametrize('count', [0, 1, MAX_WORKERS])
def test_a_possible_worker_count_is_kept(phenome: PhenoMeInterface, count: int) -> None:
    phenome.params.process.num_workers = count

    assert phenome.params.process.num_workers == count


@pytest.fixture
def extraction(phenome: PhenoMeInterface, monkeypatch: pytest.MonkeyPatch) -> dict:
    """ Catch the arguments an extraction would be run under, rather than running one.

    The interface hands a model's own arguments down alongside the form's, so the interface is made to
    look as it does once a model has been loaded: one that brings no arguments of its own.

    Returns:
        dict: Filled with the keyword arguments handed down, once an extraction has been asked for.
    """
    phenome.model = object()
    phenome.process_options = {}

    passed = {}

    def _capture(model, checkpoint_path, **kwargs) -> None:
        passed.update(kwargs)

    monkeypatch.setattr(phenome.image_processing, 'process_images', _capture)
    return passed


@pytest.mark.parametrize('count', [0, 2, MAX_WORKERS])
def test_the_worker_count_reaches_phenome(phenome: PhenoMeInterface, extraction: dict, count: int) -> None:
    """The whole point of the parameter: a count set in the form and never passed down does nothing."""
    phenome.params.process.num_workers = count

    phenome.process_images()

    assert extraction['num_workers'] == count


def test_the_batch_size_still_reaches_phenome(phenome: PhenoMeInterface, extraction: dict) -> None:
    """Handed down beside the worker count, and neither may push the other out."""
    phenome.params.process.batch_size = 64

    phenome.process_images()

    assert extraction['batch_size'] == 64


# ----------------------------------------------------------------------------
# Data loading and metadata access
# ----------------------------------------------------------------------------


def test_load_data_csv_finds_every_image(loaded: PhenoMeInterface) -> None:
    assert len(loaded.metadata_df) == 6
    assert {'file_path', 'mask_path', 'drug_name', 'time'} <= set(loaded.metadata_df.columns)


def test_metadata_columns_are_empty_before_loading(phenome: PhenoMeInterface) -> None:
    assert phenome.get_metadata_columns() == []


def test_color_by_columns_hold_the_metadata_then_the_properties(loaded: PhenoMeInterface, processed) -> None:
    """A property is as worth colouring by as a metadata column, which is how PhenoMe fills the same list."""
    results = processed(loaded, 6)
    results.properties = [{'area': 12.0, 'solidity': 0.9}] * 6

    assert loaded.get_color_by_columns() == [*loaded.get_metadata_columns(), 'area', 'solidity']


def test_color_by_columns_are_the_metadata_until_the_properties_are_computed(loaded: PhenoMeInterface) -> None:
    assert loaded.get_color_by_columns() == loaded.get_metadata_columns()


def test_color_by_columns_are_empty_before_loading(phenome: PhenoMeInterface) -> None:
    assert phenome.get_color_by_columns() == []


def test_color_by_columns_pick_up_the_cluster_labels(loaded: PhenoMeInterface) -> None:
    """The clustering writes them into the metadata, so they need no case of their own."""
    loaded.metadata_df['cluster'] = [0, 1] * 3

    assert 'cluster' in loaded.get_color_by_columns()


def test_a_column_holding_one_value_is_not_offered(loaded: PhenoMeInterface) -> None:
    """It would paint every point the same colour, saying nothing."""
    loaded.metadata_df['batch'] = 'first'

    assert 'batch' not in loaded.get_color_by_columns()
    assert 'drug_name' in loaded.get_color_by_columns()


def test_a_column_naming_each_image_is_not_offered(loaded: PhenoMeInterface) -> None:
    """One colour per image says as little as one colour for all of them, and draws far more series."""
    rows = 20
    loaded.metadata_df = pd.DataFrame(
        {
            'file_path': [f'/images/img_{row}.tif' for row in range(rows)],
            'filename': [f'img_{row}' for row in range(rows)],
            'drug_name': ['ctrl', 'drugA'] * (rows // 2),
        }
    )

    assert loaded.get_color_by_columns() == ['drug_name']


def test_an_identifier_survives_a_dataset_too_small_to_judge_it(loaded: PhenoMeInterface) -> None:
    """Under a handful of images there is no telling an identifier from a grouping key by uniqueness."""
    assert len(loaded.metadata_df) <= 10

    assert 'filename' in loaded.get_color_by_columns()


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
# Letting go of the results when another dataset is loaded
# ----------------------------------------------------------------------------

RESULT_FRAMES = ['properties_df', 'clustering_df', 'prototypes_df', 'outliers_df', 'projection_df']
SOURCE_SECTIONS = ['clustering', 'prototypes', 'outliers', 'explorer']


def analysed(interface: PhenoMeInterface, processed) -> PhenoMeInterface:
    """ Leave a result of every kind on an interface, as a full session would.

    Args:
        interface (PhenoMeInterface): The interface to fill, whose paths are already pointed at a dataset.
        processed: The fixture making the pipeline report processed images.
    Returns:
        PhenoMeInterface: The same interface, now holding results and parameters naming them.
    """
    processed(interface, 6)
    for attribute in RESULT_FRAMES:
        setattr(interface, attribute, pd.DataFrame({'image_path': ['img_000.tif']}))

    interface.model = 'a loaded model'
    interface.params.paths.checkpoint = Path('previous.hdf5')
    interface.params.explorer.color_by = 'drug_name'
    interface.params.explorer.filter = {'drug_name': ['ctrl']}
    interface.params.explorer.exclude = {'time': [24]}
    interface.params.prototypes.group_by = 'drug_name'
    interface.params.outliers.group_by = 'time'
    for section in SOURCE_SECTIONS:
        getattr(interface.params, section).source = 'properties'

    return interface


def test_has_results_is_false_before_anything_is_computed(phenome: PhenoMeInterface) -> None:
    assert phenome.has_results is False


def test_has_results_ignores_the_metadata(loaded: PhenoMeInterface) -> None:
    """Replacing it is what loading does, rather than something loading costs."""
    assert loaded.has_results is False


@pytest.mark.parametrize('attribute', RESULT_FRAMES)
def test_has_results_notices_each_kind_of_result(phenome: PhenoMeInterface, attribute: str) -> None:
    setattr(phenome, attribute, pd.DataFrame({'image_path': []}))

    assert phenome.has_results is True


def test_has_results_notices_the_embeddings(phenome: PhenoMeInterface, processed) -> None:
    processed(phenome, 6)

    assert phenome.has_results is True


@pytest.mark.parametrize('attribute', RESULT_FRAMES)
def test_loading_drops_every_computed_frame(phenome: PhenoMeInterface, processed, attribute: str) -> None:
    analysed(phenome, processed)

    phenome.load_data_csv()

    assert getattr(phenome, attribute) is None


def test_loading_empties_the_pipeline(phenome: PhenoMeInterface, processed) -> None:
    """PhenoMe.find_files() only replaces the files it found, leaving the embeddings behind them."""
    analysed(phenome, processed)

    phenome.load_data_csv()

    assert phenome.pheno.results.n_images == 0
    assert phenome.has_results is False


def test_loading_puts_the_checkpoint_back(phenome: PhenoMeInterface, processed) -> None:
    """A checkpoint answers only for the images it was filled from."""
    analysed(phenome, processed)

    phenome.load_data_csv()

    assert phenome.params.paths.checkpoint == DEFAULT_CHECKPOINT


def test_loading_drops_the_columns_and_values_the_previous_metadata_named(
    phenome: PhenoMeInterface, processed
) -> None:
    analysed(phenome, processed)

    phenome.load_data_csv()

    assert phenome.params.explorer.color_by is None
    assert phenome.params.explorer.filter is None
    assert phenome.params.explorer.exclude is None
    assert phenome.params.prototypes.group_by is None
    assert phenome.params.outliers.group_by is None


@pytest.mark.parametrize('section', SOURCE_SECTIONS)
def test_loading_puts_every_feature_space_back(
    phenome: PhenoMeInterface, processed, section: str
) -> None:
    """The properties a source reads went with the results, and the step would raise on them."""
    analysed(phenome, processed)

    phenome.load_data_csv()

    assert getattr(phenome.params, section).source == 'embeddings'


def test_loading_keeps_the_loaded_model(phenome: PhenoMeInterface, processed) -> None:
    """A downloaded network is not read off the images, and would have to be fetched again."""
    analysed(phenome, processed)

    phenome.load_data_csv()

    assert phenome.model == 'a loaded model'


def test_loading_keeps_the_paths_it_is_about_to_read(phenome: PhenoMeInterface, processed, dataset) -> None:
    analysed(phenome, processed)

    phenome.load_data_csv()

    assert phenome.params.paths.images == dataset['images']
    assert phenome.params.paths.metadata == dataset['metadata']


def test_loading_from_the_paths_clears_the_results_too(
    phenome: PhenoMeInterface, processed, path_dataset
) -> None:
    analysed(phenome, processed)
    phenome.params.paths.images = path_dataset['images']
    phenome.params.paths.masks = path_dataset['masks']
    phenome.params.paths.template = path_dataset['template']

    phenome.load_data_path()

    assert len(phenome.metadata_df) == 6
    assert phenome.has_results is False


def test_a_load_that_raises_still_leaves_nothing_behind(phenome: PhenoMeInterface, processed) -> None:
    """The results are dropped before a single file is read, so a failed load cannot keep them alive."""
    analysed(phenome, processed)
    phenome.params.paths.metadata = Path('no_such_file.csv')

    with pytest.raises(FileNotFoundError):
        phenome.load_data_csv()

    assert phenome.metadata_df is None
    assert phenome.has_results is False


def test_clearing_a_fresh_interface_changes_nothing(phenome: PhenoMeInterface) -> None:
    """Loading the first dataset of a session goes through the same path as loading the next one."""
    phenome.clear_results()

    assert phenome.has_results is False
    assert phenome.metadata_df is None


# ----------------------------------------------------------------------------
# Clustering, and the labels it writes back into the metadata
# ----------------------------------------------------------------------------


def test_compute_clustering_forwards_its_parameters(loaded: PhenoMeInterface, spy) -> None:
    loaded.pheno.compute_clustering = spy(pd.DataFrame({'image_path': [], 'cluster': []}))
    loaded.params.clustering.clustering_method = 'dbscan'
    loaded.params.clustering.source = 'combined'
    loaded.params.clustering.n_clusters = 7
    loaded.params.clustering.dbscan_eps = 0.25
    loaded.params.clustering.dbscan_min_samples = 11

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
    loaded.params.prototypes.group_by = 'drug_name'
    loaded.params.prototypes.n_prototypes = 4
    loaded.params.prototypes.source = 'properties'

    loaded.get_prototype_images()

    assert loaded.pheno.find_prototypes.last == {
        'cluster_col': 'drug_name',
        'n_prototypes': 4,
        'source': 'properties',
        'plot': False,
    }


def test_outliers_forward_their_parameters(loaded: PhenoMeInterface, spy) -> None:
    loaded.pheno.detect_outliers = spy(pd.DataFrame())
    loaded.params.outliers.group_by = 'time'
    loaded.params.outliers.method = 'iqr'
    loaded.params.outliers.threshold = 1.5
    loaded.params.outliers.source = 'combined'

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


# ----------------------------------------------------------------------------
# The explorer projection, whose names are the GUI's rather than PhenoMe's
# ----------------------------------------------------------------------------


def coordinates(n_components: int = 2) -> pd.DataFrame:
    """ A frame shaped like the one the reduction returns. """
    return pd.DataFrame({f'Component {i + 1}': [0.0, 1.0] for i in range(n_components)})


def test_the_projection_translates_its_parameters(loaded: PhenoMeInterface, reduction) -> None:
    stand_in = reduction(coordinates(3))
    loaded.params.explorer.method = 't-SNE'
    loaded.params.explorer.dims = '3D'
    loaded.params.explorer.source = 'properties'

    loaded.compute_projection()

    assert stand_in.pipelines[-1] is loaded.pheno
    assert stand_in.last == {
        'method': 'tsne',
        'n_components': 3,
        'source': 'properties',
        'filters': None,
        'exclude': None,
    }


def test_the_projection_is_kept_to_the_filtered_images(loaded: PhenoMeInterface, reduction) -> None:
    """PhenoMe names it in the plural, and takes the values as the metadata holds them."""
    stand_in = reduction(coordinates())
    loaded.params.explorer.filter = {'drug_name': ['ctrl'], 'cluster': [0, 1]}

    loaded.compute_projection()

    assert stand_in.last['filters'] == {'drug_name': ['ctrl'], 'cluster': [0, 1]}


def test_the_projection_leaves_the_excluded_images_out(loaded: PhenoMeInterface, reduction) -> None:
    """The exclusion goes down under its own name, PhenoMe applying it after the filter."""
    stand_in = reduction(coordinates())
    loaded.params.explorer.filter = {'drug_name': ['ctrl', 'drugA']}
    loaded.params.explorer.exclude = {'time': [24]}

    loaded.compute_projection()

    assert stand_in.last['filters'] == {'drug_name': ['ctrl', 'drugA']}
    assert stand_in.last['exclude'] == {'time': [24]}


@pytest.mark.parametrize(
    ('method', 'phenome_name'),
    [('t-SNE', 'tsne'), ('PCA', 'pca'), ('UMAP', 'umap')],
)
def test_every_method_goes_down_under_its_phenome_name(
    loaded: PhenoMeInterface, reduction, method: str, phenome_name: str
) -> None:
    """The GUI spells them as the papers do, where PhenoMe knows them by a plain lower case key."""
    stand_in = reduction(coordinates())
    loaded.params.explorer.method = method

    loaded.compute_projection()

    assert stand_in.last['method'] == phenome_name


@pytest.mark.parametrize(('dims', 'n_components'), [('2D', 2), ('3D', 3)])
def test_the_canvas_is_read_as_a_number_of_components(
    loaded: PhenoMeInterface, reduction, dims: str, n_components: int
) -> None:
    stand_in = reduction(coordinates(n_components))
    loaded.params.explorer.dims = dims

    loaded.compute_projection()

    assert stand_in.last['n_components'] == n_components


def test_the_translations_cover_every_choice() -> None:
    """A choice added to the parameter model without a translation would raise on the first compute."""
    assert set(DR_METHODS) == set(EMBEDDING_METHOD_CHOICES)
    assert set(DIMS_COMPONENTS) == set(DIMS_CHOICES)


def test_the_projection_is_stored_and_returned(loaded: PhenoMeInterface, reduction) -> None:
    frame = coordinates()
    reduction(frame)

    returned = loaded.compute_projection()

    assert returned is loaded.projection_df
    assert returned is frame


def test_a_projected_point_is_resolved_through_its_index(loaded: PhenoMeInterface, reduction, processed) -> None:
    """The projection names each image but does not carry its path, only where PhenoMe holds it."""
    paths = list(loaded.metadata_df['file_path'])
    processed(loaded, 6).img_path = paths

    frame = coordinates()
    frame['Index'] = [4, 0]
    reduction(frame)
    loaded.compute_projection()

    image_paths, mask_paths, metadata = loaded.get_projection_image_data(0)

    # The row is not the image: row 0 of the projection is the fifth image PhenoMe holds
    assert image_paths == [paths[4]]
    assert mask_paths is not None
    assert metadata[0]['drug_name'] == loaded.metadata_df.iloc[4]['drug_name']


def test_an_image_is_found_by_where_phenome_holds_it(loaded: PhenoMeInterface, processed) -> None:
    """What a selection is made of: the place, with neither the path nor a row of the projection."""
    paths = list(loaded.metadata_df['file_path'])
    processed(loaded, 6).img_path = paths

    image_paths, mask_paths, metadata = loaded.get_image_data_at(3)

    assert image_paths == [paths[3]]
    assert mask_paths is not None
    assert metadata[0]['drug_name'] == loaded.metadata_df.iloc[3]['drug_name']


def test_an_image_of_several_channels_is_found_by_its_first(loaded: PhenoMeInterface, processed) -> None:
    paths = list(loaded.metadata_df['file_path'])
    processed(loaded, 6).img_path = [[path, path] for path in paths]

    image_paths, _, _ = loaded.get_image_data_at(1)

    assert image_paths == [paths[1]]


def test_a_projected_point_of_a_multi_channel_image(loaded: PhenoMeInterface, reduction, processed) -> None:
    """PhenoMe holds a list of paths for the images it reads a channel at a time."""
    paths = list(loaded.metadata_df['file_path'])
    processed(loaded, 6).img_path = [[path, path] for path in paths]

    frame = coordinates()
    frame['Index'] = [2, 0]
    reduction(frame)
    loaded.compute_projection()

    image_paths, _, _ = loaded.get_projection_image_data(0)

    assert image_paths == [paths[2]]


def test_nothing_to_project_leaves_no_projection(loaded: PhenoMeInterface, reduction) -> None:
    """An empty answer is a normal outcome, the pipeline holding no embeddings until they are extracted."""
    reduction(None)

    assert loaded.compute_projection() is None
    assert loaded.projection_df is None
