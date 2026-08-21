from logging import getLogger

import pandas as pd
from phenome import PhenoMe
from phenome.utils import ProgressCallback
from torch import cuda

from ..registries.models import MODEL_OPTIONS
from .analysis import PhenoMeAnalysis
from .data_loading import DataLoading
from .image_processing import ImageProcessing
from .params import Params

logger = getLogger(__name__)

# The GUI names its algorithms as the papers do, where PhenoMe knows them by a plain lower case key.
# Keyed by the EmbeddingMethod choices, so every method the form can offer has a translation.
DR_METHODS = {'t-SNE': 'tsne', 'PCA': 'pca', 'UMAP': 'umap'}

# The GUI picks a canvas to draw on, which PhenoMe reads as a number of components to reduce to.
DIMS_COMPONENTS = {'2D': 2, '3D': 3}

# Columns naming one image rather than describing it, which colour every point differently and say
# nothing by it. Dropped only from a dataset large enough for their uniqueness to mean something, the
# same way PhenoMe's build_informative_color_columns() reads them.
IDENTIFIER_COLUMNS = {'id', 'filename', 'file_path', 'path', 'index'}
IDENTIFIER_UNIQUENESS = 0.9
IDENTIFIER_MIN_ROWS = 10

# Column of a projection holding where each of its rows sits in the PhenoMe results, which is the
# only way back from a projected point to the image it stands for: the projection carries the name of
# the image but not its path.
PROJECTION_INDEX_COLUMN = 'Index'


class PhenoMeInterface:
    """
    Single entry point to PhenoMe for the GUI.

    Holds the pipeline object, the parameters and the metadata dataframe, and
    delegates the actual PhenoMe calls to one component per tab (`DataLoading`,
    and later the processing and analysis ones).
    """

    def __init__(self, seed: int = 42) -> None:
        """ Initialize the PhenoMe pipeline, the parameters and the per-tab components.

        Args:
            seed (int): Random seed passed to the PhenoMe pipeline, for reproducibility.
        """
        self.pheno = PhenoMe(seed=seed)
        self.params = Params()
        self.metadata_df = None
        self.properties_df = None
        self.clustering_df = None
        self.prototypes_df = None
        self.outliers_df = None
        self.projection_df = None
        self.model = None
        self.process_options = None
        self.transforms = None
        self.cuda_available = cuda.is_available()

        self.data_loading = DataLoading(self.pheno)
        self.image_processing = ImageProcessing(self.pheno)
        self.phenome_analysis = PhenoMeAnalysis(self.pheno)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data_csv(self) -> pd.DataFrame:
        """ Load the dataset using a .csv file as metadata source.

        Returns:
            pd.DataFrame: The metadata of the files found, also stored in self.metadata_df.
        """
        images_folder = str(self.params.paths.images)
        masks_folder = str(self.params.paths.masks)
        metadata_csv = str(self.params.paths.metadata)
        self.metadata_df = self.data_loading.load_data_csv(images_folder, metadata_csv, masks_folder)
        return self.metadata_df

    def load_data_path(self) -> pd.DataFrame:
        """ Load the dataset by extracting the metadata from the file paths.

        Returns:
            pd.DataFrame: The metadata of the files found, also stored in self.metadata_df.
        """
        images_folder = str(self.params.paths.images)
        masks_folder = str(self.params.paths.masks)
        template = self.params.paths.template
        self.metadata_df = self.data_loading.load_data_path(images_folder, template, masks_folder)
        return self.metadata_df

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """ Find the real model name from the user-friendly name and load it in the PhenoMe pipeline.

        A model arrives with the transformations its images are to be prepared with and the arguments
        its embeddings are to be extracted under, both read from the registries by the name it is
        known by. They are kept here until the images are processed, which is where they are used, so
        both are written to the log as they arrive: what a run was made under is otherwise nowhere.
        """
        model_name = self.params.process.model

        real_model_name = MODEL_OPTIONS.get(model_name)
        if real_model_name:
            device = self.params.process.device
            self.model, self.transforms, self.process_options = self.image_processing.load_model(real_model_name, device=device)

            logger.info("Loaded '%s' (%s) on the %s.", model_name, real_model_name, device)
            logger.info(
                "Its images are prepared with %s, and processed with %s.",
                'its own transformations' if self.transforms is not None else "PhenoMe's own transformations",
                str(self.process_options) if self.process_options else 'no arguments of its own'
            )
        else:
            logger.error("Model name '%s' not recognized. Available options: %s", model_name, list(MODEL_OPTIONS.keys()))

    def process_images(self, progress_callback: ProgressCallback | None=None) -> None:
        """ Process the images using PhenoMe.process_images(). Saves the embeddings in-place in the PhenoMe object.

        Uses the params to feed arguments.

        TODO: Include the remaining advanced parameters from the form.

        Args:
            progress_callback: Hook called as (current, total, desc) after each batch. Called from the
            calling thread, so a caller running this in a worker thread must not touch widgets in it.
        """
        checkpoint_path = self.params.paths.checkpoint

        # Written out rather than left to the model that was loaded: the arguments an extraction ran
        # under are what its embeddings can be read against, and they are the registry's rather than
        # anything the form shows.
        logger.info(
            "Extracting embeddings with '%s', in batches of %d, %s.",
            self.params.process.model,
            self.params.process.batch_size,
            str(self.process_options) if self.process_options else 'with no arguments of its own'
        )

        self.image_processing.process_images(
            self.model,
            checkpoint_path,
            progress_callback=progress_callback,
            batch_size=self.params.process.batch_size,
            custom_transformations=self.transforms,
            **self.process_options
        )

    @property
    def has_processed_images(self) -> bool:
        """ Whether images have been processed, which the properties step needs to run.

        Reads the same state PhenoMe itself checks in compute_properties(), rather than tracking
        the processing step separately.

        Returns:
            bool: True once the pipeline holds processed images.
        """
        return self.pheno.results.n_images > 0

    def compute_properties(self, progress_callback: ProgressCallback | None=None) -> pd.DataFrame:
        """ Compute the per-image properties using PhenoMe.compute_properties(). Stores them in self.properties_df.

        Uses the params to feed arguments.

        TODO: Computing a different preset over an existing checkpoint may need force_update=True, otherwise
        PhenoMe warns about the property keys missing from the checkpoint.

        Args:
            progress_callback: Hook called as (current, total, desc) after each image. Called from the
                calling thread, so a caller running this in a worker thread must not touch widgets in it.
        Returns:
            pd.DataFrame: One row per image, holding the image path and the computed properties.
        """
        checkpoint_path = self.params.paths.checkpoint

        # PhenoMe only knows its presets by their lower case name, while the GUI displays them capitalized
        preset = self.params.properties.preset.lower()

        self.properties_df = self.image_processing.compute_properties(
            preset, checkpoint_path, progress_callback=progress_callback
        )
        return self.properties_df

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def compute_clustering(self) -> pd.DataFrame:
        """ Cluster the images using PhenoMe.compute_clustering(). Stores the labels in self.clustering_df.

        Uses the params to feed arguments. The fields of the 'clustering' section are named after the
        PhenoMe arguments they stand for, so they are passed straight through. PhenoMe ignores the ones
        that do not apply to the chosen method, and falls back on the pipeline seed for the random state.

        The labels are also added to the metadata dataframe as a 'cluster' column, which puts them in the
        metadata table and makes them available as a grouping key like any other metadata column.

        Returns:
            pd.DataFrame: One row per image, holding the image path and its cluster label.
        """
        clustering = self.params.clustering

        self.clustering_df = self.phenome_analysis.compute_clustering(
            n_clusters=clustering.n_clusters,
            clustering_method=clustering.clustering_method,
            source=clustering.source,
            dbscan_eps=clustering.dbscan_eps,
            dbscan_min_samples=clustering.dbscan_min_samples
        )
        self._append_clusters_to_metadata()
        return self.clustering_df

    def _append_clusters_to_metadata(self) -> None:
        """ Add the cluster labels of the last clustering run to the metadata dataframe.

        Matched on the image path rather than on the row order, as PhenoMe skips the images it fails to
        load and the labels would otherwise be shifted onto the wrong rows. Images PhenoMe never saw are
        left empty. Re-running the clustering overwrites the column, and loading data again drops it, as
        the dataframe is then rebuilt from scratch.
        """
        if self.metadata_df is None or self.metadata_df.empty:
            return
        if self.clustering_df is None or self.clustering_df.empty:
            return

        labels = dict(zip(self.clustering_df['image_path'], self.clustering_df['cluster'], strict=True))

        # PhenoMe leaves the label empty for the DBSCAN noise points, which would turn the whole column
        # to floats and show the labels as '0.0'. The nullable integer type keeps them whole.
        self.metadata_df['cluster'] = self.metadata_df['file_path'].map(labels).astype('Int64')

    def get_prototype_images(self) -> pd.DataFrame:
        """ Find the prototype images of each group using PhenoMe.find_prototypes(). Stores them in self.prototypes_df.

        Uses the params to feed arguments. The images are grouped by the metadata column selected in the
        parameter form, which is None until data has been loaded and a column picked. PhenoMe reads no
        grouping key as a request to treat every image as one group.

        The GUI calls the grouping key 'group_by' for both this step and the outlier one, while PhenoMe
        names it 'cluster_col' here and 'group_by' there, so the name is translated on the way down.

        Returns:
            pd.DataFrame: One row per prototype, holding the grouping key, the image path and its rank in the group.
        """
        prototypes = self.params.prototypes

        self.prototypes_df = self.phenome_analysis.get_prototype_images(
            cluster_col=prototypes.group_by,
            n_prototypes=prototypes.n_prototypes,
            source=prototypes.source
        )
        return self.prototypes_df

    def get_outlier_images(self) -> pd.DataFrame:
        """ Detect the outlier images of each group using PhenoMe.detect_outliers(). Stores them in self.outliers_df.

        Groups the images by its own metadata column, the same way the prototype step does. PhenoMe
        returns every image past the distance threshold, so there is no count to set here: how many
        outliers come back is decided by the method and the threshold together.

        Returns:
            pd.DataFrame: One row per outlier, holding the grouping key, the image path and its distance to the group centroid.
        """
        outliers = self.params.outliers

        self.outliers_df = self.phenome_analysis.get_outlier_images(
            group_by=outliers.group_by,
            method=outliers.method,
            threshold=outliers.threshold,
            source=outliers.source
        )
        return self.outliers_df

    def compute_projection(self) -> pd.DataFrame:
        """ Project the images down for the explorer plot. Stores the coordinates in self.projection_df.

        Uses the params to feed arguments. Both the algorithm and the number of dimensions are named
        for the GUI rather than for PhenoMe - the papers' spelling for the one, a kind of canvas for
        the other - so both are translated on the way down. The filter and the exclusion go down as
        they are, PhenoMe naming the first in the plural and reading an unset one and an empty one
        the same way.

        Returns:
            pd.DataFrame: One row per image, holding its coordinates, name, index and metadata. None
                when there was nothing to project.
        """
        explorer = self.params.explorer

        self.projection_df = self.phenome_analysis.run_dimensionality_reduction(
            method=DR_METHODS[explorer.method],
            n_components=DIMS_COMPONENTS[explorer.dims],
            source=explorer.source,
            filters=explorer.filter,
            exclude=explorer.exclude
        )
        return self.projection_df

    # ------------------------------------------------------------------
    # Metadata access
    # ------------------------------------------------------------------

    def get_metadata_columns(self) -> list[str]:
        """ Return the metadata columns that can be used to group the images.

        Leaves out the file_path and mask_path columns PhenoMe.find_files() adds itself, as they hold one
        distinct value per row and are of no use as grouping keys. Every remaining column comes from the
        metadata source chosen in the loading tab.

        Returns:
            list[str]: The groupable column names, empty when no data has been loaded yet.
        """
        if self.metadata_df is None or self.metadata_df.empty:
            return []

        reserved = {'file_path', 'mask_path'}
        return [column for column in self.metadata_df.columns if column not in reserved]

    def get_color_by_columns(self) -> list[str]:
        """ Return everything the explorer plot can colour its points by.

        The metadata columns and the computed properties together, which is how PhenoMe fills the
        same dropdown in its own explorer: metadata says what an image is, properties say what was
        measured on it, and either can be worth colouring by. The metadata comes first, the
        properties after it in the order PhenoMe holds them.

        The columns that cannot inform a colouring are left out, as PhenoMe leaves them out: one
        holding a single value paints every point alike, and one naming each image paints them all
        differently. The naming ones are only dropped from a dataset big enough for that to be
        telling, a handful of images being too few to tell an identifier from a grouping key.

        Returns:
            list[str]: The columns to offer, empty when no data has been loaded yet.
        """
        columns = [column for column in self.get_metadata_columns() if self._is_informative(column)]

        # Properties are measurements, so they are neither identifiers nor, in practice, constant
        properties = self.pheno.get_available_property_keys()

        return columns + [key for key in properties if key not in columns]

    def _is_informative(self, column: str) -> bool:
        """ Whether a metadata column says enough about the images to be worth colouring by.

        Args:
            column (str): Name of the column, which is read alongside its values: only the columns
                named after an identifier are held to the uniqueness test.
        Returns:
            bool: False for a column holding one value throughout, or one naming each image.
        """
        values = self.metadata_df[column]
        distinct = values.nunique(dropna=True)

        if distinct <= 1:
            return False

        if column.lower() in IDENTIFIER_COLUMNS and len(values) > IDENTIFIER_MIN_ROWS:
            return distinct <= IDENTIFIER_UNIQUENESS * len(values)

        return True

    def get_row_display_data(self, row: int) -> tuple[list[str], list[str] | None, list[dict]]:
        """ Return the content of a metadata row, in the form expected by the image viewer.

        Args:
            row (int): Positional index of the row in the metadata dataframe.
        Returns:
            tuple: The image paths, the mask paths (None when the dataset has no masks) and the row data as a list of dicts.
        """
        row_data = self.metadata_df.iloc[row]

        image_paths = [row_data['file_path']]
        if 'mask_path' in row_data:
            mask_paths = [row_data['mask_path']]
        else:
            mask_paths = None

        return image_paths, mask_paths, [row_data.to_dict()]

    def get_image_display_data(self, image_path: str) -> tuple[list[str], list[str] | None, list[dict]]:
        """ Return the content of a metadata row found by image path, in the form expected by the image viewer.

        The analysis steps return the path of the images they retrieved but no mask, so the mask and the
        rest of the metadata are looked up here. Matched on the path rather than on the row position, as
        the returned images are indexed against the PhenoMe results rather than against the metadata.

        Args:
            image_path (str): Path of the image, as returned by the prototype or outlier step.
        Returns:
            tuple: The image paths, the mask paths (None when the dataset has no masks) and the row data as a list of dicts.
        """
        if self.metadata_df is None or self.metadata_df.empty:
            return [image_path], None, [{}]

        matches = self.metadata_df[self.metadata_df['file_path'] == image_path]
        if matches.empty:
            # PhenoMe returns whatever its results hold, which can outlive a reload of the metadata
            logger.warning("Image '%s' not found in the loaded metadata", image_path)
            return [image_path], None, [{}]

        row_data = matches.iloc[0]
        mask_paths = [row_data['mask_path']] if 'mask_path' in row_data else None

        return [image_path], mask_paths, [row_data.to_dict()]

    def get_projection_image_data(self, row: int) -> tuple[list[str], list[str] | None, list[dict]]:
        """ Return the content of a projection row, in the form expected by the image viewer.

        A projected point is one image, but the projection only carries its name and the place it is
        held at in the PhenoMe results, so the path is looked up from that place and the mask and the
        metadata from the path, the way a retrieved image has them looked up.

        Args:
            row (int): Positional index of the row in the projection dataframe.
        Returns:
            tuple: The image paths, the mask paths (None when the dataset has no masks) and the row data as a list of dicts.
        """
        return self.get_image_data_at(int(self.projection_df.iloc[row][PROJECTION_INDEX_COLUMN]))

    def get_image_data_at(self, index: int) -> tuple[list[str], list[str] | None, list[dict]]:
        """ Return the content of one image, found by where PhenoMe holds it, for the image viewer.

        The place is what a projection carries and what a selection is made of, neither of them
        holding the path: PhenoMe writes the name of an image and its index and no more, so the path
        is looked up here and the mask and the metadata from the path.

        Args:
            index (int): Place of the image in the PhenoMe results.
        Returns:
            tuple: The image paths, the mask paths (None when the dataset has no masks) and the row data as a list of dicts.
        """
        image_path = self.pheno.results.img_path[index]

        # PhenoMe holds a list of paths rather than one for the images it reads a channel at a time
        if isinstance(image_path, list | tuple):
            image_path = image_path[0]

        return self.get_image_display_data(str(image_path))

    def get_metadata_table(self) -> "pd.Dataframe":
        """ Get the metadata dataframe in its current state.

        Returns:
            pd.Dataframe: The metadata dataframe
        """

        return self.metadata_df

    @property
    def has_properties(self) -> bool:
        """ Whether the properties have been computed, which the property-based analysis sources need.

        Returns:
            bool: True if properties have been calculated
        """
        properties_available = self.properties_df is not None
        logger.info('Properties are available: %s', properties_available)
        return properties_available
