from logging import getLogger
from pathlib import Path

import pandas as pd
from phenome import PhenoMe
from phenome.utils import ProgressCallback
from torch import cuda

from ..registries.models import MODEL_OPTIONS
from ..registries.process_options import OPTIONS_LOADERS
from .analysis import PhenoMeAnalysis
from .checkpoint import CheckpointMismatch, inspect_checkpoint
from .data_loading import DataLoading
from .image_processing import ImageProcessing
from .params import Params

logger = getLogger(__name__)

# What PhenoMe.process_images() defaults the arguments the GUI never sets to. A checkpoint is validated
# against these alongside the ones a model brings, so they are written out here rather than left
# implicit: reading a checkpoint before a run means comparing it against what the run will really use.
# The channel mode is not among them, being the form's to set.
PROCESS_DEFAULTS = {
    'channels': None,
    'resize_size': 224,
    'pad_size': None,
    'force_rgb': True,
    'l2_normalize_channels': True,
}

# The GUI names its algorithms as the papers do, where PhenoMe knows them by a plain lower case key.
# Keyed by the EmbeddingMethod choices, so every method the form can offer has a translation.
DR_METHODS = {'t-SNE': 'tsne', 'PCA': 'pca', 'UMAP': 'umap'}

# The canvas draws a flat cloud, so the images are always reduced to the two components it plots.
PROJECTION_COMPONENTS = 2

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

# Metadata column the clustering writes its labels into, which puts them wherever the metadata goes
CLUSTER_COLUMN = 'cluster'


def _without_cluster(criteria: dict[str, list] | None) -> dict[str, list] | None:
    """ A projection criterion with the cluster labels taken out of it.

    Args:
        criteria: Values picked per metadata column, or None for no criterion.
    Returns:
        The same criterion without the cluster labels, or None when nothing else is left of it.
    """
    if not criteria:
        return None

    kept = {column: values for column, values in criteria.items() if column != CLUSTER_COLUMN}
    return kept or None


def _coverage_mismatch(held: int, matched: int, loaded: int) -> str:
    """ Say how a checkpoint fails to answer for the dataset loaded.

    Args:
        held: The images the checkpoint holds.
        matched: How many of them match a loaded file.
        loaded: How many files are loaded.
    Returns:
        One sentence naming the mismatch, with the counts it was read from.
    """
    if matched == 0:
        return (
            f'It was filled from a different dataset: it holds {held} image(s), none of them among '
            f'the {loaded} currently loaded.'
        )

    return (
        f'It does not match the loaded dataset: it holds {held} image(s), {matched} of which are '
        f'among the {loaded} currently loaded.'
    )


def _as_phenome_path(path: Path | None) -> str | None:
    """ A directory as PhenoMe takes it, or None when none was named.

    Stringified rather than handed over as a Path: find_files() reads its image directory as text,
    looking for the wildcards a glob would carry, and a Path raises there. None is kept as it is,
    being what says there is no directory at all - str(None) would be the path 'None', which PhenoMe
    would then go looking for.

    Args:
        path: The directory named, or None when the box was left empty.
    Returns:
        The directory as text, or None.
    """
    return None if path is None else str(path)



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
            seed: Random seed passed to the PhenoMe pipeline, for reproducibility.
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
        self.transforms = None

        # The model and the device the network in self.model was loaded for, as (GUI name, device),
        # or None before any load. Read to tell whether a run can reuse it or has to load again.
        self._loaded_model: tuple[str, str] | None = None
        self.cuda_available = cuda.is_available()

        self.data_loading = DataLoading(self.pheno)
        self.image_processing = ImageProcessing(self.pheno)
        self.phenome_analysis = PhenoMeAnalysis(self.pheno)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def has_results(self) -> bool:
        """ Whether anything has been computed from the dataset currently loaded.

        The metadata is left out: replacing it is what loading a dataset does, rather than something
        loading one costs. Read before a load, to know whether there is anything to lose by it.

        Returns:
            True once a step has run on the loaded images.
        """
        return self.has_processed_images or any(
            frame is not None for frame in (
                self.properties_df,
                self.clustering_df,
                self.prototypes_df,
                self.outliers_df,
                self.projection_df,
            )
        )

    def clear_results(self) -> None:
        """ Drop everything computed from the dataset currently loaded.

        Called at the start of every load, so that what the interface holds always answers for the
        files currently loaded: PhenoMe.find_files() only replaces the files it found, and would
        otherwise leave the embeddings and the properties of the previous dataset behind them.

        The loaded model is kept, being a downloaded network rather than anything read off the
        images. Everything else goes, the parameters naming a result included.
        """
        # Closes the open checkpoint, and empties the embeddings, the properties and the found files
        self.pheno.reset(clear_file_df=True)

        self.metadata_df = None
        self.properties_df = None
        self.clustering_df = None
        self.prototypes_df = None
        self.outliers_df = None
        self.projection_df = None

        # A checkpoint holds the results of the images it was filled from, and would be read back for
        # the next dataset as though it answered for those images too. Nothing is processed until
        # another has been created or loaded against the images now loaded.
        self.params.paths.checkpoint = None

        # Columns and values of the metadata just dropped, which name nothing in the next one
        self.params.explorer.color_by = None
        self.params.explorer.filter = None
        self.params.explorer.exclude = None
        self.params.prototypes.group_by = None
        self.params.outliers.group_by = None

        self._reset_sources()

        logger.info('Cleared the results of the previously loaded dataset.')

    def clear_analysis(self) -> None:
        """ Drop everything computed from the embeddings, which a new run is about to replace.

        The dataset and the selected checkpoint stay, as does what the pipeline holds: the run replaces
        the embeddings and the properties itself, PhenoMe emptying them as it starts. What goes is the
        properties frame and the analysis results read off the embeddings - the cluster labels, which
        also leave the metadata, the retrieved prototypes and outliers, and the projection - along with
        the parameters naming them.
        """
        self.properties_df = None
        self.clustering_df = None
        self.prototypes_df = None
        self.outliers_df = None
        self.projection_df = None

        if self.metadata_df is not None and CLUSTER_COLUMN in self.metadata_df.columns:
            self.metadata_df = self.metadata_df.drop(columns=CLUSTER_COLUMN)

        # The cluster labels were a metadata column like any other, so the parameters may name them
        explorer = self.params.explorer
        if explorer.color_by == CLUSTER_COLUMN:
            explorer.color_by = None
        explorer.filter = _without_cluster(explorer.filter)
        explorer.exclude = _without_cluster(explorer.exclude)
        for section in (self.params.prototypes, self.params.outliers):
            if section.group_by == CLUSTER_COLUMN:
                section.group_by = None

        self._reset_sources()

        logger.info('Cleared the analysis results computed from the previous embeddings.')

    def _reset_sources(self) -> None:
        """ Put each step's feature space back to the one needing no properties.

        The parameter forms close the choice when there are no properties, but leave whatever was
        picked before standing, and the step would then be run against properties that are no longer
        there.
        """
        for section in (
            self.params.clustering, self.params.prototypes, self.params.outliers, self.params.explorer
        ):
            section.source = type(section).model_fields['source'].default

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data_csv(self) -> pd.DataFrame:
        """ Load the dataset using a .csv file as metadata source.

        Anything computed from the dataset loaded before is dropped first, so the results never
        outlive the images they were computed from.

        Returns:
            The metadata of the files found, also stored in self.metadata_df.
        """
        self.clear_results()

        images_folder = _as_phenome_path(self.params.paths.images)
        masks_folder = _as_phenome_path(self.params.paths.masks)
        metadata_csv = self.params.paths.metadata
        self.metadata_df = self.data_loading.load_data_csv(images_folder, metadata_csv, masks_folder)
        return self.metadata_df

    def load_data_path(self) -> pd.DataFrame:
        """ Load the dataset by extracting the metadata from the file paths.

        Anything computed from the dataset loaded before is dropped first, so the results never
        outlive the images they were computed from.

        Returns:
            The metadata of the files found, also stored in self.metadata_df.
        """
        self.clear_results()

        images_folder = _as_phenome_path(self.params.paths.images)
        masks_folder = _as_phenome_path(self.params.paths.masks)
        template = self.params.paths.template
        self.metadata_df = self.data_loading.load_data_path(images_folder, template, masks_folder)
        return self.metadata_df

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """ Find the real model name from the user-friendly name and load it in the PhenoMe pipeline.

        A model arrives with the transformations its images are to be prepared with, read from the
        registries by the name it is known by. They are kept here until the images are processed,
        which is where they are used, and written to the log as they arrive: what a run was made
        under is otherwise nowhere.

        Raises:
            ValueError: The selected model is not one the registries know, so there is nothing to load.
        """
        model_name = self.params.process.model

        real_model_name = MODEL_OPTIONS.get(model_name)
        if not real_model_name:
            raise ValueError(
                f"Model name '{model_name}' not recognized. Available options: {list(MODEL_OPTIONS)}"
            )

        device = self.params.process.device

        # Forgotten before the load rather than after it, so a load that raises leaves nothing
        # claiming to be loaded
        self._loaded_model = None
        self.model, self.transforms, _ = self.image_processing.load_model(real_model_name, device=device)
        self._loaded_model = (model_name, device)

        logger.info("Loaded '%s' (%s) on the %s.", model_name, real_model_name, device)
        logger.info(
            "Its images are prepared with %s.",
            'its own transformations' if self.transforms is not None else "PhenoMe's own transformations",
        )

    @property
    def needs_model_load(self) -> bool:
        """ Whether the next run has to load a model before it can extract anything.

        True before any load, and whenever the model or the device has been picked again since: the
        network held is then not the one the run is to be made with.

        Returns:
            True when the loaded network does not answer for the selected model and device.
        """
        return self._loaded_model != (self.params.process.model, self.params.process.device)

    def ensure_model_loaded(self) -> bool:
        """ Load the selected model, unless the one already held answers for it.

        Called at the start of a run rather than when a model is picked: picking is only a choice,
        and the download is left to the moment it is needed.

        Returns:
            True when a model was loaded, False when the one held was kept.
        Raises:
            ValueError: The selected model is not one the registries know.
        """
        if not self.needs_model_load:
            return False

        self.load_model()
        return True

    def model_options(self) -> dict:
        """ The arguments the selected model's embeddings are extracted under, read from the registry.

        Read by the name selected rather than off a loaded model: they are fixed per model, and a
        checkpoint is compared against them before the run that loads it.

        Returns:
            The model's own process_images() arguments, or none for a model the registries do not know.
        """
        real_model_name = MODEL_OPTIONS.get(self.params.process.model)
        options_loader = OPTIONS_LOADERS.get(real_model_name)
        return options_loader() if options_loader else {}

    def process_arguments(self) -> dict:
        """ The arguments a run is made with: the model's own, under what the form settles.

        The form has the last word, so what was picked in it is what runs. Gathered here rather than
        spelled out at the call below, which is also what keeps a model naming the same argument from
        being handed down twice.

        Returns:
            The arguments handed to PhenoMe on top of the ones the run itself names.
        """
        return {**self.model_options(), 'channel_mode': self.params.process.channel_mode}

    def processing_params(self) -> dict:
        """ The arguments an embedding run would be made under, as PhenoMe stores them in a checkpoint.

        The arguments the run is really made with over PhenoMe's defaults, which is how
        process_images() below hands them down: what neither the form nor the model sets, PhenoMe
        fills in itself. Read before a run to compare a checkpoint against it, so the comparison is
        made against what the run will really use - the channel mode included, which is the first
        thing PhenoMe refuses to resume a checkpoint across.

        Returns:
            The processing parameters, whether or not a model has been loaded.
        """
        return {**PROCESS_DEFAULTS, **self.process_arguments()}

    def run_checkpoint(self) -> Path | None:
        """ The checkpoint a run writes to as it goes, if any.

        Only with lazy checkpointing on: by default a run keeps its results in memory, and the
        selected checkpoint merely names the file last saved or loaded, which the run never touches.

        Returns:
            The selected checkpoint when lazy checkpointing is on, None otherwise.
        """
        return self.params.paths.checkpoint if self.params.process.lazy_checkpoint else None

    def inspect_checkpoint(self) -> CheckpointMismatch | None:
        """ Read the checkpoint an embedding run would be written to, before the run is started.

        PhenoMe raises on a checkpoint it cannot resume from, from the thread the run is made on and
        halfway through it. Asked here beforehand, the same disagreement can be put to the user while
        there is still a choice to make about it.

        Asked at the run rather than when the checkpoint is come by, as what it compares can change
        after the fact: the model and the channel mode both reach this comparison, and either can be
        picked again between choosing a checkpoint and running against it.

        Returns:
            What the checkpoint holds that this run would not put there, or None when it agrees, is
            empty, does not exist, could not be read, or is not one the run writes to at all.
        """
        checkpoint = self.run_checkpoint()
        if checkpoint is None:
            return None

        image_paths = [] if self.metadata_df is None else self.metadata_df['file_path'].tolist()

        return inspect_checkpoint(checkpoint, self.processing_params(), image_paths)

    def save_checkpoint(self, path: Path | str) -> None:
        """ Write every result the pipeline holds - embeddings and properties - to a checkpoint.

        The file can be loaded back later to pick the dataset up where it was left. It becomes the
        selected checkpoint once written, the one the panel names.

        Args:
            path: The checkpoint file to write, replaced if it already exists.
        Raises:
            ValueError: There are no results to save.
            OSError: The file could not be written.
        """
        self.image_processing.save_results(str(path))

        # Adopted last, so a save that raised leaves the previous checkpoint selected
        self.params.paths.checkpoint = Path(path)

        logger.info(
            "Saved the checkpoint at '%s': %d image(s), properties %s.",
            path, self.pheno.results.n_images, 'included' if self.has_stored_properties else 'not computed'
        )

    def load_checkpoint(self, path: Path | str) -> None:
        """ Read an existing checkpoint into the pipeline.

        The embeddings and the properties it holds are the ones the rest of the plugin then works
        from, so a dataset processed before is picked up where it was left rather than processed
        again. PhenoMe resolves the paths the checkpoint stores against the files currently found,
        which is why a dataset has to have been loaded first.

        Nothing is compared here: PhenoMe.load_results() reads a checkpoint whatever settings it was
        filled under, and the disagreement is the run's to find - by which time the settings may have
        been picked again anyway.

        A checkpoint has to answer for exactly the dataset loaded, every image it holds matching a file
        found and every file found being in it. PhenoMe would otherwise load the part that matches and
        drop the rest without a word - nothing at all for a checkpoint of another dataset - so the match
        is checked first, and a checkpoint failing it is refused before anything is read into the
        pipeline.

        Args:
            path: The checkpoint file to read.
        Raises:
            ValueError: The checkpoint does not answer for the dataset loaded, or is not a checkpoint.
            RuntimeError: No dataset has been loaded, so there are no files to resolve against.
            OSError: The file could not be read, or is held open by another process.
        """
        held, matched, loaded = self.image_processing.checkpoint_coverage(str(path))
        if not held == matched == loaded:
            raise ValueError(_coverage_mismatch(held, matched, loaded))

        self.image_processing.load_results(str(path))

        # Read back rather than left to the next run: the frame is what every property-based step
        # reads, and a checkpoint holding properties has to be as good as having just computed them.
        self.properties_df = self.image_processing.get_properties() if self.has_stored_properties else None

        # Adopted last, so a load that raised leaves nothing selected
        self.params.paths.checkpoint = Path(path)

        logger.info(
            "Loaded the checkpoint at '%s': %d image(s), properties %s.",
            path, self.pheno.results.n_images, 'included' if self.properties_df is not None else 'not yet computed'
        )

    @property
    def has_stored_properties(self) -> bool:
        """ Whether the pipeline holds properties, computed here or read from a checkpoint.

        Asked of PhenoMe rather than of the frame the interface keeps, this being what says whether
        there is a frame to be read back at all.

        Returns:
            True when the loaded results carry properties.
        """
        return self.pheno.results.has_properties

    def process_images(self, progress_callback: ProgressCallback | None=None, force_reprocess: bool=False) -> None:
        """ Process the images using PhenoMe.process_images(). Saves the embeddings in-place in the PhenoMe object.

        Uses the params to feed arguments.

        Args:
            progress_callback: Hook called as (current, total, desc) after each batch. Called from the
            calling thread, so a caller running this in a worker thread must not touch widgets in it.
            force_reprocess: Whether to delete the checkpoint and process every image again. PhenoMe
                removes the file rather than writing over it, so everything it held is lost: only ever
                passed for a checkpoint the user has been asked about. Without lazy checkpointing
                there is no checkpoint to delete, and PhenoMe ignores it.
        """
        checkpoint_path = self.run_checkpoint()

        # Written out rather than left to the model that was loaded: the arguments an extraction ran
        # under are what its embeddings can be read against, and they are the registry's and the
        # form's rather than only what the form shows.
        num_workers = self.params.process.num_workers
        arguments = self.process_arguments()
        logger.info(
            "Extracting embeddings with '%s', in batches of %d, read %s, %s.",
            self.params.process.model,
            self.params.process.batch_size,
            f'in {num_workers} worker process(es)' if num_workers else 'in the calling thread',
            str(arguments)
        )

        self.image_processing.process_images(
            self.model,
            checkpoint_path,
            progress_callback=progress_callback,
            batch_size=self.params.process.batch_size,
            num_workers=num_workers,
            custom_transformations=self.transforms,
            force_reprocess=force_reprocess,
            lazy_checkpoint=self.params.process.lazy_checkpoint,
            **arguments
        )

    @property
    def has_processed_images(self) -> bool:
        """ Whether images have been processed, which the properties step needs to run.

        Reads the same state PhenoMe itself checks in compute_properties(), rather than tracking
        the processing step separately.

        Returns:
            True once the pipeline holds processed images.
        """
        return self.pheno.results.n_images > 0

    def compute_properties(self, progress_callback: ProgressCallback | None=None, force_update: bool=False) -> pd.DataFrame:
        """ Compute the per-image properties using PhenoMe.compute_properties(). Stores them in self.properties_df.

        Uses the params to feed arguments.

        Args:
            progress_callback: Hook called as (current, total, desc) after each image. Called from the
                calling thread, so a caller running this in a worker thread must not touch widgets in it.
            force_update: Whether to drop the properties the checkpoint holds and compute them all
                again, which a preset computed over a checkpoint filled under another one needs:
                PhenoMe otherwise resumes, and warns about the property keys it cannot find. Only the
                properties are cleared, the embeddings in the same file being left where they are.
                Without lazy checkpointing there is no checkpoint to clear, and PhenoMe ignores it.
        Returns:
            One row per image, holding the image path and the computed properties.
        """
        checkpoint_path = self.run_checkpoint()

        # PhenoMe only knows its presets by their lower case name, while the GUI displays them capitalized
        preset = self.params.properties.preset.lower()

        self.properties_df = self.image_processing.compute_properties(
            preset,
            checkpoint_path,
            progress_callback=progress_callback,
            force_update=force_update,
            lazy_checkpoint=self.params.process.lazy_checkpoint,
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
            One row per image, holding the image path and its cluster label.
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
        self.metadata_df[CLUSTER_COLUMN] = self.metadata_df['file_path'].map(labels).astype('Int64')

    def get_prototype_images(self) -> pd.DataFrame:
        """ Find the prototype images of each group using PhenoMe.find_prototypes(). Stores them in self.prototypes_df.

        Uses the params to feed arguments. The images are grouped by the metadata column selected in the
        parameter form, which is None until data has been loaded and a column picked. PhenoMe reads no
        grouping key as a request to treat every image as one group.

        The GUI calls the grouping key 'group_by' for both this step and the outlier one, while PhenoMe
        names it 'cluster_col' here and 'group_by' there, so the name is translated on the way down.

        Returns:
            One row per prototype, holding the grouping key, the image path and its rank in the group.
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
            One row per outlier, holding the grouping key, the image path and its distance to the group centroid.
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

        Uses the params to feed arguments. The algorithm is named for the GUI rather than for PhenoMe
        - the papers' spelling rather than a plain key - so it is translated on the way down. The
        number of components is not a parameter at all, the canvas drawing a flat cloud and nothing
        else. The filter and the exclusion go down as they are, PhenoMe naming the first in the
        plural and reading an unset one and an empty one the same way.

        Returns:
            One row per image, holding its coordinates, name, index and metadata. None
            when there was nothing to project.
        """
        explorer = self.params.explorer

        self.projection_df = self.phenome_analysis.run_dimensionality_reduction(
            method=DR_METHODS[explorer.method],
            n_components=PROJECTION_COMPONENTS,
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
            The groupable column names, empty when no data has been loaded yet.
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
            The columns to offer, empty when no data has been loaded yet.
        """
        columns = [column for column in self.get_metadata_columns() if self._is_informative(column)]

        # Properties are measurements, so they are neither identifiers nor, in practice, constant
        properties = self.pheno.get_available_property_keys()

        return columns + [key for key in properties if key not in columns]

    def _is_informative(self, column: str) -> bool:
        """ Whether a metadata column says enough about the images to be worth colouring by.

        Args:
            column: Name of the column, which is read alongside its values: only the columns
                named after an identifier are held to the uniqueness test.
        Returns:
            False for a column holding one value throughout, or one naming each image.
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
            row: Positional index of the row in the metadata dataframe.
        Returns:
            The image paths, the mask paths (None when the dataset has no masks) and the row data as a list of dicts.
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
            image_path: Path of the image, as returned by the prototype or outlier step.
        Returns:
            The image paths, the mask paths (None when the dataset has no masks) and the row data as a list of dicts.
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
            row: Positional index of the row in the projection dataframe.
        Returns:
            The image paths, the mask paths (None when the dataset has no masks) and the row data as a list of dicts.
        """
        return self.get_image_data_at(int(self.projection_df.iloc[row][PROJECTION_INDEX_COLUMN]))

    def get_image_data_at(self, index: int) -> tuple[list[str], list[str] | None, list[dict]]:
        """ Return the content of one image, found by where PhenoMe holds it, for the image viewer.

        The place is what a projection carries and what a selection is made of, neither of them
        holding the path: PhenoMe writes the name of an image and its index and no more, so the path
        is looked up here and the mask and the metadata from the path.

        Args:
            index: Place of the image in the PhenoMe results.
        Returns:
            The image paths, the mask paths (None when the dataset has no masks) and the row data as a list of dicts.
        """
        image_path = self.pheno.results.img_path[index]

        # PhenoMe holds a list of paths rather than one for the images it reads a channel at a time
        if isinstance(image_path, list | tuple):
            image_path = image_path[0]

        return self.get_image_display_data(str(image_path))

    def get_metadata_table(self) -> "pd.Dataframe":
        """ Get the metadata dataframe in its current state.

        Returns:
            The metadata dataframe
        """

        return self.metadata_df

    @property
    def has_properties(self) -> bool:
        """ Whether the properties have been computed, which the property-based analysis sources need.

        Returns:
            True if properties have been calculated
        """
        properties_available = self.properties_df is not None
        logger.info('Properties are available: %s', properties_available)
        return properties_available
