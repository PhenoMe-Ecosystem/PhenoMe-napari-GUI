from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Any

import pandas as pd
from phenome import PhenoMe
from phenome.utils import ProgressCallback
from torch import cuda

from .analysis import PhenoMeAnalysis
from .data_loading import DataLoading
from .image_processing import ImageProcessing
from .models import MODEL_OPTIONS

logger = getLogger(__name__)

# This dictionary will be used to initialize the parameters for every function.
DEFAULT_PARAMS = {
    'paths':{
        'images':Path("~/datasets/toy_dataset/Ecoli_toy_example/Ecoli_images"),
        'masks':Path("~/datasets/toy_dataset/Ecoli_toy_example/Ecoli_masks"),
        'metadata':Path("~/datasets/toy_dataset/Ecoli_toy_example/metadata.csv"),
        'template':".../(drug_name)/(time)/(filename).*",
        'checkpoint':Path("./checkpoint.hdf5")
    },
    'process':{
        'model':'Small (ViT-S/14)',
        'device':'cpu',
        'batch_size':32
    },
    'properties':{
        'preset':'Basic'
    },
    'clustering':{
        'clustering_method':'kmeans',
        'n_clusters':5,
        'dbscan_eps':0.5,
        'dbscan_min_samples':50
    },
    'view_images':{
        'group_by':'',
        'no_images':3
    }
}


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
        self.params = deepcopy(DEFAULT_PARAMS)
        self.metadata_df = None
        self.properties_df = None
        self.clustering_df = None
        self.model = None
        self.cuda_available = cuda.is_available()

        self.data_loading = DataLoading(self.pheno)
        self.image_processing = ImageProcessing(self.pheno)
        self.phenome_analysis = PhenoMeAnalysis(self.pheno)
    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_param(self, section: str, key: str) -> Any:
        """ Return the value of a parameter.

        Args:
            section (str): Name of the parameter section, e.g. 'paths' or 'clustering'.
            key (str): Name of the parameter inside that section.
        Returns:
            The current value of the parameter.
        """
        try:
            return self.params[section][key]
        except KeyError:
            logger.error("Parameter '%s' not found in section '%s'", key, section)
            return ""

    def set_param(self, section: str, key: str, value: Any) -> None:
        """ Update a parameter in place.

        Args:
            section (str): Name of the parameter section, e.g. 'paths' or 'clustering'.
            key (str): Name of the parameter inside that section.
            value: New value of the parameter.
        """
        self.params[section][key] = value

    def get_path(self, key: str) -> str | Path:
        """ Return a path parameter.

        Args:
            key (str): Name of the path parameter, e.g. 'images' or 'metadata'.
        Returns:
            The current value of the path parameter.
        """
        try:
            return self.get_param('paths', key)
        except KeyError:
            logger.error("Path parameter '%s' not found", key)
            return ""

    def set_path(self, key: str, value: str | Path) -> None:
        """ Update a path parameter, coercing it to a string.

        Args:
            key (str): Name of the path parameter, e.g. 'images' or 'metadata'.
            value: New path, coerced to a string before being stored.
        """
        self.set_param('paths', key, Path(value))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data_csv(self) -> pd.DataFrame:
        """ Load the dataset using a .csv file as metadata source.

        Returns:
            pd.DataFrame: The metadata of the files found, also stored in self.metadata_df.
        """
        images_folder = str(self.get_path('images'))
        masks_folder = str(self.get_path('masks'))
        metadata_csv = str(self.get_path('metadata'))
        self.metadata_df = self.data_loading.load_data_csv(images_folder, metadata_csv, masks_folder)
        return self.metadata_df

    def load_data_path(self) -> pd.DataFrame:
        """ Load the dataset by extracting the metadata from the file paths.

        Returns:
            pd.DataFrame: The metadata of the files found, also stored in self.metadata_df.
        """
        images_folder = str(self.get_path('images'))
        masks_folder = str(self.get_path('masks'))
        template = self.get_path('template')
        self.metadata_df = self.data_loading.load_data_path(images_folder, template, masks_folder)
        return self.metadata_df

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """ Find the real model name from the user-friendly name and load it in the PhenoMe pipeline. """
        model_name = self.get_param('process', 'model')

        real_model_name = MODEL_OPTIONS.get(model_name)
        if real_model_name:
            self.model = self.image_processing.load_model(real_model_name, device=self.get_param('process', 'device'))
        else:
            logger.error("Model name '%s' not recognized. Available options: %s", model_name, list(MODEL_OPTIONS.keys()))

    def process_images(self, progress_callback: ProgressCallback | None=None) -> None:
        """ Process the images using PhenoMe.process_images(). Saves the embeddings in-place in the PhenoMe object.

        Uses the params dict to feed arguments.

        TODO: Include the remaining advanced parameters from the form.

        Args:
            progress_callback: Hook called as (current, total, desc) after each batch. Called from the
            calling thread, so a caller running this in a worker thread must not touch widgets in it.
        """
        checkpoint_path = self.get_path('checkpoint')

        self.image_processing.process_images(
            self.model, checkpoint_path, progress_callback=progress_callback,
            batch_size=self.get_param('process', 'batch_size')
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

        Uses the params dict to feed arguments.

        TODO: Computing a different preset over an existing checkpoint may need force_update=True, otherwise
        PhenoMe warns about the property keys missing from the checkpoint.

        Args:
            progress_callback: Hook called as (current, total, desc) after each image. Called from the
                calling thread, so a caller running this in a worker thread must not touch widgets in it.
        Returns:
            pd.DataFrame: One row per image, holding the image path and the computed properties.
        """
        checkpoint_path = self.get_path('checkpoint')

        # PhenoMe only knows its presets by their lower case name, while the GUI displays them capitalized
        preset = self.get_param('properties', 'preset').lower()

        self.properties_df = self.image_processing.compute_properties(
            preset, checkpoint_path, progress_callback=progress_callback
        )
        return self.properties_df

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def compute_clustering(self) -> pd.DataFrame:
        """ Cluster the images using PhenoMe.compute_clustering(). Stores the labels in self.clustering_df.

        Uses the params dict to feed arguments. The keys of the 'clustering' section are named after the
        PhenoMe arguments they stand for, so they are passed straight through. PhenoMe ignores the ones
        that do not apply to the chosen method, and falls back on the pipeline seed for the random state.

        Returns:
            pd.DataFrame: One row per image, holding the image path and its cluster label.
        """
        self.clustering_df = self.phenome_analysis.compute_clustering(
            n_clusters=self.get_param('clustering', 'n_clusters'),
            clustering_method=self.get_param('clustering', 'clustering_method'),
            dbscan_eps=self.get_param('clustering', 'dbscan_eps'),
            dbscan_min_samples=self.get_param('clustering', 'dbscan_min_samples')
        )
        return self.clustering_df

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
