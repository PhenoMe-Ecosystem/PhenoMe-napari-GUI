"""
Interface between the GUI and the PhenoMe API.

This module owns the `PhenoMe` pipeline object and every parameter feeding it.
The GUI talks only to `PhenoMeInterface` and never imports `phenome` itself, so
that widget code stays free of pipeline logic and of the metadata schema.
"""
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Any

import pandas as pd
from phenome import PhenoMe

from .data_loading import DataLoading

logger = getLogger(__name__)

# This dictionary will be used to initialize the parameters for every function.
DEFAULT_PARAMS = {
    'paths':{
        'images':"/Users/renaud/python-projects/phenome-project/PhenoMe/Notebooks/tutorials/toy_dataset/Ecoli_toy_example/Ecoli_images",
        'masks':"/Users/renaud/python-projects/phenome-project/PhenoMe/Notebooks/tutorials/toy_dataset/Ecoli_toy_example/Ecoli_masks",
        'metadata':"/Users/renaud/python-projects/phenome-project/PhenoMe/Notebooks/tutorials/toy_dataset/Ecoli_toy_example/metadata.csv",
        'template':".../(drug_name)/(time)/(filename).*",
        'checkpoint':'./checkpoint.hdf5'
    },
    'process':{
        'model':''
    },
    'properties':{
        'preset':''
    },
    'clustering':{
        'clustering_method':'kmeans',
        'n_clusters':5,
        'dbscan_eps':0.5,
        'dbscan_min_samples':50
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
        Returns:
            None
        """
        self.pheno = PhenoMe(seed=seed)
        self.params = deepcopy(DEFAULT_PARAMS)
        self.metadata_df = None

        # Components hold no parameter: the interface reads the params dict and
        # passes the values they need as arguments.
        self.data_loading = DataLoading(self.pheno)

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
        Returns:
            None
        """
        self.params[section][key] = value

    def get_path(self, key: str) -> str:
        """ Return a path parameter.

        Args:
            key (str): Name of the path parameter, e.g. 'images' or 'metadata'.
        Returns:
            str: The current value of the path parameter.
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
        Returns:
            None
        """
        self.set_param('paths', key, str(value))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data_csv(self) -> pd.DataFrame:
        """ Load the dataset using a .csv file as metadata source.

        Args:
            None
        Returns:
            pd.DataFrame: The metadata of the files found, also stored in self.metadata_df.
        """
        images_folder = self.get_path('images')
        masks_folder = self.get_path('masks')
        metadata_csv = self.get_path('metadata')
        self.metadata_df = self.data_loading.load_data_csv(images_folder, metadata_csv, masks_folder)
        return self.metadata_df

    def load_data_path(self) -> pd.DataFrame:
        """ Load the dataset by extracting the metadata from the file paths.

        Args:
            None
        Returns:
            pd.DataFrame: The metadata of the files found, also stored in self.metadata_df.
        """
        images_folder = self.get_path('images')
        masks_folder = self.get_path('masks')
        template = self.get_path('template')
        self.metadata_df = self.data_loading.load_data_path(images_folder, template, masks_folder)
        return self.metadata_df

    # ------------------------------------------------------------------
    # Metadata access
    # ------------------------------------------------------------------

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
