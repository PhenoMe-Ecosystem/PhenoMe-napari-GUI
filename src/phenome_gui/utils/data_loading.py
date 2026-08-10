from logging import getLogger

import pandas as pd
from phenome import PhenoMe, get_metadata_from_path, make_dataframe_metadata_fn

logger = getLogger(__name__)

class DataLoading:
    """ Data loading component of the PhenoMe interface.

    Turns the loading parameters into a metadata dataframe through the PhenoMe API.
    """

    def __init__(self, pheno: PhenoMe) -> None:
        """ Store the PhenoMe pipeline object to run the file discovery on.

        Args:
            pheno: The PhenoMe pipeline object owned by the interface.
        """
        self.pheno = pheno

    def load_data_csv(self, images_folder: str, metadata_csv: str, masks_folder: str | None = None) -> pd.DataFrame:
        """ Find the image files and corresponding masks, using a .csv file as metadata source.

        Args:
            images_folder: Path to the folder containing the images.
            metadata_csv: Path to the .csv file holding the metadata.
            masks_folder: Path to the folder containing the masks. None when no masks are present.
        Returns:
            pd.DataFrame: The metadata of the files found, empty when no file matched.
        """
        metadata_df = pd.read_csv(metadata_csv)
        metadata_fn = make_dataframe_metadata_fn(metadata_df)
        logger.info("Loading data with .csv metadata: images_folder=%s, metadata_csv=%s, masks_folder=%s", images_folder, metadata_csv, masks_folder)
        logger.info("Metadata columns: %s", metadata_df.columns.tolist())

        return self.pheno.find_files(images_folder, mask_dir=masks_folder, metadata_fn=metadata_fn)

    def load_data_path(self, images_folder: str, template: str, masks_folder: str | None = None) -> pd.DataFrame:
        """ Find the image files and corresponding masks, extracting the metadata from the file paths.

        Args:
            images_folder: Path to the folder containing the images.
            template: Path template to parse the metadata from the file paths.
            masks_folder: Path to the folder containing the masks. None when no masks are present.
        Returns:
            pd.DataFrame: The metadata of the files found, empty when no file matched.
        """
        metadata_fn = get_metadata_from_path(template)
        logger.info("Loading data with path template: images_folder=%s, template=%s, masks_folder=%s", images_folder, template, masks_folder)
        return self.pheno.find_files(images_folder, mask_dir=masks_folder, metadata_fn=metadata_fn)
