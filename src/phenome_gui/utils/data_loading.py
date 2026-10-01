from logging import getLogger
from pathlib import Path

import pandas as pd
from phenome import PhenoMe, get_metadata_from_path, make_dataframe_metadata_fn

logger = getLogger(__name__)

# The files a load takes for images: what PhenoMe's read_image() can open, by its own branches for
# TIFF, NumPy and NIfTI, and through OpenCV for the rest. Passed to find_files() explicitly, as its
# default of None filters nothing - every file in the folder would be taken, notes and all, whatever
# its docstring says. '.nii.gz' is left out: find_files() filters on the last suffix alone, so it
# could only be allowed by allowing every '.gz'.
EXTENSIONS = [
    '.tif', '.tiff',
    '.png', '.jpg', '.jpeg', '.bmp', '.webp', '.jp2',
    '.npy', '.npz',
    '.nii',
]

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

    def load_data_csv(self, images_folder: Path, metadata_csv: Path | None = None, masks_folder: Path | None = None) -> pd.DataFrame:
        """ Find the image files and corresponding masks, using a .csv file as metadata source.

        The images are the only thing needed. Without a .csv the files are found with no metadata
        beyond what PhenoMe reads off them, and without a masks folder they are found with no masks
        and no mask_path column.

        Args:
            images_folder: Path to the folder containing the images.
            metadata_csv: Path to the .csv file holding the metadata. None when there is none.
            masks_folder: Path to the folder containing the masks. None when no masks are present.
        Returns:
            The metadata of the files found, empty when no file matched.
        """
        logger.info(
            "Loading data with .csv metadata: images_folder=%s, metadata_csv=%s, masks_folder=%s",
            images_folder, metadata_csv or 'none', masks_folder or 'none'
        )

        metadata_fn = None
        if metadata_csv is not None:
            metadata_df = pd.read_csv(metadata_csv)
            metadata_fn = make_dataframe_metadata_fn(metadata_df)

            # Read from the frame rather than from what is found, this being what was asked for: a
            # column the images cannot be matched on reaches the log here and nowhere else.
            logger.info("Metadata columns: %s", metadata_df.columns.tolist())

        return self.pheno.find_files(
            images_folder, mask_dir=masks_folder, extensions=EXTENSIONS, metadata_fn=metadata_fn
        )

    def load_data_path(self, images_folder: str, template: str | None = None, masks_folder: str | None = None) -> pd.DataFrame:
        """ Find the image files and corresponding masks, extracting the metadata from the file paths.

        The images are the only thing needed. Without a template there is nothing to read the
        metadata off the paths with, so the files are found without any, and without a masks folder
        they are found with no masks and no mask_path column.

        Args:
            images_folder: Path to the folder containing the images.
            template: Path template to parse the metadata from the file paths. None when there is none.
            masks_folder: Path to the folder containing the masks. None when no masks are present.
        Returns:
            The metadata of the files found, empty when no file matched.
        """
        logger.info(
            "Loading data with path template: images_folder=%s, template=%s, masks_folder=%s",
            images_folder, template or 'none', masks_folder or 'none'
        )

        metadata_fn = get_metadata_from_path(template) if template else None

        return self.pheno.find_files(
            images_folder, mask_dir=masks_folder, extensions=EXTENSIONS, metadata_fn=metadata_fn
        )
