from logging import getLogger
from typing import TYPE_CHECKING

import pandas as pd
from phenome import PhenoMe
from phenome.utils import ProgressCallback

from ..registries.models import get_model

if TYPE_CHECKING:
    import torchvision.transforms as T
    from phenome import ModelWrapper

logger = getLogger(__name__)

class ImageProcessing:
    """ Image processing steps of the PhenoMe pipeline.

    Includes the model loading, embedding extraction and properties calculation steps.
    """
    def __init__(self, pheno: PhenoMe) -> None:
        """ Store the PhenoMe pipeline object to run the processing on.

        Args:
            pheno: The PhenoMe pipeline object owned by the interface.
        """
        self.pheno = pheno

    def load_model(self, model_name: str, device: str='cpu') -> None:
        """ Load the model for embedding extraction.

        Args:
            model_name: Name of the model to load.
            device: The device to load the model on.
        """
        return get_model(model_name, device=device)

    def process_images(self, model_wrapper: ModelWrapper, checkpoint_path: str | None=None, progress_callback: ProgressCallback | None=None, custom_transformations: "T.Compose" | None=None, **kwargs) -> None:
        """ Process the images using PhenoMe.process_images(). Saves the embeddings in-place in the PhenoMe object.

        Args:
            model_wrapper: PhenoMe wrapper of the used model.
            checkpoint_path: Path of the .hdf5 file to save results in. If None, no checkpoint is created.
            progress_callback: Hook called as (current, total, desc) after each batch. PhenoMe reports the
                1-based batch index and the total number of batches, and disables its tqdm bar when set.
                If None, PhenoMe keeps its default tqdm bar.
            **kwargs: Any arguments included in the PhenoMe process_images() method.
        """
        self.pheno.process_images(model_wrapper=model_wrapper,
                                  checkpoint_path=checkpoint_path,
                                  progress_callback=progress_callback,
                                  custom_transformations=custom_transformations,
                                  **kwargs)

    def compute_properties(self, property_preset: str, checkpoint_path: str | None=None, progress_callback: ProgressCallback | None=None, **kwargs) -> pd.DataFrame:
        """ Compute the per-image properties using PhenoMe.compute_properties(). Also saves them in-place in the PhenoMe object.

        Args:
            property_preset: Name of the PhenoMe property preset to compute, e.g. 'basic'. Lower case only.
            checkpoint_path: Path of the .hdf5 file to save results in. If None, no checkpoint is created.
            progress_callback: Hook called as (current, total, desc) after each image. PhenoMe reports the
                number of images computed and the total number of images to compute, and disables its
                tqdm bar when set. If None, PhenoMe keeps its default tqdm bar.
            **kwargs: Any arguments included in the PhenoMe compute_properties() method.
        Returns:
            pd.DataFrame: One row per image, holding the image path and the computed properties.
        """
        return self.pheno.compute_properties(property_preset=property_preset,
                                             checkpoint_path=checkpoint_path,
                                             progress_callback=progress_callback,
                                             **kwargs)
