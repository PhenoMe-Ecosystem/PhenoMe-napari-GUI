from logging import getLogger

from phenome import PhenoMe

from .models import get_model

logger = getLogger(__name__)

class ImageProcessing:
    """ Image processing steps of the PhenoMe pipeline.

    Includes the model loading, embedding extraction and properties calculation steps.
    """
    def __init__(self, pheno: PhenoMe) -> None:
        """ Store the PhenoMe pipeline object to run the processing on.

        Args:
            pheno: The PhenoMe pipeline object owned by the interface.
        Returns:
            None
        """
        self.pheno = pheno

    def load_model(self, model_name: str, device: str='cpu') -> None:
        """ Load the model for embedding extraction.

        Args:
            model_name: Name of the model to load.
            device: The device to load the model on.
        Returns:
            None
        """
        return get_model(model_name, device=device)
