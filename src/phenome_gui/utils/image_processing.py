from logging import getLogger
from typing import TYPE_CHECKING

import pandas as pd
from phenome import PhenoMe
from phenome.io import CheckpointManager
from phenome.utils import ProgressCallback

from ..registries.models import get_model
from .checkpoint import image_key

if TYPE_CHECKING:
    import torchvision.transforms as T
    from phenome.utils import ModelWrapper

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

    def process_images(self, model_wrapper: "ModelWrapper", progress_callback: ProgressCallback | None=None, custom_transformations: "T.Compose | None" = None, **kwargs) -> None:
        """ Process the images using PhenoMe.process_images(). Saves the embeddings in-place in the PhenoMe object.

        Always run in memory: no checkpoint is written while the run goes, so the one a dataset was
        loaded from is never touched. A run is resumed by appending to the embeddings held instead.

        Args:
            model_wrapper: PhenoMe wrapper of the used model.
            progress_callback: Hook called as (current, total, desc) after each batch. PhenoMe reports the
                1-based batch index and the total number of batches, and disables its tqdm bar when set.
                If None, PhenoMe keeps its default tqdm bar.
            custom_transformations: The transformations the images are prepared with, or None for
                PhenoMe's own.
            **kwargs: Any arguments included in the PhenoMe process_images() method.
        """
        self.pheno.process_images(model_wrapper=model_wrapper,
                                  progress_callback=progress_callback,
                                  custom_transformations=custom_transformations,
                                  **kwargs)

    def loaded_paths_among(self, held_paths: list) -> list:
        """ The loaded images' paths, as PhenoMe's file table holds them, for the images among the ones held.

        What a resumed run excludes: PhenoMe filters its file table by value, so the paths are handed
        back in the table's own form rather than in the form the results hold them in.

        Reaches for PhenoMe's private file table, the only place that form is found, kept here beside
        the other private calls.

        Args:
            held_paths: The images embeddings are held for.
        Returns:
            The file table's path of every loaded image held.
        """
        held = {image_key(path) for path in held_paths}
        return [path for path in self.pheno._file_df['file_path'] if image_key(path) in held]

    def keep_stopped_results(self) -> bool:
        """ Make the embeddings an in-memory extraction computed before it was interrupted part of the results.

        PhenoMe appends the paths and the metadata to the results as it goes, but holds the embeddings
        in a buffer it only stacks into the results once the run is done, which an interruption unwinds
        before. Stacked here instead, so the results answer for the images they name. The rows agree
        when the run was unwound between two batches, which is where a stop is raised; a failure
        caught halfway through appending one is let go of whole rather than kept misaligned.

        Reaches for PhenoMe's private buffer and finalising step, kept here beside the other ones.

        Returns:
            True when results were kept, False when there were none or they could not be kept.
        """
        n_named = self.pheno.results.n_images
        if n_named != len(self.pheno._emb_buffer):
            logger.warning(
                'Dropped what the interrupted run computed: %d image(s) named against %d embedding(s).',
                n_named, len(self.pheno._emb_buffer)
            )
            self.pheno.reset()
            return False

        self.pheno._finalize_results()
        return n_named > 0

    def save_results(self, checkpoint_path: str) -> None:
        """ Write everything the pipeline holds - embeddings, properties and processing parameters - to a checkpoint.

        PhenoMe copies the checkpoint it is reading from when one is open, which is what a loaded
        checkpoint leaves behind, adding the properties computed in memory since. Otherwise it writes
        the results held in memory, replacing whatever file is already at the path.

        Args:
            checkpoint_path: Path of the .hdf5 file to write.
        """
        self.pheno.save_results(path=str(checkpoint_path))

    def load_results(self, checkpoint_path: str) -> None:
        """ Load an existing checkpoint's results into the PhenoMe pipeline.

        The metadata and the properties are read straight away, the embeddings left on disk and read
        as they are needed: a checkpoint is opened to pick a dataset up where it was left, and its
        embeddings may be more than fits in memory.

        PhenoMe resolves the paths the checkpoint stores against the files it currently holds, so
        find_files() must have been called first - it raises rather than guessing otherwise.

        Args:
            checkpoint_path: Path of the .hdf5 file to read.
        """
        self.pheno.load_results(checkpoint_path, lazy_checkpoint=True)

    def checkpoint_coverage(self, checkpoint_path: str) -> tuple[int, int, int]:
        """ How a checkpoint's images line up with the ones currently loaded, without loading it.

        PhenoMe never refuses a checkpoint filled from other images: it keeps the rows it can match to
        the files currently found, and drops the rest without a word, down to none at all. The match is
        made here the same way, by loading the checkpoint into a pipeline of its own that is given the
        same files, so a dataset moved to another folder still matches where its paths no longer do.
        That pipeline is let go of afterwards, leaving this one exactly as it was. Only the metadata is
        read, the embeddings staying on disk.

        Reaches for PhenoMe's private file table, which set_file_df() takes back as it is: it is the
        only way to hand another pipeline the same files, and it is kept here beside the other PhenoMe
        calls so there is one place to change should PhenoMe offer it publicly.

        Args:
            checkpoint_path: Path of the .hdf5 file to compare.
        Returns:
            The images the checkpoint holds, how many of them match a loaded file, and how many files
            are loaded.
        Raises:
            RuntimeError: No dataset has been loaded, so there are no files to match against.
            FileNotFoundError: The file does not exist.
            ValueError: The file is not a checkpoint.
            OSError: The file could not be read, or is held open by another process.
        """
        file_df = self.pheno._file_df
        if file_df is None:
            raise RuntimeError('No dataset is loaded to match the checkpoint against.')

        held = CheckpointManager(str(checkpoint_path), lazy=True)
        try:
            n_held = held.n_committed
        finally:
            held.close()

        scratch = PhenoMe()
        try:
            scratch.set_file_df(file_df)
            scratch.load_results(str(checkpoint_path), lazy_checkpoint=True)
            n_matched = scratch.results.n_images
        finally:
            # Closes the checkpoint it opened, which the real load opens again straight after
            scratch.reset()

        return n_held, n_matched, len(file_df)

    def get_properties(self) -> pd.DataFrame:
        """ The properties the pipeline currently holds, as a frame.

        Reaches for a private PhenoMe method, which compute_properties() itself returns the result of:
        it is the only way to the frame for properties that were read from a checkpoint rather than
        just computed, and any other reading of them would be a second, differently shaped answer to
        the same question. Kept here beside the other PhenoMe calls rather than in the interface, so
        there is one place to change should PhenoMe offer it publicly.

        Returns:
            One row per image, holding the image path and its properties.
        """
        return self.pheno._build_properties_dataframe()

    def compute_properties(self, property_preset: str, progress_callback: ProgressCallback | None=None, **kwargs) -> pd.DataFrame:
        """ Compute the per-image properties using PhenoMe.compute_properties(). Also saves them in-place in the PhenoMe object.

        Always run in memory, every image's properties being computed anew.

        Args:
            property_preset: Name of the PhenoMe property preset to compute, e.g. 'basic'. Lower case only.
            progress_callback: Hook called as (current, total, desc) after each image. PhenoMe reports the
                number of images computed and the total number of images to compute, and disables its
                tqdm bar when set. If None, PhenoMe keeps its default tqdm bar.
            **kwargs: Any arguments included in the PhenoMe compute_properties() method.
        Returns:
            One row per image, holding the image path and the computed properties.
        """
        return self.pheno.compute_properties(property_preset=property_preset,
                                             progress_callback=progress_callback,
                                             **kwargs)
