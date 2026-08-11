from logging import getLogger

import pandas as pd
from phenome import PhenoMe

logger = getLogger(__name__)

class PhenoMeAnalysis:
    """ Image processing steps of the PhenoMe pipeline.

    Includes the model loading, embedding extraction and properties calculation steps.
    """
    def __init__(self, pheno: PhenoMe) -> None:
        """ Store the PhenoMe pipeline object to run the processing on.

        Args:
            pheno: The PhenoMe pipeline object owned by the interface.
        """
        self.pheno = pheno


    def compute_clustering(self, n_clusters: int, clustering_method: str, dbscan_eps: float, dbscan_min_samples: int) -> "pd.DataFrame":
        """ Compute clustering using PhenoMe's compute_clustering method, with the chosen clustering algorithm and parameters

        Args:
            n_clusters: Number of clusters for k-means and GMM
            clustering_method:
        """
        return self.pheno.compute_clustering(
            source="embeddings",
            n_clusters=n_clusters,
            clustering_method=clustering_method,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples
        )


    def get_prototype_images(self):
        """
        """


    def get_outlier_images(self):
        """
        """

