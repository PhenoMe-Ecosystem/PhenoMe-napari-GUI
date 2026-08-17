from logging import getLogger

import pandas as pd
from phenome import PhenoMe

# A free function taking the pipeline as its first argument rather than a PhenoMe method, unlike every
# other step wrapped here.
from phenome.core import run_dimensionality_reduction

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


    def compute_clustering(self, n_clusters: int, clustering_method: str, source: str, dbscan_eps: float, dbscan_min_samples: int) -> "pd.DataFrame":
        """ Compute clustering using PhenoMe's compute_clustering method, with the chosen clustering algorithm and parameters

        Args:
            n_clusters: Number of clusters for k-means and GMM
            clustering_method: Algorithm to cluster with: 'kmeans', 'dbscan' or 'gmm'.
            source: Feature space to cluster in: 'embeddings', 'properties' or 'combined'.
            dbscan_eps: Maximum distance between two neighbouring samples, for DBSCAN only.
            dbscan_min_samples: Minimum number of samples in a neighbourhood, for DBSCAN only.
        Returns:
            pd.DataFrame: One row per image, holding the image path and its cluster label.
        """
        return self.pheno.compute_clustering(
            source=source,
            n_clusters=n_clusters,
            clustering_method=clustering_method,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples
        )


    def get_prototype_images(self, cluster_col: str | None, n_prototypes: int, source: str) -> "pd.DataFrame":
        """ Find the prototype images of each group using PhenoMe's find_prototypes method.

        Args:
            cluster_col: Metadata key the images are grouped by. None to treat them as a single group.
            n_prototypes: Number of prototypes to return for each group.
            source: Feature space to compare the images in: 'embeddings', 'properties' or 'combined'.
        Returns:
            pd.DataFrame: One row per prototype, holding the grouping key, the image path and its rank in the group.
        """
        # PhenoMe plots one matplotlib figure per group by default, which must not happen from the worker
        # thread this runs on. The caller displays the returned images instead.
        return self.pheno.find_prototypes(
            cluster_col=cluster_col,
            n_prototypes=n_prototypes,
            source=source,
            plot=False
        )


    def get_outlier_images(self, group_by: str | None, method: str, threshold: float, source: str) -> "pd.DataFrame":
        """ Detect the outlier images of each group using PhenoMe's detect_outliers method.

        Every image further from its group centroid than the threshold is returned, so the number of
        outliers is not known in advance and varies from one group to the next.

        Args:
            group_by: Metadata key the images are grouped by. None to treat them as a single group.
            method: How the threshold is applied: 'z-score' or 'iqr'.
            threshold: Threshold multiplier, read against the method used.
            source: Feature space to measure the distances in: 'embeddings', 'properties' or 'combined'.
        Returns:
            pd.DataFrame: One row per outlier, holding the grouping key, the image path and its distance to the group centroid.
        """
        # As for the prototypes, PhenoMe plots one matplotlib figure per group by default, which must not
        # happen from the worker thread this runs on. The caller displays the returned images instead.
        return self.pheno.detect_outliers(
            group_by=group_by,
            method=method,
            threshold=threshold,
            source=source,
            plot=False
        )


    def run_dimensionality_reduction(self, method: str, n_components: int, source: str) -> "pd.DataFrame | None":
        """ Project the images down to a few dimensions using PhenoMe's run_dimensionality_reduction.

        PhenoMe falls back on the pipeline seed, so the same images projected twice land in the same
        place. It also hands back the fitted reducer and which feature space it read, neither of which
        the plot needs: only the coordinates are returned, the data type being logged instead.

        Args:
            method: Algorithm to project with, under the name PhenoMe knows it by: 'pca', 'tsne' or 'umap'.
            n_components: Number of dimensions to project down to.
            source: Feature space to project: 'embeddings', 'properties' or 'combined'.
        Returns:
            pd.DataFrame: One row per image, holding its coordinates, name, index and metadata. None when
                there was nothing to project.
        """
        coordinates, data_type, _ = run_dimensionality_reduction(
            self.pheno,
            method=method,
            n_components=n_components,
            source=source
        )

        if coordinates is None:
            logger.warning("Dimensionality reduction returned nothing to plot.")
        else:
            logger.info("Projected %d images of the %s.", len(coordinates), data_type.lower())

        return coordinates

