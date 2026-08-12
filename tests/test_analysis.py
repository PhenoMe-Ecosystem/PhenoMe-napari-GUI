"""Tests for PhenoMeAnalysis, the thin layer naming the arguments of the PhenoMe analysis calls."""

import pandas as pd
import pytest

from phenome_gui.utils.analysis import PhenoMeAnalysis


class FakePipeline:
    """ Stands in for the PhenoMe pipeline, recording the calls made to it. """

    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    def compute_clustering(self, **kwargs) -> pd.DataFrame:
        self.calls['compute_clustering'] = kwargs
        return pd.DataFrame()

    def find_prototypes(self, **kwargs) -> pd.DataFrame:
        self.calls['find_prototypes'] = kwargs
        return pd.DataFrame()

    def detect_outliers(self, **kwargs) -> pd.DataFrame:
        self.calls['detect_outliers'] = kwargs
        return pd.DataFrame()


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def analysis(pipeline: FakePipeline) -> PhenoMeAnalysis:
    return PhenoMeAnalysis(pipeline)


def test_compute_clustering_passes_everything_through(analysis: PhenoMeAnalysis, pipeline: FakePipeline) -> None:
    analysis.compute_clustering(
        n_clusters=3, clustering_method='gmm', source='properties', dbscan_eps=0.4, dbscan_min_samples=8
    )

    assert pipeline.calls['compute_clustering'] == {
        'source': 'properties',
        'n_clusters': 3,
        'clustering_method': 'gmm',
        'dbscan_eps': 0.4,
        'dbscan_min_samples': 8,
    }


def test_get_prototype_images_passes_everything_through(analysis: PhenoMeAnalysis, pipeline: FakePipeline) -> None:
    analysis.get_prototype_images(cluster_col='drug_name', n_prototypes=2, source='embeddings')

    assert pipeline.calls['find_prototypes'] == {
        'cluster_col': 'drug_name',
        'n_prototypes': 2,
        'source': 'embeddings',
        'plot': False,
    }


def test_get_outlier_images_passes_everything_through(analysis: PhenoMeAnalysis, pipeline: FakePipeline) -> None:
    analysis.get_outlier_images(group_by='time', method='iqr', threshold=2.5, source='combined')

    assert pipeline.calls['detect_outliers'] == {
        'group_by': 'time',
        'method': 'iqr',
        'threshold': 2.5,
        'source': 'combined',
        'plot': False,
    }


@pytest.mark.parametrize(
    ('call', 'kwargs', 'pheno_method'),
    [
        ('get_prototype_images', {'cluster_col': None, 'n_prototypes': 1, 'source': 'embeddings'}, 'find_prototypes'),
        (
            'get_outlier_images',
            {'group_by': None, 'method': 'z-score', 'threshold': 3.0, 'source': 'embeddings'},
            'detect_outliers',
        ),
    ],
)
def test_plotting_is_always_off(
    analysis: PhenoMeAnalysis, pipeline: FakePipeline, call: str, kwargs: dict, pheno_method: str
) -> None:
    """These run on a worker thread, where PhenoMe's matplotlib figures must never be opened."""
    getattr(analysis, call)(**kwargs)

    assert pipeline.calls[pheno_method]['plot'] is False
