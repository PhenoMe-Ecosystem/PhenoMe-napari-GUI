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


def test_run_dimensionality_reduction_passes_the_pipeline_and_the_arguments(
    analysis: PhenoMeAnalysis, pipeline: FakePipeline, reduction
) -> None:
    """PhenoMe's reduction is a free function, so the pipeline is an argument rather than the receiver."""
    stand_in = reduction(pd.DataFrame({'Component 1': [0.0], 'Component 2': [1.0]}))

    analysis.run_dimensionality_reduction(
        method='umap',
        n_components=3,
        source='properties',
        filters={'drug_name': ['ctrl']},
        exclude={'cluster': [0]},
    )

    assert stand_in.pipelines[-1] is pipeline
    assert stand_in.last == {
        'method': 'umap',
        'n_components': 3,
        'source': 'properties',
        'filters': {'drug_name': ['ctrl']},
        'exclude': {'cluster': [0]},
    }


def test_projecting_everything_is_the_default(analysis: PhenoMeAnalysis, reduction) -> None:
    """Unset criteria and empty ones are the same to PhenoMe: every image is projected."""
    stand_in = reduction(pd.DataFrame({'Component 1': [0.0], 'Component 2': [1.0]}))

    analysis.run_dimensionality_reduction(method='pca', n_components=2, source='embeddings')

    assert stand_in.last['filters'] is None
    assert stand_in.last['exclude'] is None


def test_run_dimensionality_reduction_returns_the_coordinates_alone(
    analysis: PhenoMeAnalysis, reduction
) -> None:
    """The fitted reducer and the data type come back too, neither of which the plot needs."""
    frame = pd.DataFrame({'Component 1': [0.0, 1.0], 'Component 2': [1.0, 2.0]})
    reduction(frame)

    returned = analysis.run_dimensionality_reduction(method='pca', n_components=2, source='embeddings')

    assert returned is frame


def test_nothing_to_project_comes_back_as_none(analysis: PhenoMeAnalysis, reduction) -> None:
    """PhenoMe answers with three Nones when the pipeline leaves it nothing to reduce."""
    reduction(None)

    returned = analysis.run_dimensionality_reduction(method='pca', n_components=2, source='embeddings')

    assert returned is None


