"""Typed configuration for the PhenoMe GUI.

One model per pipeline step, composed into a single `Params` object owned by `PhenoMeInterface`.
Validation happens on assignment, so an out-of-range or misspelled parameter fails at the point it is
set rather than deep inside PhenoMe.

The `Literal` types below are the single source of truth for the choices offered by the parameter
forms: the `*_CHOICES` lists are derived from them, so a value the model rejects can never appear in a
dropdown.
"""

from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

ClusteringMethod = Literal['kmeans', 'dbscan', 'gmm']
Source = Literal['embeddings', 'properties', 'combined']
OutlierMethod = Literal['z-score', 'iqr']
PropertiesPreset = Literal['Basic', 'Standard', 'Complete', 'Shape', 'Intensity']
Device = Literal['cpu', 'cuda']
EmbeddingMethod = Literal['t-SNE', 'PCA', 'UMAP']
ChannelMode = Literal['split', 'combined']

# Where the embeddings and the properties are written when no other file has been chosen. Named
# rather than written into the field alone, as loading another dataset puts the path back to it: a
# checkpoint holds the results of the images it was filled from, and answers for no others.
DEFAULT_CHECKPOINT = Path('./checkpoint.hdf5')

# The most processes the images may be read in. Held to a number the parameter form can offer and the
# model can be given, so neither can be set to a count no machine would gain by.
MAX_WORKERS = 8

CLUSTERING_METHOD_CHOICES = list(get_args(ClusteringMethod))
SOURCE_CHOICES = list(get_args(Source))
METHOD_CHOICES = list(get_args(OutlierMethod))
PRESET_CHOICES = list(get_args(PropertiesPreset))
EMBEDDING_METHOD_CHOICES = list(get_args(EmbeddingMethod))
CHANNEL_MODE_CHOICES = list(get_args(ChannelMode))


class ParamsBase(BaseModel):
    """ Shared model configuration.

    `validate_assignment` re-validates on every write, and `extra='forbid'` rejects unknown fields at
    construction. Together with pydantic's refusal to set an undefined attribute, a mistyped parameter
    name raises instead of silently landing in a slot nobody reads.
    """

    model_config = ConfigDict(validate_assignment=True, extra='forbid')


class PathParams(ParamsBase):
    """ Filesystem inputs and outputs.

    `template` is a filename pattern rather than a path, and is typed as such so it is never coerced
    into a Path.
    """

    images: Path = Path('/Users/renaud/datasets/PhenoMe/toy_dataset/Ecoli_toy_example/Ecoli_images')
    masks: Path = Path('/Users/renaud/datasets/PhenoMe/toy_dataset/Ecoli_toy_example/Ecoli_masks')
    metadata: Path = Path('/Users/renaud/datasets/PhenoMe/toy_dataset/Ecoli_toy_example/metadata.csv')
    template: str = '.../(drug_name)/(time)/(filename).*'
    checkpoint: Path = DEFAULT_CHECKPOINT


class ProcessParams(ParamsBase):
    """ Embedding extraction settings.

    `model` holds the user-friendly name shown in the GUI; `PhenoMeInterface.load_model` maps it onto
    the real checkpoint name through MODEL_OPTIONS.

    `channel_mode` decides what a multi-channel image is embedded as. 'split' embeds each channel on
    its own and joins the results, so a channel keeps its own place in the embedding; 'combined' feeds
    the channels as one image, so what the model reads is the channels together. The two answer
    different questions of the same images and are not comparable, which is why a checkpoint filled
    under one is not resumed under the other.

    `num_workers` is the number of processes the images are read in, and defaults to none of them:
    PhenoMe reads them in the thread the run is made on when there are no workers, and every worker is
    otherwise a Python started from nothing, which imports torch before it reads its first image. The
    run waits on all of them before its first batch, which is most of the wait before a run appears to
    start. Worth raising where starting a process is cheap and the images are slow to read, which is
    why the choice is offered rather than settled here.
    """

    model: str = 'DinoV2 - Small'
    device: Device = 'cpu'
    channel_mode: ChannelMode = 'split'
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0, le=MAX_WORKERS)


class PropertiesParams(ParamsBase):
    """ Property computation settings.

    The preset is stored capitalised as the GUI displays it; PhenoMe knows its presets by their lower
    case name, and the interface folds the case on the way down.
    """

    preset: PropertiesPreset = 'Basic'


class ClusteringParams(ParamsBase):
    """ Clustering settings.

    The fields are named after the PhenoMe arguments they stand for and passed straight through.
    PhenoMe ignores the ones that do not apply to the chosen method.
    """

    clustering_method: ClusteringMethod = 'kmeans'
    source: Source = 'embeddings'
    n_clusters: int = Field(default=5, ge=1)
    dbscan_eps: float = Field(default=0.5, gt=0)
    dbscan_min_samples: int = Field(default=50, ge=1)


class PrototypesParams(ParamsBase):
    """ Prototype image search settings.

    `group_by` is None until a metadata column has been picked, which PhenoMe reads as a request to
    treat every image as one group.
    """

    group_by: str | None = None
    n_prototypes: int = Field(default=3, ge=1, le=100)
    source: Source = 'embeddings'


class OutliersParams(ParamsBase):
    """ Outlier detection settings.

    The threshold is read as a number of standard deviations for 'z-score' and as a multiple of the
    interquartile range for 'iqr', so the same value means different things per method.
    """

    group_by: str | None = None
    method: OutlierMethod = 'z-score'
    threshold: float = Field(default=3.0, ge=0.1, le=100)
    source: Source = 'embeddings'


class ExplorerParams(ParamsBase):
    """ Interactive embedding explorer settings.

    `color_by` is None until a metadata column has been picked, the same way the analysis steps treat
    their grouping key.

    `filter` names the metadata values an image must have for the projection to be computed from it,
    one list per field, and is None until a filter has been picked - which PhenoMe reads the same way
    as an empty one, every image being kept. Its values are typed as the metadata holds them rather
    than as text: PhenoMe matches them by equality, where the cluster labels it writes are numbers.

    `exclude` is the same mapping read the other way round, naming what drops an image, and applied
    after the filter. The two combine their fields differently: an image is kept for answering every
    filtered field, and dropped for answering any excluded one.
    """

    method: EmbeddingMethod = 't-SNE'
    source: Source = 'embeddings'
    color_by: str | None = None
    filter: dict[str, list[Any]] | None = None
    exclude: dict[str, list[Any]] | None = None


class Params(ParamsBase):
    """ Every parameter the GUI holds, one section per pipeline step. """

    paths: PathParams = Field(default_factory=PathParams)
    process: ProcessParams = Field(default_factory=ProcessParams)
    properties: PropertiesParams = Field(default_factory=PropertiesParams)
    clustering: ClusteringParams = Field(default_factory=ClusteringParams)
    prototypes: PrototypesParams = Field(default_factory=PrototypesParams)
    outliers: OutliersParams = Field(default_factory=OutliersParams)
    explorer: ExplorerParams = Field(default_factory=ExplorerParams)
