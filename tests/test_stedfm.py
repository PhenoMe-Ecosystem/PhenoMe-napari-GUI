"""Tests for the STED-FM model as this repository builds it.

The model has to be the one the stedfm package builds, part for part, so that its state_dict and its
embeddings are the same. What these tests hold is the behaviour that makes them the same: the tokens
the embedding is pooled from, what the classification head is given, and what stays frozen.

The model is built with the weights timm initialises it with: nothing is downloaded, the real weights
mattering to none of this. Where the weights themselves are tested, the download is stood in for by
an archive built here, laid out as the published one is.
"""

import hashlib
import shutil
import zipfile
from pathlib import Path
from urllib.error import URLError

import pytest
import torch

from phenome_gui.registries import stedfm
from phenome_gui.registries.stedfm import STEDFM

# Two single-channel images, at the size the model was trained on
BATCH = 2
IMAGES = torch.rand(BATCH, 1, 224, 224)

# A ViT-S/16 on a 224x224 image: 14 x 14 patches of 384 features, and the class token ahead of them
PATCHES = 196
DIM = 384


@pytest.fixture(scope='module')
def model() -> STEDFM:
    """ A model in evaluation, shared by the tests that only read from it. """
    return STEDFM().eval()


def pooled_with(model: STEDFM, global_pool: str, **kwargs):
    """ The features of IMAGES, pooled the given way by the same network.

    Args:
        model: The model whose weights the features are computed with.
        global_pool: The pooling to read the features with.
        kwargs: Forwarded to forward_features.
    """
    pooled = STEDFM(global_pool=global_pool).eval()
    pooled.load_state_dict(model.state_dict())
    with torch.no_grad():
        return pooled.forward_features(IMAGES, **kwargs)


def tokens_of(model: STEDFM) -> torch.Tensor:
    """ Every token the backbone gives for IMAGES, before any pooling. """
    with torch.no_grad():
        return model.backbone.forward_features(IMAGES)


# ----------------------------------------------------------------------------
# The embedding of an image
# ----------------------------------------------------------------------------


def test_the_embedding_is_the_mean_of_the_patch_tokens(model: STEDFM) -> None:
    """The default, which the embeddings are extracted with: the class token is left out of it."""
    features = pooled_with(model, 'avg')

    assert features.shape == (BATCH, DIM)
    assert torch.allclose(features, tokens_of(model)[:, 1:].mean(dim=1))


def test_the_class_token_can_be_the_embedding(model: STEDFM) -> None:
    features = pooled_with(model, 'token')

    assert features.shape == (BATCH, DIM)
    assert torch.equal(features, tokens_of(model)[:, 0])


def test_every_patch_token_can_be_kept(model: STEDFM) -> None:
    """One embedding per patch, the class token still left out."""
    features = pooled_with(model, 'patch')

    assert features.shape == (BATCH, PATCHES, DIM)
    assert torch.equal(features, tokens_of(model)[:, 1:])


def test_the_tokens_can_be_returned_with_the_embedding(model: STEDFM) -> None:
    features, tokens = pooled_with(model, 'avg', return_patches=True)

    assert features.shape == (BATCH, DIM)
    assert torch.equal(tokens, tokens_of(model))
    assert tokens.shape == (BATCH, 1 + PATCHES, DIM)


def test_an_unknown_pooling_is_refused() -> None:
    """Refused as the model is built, rather than once images are already on their way through it."""
    with pytest.raises(ValueError, match='avg'):
        STEDFM(global_pool='max')


# ----------------------------------------------------------------------------
# The classification head
# ----------------------------------------------------------------------------


def test_the_head_classifies_the_embedding(model: STEDFM) -> None:
    """The features come back beside the logits, as the stedfm probe returns them."""
    with torch.no_grad():
        logits, features = model(IMAGES)

    assert logits.shape == (BATCH, 4)
    assert torch.equal(features, model.forward_features(IMAGES))


def test_the_head_predicts_the_number_of_classes_asked_for() -> None:
    with torch.no_grad():
        logits, _ = STEDFM(num_classes=7).eval()(IMAGES)

    assert logits.shape == (BATCH, 7)


# ----------------------------------------------------------------------------
# What is frozen
# ----------------------------------------------------------------------------


def test_the_backbone_is_frozen(model: STEDFM) -> None:
    """The pretrained network is kept as it is: only the head could ever learn."""
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classification_head.parameters())


def test_training_leaves_the_backbone_in_evaluation() -> None:
    """In training mode the backbone would run its dropout, and give the same image other features."""
    model = STEDFM()

    assert model.train() is model
    assert model.training
    assert model.classification_head.training
    assert not any(module.training for module in model.backbone.modules())


# ----------------------------------------------------------------------------
# The names its weights are saved under
# ----------------------------------------------------------------------------


def test_the_weights_are_saved_under_the_names_of_the_stedfm_probe(model: STEDFM) -> None:
    """The names a state_dict of the stedfm package's linear probe is saved under, so it loads here."""
    keys = set(model.state_dict())
    head_keys = {key for key in keys if key.startswith('classification_head.')}

    assert head_keys == {
        'classification_head.0.running_mean',
        'classification_head.0.running_var',
        'classification_head.0.num_batches_tracked',
        'classification_head.1.weight',
        'classification_head.1.bias',
    }
    assert all(key.startswith('backbone.') for key in keys - head_keys)


def test_the_backbone_keeps_the_head_of_its_own(model: STEDFM) -> None:
    """timm's 1000-class head is part of the pretrained checkpoint, and loads with the rest of it."""
    assert model.state_dict()['backbone.head.weight'].shape == (1000, DIM)


# ----------------------------------------------------------------------------
# The pretrained weights
# ----------------------------------------------------------------------------


@pytest.fixture(scope='module')
def archive(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """ An archive laid out as the published STED-FM weights are, holding a randomly initialised backbone.

    A Lightning checkpoint of the MAE the backbone was pretrained in: the backbone's tensors under
    backbone.vit., beside the MAE's own mask token and decoder, which the cache has no use for.

    Returns:
        The archive's path and SHA-256, and the backbone's tensors as they should come out of it.
    """
    backbone = STEDFM().backbone.state_dict()
    state_dict = {f'backbone.vit.{name}': tensor for name, tensor in backbone.items()}
    state_dict['backbone.mask_token'] = torch.zeros(1, 1, DIM)
    state_dict['decoder.decoder_pred.weight'] = torch.zeros(256, 512)

    folder = tmp_path_factory.mktemp('published')
    checkpoint = folder / 'pl_checkpoint-999.pth'
    torch.save({'epoch': 999, 'state_dict': state_dict}, checkpoint)

    path = folder / 'mae-small-sted.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as files:
        files.write(checkpoint, 'baselines/mae-small_STED/pl_checkpoint-999.pth')

    return {
        'path': path,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'backbone': backbone,
    }


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, archive: dict) -> dict:
    """ A torch hub of the test's own, and a download serving the archive rather than the network.

    Returns:
        The hub's checkpoints folder, and the downloads made, one destination per download.
    """
    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path / 'hub'))
    monkeypatch.setattr(stedfm, 'WEIGHTS_SHA256', archive['sha256'])

    downloads = []

    def _download(url: str, destination: str, *args, **kwargs) -> None:
        downloads.append(destination)
        shutil.copyfile(archive['path'], destination)

    monkeypatch.setattr(stedfm, 'download_url_to_file', _download)
    return {'checkpoints': tmp_path / 'hub' / 'checkpoints', 'downloads': downloads}


def cached_files(hub: dict) -> list[str]:
    """ Everything left in the hub's checkpoints folder, folders included. """
    return sorted(path.name for path in hub['checkpoints'].iterdir())


def test_the_weights_are_downloaded_once_and_kept_as_the_backbone_alone(hub: dict) -> None:
    """The archive and anything written on the way to the backbone file are let go of."""
    stedfm.load_backbone_weights()

    assert len(hub['downloads']) == 1
    assert cached_files(hub) == [stedfm.BACKBONE_FILENAME]


def test_the_weights_kept_are_read_back_without_the_network(hub: dict) -> None:
    first = stedfm.load_backbone_weights()
    second = stedfm.load_backbone_weights()

    assert len(hub['downloads']) == 1
    assert first.keys() == second.keys()


def test_the_backbone_is_kept_without_what_it_was_pretrained_with(hub: dict, archive: dict) -> None:
    """Only the backbone's tensors, under the timm ViT's own names: no mask token, no decoder."""
    weights = stedfm.load_backbone_weights()

    assert weights.keys() == archive['backbone'].keys()
    assert all(torch.equal(weights[name], tensor) for name, tensor in archive['backbone'].items())


def test_the_backbone_kept_loads_into_the_model(hub: dict) -> None:
    """Strictly: a name missing or left over would mean weights left out, or put where they do not go."""
    model = STEDFM()

    model.backbone.load_state_dict(stedfm.load_backbone_weights(), strict=True)


def test_weights_that_are_not_the_published_ones_are_refused(
    hub: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file replaced at the address, or corrupted on the way, is never run nor kept."""
    monkeypatch.setattr(stedfm, 'WEIGHTS_SHA256', '0' * 64)

    with pytest.raises(RuntimeError, match='SHA-256'):
        stedfm.load_backbone_weights()

    assert cached_files(hub) == []


def test_a_failed_download_says_where_from_and_where_to(hub: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    def _offline(url: str, destination: str, *args, **kwargs) -> None:
        raise URLError('nodename nor servname provided')

    monkeypatch.setattr(stedfm, 'download_url_to_file', _offline)

    with pytest.raises(RuntimeError) as raised:
        stedfm.load_backbone_weights()

    assert stedfm.WEIGHTS_URL in str(raised.value)
    assert str(stedfm.backbone_cache_path()) in str(raised.value)
    assert cached_files(hub) == []


def test_a_download_cut_off_partway_leaves_nothing_behind(
    hub: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever it wrote goes with it, so the next call downloads the whole file again."""
    def _cut_off(url: str, destination: str, *args, **kwargs) -> None:
        Path(destination).write_bytes(b'PK\x03\x04 the first few bytes')
        raise ConnectionResetError('Connection reset by peer')

    monkeypatch.setattr(stedfm, 'download_url_to_file', _cut_off)

    with pytest.raises(RuntimeError, match='Could not download'):
        stedfm.load_backbone_weights()

    assert cached_files(hub) == []
