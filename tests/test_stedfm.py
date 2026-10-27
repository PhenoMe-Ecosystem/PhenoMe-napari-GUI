"""Tests for the STED-FM model as this repository builds it.

The model has to be the one the stedfm package builds, part for part, so that its state_dict and its
embeddings are the same. What these tests hold is the behaviour that makes them the same: the tokens
the embedding is pooled from, what the classification head is given, and what stays frozen.

The model is built with the weights timm initialises it with: nothing is downloaded, the real weights
mattering to none of this.
"""

import pytest
import torch

from phenome_gui.registries.stedfm import STEDFMClassifier

# Two single-channel images, at the size the model was trained on
BATCH = 2
IMAGES = torch.rand(BATCH, 1, 224, 224)

# A ViT-S/16 on a 224x224 image: 14 x 14 patches of 384 features, and the class token ahead of them
PATCHES = 196
DIM = 384


@pytest.fixture(scope='module')
def model() -> STEDFMClassifier:
    """ A model in evaluation, shared by the tests that only read from it. """
    return STEDFMClassifier().eval()


def pooled_with(model: STEDFMClassifier, global_pool: str, **kwargs):
    """ The features of IMAGES, pooled the given way by the same network.

    Args:
        model: The model whose weights the features are computed with.
        global_pool: The pooling to read the features with.
        kwargs: Forwarded to forward_features.
    """
    pooled = STEDFMClassifier(global_pool=global_pool).eval()
    pooled.load_state_dict(model.state_dict())
    with torch.no_grad():
        return pooled.forward_features(IMAGES, **kwargs)


def tokens_of(model: STEDFMClassifier) -> torch.Tensor:
    """ Every token the backbone gives for IMAGES, before any pooling. """
    with torch.no_grad():
        return model.backbone.forward_features(IMAGES)


# ----------------------------------------------------------------------------
# The embedding of an image
# ----------------------------------------------------------------------------


def test_the_embedding_is_the_mean_of_the_patch_tokens(model: STEDFMClassifier) -> None:
    """The default, which the embeddings are extracted with: the class token is left out of it."""
    features = pooled_with(model, 'avg')

    assert features.shape == (BATCH, DIM)
    assert torch.allclose(features, tokens_of(model)[:, 1:].mean(dim=1))


def test_the_class_token_can_be_the_embedding(model: STEDFMClassifier) -> None:
    features = pooled_with(model, 'token')

    assert features.shape == (BATCH, DIM)
    assert torch.equal(features, tokens_of(model)[:, 0])


def test_every_patch_token_can_be_kept(model: STEDFMClassifier) -> None:
    """One embedding per patch, the class token still left out."""
    features = pooled_with(model, 'patch')

    assert features.shape == (BATCH, PATCHES, DIM)
    assert torch.equal(features, tokens_of(model)[:, 1:])


def test_the_tokens_can_be_returned_with_the_embedding(model: STEDFMClassifier) -> None:
    features, tokens = pooled_with(model, 'avg', return_patches=True)

    assert features.shape == (BATCH, DIM)
    assert torch.equal(tokens, tokens_of(model))
    assert tokens.shape == (BATCH, 1 + PATCHES, DIM)


def test_an_unknown_pooling_is_refused() -> None:
    """Refused as the model is built, rather than once images are already on their way through it."""
    with pytest.raises(ValueError, match='avg'):
        STEDFMClassifier(global_pool='max')


# ----------------------------------------------------------------------------
# The classification head
# ----------------------------------------------------------------------------


def test_the_head_classifies_the_embedding(model: STEDFMClassifier) -> None:
    """The features come back beside the logits, as the stedfm probe returns them."""
    with torch.no_grad():
        logits, features = model(IMAGES)

    assert logits.shape == (BATCH, 4)
    assert torch.equal(features, model.forward_features(IMAGES))


def test_the_head_predicts_the_number_of_classes_asked_for() -> None:
    with torch.no_grad():
        logits, _ = STEDFMClassifier(num_classes=7).eval()(IMAGES)

    assert logits.shape == (BATCH, 7)


# ----------------------------------------------------------------------------
# What is frozen
# ----------------------------------------------------------------------------


def test_the_backbone_is_frozen(model: STEDFMClassifier) -> None:
    """The pretrained network is kept as it is: only the head could ever learn."""
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.classification_head.parameters())


def test_training_leaves_the_backbone_in_evaluation() -> None:
    """In training mode the backbone would run its dropout, and give the same image other features."""
    model = STEDFMClassifier()

    assert model.train() is model
    assert model.training
    assert model.classification_head.training
    assert not any(module.training for module in model.backbone.modules())


# ----------------------------------------------------------------------------
# The names its weights are saved under
# ----------------------------------------------------------------------------


def test_the_weights_are_saved_under_the_names_of_the_stedfm_probe(model: STEDFMClassifier) -> None:
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


def test_the_backbone_keeps_the_head_of_its_own(model: STEDFMClassifier) -> None:
    """timm's 1000-class head is part of the pretrained checkpoint, and loads with the rest of it."""
    assert model.state_dict()['backbone.head.weight'].shape == (1000, DIM)
