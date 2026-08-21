"""Tests for the registries naming what each model is loaded, prepared and processed with.

A model is three things at once: the weights, the transformations its images are prepared with, and
the arguments its embeddings are extracted under. The three are registered apart and looked up by the
same key, so what these tests mostly hold is that the three agree on which keys exist.

No test loads real weights: the loaders reach for the network and a few hundred megabytes, so the
model half is stood in for wherever a whole model would otherwise be built.
"""

import pytest
import torchvision.transforms as T

from phenome_gui.registries.models import (
    MODEL_LOADERS,
    MODEL_OPTIONS,
    get_model,
)
from phenome_gui.registries.process_options import (
    OPTIONS_LOADERS,
    get_dinov2_options,
    get_stedfm_options,
)
from phenome_gui.registries.transforms import (
    TRANSFORMS_LOADERS,
    get_dinov2_transforms,
    get_stedfm_transforms,
)

# The models the GUI offers, under the names the registries know them by
MODEL_KEYS = sorted(MODEL_OPTIONS.values())


def described(transforms) -> str:
    """ A set of transformations as text, two of them never comparing equal to one another.

    Args:
        transforms: The transformations a model's images are prepared with, or None for PhenoMe's own.
    """
    return repr(transforms)


# ----------------------------------------------------------------------------
# The three registries, keyed alike
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('registry', [MODEL_LOADERS, TRANSFORMS_LOADERS, OPTIONS_LOADERS], ids=['models', 'transforms', 'options'])
def test_every_model_the_gui_offers_is_registered(registry: dict) -> None:
    """A model offered in the form and missing from a registry raises the moment it is picked."""
    assert sorted(registry) == MODEL_KEYS


def test_nothing_is_registered_that_cannot_be_picked() -> None:
    """A key in the registries and not in the form is a model nobody can reach."""
    assert sorted(MODEL_LOADERS) == sorted(TRANSFORMS_LOADERS) == sorted(OPTIONS_LOADERS)


@pytest.mark.parametrize('model_key', MODEL_KEYS)
def test_every_model_has_something_to_load_it_with(model_key: str) -> None:
    assert callable(MODEL_LOADERS[model_key])
    assert callable(TRANSFORMS_LOADERS[model_key])
    assert callable(OPTIONS_LOADERS[model_key])


# ----------------------------------------------------------------------------
# What each model is processed with
# ----------------------------------------------------------------------------


def test_a_dinov2_model_is_given_three_channels() -> None:
    """DINOv2 reads colour images, so a single-channel one is repeated across three."""
    assert get_dinov2_options() == {'force_rgb': True}


def test_the_sted_model_is_left_the_channels_it_was_given() -> None:
    """STED-FM was trained on one channel, and normalising across three would rewrite its images."""
    assert get_stedfm_options() == {'force_rgb': False, 'l2_normalize_channels': False}


@pytest.mark.parametrize('model_key', MODEL_KEYS)
def test_the_options_of_a_model_are_arguments_it_can_be_processed_with(model_key: str) -> None:
    """They are handed to PhenoMe as keyword arguments, so their names have to be usable as such."""
    options = OPTIONS_LOADERS[model_key]()

    assert isinstance(options, dict)
    assert all(isinstance(name, str) and name.isidentifier() for name in options)


@pytest.mark.parametrize(
    'model_key', [key for key in MODEL_KEYS if key.startswith('dinov2')]
)
def test_every_dinov2_model_is_processed_alike(model_key: str) -> None:
    """They differ in size alone, which is nothing to the images or the arguments."""
    assert OPTIONS_LOADERS[model_key]() == get_dinov2_options()
    assert described(TRANSFORMS_LOADERS[model_key]()) == described(get_dinov2_transforms())


# ----------------------------------------------------------------------------
# What each model's images are prepared with
# ----------------------------------------------------------------------------


def test_a_dinov2_model_keeps_phenome_s_own_transformations() -> None:
    """None is what PhenoMe reads as a request for the ones it prepares images with itself."""
    assert get_dinov2_transforms() is None


def test_the_sted_model_brings_its_own_transformations() -> None:
    """It was trained on single-channel images of one size, which is what it has to be given."""
    prepared = get_stedfm_transforms()

    assert isinstance(prepared, T.Compose)
    assert [type(step).__name__ for step in prepared.transforms] == [
        'ToPILImage', 'Resize', 'Grayscale', 'ToTensor'
    ]


def test_the_sted_model_is_given_the_size_and_the_channel_it_was_trained_on() -> None:
    steps = {type(step).__name__: step for step in get_stedfm_transforms().transforms}

    assert steps['Resize'].size == (224, 224)
    assert steps['Grayscale'].num_output_channels == 1


# ----------------------------------------------------------------------------
# Asking the registry for a model
# ----------------------------------------------------------------------------


@pytest.fixture
def loaded_without_weights(monkeypatch: pytest.MonkeyPatch):
    """ Return a hook standing in for the model loaders, which would fetch real weights.

    Only the model half is stood in for: the transformations and the arguments come from the
    registries themselves, which is what these tests are about.

    Returns:
        Callable[[], list[dict]]: Called as loaded_without_weights(), returning the calls made.
    """
    def _replace() -> list[dict]:
        calls = []

        def _load(model_name: str, device: str = 'cpu', **kwargs):
            calls.append({'model_name': model_name, 'device': device, **kwargs})
            return f'weights of {model_name}'

        monkeypatch.setattr(
            'phenome_gui.registries.models.MODEL_LOADERS',
            dict.fromkeys(MODEL_LOADERS, _load)
        )
        return calls

    return _replace


@pytest.mark.parametrize('model_key', MODEL_KEYS)
def test_a_model_arrives_with_its_transformations_and_its_arguments(
    model_key: str, loaded_without_weights
) -> None:
    loaded_without_weights()

    model, transforms, options = get_model(model_key)

    assert model == f'weights of {model_key}'
    assert described(transforms) == described(TRANSFORMS_LOADERS[model_key]())
    assert options == OPTIONS_LOADERS[model_key]()


def test_the_model_is_loaded_onto_the_device_it_was_asked_for(loaded_without_weights) -> None:
    calls = loaded_without_weights()

    get_model('dinov2_vits14_reg', device='cuda')

    assert calls[-1] == {'model_name': 'dinov2_vits14_reg', 'device': 'cuda'}


def test_a_model_nobody_registered_is_refused(loaded_without_weights) -> None:
    """Named in the message, along with what there is, rather than raising a bare key error."""
    loaded_without_weights()

    with pytest.raises(ValueError, match='Unknown model'):
        get_model('vit-enormous')


def test_the_refusal_says_what_there_is_instead(loaded_without_weights) -> None:
    loaded_without_weights()

    with pytest.raises(ValueError, match='dinov2_vits14_reg'):
        get_model('vit-enormous')
