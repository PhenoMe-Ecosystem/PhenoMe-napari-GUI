"""The STED-FM ViT-S model, built in this repository rather than imported from the stedfm package.

The model is rebuilt on timm's ViT, with the same structure, the same state_dict keys
and the same embeddings as the linear probe the stedfm package builds around its pretrained backbone.
Only the weights are STED-FM's, and they are downloaded at runtime rather than redistributed.
"""

import hashlib
import os
import tempfile
import zipfile
from pathlib import Path

import torch
from timm.models.vision_transformer import vit_small_patch16_224
from torch.hub import download_url_to_file

# The ways the backbone's tokens can be pooled into the embedding of an image
GLOBAL_POOLS = ('avg', 'token', 'patch')

# The official STED-FM ViT-S weights, pretrained on STED images, and the digest the file is checked against
WEIGHTS_URL = 'https://s3.valeria.science/flclab-foundation-models/models/mae-small-sted.zip'
WEIGHTS_SHA256 = 'd0daba42e0087511695243a8bb5f0e756378d13a5181639ccdb40ca3a9dd793d'

# The backbone's tensors alone, as they are kept in the torch hub cache once downloaded
BACKBONE_FILENAME = 'stedfm_mae_small_sted_backbone.pth'

# What the backbone's tensors are named under in the pretraining checkpoint, beside its MAE decoder's
BACKBONE_PREFIX = 'backbone.vit.'


class STEDFM(torch.nn.Module):
    """ The STED-FM backbone, frozen, with a linear classification head on top.

    The structure is that of the stedfm package's linear probe, down to the names of its parts, so a
    state_dict saved from one loads into the other: a timm ViT-S/16 as the backbone, its own 1000-class
    head included, and a batch-normalised linear head the backbone's features are classified by. The
    head is left as it is built, randomly initialised: the embeddings are read before it.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4, global_pool: str = 'avg') -> None:
        """ Build the model, with the backbone's weights as timm initialises them.

        Args:
            in_channels: The number of channels the images are given in, a single one for STED-FM.
            num_classes: The number of classes the classification head predicts.
            global_pool: How the backbone's tokens are pooled into the features: 'avg' for the mean of
                the patch tokens, 'token' for the class token, 'patch' for every patch token.

        Raises:
            ValueError: If global_pool is not one of GLOBAL_POOLS.
        """
        super().__init__()

        if global_pool not in GLOBAL_POOLS:
            raise ValueError(
                f'Unknown pooling: {global_pool!r}. Available poolings: {", ".join(GLOBAL_POOLS)}'
            )
        self.global_pool = global_pool

        self.backbone = vit_small_patch16_224(in_chans=in_channels, num_classes=1000, pretrained=False)
        # Only the head is ever meant to learn: the backbone is the pretrained network, kept as it is
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.classification_head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=self.backbone.embed_dim, affine=False, eps=1e-6),
            torch.nn.Linear(in_features=self.backbone.embed_dim, out_features=num_classes),
        )

    def train(self, mode: bool = True) -> 'STEDFM':
        """ Switch the head between training and evaluation, the backbone staying in evaluation.

        The backbone is frozen, so it is never trained: left in training mode it would only run its
        dropout, and give different features for the same image.

        Args:
            mode: True for training, False for evaluation.

        Returns:
            The model itself.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def forward_features(
        self, x: torch.Tensor, return_patches: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """ The features of a batch of images, pooled from the backbone's tokens.

        Args:
            x: The images, as a (N, C, H, W) tensor.
            return_patches: Whether to also return every token the backbone gave, before pooling.

        Returns:
            The pooled features: (N, D) for 'avg' and 'token', (N, P, D) for 'patch'. With
            return_patches, a tuple of the pooled features and the backbone's (N, 1 + P, D) tokens.
        """
        tokens = self.backbone.forward_features(x)
        # The class token comes first, ahead of the patch tokens
        prefix = self.backbone.num_prefix_tokens

        if self.global_pool == 'avg':
            features = tokens[:, prefix:].mean(dim=1)
        elif self.global_pool == 'token':
            features = tokens[:, 0]
        else:
            features = tokens[:, prefix:]

        if return_patches:
            return features, tokens
        return features

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Classify a batch of images, returning the features they were classified from as well.

        Args:
            x: The images, as a (N, C, H, W) tensor.

        Returns:
            The (N, num_classes) logits, and the pooled features they were computed from.
        """
        features = self.forward_features(x)
        return self.classification_head(features), features


def backbone_cache_path() -> Path:
    """ Where the backbone's weights are kept once downloaded, beside the other torch hub checkpoints.

    Read from torch at every call, so a hub directory set with TORCH_HOME or torch.hub.set_dir() is
    the one used.

    Returns:
        The path of the cached backbone file, whether or not it exists yet.
    """
    return Path(torch.hub.get_dir()) / 'checkpoints' / BACKBONE_FILENAME


def load_backbone_weights() -> dict[str, torch.Tensor]:
    """ The pretrained weights of the STED-FM backbone, downloaded the first time they are asked for.

    The checkpoint published by STED-FM's authors is downloaded from WEIGHTS_URL, checked against
    WEIGHTS_SHA256, and cut down to the backbone, which is all the embeddings are read from: the MAE
    decoder it was pretrained with and the optimiser's state are left out, and the file kept in the
    torch hub cache is a third of the size. Later calls read that file, without the network. The cache
    the stedfm package keeps in ~/.stedfm is never read.

    Returns:
        The backbone's tensors, under the names of the timm ViT they load into with strict=True.
    Raises:
        RuntimeError: The weights could not be downloaded, or what was downloaded is not them.
    """
    path = backbone_cache_path()
    if not path.exists():
        _cache_backbone_weights(path)
    return torch.load(path, map_location='cpu', weights_only=True)


def _cache_backbone_weights(path: Path) -> None:
    """ Download the STED-FM checkpoint, check it, and keep its backbone at the given path.

    Everything is written to a scratch folder beside the cache, removed whatever happens, and the
    backbone file is only renamed into place once it is whole: an interrupted download, a failed
    check or a full disk leaves no file behind, and the next call downloads again.

    Args:
        path: Where the backbone file is kept.
    Raises:
        RuntimeError: The weights could not be downloaded, or what was downloaded is not them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        archive = Path(scratch) / Path(WEIGHTS_URL).name
        try:
            download_url_to_file(WEIGHTS_URL, str(archive))
        except OSError as error:
            raise RuntimeError(
                f'Could not download the STED-FM weights from {WEIGHTS_URL} to {path}: {error}'
            ) from error

        digest = _sha256(archive)
        if digest != WEIGHTS_SHA256:
            raise RuntimeError(
                f'The STED-FM weights downloaded from {WEIGHTS_URL} are not the expected file: their '
                f'SHA-256 is {digest} rather than {WEIGHTS_SHA256}. Nothing was kept at {path}.'
            )

        partial = Path(scratch) / f'{BACKBONE_FILENAME}.part'
        torch.save(_read_backbone(archive), partial)
        os.replace(partial, path)


def _sha256(path: Path) -> str:
    """ The SHA-256 digest of a file, read a chunk at a time.

    Args:
        path: The file to digest.
    Returns:
        The digest, as lower-case hexadecimal.
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_backbone(archive: Path) -> dict[str, torch.Tensor]:
    """ The backbone's tensors, read out of the STED-FM checkpoint archive.

    The archive holds a single Lightning checkpoint, read straight out of it, whose state_dict names
    the backbone's tensors under BACKBONE_PREFIX: they are kept under the names left once it is
    taken off, the timm ViT's own, its 1000-class head included.

    Args:
        archive: The zip file downloaded from WEIGHTS_URL.
    Returns:
        The backbone's tensors.
    Raises:
        RuntimeError: The archive holds no checkpoint, or no backbone in it.
    """
    with zipfile.ZipFile(archive) as files:
        checkpoints = [name for name in files.namelist() if name.endswith('.pth')]
        if len(checkpoints) != 1:
            raise RuntimeError(
                f'The STED-FM weights from {WEIGHTS_URL} hold {len(checkpoints)} checkpoints rather than one.'
            )
        with files.open(checkpoints[0]) as file:
            checkpoint = torch.load(file, map_location='cpu', weights_only=True)

    backbone = {
        name.removeprefix(BACKBONE_PREFIX): tensor
        for name, tensor in checkpoint.get('state_dict', {}).items()
        if name.startswith(BACKBONE_PREFIX)
    }
    if not backbone:
        raise RuntimeError(f'The STED-FM weights from {WEIGHTS_URL} hold no backbone.')
    return backbone
