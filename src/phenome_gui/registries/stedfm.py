"""The STED-FM ViT-S model, built in this repository rather than imported from the stedfm package.

STED-FM's own code is licensed CC BY-NC 4.0, which this MIT repository cannot take in, so none of it
is copied here: the model is rebuilt on timm's ViT, with the same structure, the same state_dict keys
and the same embeddings as the linear probe the stedfm package builds around its pretrained backbone.
Only the weights are STED-FM's, and they are downloaded at runtime rather than redistributed.
"""

import torch
from timm.models.vision_transformer import vit_small_patch16_224

# The ways the backbone's tokens can be pooled into the embedding of an image
GLOBAL_POOLS = ('avg', 'token', 'patch')


class STEDFMClassifier(torch.nn.Module):
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

    def train(self, mode: bool = True) -> 'STEDFMClassifier':
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
