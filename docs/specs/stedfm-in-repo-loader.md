---
title: Load STED-FM in-repo and drop the stedfm dependency
status: Draft            # Draft | Approved
date: 2026-10-26
jira_project:            # set by spec-to-jira
jira_epic:               # set by spec-to-jira
---

# Load STED-FM in-repo and drop the stedfm dependency

## Problem Statement

The plugin offers the STED-FM ViT-S model ("STED-FM - Small") for embedding extraction. To do this it depends on the `stedfm` package, which is installed from a Git URL rather than PyPI. That package pulls in about 33 packages that only it needs: Lightning, lightly, TensorBoard, Hydra, statsmodels, seaborn and others. This makes installation heavier, slower and more fragile.

The plugin uses one function from that package. The `stedfm` package also stores its weights in an ad-hoc `~/.stedfm/` folder: the 276 MB download zip and the 300 MB extracted checkpoint, about 576 MB together. It also prints debugging output to the console.

The STED-FM code and weights are licensed CC BY-NC 4.0. This repository is MIT. Copying their code into the repository is therefore not an option.

## Solution

The repository builds and loads the STED-FM model with its own small loader, on top of timm. No STED-FM code is copied. The model the plugin hands to PhenoMe has the same structure, the same `state_dict` keys and the same loaded weights as the one built by `stedfm` today, and its embeddings are numerically identical.

The weights are still downloaded at runtime from the official STED-FM URL. After download, the zip is verified against a fixed SHA-256, and only the backbone tensors are kept in the torch hub cache, as one file of about 86 MB.

Users see no change in the GUI. The model keeps its name and still works with saved parameters and checkpoints. Installation no longer pulls in `stedfm` or its dependency tree.

## Success Criteria

- `stedfm` no longer appears in the project dependencies or the lock file. `timm` is a direct dependency.
- On the same weights, the in-repo model's `state_dict` has the same keys, shapes and dtypes as the model built by `stedfm.get_pretrained_model_v2("mae-lightning-small", "MAE_SMALL_STED", in_channels=1, as_classifier=True)`. Every loaded tensor is equal in value. The randomly initialized 4-class head is compared by keys, shapes and dtypes only.
- On sample images, embeddings from the in-repo model equal those from the `stedfm` model within `atol=1e-5`.
- A first STED-FM run downloads the weights once and leaves a single backbone file of about 86 MB in the torch hub checkpoints folder. Later runs load it without any network access.
- An interrupted download, or one whose hash does not match, leaves nothing in the cache. The user gets an error that names the URL and the cache path.
- The full test suite passes and Ruff reports no issues.

## User Stories

1. As a plugin user, I want to install the plugin without the `stedfm` package, so that installation is lighter and does not depend on a Git URL.
2. As a plugin user, I want to keep picking "STED-FM - Small" in the model list, so that my workflow does not change.
3. As a plugin user, I want parameters and checkpoints saved with the STED-FM model to keep loading, so that earlier work is not lost.
4. As a plugin user, I want STED-FM embeddings identical to the ones I got before, so that results stay comparable with earlier runs.
5. As a plugin user, I want the weights downloaded once and cached, so that later runs start without a network.
6. As a plugin user, I want the cache to take about 86 MB instead of about 576 MB, so that the model uses less disk space.
7. As a plugin user, I want a corrupted or tampered download to be refused, so that I never run on wrong weights.
8. As a plugin user, I want a failed download to show a clear message in the existing "Extraction failed" dialog, so that I know what went wrong and where.
9. As a plugin user, I want an interrupted download to leave no corrupt file behind, so that the next run downloads it again cleanly.
10. As a plugin user upgrading from an earlier version, I want to know that `~/.stedfm/` is no longer used, so that I can delete it to free disk space.
11. As a developer, I want the STED-FM model to keep the same interface as `stedfm`'s linear probe, so that trained STED-FM linear-probe checkpoints load into it unchanged.
12. As a developer, I want to pick the pooling mode (`avg`, `token`, `patch`) and get the patch tokens back if needed, so that alternative embeddings stay possible.
13. As a maintainer, I want no code copied from STED-FM, so that the repository stays MIT-licensed.

## Implementation Decisions

- **Reimplement, don't vendor.** The loader is written from scratch on top of timm. No STED-FM source is copied.
- **Dependencies.** `stedfm` is replaced by `timm>=1.0` in the project dependencies, and the lock file is regenerated. torch and torchvision keep coming through PhenoMe.
- **New module.** A dedicated STED-FM module in the registries package holds:
  - the weights URL and the zip's SHA-256;
  - the download and cache logic;
  - the classifier class;
  - the `STEDFMWrapper` that PhenoMe uses.

  The models registry only wires the module in, the same way DINOv2 is wired from PhenoMe.
- **Weights source.** The official URL is `https://s3.valeria.science/flclab-foundation-models/models/mae-small-sted.zip`. Its SHA-256 is `d0daba42e0087511695243a8bb5f0e756378d13a5181639ccdb40ca3a9dd793d`. The weights are not re-hosted, because the CC BY-NC license would make that redistribution.
- **Caching.**
  1. The zip is downloaded to a temporary file and its hash is checked.
  2. The checkpoint is read straight out of the zip. It is deflated, and it loads with `torch.load(weights_only=True)`.
  3. Only the tensors under `state_dict` whose key starts with `backbone.vit.` are kept, prefix removed, `head.*` included.
  4. They are saved as `stedfm_mae_small_sted_backbone.pth` in the `checkpoints` folder of `torch.hub.get_dir()`. The file is written under a temporary name and renamed, so the write is atomic.
  5. The zip is deleted.

  A cache hit loads that file directly. The cached file is not re-hashed.
- **Errors.** A failed download or a hash mismatch raises a `RuntimeError` that names the URL and the cache path, and leaves no file in the cache. Model loading already runs on the processing worker, which shows the error in the existing "Extraction failed" dialog. No new UI is needed.
- **Model structure.** The model is identical to `stedfm`'s `LinearProbe` as built today:
  - `backbone`: timm `vit_small_patch16_224` with `in_chans=1`, `num_classes=1000` and `pretrained=False`. It is loaded with `strict=True` from the cached tensors. All its parameters are frozen.
  - `classification_head`: `Sequential(BatchNorm1d(384, affine=False, eps=1e-6), Linear(384, 4))`. It is randomly initialized and untrained, as it is today.
  - `train(mode)` keeps the backbone in eval mode and switches only the head.
  - `forward_features(x, return_patches=False)` runs the backbone's `forward_features` and pools by `global_pool`:
    - `avg` (default): mean of the patch tokens, without the CLS token;
    - `token`: the CLS token;
    - `patch`: all patch tokens.

    With `return_patches`, it returns the pooled output together with the raw features.
  - `forward(x)` returns `(logits, features)`.
  - The `state_dict` keys are identical to `stedfm`'s: `backbone.*`, `classification_head.0.*`, `classification_head.1.*`.
  - The partial-freezing path (`num_blocks` other than `"all"`) is not reproduced, because `stedfm` only raises `NotImplementedError` there.
- **Wrapper.** `STEDFMWrapper` keeps its current behaviour: it builds the classifier on the requested device and returns `forward_features` as the embeddings.
- **Registry.** The key `"sted-fm"`, the display name `"STED-FM - Small"`, the STED-FM transforms and the STED-FM process options stay unchanged.
- **Docs.** The README notes that `~/.stedfm/` is no longer used and can be deleted to free about 576 MB.

## Testing Decisions

- Tests check external behaviour, not implementation details. No test touches the real network.
- **Fake download.** In every permanent test, the network download is replaced by a fake. The fake serves a zip built in the test that holds random ViT-S weights with the real checkpoint key layout. The expected hash is patched to match that zip, or deliberately left unmatched for the mismatch test.
- **Seam 1, the registry: `get_model('sted-fm')`.** It covers:
  - a first load downloads once and leaves exactly one backbone file in the cache;
  - a second load is a cache hit and makes no download;
  - a hash mismatch raises and leaves no file;
  - a failed download raises with the URL and the cache path in the message, and leaves no file;
  - the returned wrapper yields `(N, 384)` embeddings.
- **Seam 2, the classifier class built directly.** It covers:
  - `avg` and `token` pooling give `(N, 384)`, and `avg` leaves out the CLS token;
  - `patch` pooling gives all patch tokens;
  - `return_patches` returns the raw features as well;
  - `forward` returns `(logits[N, 4], features)`;
  - the backbone is frozen and stays in eval after `train()`.

  This seam is needed because `get_model` only exposes the default `avg` pooling.
- **Prior art.** The existing registry tests already stand in for model loaders that would fetch real weights. The PhenoMe interface tests already stand in for the download. The new tests follow the same style.
- **One-off parity check.** Before the dependency is removed, a throwaway script compares the in-repo model with `stedfm`'s on the real weights, using the criteria under Success Criteria. It is not committed.

## Out of Scope

- Other STED-FM variants (tiny, base, large) and other pretraining datasets (JUMP, HPA, SIM).
- Training or fine-tuning the linear probe, and partial backbone freezing.
- Re-hosting or redistributing the weights, including a stripped backbone-only file.
- A README or model-description section on the STED-FM weights license.
- Reading, migrating or deleting the existing `~/.stedfm/` cache.
- Any change to the DINOv2 models.

## Open Questions

- None.

## Further Notes

- Assumption: the official download URL stays stable. If FLClab moves the file, the URL and hash constants must be updated.
- Assumption: re-hashing the stripped cache file on each load is not worth it. Integrity is checked once, on the downloaded zip.
- Users who already have `~/.stedfm/` will download the weights once more (276 MB) into the torch hub cache.
- Removing `stedfm` also removes the debugging output it printed to the console.
- The 4-class head's random values cannot match `stedfm`'s even with a fixed seed. `stedfm` uses the random generator for the decoder first, and the in-repo loader never builds the decoder. Two `stedfm` loads never matched each other either.

## Jira tickets

<Filled in by spec-to-jira.>
