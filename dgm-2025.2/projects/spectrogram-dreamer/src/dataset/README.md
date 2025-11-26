# Dataset module

This module contains utilities to work with the project's **Log-Mel spectrogram** dataset. It provides a small, focused helper to create PyTorch `DataLoader`s for datasets that are already loaded or cached in memory.

## Important Note: Log-Mel Spectrograms Only

All spectrograms in this project are **Log-Mel spectrograms** (natural log of power), NOT Power spectrograms. This is essential for audio quality. See [LOG_MEL_IMPLEMENTATION.md](../../LOG_MEL_IMPLEMENTATION.md) for details.

The dataset code in `src/dataset` expects the project's data layout (rooted at the repository `data/` directory):

- `data/1_validated-audio/` — original validated audio and metadata (e.g. `validated_metadata.tsv`).
- `data/2_mel-spectrograms/` — per-clip precomputed **Log-Mel** spectrograms organized by speaker or split.
- `data/3_style-vectors/` — optional style / embedding vectors per clip.

Purpose
-------

The dataset module provides small glue utilities that standardize how datasets are converted into PyTorch `DataLoader`s for training and evaluation. The helper `create_dataloader` configures a `DataLoader` with defaults appropriate for this project (notably `num_workers=0` to avoid duplicating large in-memory datasets).

Key API
-------

- `create_dataloader(dataset: torch.utils.data.Dataset, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0) -> DataLoader`
    - Creates and returns a `torch.utils.data.DataLoader` configured with `pin_memory=True` and a default of `num_workers=0` (critical for this project to avoid memory duplication when dataset is preloaded).

Usage example
-------------

Example minimal usage showing how to plug a `Dataset` implementation into the helper:

```python
from torch.utils.data import Dataset
from src.dataset.spectrogram_dataloader import create_dataloader

class ExampleSpectrogramDataset(Dataset):
    def __init__(self, data_root: str):
        # load or index precomputed mel-spectrogram files here
        self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # return (spectrogram_tensor, style_vector, metadata)
        return ...

# instantiate dataset
dataset = ExampleSpectrogramDataset("data/2_mel-spectrograms/")

# create dataloader — keep num_workers=0 for this project
dataloader = create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0)

for batch in dataloader:
    # training loop here
    pass
```

Notes & Recommendations
-----------------------

- `num_workers`: The project sets a default of `num_workers=0` in `create_dataloader`. This is intentional because many dataset implementations in this repository load or cache large tensors in memory; using worker processes can duplicate memory and quickly exhaust system RAM.
- `pin_memory`: The `DataLoader` returned by `create_dataloader` uses `pin_memory=True` which helps faster host->GPU transfers when using CUDA.
- Data format: the codebase generally expects precomputed mel-spectrograms and optional style vectors. Keep binary tensor files or serialized numpy arrays in `data/2_mel-spectrograms/` and `data/3_style-vectors/` respectively.

Dependencies
------------

The project declares dependencies in `requirements.txt`. The dataset utilities assume at least:

- `torch` (used for `Dataset`/`DataLoader`)
- `torchaudio` and/or audio processing libraries if you're generating spectrograms on-the-fly
- `numpy` for array handling

Troubleshooting
---------------

- Out of memory: reduce `batch_size` and ensure `num_workers=0`.
- Slow data transfer: keep `pin_memory=True` (already configured) and consider moving preprocessing off the training loop.

Where to extend
---------------

- Implement dataset classes in `src/dataset/` that subclass `torch.utils.data.Dataset` and follow simple I/O conventions (index, length, __getitem__).
- Add helper loaders for on-disk formats used in your project (e.g., `.npy`, `.pt`, or custom binary formats).

If you want, I can add a simple reference Dataset implementation to `src/dataset/` (e.g. `spectrogram_dataset.py`) and wire a small unit test showing the `DataLoader` iterates correctly.
