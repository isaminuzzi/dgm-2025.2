# Preprocessing Module

The preprocessing module is responsible for preparing audio data for the spectrogram dreamer model. It handles dataset validation, audio processing, and mel-spectrogram generation with segmentation for training.

## Overview

This module implements a complete preprocessing pipeline that:

1. **Validates audio datasets** using crowdsourced voting metadata
2. **Generates mel-spectrograms** from audio files using torchaudio
3. **Segments spectrograms** into fixed-length windows for model training
4. **Computes global normalization statistics** for consistent data preprocessing

## Module Components

### 1. `dataset_cleaner.py` - DatasetCleaner

Filters and validates audio files based on crowdsourced voting metadata.

#### Class: `DatasetCleaner`

**Purpose:** Filters audio files from a metadata file based on voting criteria and copies validated files to an output directory.

**Constructor Parameters:**
- `metadata_file_path` (str): Path to the TSV metadata file containing voting information
- `clips_dir_path` (str): Directory containing the original audio clips
- `output_dir` (str): Directory where validated audio files will be copied
- `min_votes` (int, default=2): Minimum number of up-votes required to consider an audio clip as validated

**Key Methods:**

- `run()`: Executes the complete validation pipeline
  - Loads metadata from TSV file
  - Filters clips based on voting criteria (more up-votes than down-votes, meets minimum votes)
  - Copies validated audio files to output directory
  - Saves filtered metadata as `validated_metadata.tsv`

- `_load_dataframe()`: Loads and normalizes the metadata TSV file
- `_filter_data(df)`: Applies filtering logic to identify validated clips
- `_copy_validated_files(validated_clips)`: Copies validated audio files and saves metadata

**Example Usage:**
```python
from dataset_cleaner import DatasetCleaner

cleaner = DatasetCleaner(
    metadata_file_path='data/data-file/validated.tsv',
    clips_dir_path='data/new-clip/',
    output_dir='data/1_validated-audio/',
    min_votes=2
)
cleaner.run()
```

---

### 2. `generate_spectrogram.py` - AudioFile

Processes audio files and generates mel-spectrograms with segmentation capabilities.

#### Class: `AudioFile`

**Purpose:** Loads audio files, computes mel-spectrograms, and provides methods for visualization and segmentation.

**Constructor Parameters:**

Audio Parameters:
- `waveform_path` (str): Path to audio file (.mp3 or .wav)

Spectrogram Parameters (with defaults aligned to Henrique's notebook):
- `n_fft` (int, default=1024): FFT size for STFT computation
- `win_length` (int or None, default=20): Window length in milliseconds (automatically converted to samples)
- `hop_length` (int or None, default=10): Hop length in milliseconds (automatically converted to samples)
- `n_mels` (int, default=64): Number of mel frequency bins
- `f_min` (int, default=50): Minimum frequency in Hz
- `f_max` (int, default=7600): Maximum frequency in Hz

Segmentation Parameters:
- `segment_duration` (float, default=0.1): Duration of each segment in seconds
- `overlap` (float, default=0.5): Overlap ratio between consecutive segments (0.0-1.0)

**Key Properties & Methods:**

- `mel_spectrogram` (property): Returns the computed mel-spectrogram as a PyTorch tensor
  - Shape: `[channels, n_mels, time_frames]`
  - Uses Slaney normalization and HTK mel scale

- `segment_spectrogram()`: Segments the mel-spectrogram based on `segment_duration` and `overlap`.
  - Returns: List of mel-spectrogram segments as tensors
# Preprocessing Module

The preprocessing module prepares audio for the spectrogram-dreamer pipeline. It covers dataset validation, mel-spectrogram generation, sliding-window segmentation, and creation of per-file style vectors used by the model.

## Overview

The module performs the following steps:

- Validate and filter raw audio using crowdsourced metadata (optional)
- Generate mel-spectrograms from validated audio
- Segment spectrograms into fixed-length overlapping windows
- Extract and save per-file style vectors (from metadata)
- Compute and save global normalization statistics for training

## Components

### `dataset_cleaner.py` — DatasetCleaner

Filters audio files using a TSV metadata file (crowd votes) and copies validated clips to a target directory.

Key points:
- Input: TSV metadata with vote counts
- Output: Validated audio files and `validated_metadata.tsv` in the configured output directory
- Typical use: run once to produce `data/1_validated-audio/`

Example:

```python
from src.preprocessing.dataset_cleaner import DatasetCleaner

cleaner = DatasetCleaner(
    metadata_file_path='data/data-file/validated.tsv',
    clips_dir_path='data/new-clip/',
    output_dir='data/1_validated-audio/',
    min_votes=2
)
cleaner.run()
```

### `generate_spectrogram.py` — AudioFile

Loads audio, computes mel-spectrograms, supports segmentation, visualization, and style-vector extraction.

Important behaviors:
- Supports `.mp3` and `.wav` inputs
- Uses Slaney normalization and HTK mel scale for consistency
- Converts time-based `win_length`/`hop_length` (ms) to samples internally
- Exposes `.mel_spectrogram` (PyTorch tensor) and `segment_spectrogram()`

Common parameters (constructor):
- `waveform_path` (str)
- `n_fft` (int)
- `win_length` (ms)
- `hop_length` (ms)
- `n_mels` (int)
- `f_min`, `f_max` (Hz)
- `segment_duration` (s)
- `overlap` (0.0–1.0)

Example:

```python
from src.preprocessing.generate_spectrogram import AudioFile

audio = AudioFile('path/to/audio.mp3', n_fft=512, win_length=20, hop_length=10)
mel = audio.mel_spectrogram
segments = audio.segment_spectrogram()
```

### `pipeline.py` — Pipeline

High-level orchestrator that ties dataset cleaning, mel generation, segmentation, style extraction, and stats computation.

Constructor highlights:
- `input_dir` (str): Directory with validated audio files
- `output_dir` (str): Where to save segmented mel files and `mel_stats.pt`
- `style_vector_dir` (str): Directory where per-file style vectors are written
- `file_extension` (default: `mp3`)
- Spectrogram params: `n_fft`, `win_length`, `hop_length`, `n_mels`, `f_min`, `f_max`
- Segmentation params: `segment_duration`, `overlap`

Notes about usage:
- `run_dataset_cleaner(metadata_file, clips_dir, min_votes)` will run `DatasetCleaner` and copy validated audio into `input_dir`.
- `process(metadata_file)` requires a metadata TSV to build the global style map (used to extract per-file style vectors). The method will:
  - load global styles from `AudioFile.load_global_styles(metadata_file)`
  - iterate over audio files in `input_dir`
  - compute and save overlapping mel segments to `output_dir/<audio_name>/<audio_name>_XXXX.pt`
  - save normalization stats to `output_dir/mel_stats.pt`

Example:

```python
from src.preprocessing.pipeline import Pipeline

p = Pipeline(
    input_dir='data/1_validated-audio/',
    output_dir='data/2_mel-spectrograms/',
    style_vector_dir='data/3_style-vectors/',
    file_extension='mp3',
    n_fft=512,
    win_length=20,
    hop_length=10,
    n_mels=64,
    f_min=50,
    f_max=7600,
    segment_duration=0.1,
    overlap=0.5
)

# Optional: validate raw clips first
p.run_dataset_cleaner(metadata_file='data/data-file/validated.tsv', clips_dir='data/new-clip/', min_votes=2)

# Process validated audio. `metadata_file` is required for style extraction.
p.process(metadata_file='data/data-file/validated.tsv')
```

### `launch.py` — Launcher helper

`src/preprocessing/launch.py` provides a convenience example that instantiates `Pipeline` with sensible defaults and calls `p.process(...)`. The file defines a `launch()` function — calling it will execute the example pipeline.

To run the example launcher from a Python REPL or script:

```py
from src.preprocessing.launch import launch
launch()
```

Or run on the command line using Python's `-c` flag:

```bash
python -c "from src.preprocessing.launch import launch; launch()"
```

Note: the example in `launch.py` sets `n_fft=1024` while `Pipeline` defaults to `512`. Adjust parameters as needed.

## Data flow

Raw audio → dataset cleaning (optional) → validated audio → mel generation → segmentation → style vectors + normalization stats

## Output

- Segmented spectrograms: `output_dir/<audio_name>/<audio_name>_0000.pt` (2D tensor: `[n_mels, time_frames]`)
- Style vectors: saved into the `style_vector_dir` provided to `Pipeline`
- Normalization stats: `output_dir/mel_stats.pt` (dict with `mean` and `std`, per-mel-band)
- Validation metadata: `validated_metadata.tsv` (from `DatasetCleaner`)

## Key parameters

- `n_fft` (default in `Pipeline`): 512 — FFT size for STFT
- `win_length` / `hop_length`: provided in ms, converted internally to samples
- `n_mels`: number of mel bins (default 64)
- `segment_duration`: seconds per segment (default 0.1)
- `overlap`: overlap ratio (default 0.5)

## Running the pipeline (quick steps)

1. (Optional) Validate raw clips:

```bash
python -c "from src.preprocessing.dataset_cleaner import DatasetCleaner; DatasetCleaner('data/data-file/validated.tsv','data/new-clip/','data/1_validated-audio/').run()"
```

2. Run processing (from Python):

```bash
python -c "from src.preprocessing.pipeline import Pipeline; p=Pipeline('data/1_validated-audio/','data/2_mel-spectrograms/','data/3_style-vectors/'); p.process('data/data-file/validated.tsv')"
```

3. Or call the launcher helper:

```bash
python -c "from src.preprocessing.launch import launch; launch()"
```

## Dependencies

- `torch`
- `torchaudio`
- `pandas`
- `numpy`
- `tqdm`
- `matplotlib` (optional, for plotting)

## Logging & errors

- Uses `src.utils.logger.get_logger` for informative logs
- Files that fail to load are skipped with warnings
- Empty spectrograms are skipped
- Output directories are created automatically

## Notes

- The pipeline requires a metadata TSV for style extraction — ensure `metadata_file` includes identifiers matching audio filenames (stem only).
- The `launch.py` helper provides a quick example but does not run automatically when executed as a script unless you call `launch()` (see examples above).
- The saved normalization file contains per-band means and stds; use them to normalize segments before training.

If you'd like, I can also add a small `if __name__ == '__main__': launch()` block in `launch.py` so it becomes directly runnable. Want me to do that?
