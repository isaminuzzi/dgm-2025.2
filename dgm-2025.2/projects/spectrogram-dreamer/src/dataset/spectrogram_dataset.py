# general
import os
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm

# torch imports
import torch
from torch import Tensor
from torch.utils.data import Dataset

# internal imports
from ..utils.logger import get_logger

# set up logging
_logger = get_logger("dataset", level="INFO")

class SpectrogramDataset(Dataset):
    """Generates the spectogram dataset
    
    The dataset is composed by the generated spectograms broken down into 
    sequential time patches. Each spectrogram (audio file) corresponds to one episode. 
    The actions are given by the style vectors.

    This mimics the process of sampling sequences from episodes from the orginal paper.
    """
    def __init__(self, spectrograms_dir_path: str, style_vectors_dir_path: str):
        super().__init__()
        # globals
        self.DEFAULT_EXT = (".pt") # Allowed extensions
        self.SEQUENCE_LENGTH = 50 # dreamer paper calls this T, unroll_length
        self.OVERLAP = 25         # for smooth transition
        self.CHANNELS = 1         # audio only. Dreamer has 3 (RGB)
        self.OVERLAP_STEP = self.SEQUENCE_LENGTH - self.OVERLAP

        # loading paths
        self.spec_files = self._find_files(Path(spectrograms_dir_path))
        self.style_files = self._find_files(Path(style_vectors_dir_path))

        # pairing paths
        self.paired_paths = self._pair_files(self.spec_files, self.style_files)

        # loading episode data
        self.episode_data = self._group_episodes(self.paired_paths)

        # loading all possible start indices
        self.sequence_indices = self._generate_seq_idx()

    # --- Dataset methods ---
    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, index):
        episode_name, start_index = self.sequence_indices[index]
        episode_segments = self.episode_data[episode_name]

        sequence = episode_segments[start_index : start_index + self.SEQUENCE_LENGTH]

        specs = [s["spectrogram"] for s in sequence]
        styles = [s["style"] for s in sequence]

        spec_sequence = torch.stack(specs)
        style_sequence = torch.stack(styles)

        # adding channel
        spec_sequence = spec_sequence.unsqueeze(self.CHANNELS) 
        _logger.debug(f"spec sequence shape: {spec_sequence.shape}") # (T, C, n_mels, L) -> (50, 1, 64, 10)

        # adding dummy rewards
        rewards = torch.zeros(self.SEQUENCE_LENGTH, dtype=torch.float32)

        return {
            "observation": spec_sequence,
            "action": style_sequence,
            "rewards": rewards,
        }

    # --- Helper functions ---
    def _load_torch(self, path: Path):
        return torch.load(path, map_location="cpu")
    
    def _get_segment_info(self, path: Path) -> Tuple[str, int]:
        """Extracts the base episode name from parents dir and the frame index from the last part
        of the audio stem

        Args:
            path (Path): path of the episode
        
        Returns:
            Tuple: episode name and frame index
        """
        base_name = path.parent.name # example: 'common_voice_en_10887'
        frame_idx_str = path.stem.split('_')[-1] # extracting the frame index

        if not frame_idx_str.isdigit():
            _logger.error(f"Could not extract frame index from {path.stem}")
            raise ValueError(f"Could not extract frame index from {path.stem}")
        
        return base_name, int(frame_idx_str)

    def _group_episodes(self, paired_paths: List[Tuple[Path, Path]]) -> Dict[str, List[Dict[str, Tensor]]]:
        episode_data = {}

        for spec_path, style_path in tqdm(paired_paths, desc="Loading and grouping segments"):
            try:
                base_name, frame_idx = self._get_segment_info(spec_path)
            except ValueError:
                _logger.warning(f"Error extracting information from episode {spec_path}. Skipping...")
                continue

            # load the tensors
            spec_tensor = self._load_torch(spec_path)
            style_tensor = self._load_torch(style_path)

            if base_name not in episode_data:
                episode_data[base_name] = []
            
            # store the loaded tensors
            episode_data[base_name].append({
                "spectrogram": spec_tensor,
                "style": style_tensor,
                "frame_idx": frame_idx
            })
        
        # sort by frame_idx
        for base_name in episode_data:
            episode_data[base_name].sort(key=lambda x: x["frame_idx"])
        
        return episode_data
    
    def _generate_seq_idx(self) -> List[Tuple[str, int]]:
        """Generates all possible indices that could be the starting index.

        In the original Dreamer implementation, they select a random index to start the episode.
        We must ensure that all kepisodes are T (unroll_length) sized, therefore, we need to select only the index that can
        safely produce T sized episode lengths.

        >>> Example: L=50, W=16
               Total Window Size = 50 + 16 - 1 = 65 Time Steps

               The `unfold` operation efficiently handles this:
               - Patch x_1: covers indices 0 to 15.
               - Patch x_2: covers indices 1 to 16.
               ...
               - Patch x_50: covers indices 49 to 64.
        
        Returns:
            List[Tuple[str, int]]: list of possible episode_names and starting index
        """
        indices = []

        for episode_name, segments in self.episode_data.items():
            total_frames = len(segments)
            _logger.info(f"Total frames: {total_frames}")

            for start_idx in range(0, total_frames - self.SEQUENCE_LENGTH + 1, self.OVERLAP_STEP):
                indices.append((episode_name, start_idx))
        
        _logger.info(f"Generated {len(indices)} total sequences of length {self.SEQUENCE_LENGTH} with a step of {self.OVERLAP_STEP}.")
        return indices

    def _find_files(self, path: Path) -> List[Path]:
        """Finds the extracted tensor files

        Args:
            path (Path): Directory path of the vector files

        Returns:
            List[Paths]: all the found paths inside the dir        
        """
        if not path.exists():
            _logger.error(f"Path '{path}' was not found.")
            raise FileNotFoundError(f"Path '{path}' was not found.")
        if path.is_file():
            _logger.error(f"Path is '{path}' is a file. Expected folder path.")
            raise Exception(f"Path is '{path}' is a file. Expected folder path.")
        files: List[Path] = []
        for root, _, filenames in os.walk(path):
            for fn in filenames:
                if fn.lower().endswith(self.DEFAULT_EXT):
                    files.append(Path(root) / fn)
        
        _logger.info(f"Found {len(files)} at '{path}'")
        return files
    
    def _pair_files(self, spec_files: List[Path], style_files: List[Path]) -> List[Tuple[Path, Path]]:
        """Pair the found files by file name
        
        Args:
            spec_files (List[Path]): List of spectrogram files
            style_files (List[Path]): List of style vector files
        
        Returns:
            List[Tuple[Path, Path]]: matched spectrogram with style vector
        """
        style_map = {y.stem: y for y in style_files}
        samples: List[Tuple[Path, Path]] = []

        _logger.info("Initiating files pairing...")

        for i, spec in enumerate(spec_files):
            style = style_map.get(spec.stem)
            samples.append((spec, style))

        if any(style is None for _, style in samples):
            _logger.warning(f"Not all spectrograms have matching style vectors.")

        _logger.info(f"Total pairs found: {len(samples)}")    
        return samples




    

