# --- Mel Cepstral Distance (MCD) ---
# 
# The MCD is inspired on the classic Cdpstral Distance (CD) metric and
# is essentially an implementation of CD on the Mel frequency scale.
# This aims to better measure the quality of the generated audio based on the spectrogram.
#
# To our implementation, calculating MCD is a measure of how good the generated spectrogram 
# from the world model is and, therefore, how good is the World Model itself (autoencoder).
# 
# Original paper: https://ieeexplore.ieee.org/document/407206
# For this implementation, we'll be using this library: 

# general
import os
from typing import List, Tuple
from pathlib import Path

# mel cepstral distance imports
from mel_cepstral_distance import compare_audio_files

# internal imports
from base import BaseEvaluator
from ..utils.logger import get_logger

# set up logging
_logger = get_logger("fad_evaluator", level="DEBUG")


class MCD(BaseEvaluator):
    def __init__(self, 
                 path_to_ground_audio: Path | str,
                 path_to_generated_audio: Path | str):
        """
        MCD measures the distance between the Mel-frequency cepstral coefficients (MFCCs)
        of the generated audio and the reference audio. It effectively quantifies how 
        different the "timbre" or spectral envelope of the generated speech is from 
        the ground truth.

        Parameters:
            path_to_ground_audio (Path | str): Path to the ground truth audio dir
            path_to_generated_audio (Path | str): Path to the generated audio dir
        """
        # loading audios
        if os.path.exists(path_to_ground_audio):
            path_to_ground_audio_path = Path(path_to_ground_audio)
        else:
            _logger.error(f"Path '{path_to_ground_audio}' not found")
            raise FileNotFoundError(f"Path '{path_to_ground_audio}' not found")

        if os.path.exists(path_to_generated_audio):
            path_to_generated_audio_path = Path(path_to_generated_audio)
        else:
            _logger.error(f"Path '{path_to_generated_audio}' not found")
            raise FileNotFoundError(f"Path '{path_to_generated_audio}' not found")

        generated_audio = self._find_files(path_to_generated_audio)
        ground_audio = self._find_files(path_to_ground_audio)

        self.paired_files = self._pair_files(ground_audio, generated_audio)

    def evaluate(self) -> Tuple[List[float], List[float]]:
        """
        Compute MCD between two inputs: audio files, amplitude spectrograms, Mel spectrograms, or MFCCs.
        Calculate an alignment penalty (PEN) as an additional metric to indicate the extent of alignment applied.

        Returns:
            Tuple: mcd and penalty
        """
        mcd_list = []
        pen_list = []
        for files in self.paired_files:
            mcd, penalty = compare_audio_files(
                files[0],
                files[1],
            )
            mcd_list.append(mcd)
            pen_list.append(penalty)

        _logger.info(f"Mean calculated MCD is: {mcd_list.mean():.2f}. Mean calculated penalty (PEN): {pen_list.mean():.4f}")

        return mcd_list, pen_list
    
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
