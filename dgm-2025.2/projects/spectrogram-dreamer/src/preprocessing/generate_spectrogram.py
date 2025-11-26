"""
Generating LOG-MEL Spectrograms for Dreamer Training

CRITICAL CHANGE: This module generates LOG-MEL spectrograms, NOT Power spectrograms.

Why Log-Mel is Essential:
--------------------------
Deep learning models with MSE loss struggle with Power spectrograms due to massive
dynamic range (0.001 to 1000.0 = 1,000,000x difference). This causes:
  
  1. Model ignores quiet sounds (speech texture, consonants)
  2. Only learns loud peaks (volume spikes, rhythm)  
  3. Generates metallic noise and hiss
  
Log-Mel compresses the range (e.g., -10 to +10 = 20 units), forcing the model
to learn both quiet and loud features equally, matching human hearing perception.

Do NOT switch back to Power spectrograms without understanding these consequences.
"""

# torch imports
import torch
import torchaudio
import torchaudio.transforms as T

# sklearn imports
from sklearn.preprocessing import OneHotEncoder

# viz imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# general
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# internal imports
from ..utils.logger import get_logger 

_logger = get_logger("spectrogram")


class AudioFile:
    def __init__(self, waveform_path: str, 
                 n_fft: int = 1024, # n_fft (>= win_length)
                 win_length: int = 20,
                 hop_length: int = 10, # 5 ou 10 -> 200fps ou 100fps (5 para mais "instantâneo")
                 n_mels: int = 64,
                 f_min: int = 50,
                 f_max: int = 7600,
                 segment_duration: float = 0.1, # in seconds
                 overlap: float = 0.5, 
                 ):
        permitted_extensions = [".mp3", ".wav"]
        file_path = Path(waveform_path)

        if file_path.suffix not in permitted_extensions:
            _logger.error(f"'{file_path.suffix}' not allowed.")
            raise TypeError(f"'{file_path.suffix}' not allowed.")
        
        self.name = file_path.stem
        
        try:
            self.waveform, self.sample_rate = torchaudio.load(str(file_path))
        except Exception as e:
            _logger.error(f"Error loading audio file {waveform_path}: {e}")
            raise
            
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        # convert to frames
        win_length_ms = 20 if win_length is None else win_length
        hop_length_ms = 10 if hop_length is None else hop_length

        self.win_length = max(1, int(self.sample_rate * win_length_ms / 1000))
        self.hop_length = max(1, int(self.sample_rate * hop_length_ms / 1000))

        if self.n_fft < self.win_length:
            new_n_fft = 1 << (self.win_length - 1).bit_length()
            _logger.info(f"Adjusting n_fft from {self.n_fft} -> {new_n_fft} to satisfy win_length <= n_fft")
            self.n_fft = new_n_fft

        self.L = max(1, int(segment_duration * 1000 / hop_length_ms)) # frames per segment
        self.step = max(1, int(self.L * (1 - overlap)))
        self.overlap = overlap
        
        # ============================================================================
        # PRE-CALCULATE MEL TRANSFORM (ALWAYS OUTPUTS LOG-MEL)
        # ============================================================================
        # CRITICAL: We use power=2.0 to get Power spectrogram first, then apply
        # log transformation to get Log-Mel. This avoids the massive dynamic range
        # problem that causes the model to ignore quiet sounds and produce metallic
        # noise artifacts.
        #
        # Power Domain Problem:
        #   - Quiet sound: 0.001
        #   - Loud sound: 1000.0
        #   - Range: 1,000,000x difference
        #   - Model ignores fine details (speech texture) and only learns volume spikes
        #
        # Log-Mel Solution:
        #   - Quiet sound: -7.0 (log scale)
        #   - Loud sound: +7.0 (log scale)  
        #   - Range: ~14 units (compressed, balanced)
        #   - Model learns both quiet and loud features equally
        # ============================================================================
        self._mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min = self.f_min,
            f_max = self.f_max,
            center=True,
            pad_mode="reflect",
            power=2.0,  # Power spectrogram (will be log-transformed)
            norm="slaney",
            n_mels=self.n_mels,
            mel_scale="htk",
        ).to(self.waveform.device)
    
    @property
    def mel_spectrogram(self):
        """
        Returns the LOG-Mel Spectrogram (NOT Power spectrogram).
        
        This is CRITICAL for audio quality:
        - Compresses dynamic range from ~1,000,000x to ~10-20x
        - Forces model to learn quiet details (speech texture) not just volume spikes
        - Prevents metallic noise floor artifacts in generated audio
        - Matches human hearing perception (logarithmic sensitivity)
        
        Returns:
            Log-Mel spectrogram with values typically in range [-10, 10]
            (Natural log of power spectrogram)
        """
        # 1. Get Power Spectrogram (Magnitude^2)
        mel_power = self._mel_transform(self.waveform)
        
        # 2. Apply Natural Log with epsilon to avoid log(0)
        # min=1e-5 corresponds to ~-11.5 in log space (silence floor)
        mel_log = torch.log(torch.clamp(mel_power, min=1e-5))
        
        return mel_log
    
    def view_spectrogram(self, title=None, ylabel="Mel bins", ax=None, save_path=None):
        """Display and optionally save a Log-Mel spectrogram.
        
        NOTE: The spectrogram is already in Log scale (natural log of power).
        We normalize for visualization only, not for training.
        """
        # Get Log-Mel spectrogram
        mel_log = self.mel_spectrogram 
        
        # Convert to numpy for plotting
        mel_vis = mel_log.squeeze(0).detach().cpu().numpy() 

        # Min-max normalization for visualization only
        mel_vis -= mel_vis.min()
        mel_vis /= mel_vis.max() + 1e-8

        plt.figure(figsize=(40, 10), dpi=300)
        plt.axis("off")
        plt.imshow(mel_vis, origin="lower", aspect="auto", cmap="magma")

        if save_path is not None:
            path = Path(save_path).with_suffix(".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, bbox_inches="tight", pad_inches=0)
            _logger.info(f"Saved Log-Mel spectrogram: {path}")
        
    def segment_spectrogram(self) -> List[torch.Tensor]:
        """Segments Log-Mel spectrograms to feed to the Dreamer model.
        
        Returns:
            List of Log-Mel spectrogram segments, each with shape [n_mels, L]
            where values are in natural log space (typically [-10, 10] range)
        """
        # Get Log-Mel spectrogram (already in log space)
        mel_log = self.mel_spectrogram
        
        if mel_log.dim() == 3 and mel_log.shape[0] == 1:
            mel_2d = mel_log.squeeze(0)
        elif mel_log.dim() == 2:
            mel_2d = mel_log
        else:
            mel_2d = mel_log.reshape(mel_log.shape[-2], mel_log.shape[-1])

        step = max(1, int(self.L * (1 - self.overlap)))
        segments = []

        for start in range(0, mel_2d.shape[1] - self.L + 1, step):
            segments.append(mel_2d[:, start:start + self.L])
            
        if len(segments) == 0 and mel_2d.shape[1] > 0:
            current_L = mel_2d.shape[1]
            if current_L < self.L:
                padding = self.L - current_L
                padded_mel = torch.nn.functional.pad(mel_2d, (0, padding), mode='reflect')
                segments.append(padded_mel)
            else:
                segments.append(mel_2d[:, :self.L])
                
        return segments

    @staticmethod
    def _compute_local_style(mel_segment: torch.Tensor, delta_transform: T.ComputeDeltas) -> torch.Tensor:
        """
        Calculates the local style vector from a mel spectrogram segment.
        """
        if mel_segment.dim() == 3:
            mel_segment = mel_segment.squeeze(0)
        
        mel_batch = mel_segment.unsqueeze(0) # [1, n_mels, L]
        delta1_batch = delta_transform(mel_batch)
        delta2_batch = delta_transform(delta1_batch)

        delta1 = delta1_batch.squeeze(0) # [n_mels, L]
        delta2 = delta2_batch.squeeze(0) # [n_mels, L]
        energia = mel_segment.mean(dim=0, keepdim=True) # [1, L]

        feat_delta = delta1.mean(dim=1)  # [n_mels]
        feat_delta2 = delta2.mean(dim=1) # [n_mels]
        feat_energia = energia.mean()    # scalar

        vec = torch.cat([
            feat_energia.unsqueeze(0),
            feat_delta[:10],
            feat_delta2[:10],
        ])
        
        return vec.float()

    def extract_style_vectors(self, global_style: torch.Tensor, output_dir: str):
        """
        Calculates and saves style vectors (local + global) for each segment
        of the audio file as PyTorch Tensors (.pt files).
        """
        _logger.info(f"Extracting style vectors for {self.name}...")
        
        save_path = Path(output_dir) / self.name
        save_path.mkdir(parents=True, exist_ok=True)
        
        device = self.waveform.device
        delta_transform = T.ComputeDeltas().to(device)
        global_style_tensor = global_style.to(device)
        segments = self.segment_spectrogram()

        if not segments:
            _logger.warning(f"No segments generated for {self.name}. Skipping style vector extraction.")
            return

        for i, seg_mel in enumerate(segments):
            seg_mel = seg_mel.to(device)
            local_style = self._compute_local_style(seg_mel, delta_transform)
            style_vector = torch.cat([local_style, global_style_tensor], dim=0)
            
            # Save as .pt file
            seg_name = f"{self.name}_{i:04d}.pt" 
            seg_save_path = save_path / seg_name
            torch.save(style_vector.cpu(), seg_save_path) # Save tensor to CPU
            
        _logger.info(f"Saved {len(segments)} style vectors (as .pt) to {save_path}")
    
    @staticmethod
    def load_global_styles(metadata_file: str) -> Dict[str, torch.Tensor]:
        """
        Loads metadata, creates one-hot encoded global style vectors,
        and builds a map {file_stem: global_style_tensor}.
        
        Args:
            metadata_file: Path to the .tsv metadata file.
        Returns:
            A dictionary mapping file stem (str) to global style (torch.Tensor).
        """
        _logger.info(f"Loading metadata from {metadata_file} to build global styles...")
        try:
            df = pd.read_csv(metadata_file, sep='\t', low_memory=False)
            
            df = df.drop(columns=["client_id", "sentence", "up_votes", "down_votes"], errors='ignore')

            # One-hot encoding for gender
            generos = np.array([row["gender"] for _, row in df.iterrows()]).reshape(-1, 1)
            enc_gen = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            gen_vecs = enc_gen.fit_transform(generos)

            # One-hot encoding for accent
            sotaques = np.array([row["accent"] for _, row in df.iterrows()]).reshape(-1, 1)
            enc_acc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            acc_vecs = enc_acc.fit_transform(sotaques)

            # One-hot encoding for age
            idades = np.array([row["age"] for _, row in df.iterrows()]).reshape(-1, 1)
            enc_idade = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            idade_vecs = enc_idade.fit_transform(idades)

            global_styles = np.concatenate([gen_vecs, acc_vecs, idade_vecs], axis=1)
            global_styles_tensor = torch.tensor(global_styles, dtype=torch.float32)
            
            _logger.info("Building file-to-style map...")
            style_map = {}
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building style map"):
                base = Path(row["path"]).stem
                style_map[base] = global_styles_tensor[idx]
            
            _logger.info(f"Style map built with {len(style_map)} entries.")
            return style_map

        except Exception as e:
            _logger.error(f"Error loading global styles: {e}")
            raise