"""Dataset for loading consolidated spectrograms from single file"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConsolidatedSpectrogramDataset(Dataset):
    """
    Dataset that loads all spectrograms from a single consolidated .pt file.
    Much more efficient than loading millions of individual files.
    """
    
    def __init__(
        self,
        dataset_path: str,
        normalize: bool = True,
        sequence_length: int = 10,
        return_sequences: bool = True
    ):
        self.dataset_path = Path(dataset_path)
        self.normalize = normalize
        self.sequence_length = sequence_length
        self.return_sequences = return_sequences
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        logger.info(f"📂 Loading dataset from {dataset_path}...")
        
        # Load consolidated file
        if str(dataset_path).endswith('.gz'):
            import gzip
            with gzip.open(dataset_path, 'rb') as f:
                data = torch.load(f)
        else:
            data = torch.load(dataset_path)
        
        self.data = data['data']
        self.stats = data['stats']
        self.config = data.get('config', {})
        
        if self.return_sequences:
            self._build_sequence_index()
        
        logger.info(f"✅ Loaded {len(self)} samples")
    
    def _build_sequence_index(self):
        """Build index of valid sequences grouped by file"""
        file_segments = defaultdict(list)
        for idx, item in enumerate(self.data):
            file_segments[item['file_id']].append(idx)
        
        self.sequence_index = []
        for file_id, indices in file_segments.items():
            indices_sorted = sorted(indices, key=lambda i: self.data[i]['segment_idx'])
            for i in range(len(indices_sorted) - self.sequence_length + 1):
                self.sequence_index.append(indices_sorted[i:i + self.sequence_length])
        
        logger.info(f"📦 Created {len(self.sequence_index)} sequences from {len(file_segments)} files")
    
    def __len__(self) -> int:
        if self.return_sequences:
            return len(self.sequence_index)
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if self.return_sequences:
            indices = self.sequence_index[idx]
            spectrograms = []
            for i in indices:
                spec = self.data[i]['spectrogram']
                if self.normalize:
                    spec = self._normalize(spec)
                spectrograms.append(spec)
            
            spectrograms = torch.stack(spectrograms)
            first_item = self.data[indices[0]]
            style_vector = first_item['style_vector']
            
            metadata = {
                'file_id': first_item['file_id'],
                'segment_indices': [self.data[i]['segment_idx'] for i in indices]
            }
        else:
            item = self.data[idx]
            spectrograms = item['spectrogram']
            if self.normalize:
                spectrograms = self._normalize(spectrograms)
            style_vector = item['style_vector']
            
            metadata = {
                'file_id': item['file_id'],
                'segment_idx': item['segment_idx']
            }
        
        return spectrograms, style_vector, metadata
    
    def _normalize(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Normalize Log-Mel spectrograms using per-mel-band statistics.
        
        Note: Statistics are computed from Log-Mel data during dataset creation.
        This standardizes each mel band to ~zero mean and unit variance, which
        helps training convergence.
        """
        mean = self.stats['mean'].unsqueeze(1)
        std = self.stats['std'].unsqueeze(1)
        return (spectrogram - mean) / (std + 1e-6)
    
    def get_sample_shape(self) -> Tuple[int, ...]:
        sample, _, _ = self[0]
        return sample.shape
    
    def get_file_ids(self) -> List[str]:
        return list(set(item['file_id'] for item in self.data))
