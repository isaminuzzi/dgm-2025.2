import h5py
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from ..utils.logger import get_logger

_logger = get_logger("spectrogram_hdf5_dataset", level="INFO")


class SpectrogramH5Dataset(Dataset):
    """Reads consolidated HDF5 dataset created by create_consolidated_dataset.py

    Exposes the same interface as `SpectrogramDataset` (returns sequence dicts with
    'observation', 'action', 'rewards'). It indexes episodes by `file_id` and `seg_idx`
    stored as datasets in the HDF5 file.
    """

    def __init__(self, h5_path: str):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {h5_path}")

        # Open the file in read-only mode; actual file handle is opened lazily in worker
        self.h5 = h5py.File(str(self.h5_path), 'r')

        # datasets
        self.specs = self.h5['spectrograms']  # (N, n_mels, T)
        self.styles = self.h5['styles']       # (N, style_dim)
        self.file_ids = self.h5['file_ids']   # (N,)
        self.seg_idx = self.h5['seg_idx']     # (N,)

        # group by episode name (file_id)
        self.episode_map = self._group_by_file_id()

        # sequence settings (match SpectrogramDataset)
        self.SEQUENCE_LENGTH = 50
        self.OVERLAP = 25
        self.OVERLAP_STEP = self.SEQUENCE_LENGTH - self.OVERLAP

        # precompute sequence indices
        self.sequence_indices = self._generate_seq_idx()

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        episode_name, start_idx = self.sequence_indices[idx]
        segment_indices = self.episode_map[episode_name]

        # gather a contiguous block of segment entries
        selected = segment_indices[start_idx: start_idx + self.SEQUENCE_LENGTH]

        specs = []
        styles = []
        for i in selected:
            specs.append(torch.from_numpy(self.specs[i]))
            styles.append(torch.from_numpy(self.styles[i]))

        spec_sequence = torch.stack(specs).unsqueeze(1)  # (T, C=1, n_mels, time)
        style_sequence = torch.stack(styles)

        rewards = torch.zeros(self.SEQUENCE_LENGTH, dtype=torch.float32)

        return {
            'observation': spec_sequence,
            'action': style_sequence,
            'rewards': rewards
        }

    def _group_by_file_id(self) -> Dict[str, List[int]]:
        """Group segment indices by file_id (episode)
        
        Optimized to load data in chunks instead of one-by-one
        """
        _logger.info(f"Grouping {self.specs.shape[0]:,} segments by file_id...")
        
        mapping: Dict[str, List[int]] = {}
        total_samples = self.specs.shape[0]
        
        # Load in chunks to avoid memory issues with huge datasets
        chunk_size = 100000  # Process 100k at a time
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        # Use tqdm for progress bar
        for chunk_idx in tqdm(range(num_chunks), desc="Grouping segments by file_id", ncols=100):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            
            # Load chunk of file_ids and seg_idx
            file_ids_chunk = self.file_ids[start_idx:end_idx]
            seg_idx_chunk = self.seg_idx[start_idx:end_idx]
            
            # Process chunk
            for i, (raw_fid, seg_id) in enumerate(zip(file_ids_chunk, seg_idx_chunk)):
                global_idx = start_idx + i
                
                # Normalize file_id to string
                if isinstance(raw_fid, (bytes, bytearray)):
                    file_id = raw_fid.decode('utf-8')
                else:
                    file_id = str(raw_fid)
                
                if file_id not in mapping:
                    mapping[file_id] = []
                mapping[file_id].append((global_idx, int(seg_id)))
        
        # Sort by seg_idx within each file
        for fid in mapping:
            mapping[fid].sort(key=lambda x: x[1])  # Sort by seg_idx
            mapping[fid] = [idx for idx, _ in mapping[fid]]  # Keep only indices
        
        _logger.info(f"✅ Found {len(mapping):,} episodes in {self.h5_path}")
        return mapping

    def _generate_seq_idx(self) -> List[Tuple[str, int]]:
        indices = []
        for episode_name, indices_list in self.episode_map.items():
            total_frames = len(indices_list)
            for start_idx in range(0, total_frames - self.SEQUENCE_LENGTH + 1, self.OVERLAP_STEP):
                indices.append((episode_name, start_idx))
        _logger.info(f"Generated {len(indices)} sequence indices from HDF5 dataset")
        return indices

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
