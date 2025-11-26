"""DataLoader utilities for consolidated dataset"""

import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Dict, Tuple
from .consolidated_dataset import ConsolidatedSpectrogramDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_consolidated_dataloader(
    dataset_path: str,
    batch_size: int = 32,
    sequence_length: int = 10,
    normalize: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    return_sequences: bool = True,
    **kwargs
) -> DataLoader:
    """Create DataLoader for consolidated dataset"""
    
    dataset = ConsolidatedSpectrogramDataset(
        dataset_path=dataset_path,
        normalize=normalize,
        sequence_length=sequence_length,
        return_sequences=return_sequences
    )
    
    def collate_fn(batch):
        spectrograms, style_vectors, metadatas = zip(*batch)
        spectrograms = torch.stack(spectrograms)
        style_vectors = torch.stack(style_vectors)
        
        aggregated_metadata = {
            'file_ids': [m['file_id'] for m in metadatas]
        }
        
        if return_sequences:
            aggregated_metadata['segment_indices'] = [m['segment_indices'] for m in metadatas]
        else:
            aggregated_metadata['segment_indices'] = [m['segment_idx'] for m in metadatas]
        
        return spectrograms, style_vectors, aggregated_metadata
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        **kwargs
    )
    
    logger.info(f"✅ Created DataLoader: {len(dataset)} samples, {len(dataloader)} batches")
    
    return dataloader


def get_dataset_info(dataset_path: str) -> Dict:
    """Get information about consolidated dataset without loading it fully"""
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    logger.info(f"📊 Loading dataset info from {dataset_path}...")
    
    if str(dataset_path).endswith('.gz'):
        import gzip
        with gzip.open(dataset_path, 'rb') as f:
            data = torch.load(f)
    else:
        data = torch.load(dataset_path)
    
    file_size_mb = dataset_path.stat().st_size / (1024 ** 2)
    
    info = {
        'num_samples': len(data['data']),
        'config': data.get('config', {}),
        'file_size_mb': file_size_mb,
        'num_unique_files': len(set(item['file_id'] for item in data['data']))
    }
    
    logger.info(f"✅ Dataset: {info['num_samples']} samples, {info['num_unique_files']} files, {file_size_mb:.2f} MB")
    
    return info


def create_train_val_dataloaders(
    dataset_path: str,
    val_split: float = 0.1,
    batch_size: int = 32,
    sequence_length: int = 10,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with file-level split"""
    
    dataset = ConsolidatedSpectrogramDataset(
        dataset_path=dataset_path,
        sequence_length=sequence_length,
        return_sequences=kwargs.get('return_sequences', True),
        normalize=kwargs.get('normalize', True)
    )
    
    file_ids = dataset.get_file_ids()
    num_files = len(file_ids)
    num_val_files = max(1, int(num_files * val_split))
    
    logger.info(f"📊 Splitting: {num_files} files → {num_val_files} val, {num_files - num_val_files} train")
    
    import random
    random.seed(42)
    val_file_ids = set(random.sample(file_ids, num_val_files))
    
    train_indices = []
    val_indices = []
    
    if dataset.return_sequences:
        for seq_idx, indices in enumerate(dataset.sequence_index):
            first_file_id = dataset.data[indices[0]]['file_id']
            if first_file_id in val_file_ids:
                val_indices.append(seq_idx)
            else:
                train_indices.append(seq_idx)
    else:
        for idx, item in enumerate(dataset.data):
            if item['file_id'] in val_file_ids:
                val_indices.append(idx)
            else:
                train_indices.append(idx)
    
    logger.info(f"📦 Split: {len(train_indices)} train samples, {len(val_indices)} val samples")
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=kwargs.get('pin_memory', True)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=kwargs.get('pin_memory', True)
    )
    
    logger.info(f"✅ Created: {len(train_dataloader)} train batches, {len(val_dataloader)} val batches")
    
    return train_dataloader, val_dataloader
