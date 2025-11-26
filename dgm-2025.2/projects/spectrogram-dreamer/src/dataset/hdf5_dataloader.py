"""DataLoader utilities for HDF5 consolidated dataset"""

import h5py
import torch
from torch.utils.data import DataLoader, Subset, random_split
from pathlib import Path
from typing import Tuple
from .spectrogram_hdf5_dataset import SpectrogramH5Dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_hdf5_dataset_info(dataset_path: str) -> dict:
    """Get information about the HDF5 dataset
    
    Args:
        dataset_path: Path to HDF5 file
        
    Returns:
        Dictionary with dataset information
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with h5py.File(str(dataset_path), 'r') as f:
        num_samples = f['spectrograms'].shape[0]
        
        # Get unique file IDs
        file_ids = f['file_ids'][:]
        if isinstance(file_ids[0], bytes):
            file_ids = [fid.decode('utf-8') for fid in file_ids]
        unique_files = len(set(file_ids))
        
        # Get shapes
        spec_shape = f['spectrograms'].shape
        style_shape = f['styles'].shape
        
        # Get file size
        file_size_mb = dataset_path.stat().st_size / (1024 ** 2)
    
    return {
        'num_samples': num_samples,
        'num_unique_files': unique_files,
        'file_size_mb': file_size_mb,
        'spectrogram_shape': spec_shape,
        'style_shape': style_shape,
        'format': 'HDF5'
    }


def create_hdf5_dataloaders(
    dataset_path: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from HDF5 dataset
    
    Args:
        dataset_path: Path to HDF5 file
        batch_size: Batch size
        val_split: Validation split ratio (0.0-1.0)
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        shuffle: Shuffle training data
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    logger.info(f"Loading HDF5 dataset from: {dataset_path}")
    
    # Create dataset
    dataset = SpectrogramH5Dataset(dataset_path)
    
    # Split into train/val
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Dataset split: {train_size} train, {val_size} val")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    logger.info(f"DataLoaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader
