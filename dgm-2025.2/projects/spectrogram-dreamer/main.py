import torch
import argparse
import os
from pathlib import Path
from src.dataset.spectrogram_dataset import SpectrogramDataset
from src.models import DreamerModel
from src.training import train
from src.utils.logger import get_logger

_logger = get_logger("main", level="INFO")

# run preprocessing pipeline
# launch()

def main():
    """Main entry point for training Dreamer model"""
    
    parser = argparse.ArgumentParser(description="Train Dreamer model on spectrograms")
    
    # Dataset mode
    parser.add_argument("--use-consolidated", action="store_true",
                        help="Use consolidated dataset (RECOMMENDED - 40-90%% space savings)")
    parser.add_argument("--dataset-path", type=str, default="data/dataset_consolidated.h5",
                        help="Path to consolidated dataset file (.h5 or .pt)")
    
    # Original dataset paths (deprecated)
    parser.add_argument("--spec-path", type=str, default="data/2_mel-spectrograms",
                        help="Path to spectrogram data (original mode only)")
    parser.add_argument("--style-path", type=str, default="data/3_style-vectors",
                        help="Path to style vectors (original mode only)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=10,
                        help="Sequence length for temporal modeling")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")
    
    # MLflow parameters
    parser.add_argument("--experiment-name", type=str, default="dreamer-spectrogram",
                        help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None,
                        help="MLflow run name")
    
    # Model parameters
    parser.add_argument("--h-state-size", type=int, default=200,
                        help="Size of deterministic state")
    parser.add_argument("--z-state-size", type=int, default=30,
                        help="Size of stochastic state")
    parser.add_argument("--action-size", type=int, default=128,
                        help="Size of style action vector")
    
    # Testing
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in test mode with dummy data")
    
    args = parser.parse_args()
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    _logger.info(f"Using device: {device}")
    _logger.info(f"MLflow tracking: mlruns/{args.experiment_name}")
    
    # Auto-detect action size from dataset if using consolidated
    actual_action_size = args.action_size
    input_shape = (64, 10)  # default: (n_mels, time_frames)
    
    if args.use_consolidated and os.path.exists(args.dataset_path):
        _logger.info("Detecting model parameters from HDF5 dataset...")
        import h5py
        with h5py.File(args.dataset_path, 'r') as f:
            if 'styles' in f:
                actual_action_size = f['styles'].shape[-1]
                _logger.info(f"Detected action size: {actual_action_size}")
            
            # Detect input shape from spectrograms
            if 'spectrograms' in f:
                spec_shape = f['spectrograms'].shape  # (N, H, W)
                input_shape = (spec_shape[1], spec_shape[2])  # (n_mels, time_frames)
                _logger.info(f"Detected input shape: {input_shape}")
    
    # Initialize model
    _logger.info("Initializing Dreamer model...")
    model = DreamerModel(
        h_state_size=args.h_state_size,
        z_state_size=args.z_state_size,
        action_size=actual_action_size,
        embedding_size=256,
        aux_size=5,  # pitch, energy, delta-energy, spectral centroid, onset strength
        in_channels=1,
        cnn_depth=32,
        input_shape=input_shape
    )
    
    # Load dataset
    _logger.info("Loading dataset...")
    try:
        if not args.test_mode:
            if args.use_consolidated:
                # Check if it's HDF5 or PyTorch format
                dataset_path = Path(args.dataset_path)
                is_hdf5 = dataset_path.suffix == '.h5'
                
                if is_hdf5:
                    _logger.info("Using HDF5 CONSOLIDATED dataset")
                    _logger.info(f"Loading from: {args.dataset_path}")
                    
                    from src.dataset import create_hdf5_dataloaders, get_hdf5_dataset_info
                    
                    # Get dataset info
                    info = get_hdf5_dataset_info(args.dataset_path)
                    _logger.info(f"Dataset info:")
                    _logger.info(f"   - Format: {info['format']}")
                    _logger.info(f"   - Total samples: {info['num_samples']}")
                    _logger.info(f"   - Unique files: {info['num_unique_files']}")
                    _logger.info(f"   - File size: {info['file_size_mb']:.2f} MB ({info['file_size_mb']/1024:.2f} GB)")
                    _logger.info(f"   - Spectrogram shape: {info['spectrogram_shape']}")
                    _logger.info(f"   - Style shape: {info['style_shape']}")
                    
                    # Create train/val dataloaders
                    train_dataloader, val_dataloader = create_hdf5_dataloaders(
                        dataset_path=args.dataset_path,
                        val_split=args.val_split,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=(device == "cuda")
                    )
                else:
                    _logger.info("Using PyTorch CONSOLIDATED dataset")
                    _logger.info(f"Loading from: {args.dataset_path}")
                    
                    from src.dataset import create_train_val_dataloaders, get_dataset_info
                    
                    # Get dataset info
                    info = get_dataset_info(args.dataset_path)
                    _logger.info(f"Dataset info:")
                    _logger.info(f"   - Total samples: {info['num_samples']}")
                    _logger.info(f"   - Unique files: {info['num_unique_files']}")
                    _logger.info(f"   - File size: {info['file_size_mb']:.2f} MB")
                    
                    # Create train/val dataloaders
                    train_dataloader, val_dataloader = create_train_val_dataloaders(
                        dataset_path=args.dataset_path,
                        val_split=args.val_split,
                        batch_size=args.batch_size,
                        sequence_length=args.sequence_length,
                        num_workers=args.num_workers,
                        pin_memory=(device == "cuda")
                    )
                
                _logger.info(f"Dataloaders created:")
                _logger.info(f"   - Train batches: {len(train_dataloader)}")
                _logger.info(f"   - Val batches: {len(val_dataloader)}")
                
                # Start training
                _logger.info("Starting training with MLflow tracking...")
                
                from src.training import train_consolidated
                
                train_consolidated(
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    num_epochs=args.epochs,
                    device=device,
                    experiment_name=args.experiment_name,
                    run_name=args.run_name,
                    checkpoint_freq=args.checkpoint_freq,
                    learning_rate=args.lr
                )
                
                _logger.info("\n")
                _logger.info("Training complete!")
                _logger.info("Check mlruns/ directory for results")
                _logger.info("Run 'mlflow ui' to visualize training")
                _logger.info("\n")
                
            else:
                # Original dataset mode (deprecated)
                _logger.info("Using ORIGINAL dataset mode (deprecated)")
                _logger.info("Consider using --use-consolidated for 40-90% space savings")
                
                dataset = SpectrogramDataset(args.spec_path, args.style_path)
                _logger.info(f"Dataset loaded with {len(dataset)} samples")
                
                # Show sample
                sample = dataset[0]
                _logger.info(f"Sample observation shape: {sample['observation'].shape}")
                _logger.info(f"Sample action shape: {sample['action'].shape}")
                _logger.info(f"Sample rewards shape: {sample['rewards'].shape}")
                
                # Start training with MLflow
                _logger.info("Starting training with MLflow tracking...")
                train(
                    model, 
                    dataset, 
                    num_epochs=args.epochs, 
                    batch_size=args.batch_size, 
                    device=device,
                    experiment_name=args.experiment_name,
                    run_name=args.run_name,
                    checkpoint_freq=args.checkpoint_freq
                )
                
                _logger.info("Training complete! Check mlruns/ directory for results.")
                _logger.info("To view results: mlflow ui")
        
    except FileNotFoundError as e:
        _logger.error(f"Dataset not found: {e}")
        _logger.info("Please run the preprocessing pipeline first:")
        _logger.info("  python run_pipeline.py --consolidated --use-float16")
        _logger.info("Or adjust paths with --dataset-path")
        _logger.info("You can test individual components with --test-mode")
        args.test_mode = True
    
    if args.test_mode:
        # Test mode - create dummy data
        _logger.info("Running in test mode with dummy data...")
        batch_size = 4
        seq_len = 50
        # Match the HDF5 dataset spectrogram shape: (64, 10) -> (n_mels, time_frames)
        # But we need time_frames=50 to match the model's expected input
        dummy_obs = torch.randn(batch_size, seq_len, 1, 64, 50)
        dummy_actions = torch.randn(batch_size, seq_len, 128)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_obs, dummy_actions, compute_loss=False)
            _logger.info(f"Reconstructed shape: {output['reconstructed'].shape}")
            _logger.info(f"h_states shape: {output['h_states'].shape}")
            _logger.info(f"z_states shape: {output['z_states'].shape}")
            _logger.info("Model test successful!")


if __name__ == "__main__":
    main()