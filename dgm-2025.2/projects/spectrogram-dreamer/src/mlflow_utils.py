#!/usr/bin/env python3
"""
MLflow utilities for Dreamer model management
"""

import mlflow
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from src.models import DreamerModel
from src.utils.logger import get_logger

_logger = get_logger("mlflow_utils", level="INFO")


class ModelManager:
    """Manages model loading and artifacts from MLflow"""
    
    def __init__(self, tracking_uri: str = "./mlruns"):
        """
        Initialize ModelManager
        
        Args:
            tracking_uri: Path to MLflow tracking directory
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
    
    def list_experiments(self) -> pd.DataFrame:
        """List all MLflow experiments"""
        experiments = mlflow.search_experiments()
        
        exp_data = []
        for exp in experiments:
            exp_data.append({
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'artifact_location': exp.artifact_location,
                'lifecycle_stage': exp.lifecycle_stage,
            })
        
        df = pd.DataFrame(exp_data)
        _logger.info(f"Found {len(df)} experiments")
        return df
    
    def list_runs(self, experiment_name: str) -> pd.DataFrame:
        """
        List all runs in an experiment
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            DataFrame with run information
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            _logger.error(f"Experiment '{experiment_name}' not found")
            return pd.DataFrame()
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        _logger.info(f"Found {len(runs)} runs in experiment '{experiment_name}'")
        return runs
    
    def get_best_run(self, experiment_name: str, metric: str = "epoch/world_loss") -> Optional[str]:
        """
        Get the run ID with the best (lowest) metric
        
        Args:
            experiment_name: Name of the experiment
            metric: Metric to optimize (default: world_loss)
            
        Returns:
            Run ID of the best run, or None if not found
        """
        runs = self.list_runs(experiment_name)
        
        if runs.empty:
            return None
        
        # Filter runs that have the metric
        runs_with_metric = runs[runs[f'metrics.{metric}'].notna()]
        
        if runs_with_metric.empty:
            _logger.warning(f"No runs found with metric '{metric}'")
            return None
        
        best_run = runs_with_metric.loc[runs_with_metric[f'metrics.{metric}'].idxmin()]
        best_run_id = best_run['run_id']
        best_value = best_run[f'metrics.{metric}']
        
        _logger.info(f"Best run: {best_run_id} with {metric}={best_value:.4f}")
        return best_run_id
    
    def load_model(self, run_id: str, device: str = 'cuda') -> DreamerModel:
        """
        Load a DreamerModel from an MLflow run
        
        Args:
            run_id: MLflow run ID
            device: Device to load model on
            
        Returns:
            Loaded DreamerModel
        """
        _logger.info(f"Loading model from run {run_id}...")
        
        # Load model using MLflow
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        
        _logger.info(f"Model loaded successfully on {device}")
        return model
    
    def load_checkpoint(self, run_id: str, checkpoint_name: str = "best_model.pt", 
                       device: str = 'cuda') -> Dict[str, Any]:
        """
        Load a checkpoint from an MLflow run
        
        Args:
            run_id: MLflow run ID
            checkpoint_name: Name of the checkpoint file
            device: Device to load on
            
        Returns:
            Dictionary with checkpoint data
        """
        _logger.info(f"Loading checkpoint '{checkpoint_name}' from run {run_id}...")
        
        # Download artifact
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=checkpoint_name
        )
        
        # Load checkpoint
        checkpoint = torch.load(artifact_path, map_location=device)
        
        _logger.info(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        return checkpoint
    
    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a run
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary with run information
        """
        run = mlflow.get_run(run_id)
        
        info = {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'artifact_uri': run.info.artifact_uri,
            'params': run.data.params,
            'metrics': run.data.metrics,
        }
        
        return info
    
    def compare_runs(self, experiment_name: str, metric_names: list = None) -> pd.DataFrame:
        """
        Compare runs in an experiment
        
        Args:
            experiment_name: Name of the experiment
            metric_names: List of metrics to compare
            
        Returns:
            DataFrame with comparison
        """
        if metric_names is None:
            metric_names = [
                'epoch/world_loss',
                'epoch/recon_loss',
                'epoch/kl_loss',
                'epoch/actor_loss',
                'epoch/critic_loss'
            ]
        
        runs = self.list_runs(experiment_name)
        
        if runs.empty:
            return pd.DataFrame()
        
        # Select relevant columns
        cols = ['run_id', 'start_time', 'status']
        cols += [f'metrics.{m}' for m in metric_names if f'metrics.{m}' in runs.columns]
        cols += [f'params.{p}' for p in ['num_epochs', 'batch_size', 'h_state_size', 'z_state_size']]
        
        comparison = runs[[c for c in cols if c in runs.columns]]
        
        return comparison


def print_experiment_summary(experiment_name: str = "dreamer-spectrogram"):
    """Print a summary of an experiment"""
    manager = ModelManager()
    
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}\n")
    
    runs = manager.list_runs(experiment_name)
    
    if runs.empty:
        print("No runs found.")
        return
    
    print(f"Total runs: {len(runs)}")
    print(f"\nRun Summary:")
    print("-" * 80)
    
    for _, run in runs.iterrows():
        print(f"\nRun ID: {run['run_id']}")
        print(f"Status: {run['status']}")
        print(f"Start time: {run['start_time']}")
        
        # Print metrics
        metric_cols = [c for c in run.index if c.startswith('metrics.epoch/')]
        if metric_cols:
            print("\nFinal Metrics:")
            for col in metric_cols:
                metric_name = col.replace('metrics.epoch/', '')
                value = run[col]
                if pd.notna(value):
                    print(f"  {metric_name}: {value:.4f}")
        
        print("-" * 80)
    
    # Best run
    best_run_id = manager.get_best_run(experiment_name)
    if best_run_id:
        print(f"\n🏆 Best run: {best_run_id}")


def load_best_model(experiment_name: str = "dreamer-spectrogram", 
                   device: str = 'cuda') -> DreamerModel:
    """
    Load the best model from an experiment
    
    Args:
        experiment_name: Name of the experiment
        device: Device to load on
        
    Returns:
        Best DreamerModel
    """
    manager = ModelManager()
    best_run_id = manager.get_best_run(experiment_name)
    
    if best_run_id is None:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    
    model = manager.load_model(best_run_id, device=device)
    return model


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python mlflow_utils.py summary [experiment_name]")
        print("  python mlflow_utils.py list-experiments")
        print("  python mlflow_utils.py compare [experiment_name]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "summary":
        exp_name = sys.argv[2] if len(sys.argv) > 2 else "dreamer-spectrogram"
        print_experiment_summary(exp_name)
    
    elif command == "list-experiments":
        manager = ModelManager()
        experiments = manager.list_experiments()
        print("\nExperiments:")
        print(experiments.to_string())
    
    elif command == "compare":
        exp_name = sys.argv[2] if len(sys.argv) > 2 else "dreamer-spectrogram"
        manager = ModelManager()
        comparison = manager.compare_runs(exp_name)
        print("\nRun Comparison:")
        print(comparison.to_string())
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
