#!/usr/bin/env python3
"""
Visualization utilities for MLflow experiments
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mlflow
from pathlib import Path
from typing import List, Optional
import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("visualization", level="INFO")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_training_metrics(run_id: str, output_dir: str = "plots"):
    """
    Plot training metrics for a specific run
    
    Args:
        run_id: MLflow run ID
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get run data
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    
    # Get all metrics
    metrics_data = {}
    for metric_key in run.data.metrics.keys():
        metric_history = client.get_metric_history(run_id, metric_key)
        metrics_data[metric_key] = pd.DataFrame([
            {'step': m.step, 'value': m.value, 'timestamp': m.timestamp}
            for m in metric_history
        ])
    
    if not metrics_data:
        _logger.warning(f"No metrics found for run {run_id}")
        return
    
    # Plot epoch metrics
    epoch_metrics = {k: v for k, v in metrics_data.items() if k.startswith('epoch/')}
    
    if epoch_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Metrics - Run {run_id[:8]}', fontsize=16)
        
        metric_names = [
            'epoch/world_loss',
            'epoch/recon_loss',
            'epoch/kl_loss',
            'epoch/aux_loss',
            'epoch/actor_loss',
            'epoch/critic_loss'
        ]
        
        for idx, metric_name in enumerate(metric_names):
            if metric_name in epoch_metrics:
                row = idx // 3
                col = idx % 3
                ax = axes[row, col]
                
                data = epoch_metrics[metric_name]
                ax.plot(data['step'], data['value'], linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(metric_name.replace('epoch/', '').replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_path / f"training_metrics_{run_id[:8]}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        _logger.info(f"Saved plot: {plot_path}")
        plt.close()
    
    # Plot combined losses
    if 'epoch/world_loss' in epoch_metrics and 'epoch/actor_loss' in epoch_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        world_data = epoch_metrics['epoch/world_loss']
        actor_data = epoch_metrics['epoch/actor_loss']
        critic_data = epoch_metrics['epoch/critic_loss']
        
        ax.plot(world_data['step'], world_data['value'], label='World Model Loss', linewidth=2)
        ax.plot(actor_data['step'], actor_data['value'], label='Actor Loss', linewidth=2)
        ax.plot(critic_data['step'], critic_data['value'], label='Critic Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Combined Training Losses - Run {run_id[:8]}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_path / f"combined_losses_{run_id[:8]}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        _logger.info(f"Saved plot: {plot_path}")
        plt.close()


def plot_run_comparison(experiment_name: str, metric: str = "epoch/world_loss", 
                       output_dir: str = "plots"):
    """
    Compare multiple runs in an experiment
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to compare
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        _logger.error(f"Experiment '{experiment_name}' not found")
        return
    
    # Get all runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        _logger.warning(f"No runs found in experiment '{experiment_name}'")
        return
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    client = mlflow.tracking.MlflowClient()
    
    for _, run in runs.iterrows():
        run_id = run['run_id']
        
        try:
            metric_history = client.get_metric_history(run_id, metric)
            if metric_history:
                steps = [m.step for m in metric_history]
                values = [m.value for m in metric_history]
                ax.plot(steps, values, label=f'Run {run_id[:8]}', linewidth=2, alpha=0.7)
        except Exception as e:
            _logger.warning(f"Could not get metric history for run {run_id}: {e}")
            continue
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric.replace('epoch/', '').replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric.replace("epoch/", "").replace("_", " ").title()} Comparison', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / f"comparison_{metric.replace('/', '_')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    _logger.info(f"Saved plot: {plot_path}")
    plt.close()


def plot_hyperparameter_impact(experiment_name: str, param_name: str, 
                               metric_name: str = "epoch/world_loss",
                               output_dir: str = "plots"):
    """
    Plot impact of hyperparameter on metric
    
    Args:
        experiment_name: Name of the experiment
        param_name: Hyperparameter to analyze
        metric_name: Metric to plot
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get runs
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        _logger.error(f"Experiment '{experiment_name}' not found")
        return
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        _logger.warning(f"No runs found")
        return
    
    # Extract data
    param_col = f'params.{param_name}'
    metric_col = f'metrics.{metric_name}'
    
    if param_col not in runs.columns or metric_col not in runs.columns:
        _logger.error(f"Parameter '{param_name}' or metric '{metric_name}' not found")
        return
    
    # Filter valid data
    valid_data = runs[[param_col, metric_col]].dropna()
    
    if valid_data.empty:
        _logger.warning("No valid data found")
        return
    
    # Convert param to numeric if possible
    try:
        valid_data[param_col] = pd.to_numeric(valid_data[param_col])
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(valid_data[param_col], valid_data[metric_col], s=100, alpha=0.6)
        ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(metric_name.replace('epoch/', '').replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Impact of {param_name} on {metric_name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_path / f"hyperparam_{param_name}_impact.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        _logger.info(f"Saved plot: {plot_path}")
        plt.close()
        
    except (ValueError, TypeError):
        _logger.warning(f"Could not convert parameter '{param_name}' to numeric")


def create_dashboard(experiment_name: str = "dreamer-spectrogram", 
                    output_dir: str = "plots"):
    """
    Create a comprehensive dashboard for an experiment
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Directory to save plots
    """
    _logger.info(f"Creating dashboard for experiment '{experiment_name}'...")
    
    # Get best run
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        _logger.error(f"Experiment '{experiment_name}' not found")
        return
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        _logger.warning("No runs found")
        return
    
    # Get best run
    best_run = runs.loc[runs['metrics.epoch/world_loss'].idxmin()]
    best_run_id = best_run['run_id']
    
    _logger.info(f"Best run: {best_run_id[:8]}")
    
    # Plot metrics for best run
    plot_training_metrics(best_run_id, output_dir)
    
    # Plot comparisons
    metrics = ['epoch/world_loss', 'epoch/recon_loss', 'epoch/actor_loss']
    for metric in metrics:
        try:
            plot_run_comparison(experiment_name, metric, output_dir)
        except Exception as e:
            _logger.warning(f"Could not plot comparison for {metric}: {e}")
    
    _logger.info(f"Dashboard created in {output_dir}/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize.py dashboard [experiment_name]")
        print("  python visualize.py metrics <run_id>")
        print("  python visualize.py compare [experiment_name] [metric]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "dashboard":
        exp_name = sys.argv[2] if len(sys.argv) > 2 else "dreamer-spectrogram"
        create_dashboard(exp_name)
    
    elif command == "metrics":
        if len(sys.argv) < 3:
            print("Error: run_id required")
            sys.exit(1)
        run_id = sys.argv[2]
        plot_training_metrics(run_id)
    
    elif command == "compare":
        exp_name = sys.argv[2] if len(sys.argv) > 2 else "dreamer-spectrogram"
        metric = sys.argv[3] if len(sys.argv) > 3 else "epoch/world_loss"
        plot_run_comparison(exp_name, metric)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
