# Training script for Dreamer with style control
# Implements the complete training loop with actor-critic optimization

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
import mlflow
import mlflow.pytorch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.models import DreamerModel
from src.dataset.spectrogram_dataset import SpectrogramDataset
from src.dataset.spectrogram_dataloader import create_dataloader
from src.dataset.spectrogram_hdf5_dataset import SpectrogramH5Dataset
from src.utils.logger import get_logger

_logger = get_logger("training", level="INFO")


class DreamerTrainer:
    """Trainer for Dreamer model with style-controllable generation
    
    Args:
        model: DreamerModel instance
        learning_rate: Learning rate for optimizers (default: 1e-4)
        imagination_horizon: Steps to imagine for actor-critic (default: 15)
        gamma: Discount factor (default: 0.99)
        lambda_: GAE lambda (default: 0.95)
    """
    
    def __init__(self,
                 model: DreamerModel,
                 learning_rate: float = 1e-4,
                 imagination_horizon: int = 15,
                 gamma: float = 0.99,
                 lambda_: float = 0.95):
        
        self.model = model
        self.imagination_horizon = imagination_horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # Separate optimizers for world model and actor-critic
        world_model_params = list(model.encoder.parameters()) + \
                           list(model.rssm.parameters()) + \
                           list(model.decoder.parameters()) + \
                           list(model.reward_predictor.parameters()) + \
                           list(model.style_reward_predictor.parameters()) + \
                           list(model.auxiliary_predictor.parameters())
        
        self.world_optimizer = Adam(world_model_params, lr=learning_rate)
        self.actor_optimizer = Adam(model.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = Adam(model.critic.parameters(), lr=learning_rate)
        
    def train_step(self, batch, aux_targets=None):
        """
        Single training step
        
        Args:
            batch: Dictionary with 'observation', 'action', 'rewards'
            aux_targets: Optional auxiliary feature targets
            
        Returns:
            Dictionary of losses and metrics
        """
        observations = batch['observation']  # (B, T, C, H, W)
        actions = batch['action']            # (B, T, action_size)
        true_rewards = batch['rewards']      # (B, T)
        
        # 1. Train World Model
        self.world_optimizer.zero_grad()
        
        model_output = self.model(observations, actions, compute_loss=True, aux_targets=aux_targets)
        
        world_loss = model_output['losses']['total_loss']
        world_loss.backward()
        
        # Gradient clipping for stability (especially important with He initialization)
        # Clip world model gradients (encoder, RSSM, decoder, predictors)
        world_model_params = list(self.world_optimizer.param_groups[0]['params'])
        grad_norm = torch.nn.utils.clip_grad_norm_(world_model_params, max_norm=100.0)
        
        self.world_optimizer.step()
        
        # 2. Train Actor-Critic via imagination
        # Get final states from world model (detach to stop gradients from flowing back to world model)
        h_final = model_output['h_states'][:, -1].detach()  # (B, h_state_size)
        z_final = model_output['z_states'][:, -1].detach()  # (B, z_state_size)
        
        # Imagine trajectories
        actor_loss, critic_loss = self._train_actor_critic(h_final, z_final)
        
        return {
            'world_loss': world_loss.item(),
            'recon_loss': model_output['losses']['recon_loss'].item(),
            'kl_loss': model_output['losses']['kl_loss'].item(),
            'aux_loss': model_output['losses']['aux_loss'].item(),
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            # Variance metrics for monitoring decoder health
            'obs_std': model_output['losses'].get('obs_std', 0.0),
            'recon_std': model_output['losses'].get('recon_std', 0.0),
            'variance_ratio': model_output['losses'].get('variance_ratio', 0.0),
            'grad_norm': grad_norm.item()
        }
    
    def _train_actor_critic(self, h_state, z_state):
        """
        Train actor and critic using imagined trajectories
        
        Args:
            h_state: Initial deterministic state (B, h_state_size)
            z_state: Initial stochastic state (B, z_state_size)
            
        Returns:
            Tuple of (actor_loss, critic_loss)
        """
        # Imagine trajectory with world model frozen
        with torch.no_grad():
            imagined = self.model.imagine_trajectory(h_state, z_state, self.imagination_horizon)
        
        h_imag = imagined['h_states']     # (B, H+1, h_state_size)
        z_imag = imagined['z_states']     # (B, H+1, z_state_size)
        rewards = imagined['rewards']     # (B, H, 1)
        style_rewards = imagined['style_rewards']  # (B, H, 1)
        
        # Combine rewards (you can adjust weights)
        total_rewards = rewards + 0.1 * style_rewards
        
        # Train Critic
        # Recompute values WITH gradients on frozen latent states
        self.critic_optimizer.zero_grad()
        
        values = self.model.critic(h_imag, z_imag).squeeze(-1)  # (B, H+1)
        
        # Compute TD-lambda targets
        with torch.no_grad():
            targets = self._compute_lambda_returns(total_rewards.squeeze(-1), values.detach())
        
        critic_loss = F.mse_loss(values[:, :-1], targets)
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=100.0)
        self.critic_optimizer.step()
        
        # Train Actor
        self.actor_optimizer.zero_grad()
        
        # Re-imagine trajectory WITH gradients through actor
        # The latent states from first imagination are used as starting points
        # We compute actor loss as negative expected value (to maximize value)
        h_t = h_state
        z_t = z_state
        actor_values = []
        
        for t in range(self.imagination_horizon):
            # Generate action with gradients
            action = self.model.actor(h_t, z_t)
            
            # Step through RSSM without gradients (world model is frozen)
            with torch.no_grad():
                h_t = self.model.rssm.gru(torch.cat([z_t, action], dim=-1), h_t)
                prior_params = self.model.rssm.prior(h_t)
                prior_mean, prior_std = torch.chunk(prior_params, 2, dim=-1)
                prior_std = F.softplus(prior_std) + 0.1
                z_t = Normal(prior_mean, prior_std).rsample()
            
            # Compute value estimate WITH gradients through critic
            # (but critic params won't be updated here, only actor params)
            value_t = self.model.critic(h_t.unsqueeze(1), z_t.unsqueeze(1)).squeeze()
            actor_values.append(value_t)
        
        # Actor objective: maximize expected values
        actor_loss = -torch.stack(actor_values).mean()
        actor_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), max_norm=100.0)
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def _compute_lambda_returns(self, rewards, values):
        """
        Compute TD-lambda returns using GAE
        
        Args:
            rewards: Rewards (B, H)
            values: Value estimates (B, H+1)
            
        Returns:
            Lambda returns (B, H)
        """
        batch_size, horizon = rewards.shape
        device = rewards.device
        
        lambda_returns = torch.zeros_like(rewards)
        last_value = values[:, -1]
        
        for t in reversed(range(horizon)):
            if t == horizon - 1:
                next_value = last_value
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + self.gamma * next_value - values[:, t]
            lambda_returns[:, t] = delta + self.gamma * self.lambda_ * \
                                   (lambda_returns[:, t + 1] if t < horizon - 1 else 0)
        
        # Add value estimate
        lambda_returns = lambda_returns + values[:, :-1]
        
        return lambda_returns


def train(model, dataset, num_epochs=100, batch_size=50, device='cuda', 
          experiment_name="dreamer-spectrogram", run_name=None, checkpoint_freq=10):
    """
    Main training function with MLflow tracking
    
    Args:
        model: DreamerModel instance
        dataset: SpectrogramDataset instance
        num_epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
        experiment_name: MLflow experiment name
        run_name: MLflow run name (default: auto-generated)
        checkpoint_freq: Save checkpoint every N epochs
    """
    # Setup MLflow
    mlflow.set_experiment(experiment_name)
    
    if run_name is None:
        run_name = f"dreamer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "h_state_size": model.h_state_size,
            "z_state_size": model.z_state_size,
            "action_size": model.action_size,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "device": device,
            "dataset_size": len(dataset),
        })
        
        model = model.to(device)
        trainer = DreamerTrainer(model)
        
        # Log trainer hyperparameters
        mlflow.log_params({
            "learning_rate": 1e-4,
            "imagination_horizon": trainer.imagination_horizon,
            "gamma": trainer.gamma,
            "lambda": trainer.lambda_,
            "free_nats": 3.0,  # KL free nats for posterior collapse prevention
        })
        
        # Learning rate warmup scheduler
        warmup_steps = 1000
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        
        from torch.optim.lr_scheduler import LambdaLR
        world_scheduler = LambdaLR(trainer.world_optimizer, lr_lambda)
        actor_scheduler = LambdaLR(trainer.actor_optimizer, lr_lambda)
        critic_scheduler = LambdaLR(trainer.critic_optimizer, lr_lambda)
        
        dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create checkpoint directory
        checkpoint_dir = Path("checkpoints") / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_loss = float('inf')
        
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_losses = {
                'world_loss': 0,
                'recon_loss': 0,
                'kl_loss': 0,
                'aux_loss': 0,
                'actor_loss': 0,
                'critic_loss': 0,
                'obs_std': 0,
                'recon_std': 0,
                'variance_ratio': 0,
                'grad_norm': 0
            }
            
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Training step
                losses = trainer.train_step(batch)
                
                # Step schedulers (warmup)
                world_scheduler.step()
                actor_scheduler.step()
                critic_scheduler.step()
                
                # Accumulate losses
                for k, v in losses.items():
                    epoch_losses[k] += v
                
                num_batches += 1
                global_step += 1
                
                # Variance monitoring and alerting
                recon_std = losses.get('recon_std', 0.0)
                if recon_std < 0.2 and global_step > 100:
                    _logger.warning(f"⚠️  WARNING: Decoder variance collapsed! "
                                  f"recon_std={recon_std:.4f} at step {global_step}")
                
                if batch_idx % 10 == 0:
                    _logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                               f"World Loss: {losses['world_loss']:.4f}, "
                               f"Actor Loss: {losses['actor_loss']:.4f}, "
                               f"Critic Loss: {losses['critic_loss']:.4f}, "
                               f"Recon Std: {recon_std:.4f}")
                    
                    # Log batch metrics to MLflow
                    step = epoch * len(dataloader) + batch_idx
                    mlflow.log_metrics({
                        f"batch/{k}": v for k, v in losses.items()
                    }, step=step)
                    
                    # Log learning rates
                    mlflow.log_metrics({
                        "lr/world": trainer.world_optimizer.param_groups[0]['lr'],
                        "lr/actor": trainer.actor_optimizer.param_groups[0]['lr'],
                        "lr/critic": trainer.critic_optimizer.param_groups[0]['lr'],
                    }, step=step)
            
            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= num_batches
            
            _logger.info(f"Epoch {epoch} Summary:")
            _logger.info(f"  World Loss: {epoch_losses['world_loss']:.4f}")
            _logger.info(f"  Recon Loss: {epoch_losses['recon_loss']:.4f}")
            _logger.info(f"  KL Loss: {epoch_losses['kl_loss']:.4f}")
            _logger.info(f"  Aux Loss: {epoch_losses['aux_loss']:.4f}")
            _logger.info(f"  Actor Loss: {epoch_losses['actor_loss']:.4f}")
            _logger.info(f"  Critic Loss: {epoch_losses['critic_loss']:.4f}")
            _logger.info(f"  Obs Std: {epoch_losses['obs_std']:.4f}")
            _logger.info(f"  Recon Std: {epoch_losses['recon_std']:.4f}")
            _logger.info(f"  Variance Ratio: {epoch_losses['variance_ratio']:.4f}")
            _logger.info(f"  Grad Norm: {epoch_losses['grad_norm']:.4f}")
            
            # Log epoch metrics to MLflow
            mlflow.log_metrics({
                f"epoch/{k}": v for k, v in epoch_losses.items()
            }, step=epoch)
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'world_optimizer_state_dict': trainer.world_optimizer.state_dict(),
                    'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
                    'losses': epoch_losses,
                }, checkpoint_path)
                
                _logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                # Log checkpoint as artifact
                mlflow.log_artifact(str(checkpoint_path))
                
                # Save best model
                total_loss = epoch_losses['world_loss'] + epoch_losses['actor_loss'] + epoch_losses['critic_loss']
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_model_path = checkpoint_dir / "best_model.pt"
                    torch.save(model.state_dict(), best_model_path)
                    mlflow.log_artifact(str(best_model_path))
                    _logger.info(f"New best model saved with loss: {best_loss:.4f}")
        
        # Save final model with MLflow
        _logger.info("Saving final model to MLflow...")
        mlflow.pytorch.log_model(model, "model")
        
        # Log model architecture
        model_info = {
            "architecture": "DreamerModel",
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        mlflow.log_dict(model_info, "model_info.json")
        
        _logger.info(f"Training complete! MLflow run ID: {mlflow.active_run().info.run_id}")


def train_consolidated(model, train_dataloader, val_dataloader, num_epochs=100, 
                       device='cuda', experiment_name="dreamer-spectrogram", 
                       run_name=None, checkpoint_freq=10, learning_rate=1e-4):
    """
    Training function for consolidated dataset (HDF5 or PyTorch)
    
    Args:
        model: DreamerModel instance
        train_dataloader: Training DataLoader
        val_dataloader: Validation DataLoader
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        experiment_name: MLflow experiment name
        run_name: MLflow run name (default: auto-generated)
        checkpoint_freq: Save checkpoint every N epochs
        learning_rate: Learning rate for optimizers
    """
    # Setup MLflow
    mlflow.set_experiment(experiment_name)
    
    if run_name is None:
        run_name = f"dreamer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "h_state_size": model.h_state_size,
            "z_state_size": model.z_state_size,
            "action_size": model.action_size,
            "num_epochs": num_epochs,
            "device": device,
            "train_batches": len(train_dataloader),
            "val_batches": len(val_dataloader),
            "learning_rate": learning_rate,
        })
        
        model = model.to(device)
        trainer = DreamerTrainer(model, learning_rate=learning_rate)
        
        # Learning rate warmup scheduler
        warmup_steps = 1000
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        
        from torch.optim.lr_scheduler import LambdaLR
        world_scheduler = LambdaLR(trainer.world_optimizer, lr_lambda)
        actor_scheduler = LambdaLR(trainer.actor_optimizer, lr_lambda)
        critic_scheduler = LambdaLR(trainer.critic_optimizer, lr_lambda)
        
        # Log trainer hyperparameters
        mlflow.log_params({
            "imagination_horizon": trainer.imagination_horizon,
            "free_nats": 3.0,  # KL free nats
            "gamma": trainer.gamma,
            "lambda": trainer.lambda_,
        })
        
        # Create checkpoint directory
        checkpoint_dir = Path("checkpoints") / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_loss = float('inf')
        
        _logger.info(f"Starting training for {num_epochs} epochs...")
        _logger.info(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_losses = {
                'world_loss': 0,
                'recon_loss': 0,
                'kl_loss': 0,
                'aux_loss': 0,
                'actor_loss': 0,
                'critic_loss': 0,
                'obs_std': 0,
                'recon_std': 0,
                'variance_ratio': 0,
                'grad_norm': 0
            }
            
            num_batches = 0
            
            _logger.info(f"Epoch {epoch + 1}/{num_epochs} - Training...")
            
            # Add tqdm progress bar
            train_pbar = tqdm(enumerate(train_dataloader), 
                             total=len(train_dataloader),
                             desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                             ncols=100,
                             leave=True)
            
            for batch_idx, batch in train_pbar:
                # Move to device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                else:
                    # Tuple format: (observations, actions, metadata)
                    obs, actions, meta = batch
                    batch = {
                        'observation': obs.to(device),
                        'action': actions.to(device),
                        'rewards': torch.zeros(obs.shape[0], obs.shape[1], device=device)
                    }
                
                # Training step
                try:
                    losses = trainer.train_step(batch)
                    
                    # Step schedulers (warmup)
                    world_scheduler.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    
                    # Accumulate losses
                    for k, v in losses.items():
                        epoch_losses[k] += v
                    
                    num_batches += 1
                    
                    # Variance monitoring and alerting
                    recon_std = losses.get('recon_std', 0.0)
                    if recon_std < 0.2 and batch_idx > 100:
                        _logger.warning(f"⚠️  WARNING: Decoder variance collapsed! "
                                      f"recon_std={recon_std:.4f}")
                    
                    # Update progress bar with current losses
                    train_pbar.set_postfix({
                        'World': f"{losses['world_loss']:.4f}",
                        'Actor': f"{losses['actor_loss']:.4f}",
                        'Recon_std': f"{recon_std:.3f}"
                    })
                    
                    # Log progress
                    if batch_idx % 50 == 0:
                        _logger.info(f"  Batch {batch_idx}/{len(train_dataloader)}: "
                                   f"World={losses['world_loss']:.4f}, "
                                   f"Actor={losses['actor_loss']:.4f}, "
                                   f"Critic={losses['critic_loss']:.4f}")
                        
                        # Log batch metrics to MLflow
                        step = epoch * len(train_dataloader) + batch_idx
                        mlflow.log_metrics({
                            f"train_batch/{k}": v for k, v in losses.items()
                        }, step=step)
                
                except Exception as e:
                    _logger.error(f"Error in training step: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Average training losses
            for k in epoch_losses:
                epoch_losses[k] /= num_batches
            
            # Validation phase
            model.eval()
            val_losses = {
                'world_loss': 0,
                'recon_loss': 0,
                'kl_loss': 0,
            }
            
            num_val_batches = 0
            
            _logger.info(f"Epoch {epoch + 1}/{num_epochs} - Validation...")
            
            # Add tqdm progress bar for validation
            val_limit = min(10, len(val_dataloader))  # Only validate on first 10 batches for speed
            val_pbar = tqdm(enumerate(val_dataloader),
                           total=val_limit,
                           desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
                           ncols=100,
                           leave=True)
            
            with torch.no_grad():
                for batch_idx, batch in val_pbar:
                    if batch_idx >= val_limit:
                        break
                    
                    # Move to device
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                    else:
                        obs, actions, meta = batch
                        batch = {
                            'observation': obs.to(device),
                            'action': actions.to(device),
                            'rewards': torch.zeros(obs.shape[0], obs.shape[1], device=device)
                        }
                    
                    try:
                        # Forward pass only
                        observations = batch['observation']
                        actions = batch['action']
                        
                        model_output = model(observations, actions, compute_loss=True)
                        
                        val_losses['world_loss'] += model_output['losses']['total_loss'].item()
                        val_losses['recon_loss'] += model_output['losses']['recon_loss'].item()
                        val_losses['kl_loss'] += model_output['losses']['kl_loss'].item()
                        
                        num_val_batches += 1
                        
                        # Update progress bar with validation losses
                        val_pbar.set_postfix({
                            'World': f"{model_output['losses']['total_loss'].item():.4f}",
                            'Recon': f"{model_output['losses']['recon_loss'].item():.4f}",
                            'KL': f"{model_output['losses']['kl_loss'].item():.4f}"
                        })
                    
                    except Exception as e:
                        _logger.error(f"Error in validation: {e}")
                        continue
            
            # Average validation losses
            for k in val_losses:
                if num_val_batches > 0:
                    val_losses[k] /= num_val_batches
            
            # Log epoch summary
            _logger.info("=" * 80)
            _logger.info(f"Epoch {epoch + 1}/{num_epochs} Summary:")
            _logger.info(f"  Train World Loss: {epoch_losses['world_loss']:.4f}")
            _logger.info(f"  Train Recon Loss: {epoch_losses['recon_loss']:.4f}")
            _logger.info(f"  Train KL Loss: {epoch_losses['kl_loss']:.4f}")
            _logger.info(f"  Train Actor Loss: {epoch_losses['actor_loss']:.4f}")
            _logger.info(f"  Train Critic Loss: {epoch_losses['critic_loss']:.4f}")
            if num_val_batches > 0:
                _logger.info(f"  Val World Loss: {val_losses['world_loss']:.4f}")
                _logger.info(f"  Val Recon Loss: {val_losses['recon_loss']:.4f}")
            _logger.info("=" * 80)
            
            # Log epoch metrics to MLflow
            mlflow.log_metrics({
                f"train_epoch/{k}": v for k, v in epoch_losses.items()
            }, step=epoch)
            
            if num_val_batches > 0:
                mlflow.log_metrics({
                    f"val_epoch/{k}": v for k, v in val_losses.items()
                }, step=epoch)
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'world_optimizer_state_dict': trainer.world_optimizer.state_dict(),
                    'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
                    'train_losses': epoch_losses,
                    'val_losses': val_losses,
                }, checkpoint_path)
                
                _logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                # Log checkpoint as artifact
                mlflow.log_artifact(str(checkpoint_path))
                
                # Save best model
                total_loss = epoch_losses['world_loss'] + epoch_losses['actor_loss'] + epoch_losses['critic_loss']
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_model_path = checkpoint_dir / "best_model.pt"
                    torch.save(model.state_dict(), best_model_path)
                    mlflow.log_artifact(str(best_model_path))
                    _logger.info(f"New best model saved with loss: {best_loss:.4f}")
        
        # Save final model with MLflow
        _logger.info("Saving final model to MLflow...")
        mlflow.pytorch.log_model(model, "model")
        
        # Log model architecture
        model_info = {
            "architecture": "DreamerModel",
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        mlflow.log_dict(model_info, "model_info.json")
        
        _logger.info(f"Training complete! MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    # Example usage
    _logger.info("Initializing Dreamer model...")
    
    # Default shape, will be updated if loading from HDF5
    input_shape = (64, 10)
    
    model = DreamerModel(
        h_state_size=200,
        z_state_size=30,
        action_size=128,
        embedding_size=256,
        aux_size=5,
        in_channels=1,
        cnn_depth=32,
        input_shape=input_shape
    )
    
    _logger.info("Loading dataset...")
    spec_path = "data/2_mel-spectrograms"
    style_path = "data/3_style-vectors"

    try:
        # allow passing an HDF5 consolidated dataset file instead of two folders
        if Path(spec_path).suffix == '.h5' or Path(style_path).suffix == '.h5':
            # if a single .h5 file is used, prefer it
            h5_path = spec_path if Path(spec_path).suffix == '.h5' else style_path
            dataset = SpectrogramH5Dataset(h5_path)
        else:
            dataset = SpectrogramDataset(spec_path, style_path)

        _logger.info(f"Dataset loaded with {len(dataset)} samples")
        _logger.info("Starting training with MLflow tracking...")

        train(
            model,
            dataset,
            num_epochs=100,
            batch_size=50,
            device='cuda',
            experiment_name="dreamer-spectrogram",
            checkpoint_freq=10
        )
        
        _logger.info("Training complete! Check mlruns/ for results.")
        
    except FileNotFoundError as e:
        _logger.error(f"Dataset not found: {e}")
        _logger.error("Please ensure the dataset paths are correct")
