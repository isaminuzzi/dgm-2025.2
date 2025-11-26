# Main Dreamer Model
# Integrates all components: Encoder, RSSM, Decoder, Actor, Critic, and Predictors

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .rssm import RSSM
from .actor_critic import Actor, Critic
from .predictors import RewardPredictor, StyleRewardPredictor, AuxiliaryPredictor
from ..utils.logger import get_logger

_logger = get_logger("dreamer_model", level="INFO")


class DreamerModel(nn.Module):
    """Complete Dreamer model with style-controllable generation
    
    This model implements the Dreamer architecture adapted for spectrogram generation
    with style control via action vectors a_t.
    
    Args:
        h_state_size: Size of deterministic recurrent state (default: 200)
        z_state_size: Size of stochastic state (default: 30)
        action_size: Size of style action vector (default: 128)
        embedding_size: Size of observation embedding (default: 256)
        aux_size: Size of auxiliary feature vector (default: 5)
        in_channels: Input spectrogram channels (default: 1)
        cnn_depth: CNN depth multiplier (default: 32)
    """
    
    def __init__(self,
                 h_state_size: int = 200,
                 z_state_size: int = 30,
                 action_size: int = 128,
                 embedding_size: int = 256,
                 aux_size: int = 5,
                 in_channels: int = 1,
                 cnn_depth: int = 32,
                 input_shape: tuple = (64, 10)):  # (n_mels, time_frames)
        super().__init__()
        
        self.h_state_size = h_state_size
        self.z_state_size = z_state_size
        self.action_size = action_size
        
        # World Model Components
        self.encoder = Encoder(
            h_state_size=h_state_size,
            in_channels=in_channels,
            cnn_depth=cnn_depth,
            embedding_size=embedding_size,
            input_shape=input_shape
        )
        
        self.rssm = RSSM(
            h_state_size=h_state_size,
            z_state_size=z_state_size,
            action_size=action_size,
            embedding_size=embedding_size
        )
        
        self.decoder = Decoder(
            h_state_size=h_state_size,
            z_state_size=z_state_size,
            out_channels=in_channels,
            cnn_depth=cnn_depth,
            output_shape=input_shape  # decoder outputs same shape as encoder input
        )
        
        # Predictors
        self.reward_predictor = RewardPredictor(
            h_state_size=h_state_size,
            z_state_size=z_state_size
        )
        
        self.style_reward_predictor = StyleRewardPredictor(
            h_state_size=h_state_size,
            z_state_size=z_state_size,
            action_size=action_size
        )
        
        self.auxiliary_predictor = AuxiliaryPredictor(
            h_state_size=h_state_size,
            z_state_size=z_state_size,
            aux_size=aux_size
        )
        
        # Actor-Critic for policy learning
        self.actor = Actor(
            h_state_size=h_state_size,
            z_state_size=z_state_size,
            action_size=action_size
        )
        
        self.critic = Critic(
            h_state_size=h_state_size,
            z_state_size=z_state_size
        )
        
    def forward(self, observations, actions, compute_loss=True, aux_targets=None):
        """
        Forward pass through the Dreamer model
        
        Args:
            observations: Input spectrograms (B, T, C, H, W)
            actions: Style action vectors a_t (B, T, action_size)
            compute_loss: Whether to compute losses (default: True)
            aux_targets: Target auxiliary features for loss computation (B, T, aux_size)
            
        Returns:
            Dictionary with outputs and losses
        """
        batch_size, seq_len = observations.shape[:2]
        device = observations.device
        
        # Initialize states
        h_state = torch.zeros(batch_size, self.h_state_size, device=device)
        z_state = torch.zeros(batch_size, self.z_state_size, device=device)
        
        # Step 1: Encode observations (requires h_state, so we do it iteratively with RSSM)
        # We need to encode alongside RSSM since encoder needs h_state
        
        # Collect all embeddings first with initial h_state
        obs_embeddings = []
        h_states_for_encoding = [h_state]
        
        # First pass: collect deterministic states for encoding
        for t in range(seq_len):
            if t > 0:
                # Update h_state for encoding
                h_state = self.rssm.gru(
                    torch.cat([z_state, actions[:, t-1]], dim=-1), 
                    h_state
                )
                h_states_for_encoding.append(h_state)
        
        # Stack h_states for encoding
        h_states_stacked = torch.stack(h_states_for_encoding[:seq_len], dim=1)
        
        # Encode all observations
        obs_embedding = self.encoder(observations, h_states_stacked)
        
        _logger.debug(f"Observation embedding shape: {obs_embedding.shape}")
        
        # Step 2: RSSM forward pass
        rssm_output = self.rssm(obs_embedding, actions)
        
        h_states = rssm_output['h_states']
        z_states = rssm_output['z_states']
        
        _logger.debug(f"h_states shape: {h_states.shape}")
        _logger.debug(f"z_states shape: {z_states.shape}")
        
        # Step 3: Decode to reconstruct observations
        reconstructed = self.decoder(h_states, z_states)
        
        _logger.debug(f"Reconstructed shape: {reconstructed.shape}")
        
        # Step 4: Predict rewards and auxiliary features
        predicted_rewards = self.reward_predictor(h_states, z_states)
        predicted_style_rewards = self.style_reward_predictor(h_states, z_states, actions)
        predicted_aux = self.auxiliary_predictor(h_states, z_states)
        
        output = {
            'reconstructed': reconstructed,
            'h_states': h_states,
            'z_states': z_states,
            'predicted_rewards': predicted_rewards,
            'predicted_style_rewards': predicted_style_rewards,
            'predicted_aux': predicted_aux,
            'prior_means': rssm_output['prior_means'],
            'prior_stds': rssm_output['prior_stds'],
            'posterior_means': rssm_output['posterior_means'],
            'posterior_stds': rssm_output['posterior_stds']
        }
        
        if compute_loss:
            losses = self.compute_losses(
                observations, reconstructed,
                rssm_output['prior_means'], rssm_output['prior_stds'],
                rssm_output['posterior_means'], rssm_output['posterior_stds'],
                predicted_aux, aux_targets
            )
            output['losses'] = losses
        
        return output
    
    def compute_losses(self, observations, reconstructed, 
                      prior_means, prior_stds, posterior_means, posterior_stds,
                      predicted_aux, aux_targets=None):
        """
        Compute all losses for the world model
        
        Args:
            observations: Ground truth Log-Mel spectrograms (in log space)
            reconstructed: Reconstructed Log-Mel spectrograms
            prior_means, prior_stds: Prior distribution parameters
            posterior_means, posterior_stds: Posterior distribution parameters
            predicted_aux: Predicted auxiliary features
            aux_targets: Ground truth auxiliary features (optional)
            
        Returns:
            Dictionary of losses
            
        Note:
            MSE loss is appropriate for Log-Mel spectrograms because the log
            transformation compresses the dynamic range, making all frequency
            content equally important to the loss function. This prevents the
            model from ignoring quiet sounds (speech texture) and focusing only
            on loud peaks (volume spikes).
        """
        # Reconstruction loss (MSE on Log-Mel spectrograms)
        recon_loss = F.mse_loss(reconstructed, observations)
        
        # KL divergence loss with free nats to prevent posterior collapse
        # Free nats allows the latent space to maintain minimum information capacity
        prior_dist = torch.distributions.Normal(prior_means, prior_stds)
        posterior_dist = torch.distributions.Normal(posterior_means, posterior_stds)
        kl_divergence = torch.distributions.kl_divergence(posterior_dist, prior_dist)
        
        # Apply free nats: minimum KL per dimension to maintain representation capacity
        # This prevents the posterior from collapsing to the prior too much
        free_nats = 3.0  # Standard value for latent variable models
        kl_loss = torch.maximum(
            kl_divergence, 
            torch.tensor(free_nats, device=observations.device)
        ).mean()
        
        # Auxiliary loss (if targets provided)
        aux_loss = torch.tensor(0.0, device=observations.device)
        if aux_targets is not None:
            aux_loss = F.mse_loss(predicted_aux, aux_targets)
        
        # Total loss
        total_loss = recon_loss + kl_loss + aux_loss
        
        # Compute variance metrics for monitoring (detached, no gradients)
        with torch.no_grad():
            obs_std = observations.std().item()
            recon_std = reconstructed.std().item()
            variance_ratio = recon_std / (obs_std + 1e-8)
            
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'aux_loss': aux_loss,
            # Variance metrics for monitoring
            'obs_std': obs_std,
            'recon_std': recon_std,
            'variance_ratio': variance_ratio
        }
    
    def imagine_trajectory(self, initial_h_state, initial_z_state, horizon):
        """
        Imagine future trajectory using the actor policy
        
        Args:
            initial_h_state: Initial deterministic state (B, h_state_size)
            initial_z_state: Initial stochastic state (B, z_state_size)
            horizon: Number of steps to imagine
            
        Returns:
            Dictionary with imagined trajectory and predictions
        """
        imagined = self.rssm.imagine(self.actor, initial_h_state, initial_z_state, horizon)
        
        h_states = imagined['h_states']
        z_states = imagined['z_states']
        actions = imagined['actions']
        
        # Predict values and rewards along the trajectory
        values = self.critic(h_states, z_states)
        rewards = self.reward_predictor(h_states[:, :-1], z_states[:, :-1])
        style_rewards = self.style_reward_predictor(h_states[:, :-1], z_states[:, :-1], actions)
        
        # Decode imagined spectrograms
        imagined_specs = self.decoder(h_states, z_states)
        
        return {
            'h_states': h_states,
            'z_states': z_states,
            'actions': actions,
            'values': values,
            'rewards': rewards,
            'style_rewards': style_rewards,
            'imagined_specs': imagined_specs
        }
