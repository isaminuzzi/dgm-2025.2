# RSSM (Recurrent State Space Model) implementation for Dreamer
# This module implements the world model's recurrent state space dynamics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ..utils.logger import get_logger
from ..utils.functions import flatten_batch, unflatten_batch

_logger = get_logger("rssm", level="INFO")


class RSSM(nn.Module):
    """Recurrent State Space Model for Dreamer
    
    The RSSM maintains both deterministic (h_t) and stochastic (z_t) states.
    It uses style vectors (a_t) as actions to drive the dynamics.
    
    Args:
        h_state_size: Size of deterministic recurrent state (default: 200)
        z_state_size: Size of stochastic state (default: 30)
        action_size: Size of style vector (a_t) (default: 128)
        embedding_size: Size of observation embedding (default: 256)
        hidden_size: Size of hidden layers (default: 200)
    """
    
    def __init__(self,
                 h_state_size: int = 200,
                 z_state_size: int = 30,
                 action_size: int = 128,
                 embedding_size: int = 256,
                 hidden_size: int = 200):
        super().__init__()
        
        self.h_state_size = h_state_size
        self.z_state_size = z_state_size
        self.action_size = action_size
        
        # Recurrent model: h_t = f(h_{t-1}, z_{t-1}, a_t)
        # Using GRU as per Dreamer implementation
        self.gru = nn.GRUCell(z_state_size + action_size, h_state_size)
        
        # Prior: p(z_t | h_t)
        self.prior = nn.Sequential(
            nn.Linear(h_state_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * z_state_size)  # mean and std
        )
        
        # Posterior: q(z_t | h_t, o_t)
        self.posterior = nn.Sequential(
            nn.Linear(h_state_size + embedding_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * z_state_size)  # mean and std
        )
        
    def forward(self, obs_embedding, style_action, prev_h_state=None, prev_z_state=None):
        """
        Forward pass through RSSM
        
        Args:
            obs_embedding: Encoded observation (B, T, embedding_size)
            style_action: Style vector a_t (B, T, action_size)
            prev_h_state: Previous deterministic state (B, h_state_size)
            prev_z_state: Previous stochastic state (B, z_state_size)
            
        Returns:
            Dictionary with h_states, z_states, prior_dists, posterior_dists
        """
        batch_size, seq_len = obs_embedding.shape[:2]
        device = obs_embedding.device
        
        # Initialize states if not provided
        if prev_h_state is None:
            prev_h_state = torch.zeros(batch_size, self.h_state_size, device=device)
        if prev_z_state is None:
            prev_z_state = torch.zeros(batch_size, self.z_state_size, device=device)
        
        h_states = []
        z_states = []
        prior_means = []
        prior_stds = []
        posterior_means = []
        posterior_stds = []
        
        for t in range(seq_len):
            # Get current inputs
            obs_t = obs_embedding[:, t]
            action_t = style_action[:, t]
            
            # Recurrent step: h_t = f(h_{t-1}, z_{t-1}, a_t)
            h_t = self.gru(torch.cat([prev_z_state, action_t], dim=-1), prev_h_state)
            
            # Prior: p(z_t | h_t)
            prior_params = self.prior(h_t)
            prior_mean, prior_std = torch.chunk(prior_params, 2, dim=-1)
            prior_std = F.softplus(prior_std) + 0.1  # Ensure positive std
            prior_dist = Normal(prior_mean, prior_std)
            
            # Posterior: q(z_t | h_t, o_t)
            posterior_params = self.posterior(torch.cat([h_t, obs_t], dim=-1))
            posterior_mean, posterior_std = torch.chunk(posterior_params, 2, dim=-1)
            posterior_std = F.softplus(posterior_std) + 0.1
            posterior_dist = Normal(posterior_mean, posterior_std)
            
            # Sample from posterior during training
            z_t = posterior_dist.rsample()
            
            # Store states
            h_states.append(h_t)
            z_states.append(z_t)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)
            
            # Update for next step
            prev_h_state = h_t
            prev_z_state = z_t
        
        # Stack all states
        h_states = torch.stack(h_states, dim=1)  # (B, T, h_state_size)
        z_states = torch.stack(z_states, dim=1)  # (B, T, z_state_size)
        prior_means = torch.stack(prior_means, dim=1)
        prior_stds = torch.stack(prior_stds, dim=1)
        posterior_means = torch.stack(posterior_means, dim=1)
        posterior_stds = torch.stack(posterior_stds, dim=1)
        
        return {
            'h_states': h_states,
            'z_states': z_states,
            'prior_means': prior_means,
            'prior_stds': prior_stds,
            'posterior_means': posterior_means,
            'posterior_stds': posterior_stds
        }
    
    def imagine(self, actor, h_state, z_state, horizon):
        """
        Imagine future trajectories using the actor policy
        
        Args:
            actor: Actor network to generate style actions
            h_state: Initial deterministic state (B, h_state_size)
            z_state: Initial stochastic state (B, z_state_size)
            horizon: Number of steps to imagine
            
        Returns:
            Dictionary with imagined h_states, z_states, and actions
        """
        batch_size = h_state.shape[0]
        device = h_state.device
        
        h_states = [h_state]
        z_states = [z_state]
        actions = []
        
        for _ in range(horizon):
            # Generate action from actor
            action = actor(h_state, z_state)
            
            # Predict next state
            h_state = self.gru(torch.cat([z_state, action], dim=-1), h_state)
            
            # Sample from prior
            prior_params = self.prior(h_state)
            prior_mean, prior_std = torch.chunk(prior_params, 2, dim=-1)
            prior_std = F.softplus(prior_std) + 0.1
            prior_dist = Normal(prior_mean, prior_std)
            z_state = prior_dist.rsample()
            
            h_states.append(h_state)
            z_states.append(z_state)
            actions.append(action)
        
        return {
            'h_states': torch.stack(h_states, dim=1),  # (B, horizon+1, h_state_size)
            'z_states': torch.stack(z_states, dim=1),  # (B, horizon+1, z_state_size)
            'actions': torch.stack(actions, dim=1)      # (B, horizon, action_size)
        }
