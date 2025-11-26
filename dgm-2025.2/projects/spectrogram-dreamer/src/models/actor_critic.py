# Actor and Critic implementations for Dreamer
# Actor generates style actions, Critic estimates value function

import torch
import torch.nn as nn
from torch.distributions import Normal

from ..utils.logger import get_logger
from ..utils.functions import flatten_batch, unflatten_batch

_logger = get_logger("actor_critic", level="INFO")


class Actor(nn.Module):
    """Actor network for generating style actions
    
    The Actor predicts the dynamic part of the style vector a_t from latent states.
    This follows the Dreamer policy architecture.
    
    Args:
        h_state_size: Size of deterministic state (default: 200)
        z_state_size: Size of stochastic state (default: 30)
        action_size: Size of style vector output (default: 128)
        hidden_size: Size of hidden layers (default: 400)
        hidden_layers: Number of hidden layers (default: 4)
    """
    
    def __init__(self,
                 h_state_size: int = 200,
                 z_state_size: int = 30,
                 action_size: int = 128,
                 hidden_size: int = 400,
                 hidden_layers: int = 4):
        super().__init__()
        
        self.action_size = action_size
        latent_size = h_state_size + z_state_size
        
        # Build MLP
        layers = []
        input_size = latent_size
        for _ in range(hidden_layers):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ELU()
            ])
            input_size = hidden_size
        
        # Output layer produces mean and std for action distribution
        layers.append(nn.Linear(hidden_size, 2 * action_size))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, h_state, z_state, sample=True):
        """
        Generate style action from latent states
        
        Args:
            h_state: Deterministic state (B, [T,] h_state_size)
            z_state: Stochastic state (B, [T,] z_state_size)
            sample: Whether to sample from distribution or use mean
            
        Returns:
            Style action a_t (B, [T,] action_size)
        """
        # Concatenate states
        latent = torch.cat([h_state, z_state], dim=-1)
        
        _logger.debug(f"Actor input shape: {latent.shape}")
        
        # Handle both (B, latent_size) and (B, T, latent_size)
        needs_unflatten = len(latent.shape) > 2
        if needs_unflatten:
            latent, batch_dim = flatten_batch(latent)
        
        # Forward pass
        output = self.model(latent)
        
        # Split into mean and std
        mean, std = torch.chunk(output, 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 0.1  # Ensure positive
        
        if sample:
            # Sample from distribution
            dist = Normal(mean, std)
            action = dist.rsample()
        else:
            # Use mean for deterministic action
            action = mean
        
        # Apply tanh to bound actions
        action = torch.tanh(action)
        
        if needs_unflatten:
            action = unflatten_batch(action, batch_dim)
        
        _logger.debug(f"Actor output shape: {action.shape}")
        
        return action


class Critic(nn.Module):
    """Critic network for value estimation
    
    Estimates V(h_t, z_t) for the value function.
    
    Args:
        h_state_size: Size of deterministic state (default: 200)
        z_state_size: Size of stochastic state (default: 30)
        hidden_size: Size of hidden layers (default: 400)
        hidden_layers: Number of hidden layers (default: 4)
    """
    
    def __init__(self,
                 h_state_size: int = 200,
                 z_state_size: int = 30,
                 hidden_size: int = 400,
                 hidden_layers: int = 4):
        super().__init__()
        
        latent_size = h_state_size + z_state_size
        
        # Build MLP
        layers = []
        input_size = latent_size
        for _ in range(hidden_layers):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ELU()
            ])
            input_size = hidden_size
        
        # Output single value
        layers.append(nn.Linear(hidden_size, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, h_state, z_state):
        """
        Estimate value function
        
        Args:
            h_state: Deterministic state (B, [T,] h_state_size)
            z_state: Stochastic state (B, [T,] z_state_size)
            
        Returns:
            Value estimates (B, [T,] 1)
        """
        # Concatenate states
        latent = torch.cat([h_state, z_state], dim=-1)
        
        _logger.debug(f"Critic input shape: {latent.shape}")
        
        # Handle both (B, latent_size) and (B, T, latent_size)
        needs_unflatten = len(latent.shape) > 2
        if needs_unflatten:
            latent, batch_dim = flatten_batch(latent)
        
        # Forward pass
        value = self.model(latent)
        
        if needs_unflatten:
            value = unflatten_batch(value, batch_dim)
        
        _logger.debug(f"Critic output shape: {value.shape}")
        
        return value
