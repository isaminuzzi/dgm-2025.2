# Reward and Auxiliary Predictors for Dreamer
# Predicts rewards and auxiliary acoustic features from latent states

import torch
import torch.nn as nn

from ..utils.logger import get_logger
from ..utils.functions import flatten_batch, unflatten_batch

_logger = get_logger("predictors", level="INFO")


class RewardPredictor(nn.Module):
    """Predicts rewards from latent states
    
    Args:
        h_state_size: Size of deterministic state (default: 200)
        z_state_size: Size of stochastic state (default: 30)
        hidden_size: Size of hidden layer (default: 400)
    """
    
    def __init__(self,
                 h_state_size: int = 200,
                 z_state_size: int = 30,
                 hidden_size: int = 400):
        super().__init__()
        
        latent_size = h_state_size + z_state_size
        
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, h_state, z_state):
        """
        Predict reward from latent states
        
        Args:
            h_state: Deterministic state (B, [T,] h_state_size)
            z_state: Stochastic state (B, [T,] z_state_size)
            
        Returns:
            Predicted rewards (B, [T,] 1)
        """
        latent = torch.cat([h_state, z_state], dim=-1)
        
        _logger.debug(f"RewardPredictor input shape: {latent.shape}")
        
        # Handle both (B, latent_size) and (B, T, latent_size)
        needs_unflatten = len(latent.shape) > 2
        if needs_unflatten:
            latent, batch_dim = flatten_batch(latent)
        
        reward = self.model(latent)
        
        if needs_unflatten:
            reward = unflatten_batch(reward, batch_dim)
        
        _logger.debug(f"RewardPredictor output shape: {reward.shape}")
        
        return reward


class StyleRewardPredictor(nn.Module):
    """Predicts style-based rewards from latent states and actions
    
    This predictor incorporates the style action a_t to predict style-specific rewards.
    
    Args:
        h_state_size: Size of deterministic state (default: 200)
        z_state_size: Size of stochastic state (default: 30)
        action_size: Size of style action (default: 128)
        hidden_size: Size of hidden layer (default: 400)
    """
    
    def __init__(self,
                 h_state_size: int = 200,
                 z_state_size: int = 30,
                 action_size: int = 128,
                 hidden_size: int = 400):
        super().__init__()
        
        input_size = h_state_size + z_state_size + action_size
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, h_state, z_state, action):
        """
        Predict style reward from latent states and action
        
        Args:
            h_state: Deterministic state (B, [T,] h_state_size)
            z_state: Stochastic state (B, [T,] z_state_size)
            action: Style action (B, [T,] action_size)
            
        Returns:
            Predicted style rewards (B, [T,] 1)
        """
        latent = torch.cat([h_state, z_state, action], dim=-1)
        
        _logger.debug(f"StyleRewardPredictor input shape: {latent.shape}")
        
        # Handle both (B, latent_size) and (B, T, latent_size)
        needs_unflatten = len(latent.shape) > 2
        if needs_unflatten:
            latent, batch_dim = flatten_batch(latent)
        
        reward = self.model(latent)
        
        if needs_unflatten:
            reward = unflatten_batch(reward, batch_dim)
        
        _logger.debug(f"StyleRewardPredictor output shape: {reward.shape}")
        
        return reward


class AuxiliaryPredictor(nn.Module):
    """Predicts auxiliary acoustic features from latent states
    
    This predictor estimates acoustic features derived from the input:
    - pitch (F0)
    - energy
    - delta-energy
    - spectral centroid
    - onset strength
    
    Args:
        h_state_size: Size of deterministic state (default: 200)
        z_state_size: Size of stochastic state (default: 30)
        aux_size: Size of auxiliary feature vector (default: 5)
        hidden_size: Size of hidden layer (default: 400)
    """
    
    def __init__(self,
                 h_state_size: int = 200,
                 z_state_size: int = 30,
                 aux_size: int = 5,
                 hidden_size: int = 400):
        super().__init__()
        
        latent_size = h_state_size + z_state_size
        
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, aux_size)
        )
        
    def forward(self, h_state, z_state):
        """
        Predict auxiliary features from latent states
        
        Args:
            h_state: Deterministic state (B, [T,] h_state_size)
            z_state: Stochastic state (B, [T,] z_state_size)
            
        Returns:
            Predicted auxiliary features (B, [T,] aux_size)
        """
        latent = torch.cat([h_state, z_state], dim=-1)
        
        _logger.debug(f"AuxiliaryPredictor input shape: {latent.shape}")
        
        # Handle both (B, latent_size) and (B, T, latent_size)
        needs_unflatten = len(latent.shape) > 2
        if needs_unflatten:
            latent, batch_dim = flatten_batch(latent)
        
        aux_features = self.model(latent)
        
        if needs_unflatten:
            aux_features = unflatten_batch(aux_features, batch_dim)
        
        _logger.debug(f"AuxiliaryPredictor output shape: {aux_features.shape}")
        
        return aux_features
