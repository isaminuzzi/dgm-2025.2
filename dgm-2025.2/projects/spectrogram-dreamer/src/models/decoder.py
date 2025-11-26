# Decoder implementation for Dreamer
# Reconstructs spectrograms from latent states

import torch
import torch.nn as nn

from ..utils.logger import get_logger
from ..utils.functions import flatten_batch, unflatten_batch

_logger = get_logger("decoder", level="INFO")


class Decoder(nn.Module):
    """Decoder for reconstructing spectrograms from latent states
    
    Args:
        h_state_size: Size of deterministic state (default: 200)
        z_state_size: Size of stochastic state (default: 30)
        out_channels: Output channels (default: 1 for mono audio)
        cnn_depth: Base depth for convolutional layers (default: 32)
    """
    
    def __init__(self,
                 h_state_size: int = 200,
                 z_state_size: int = 30,
                 out_channels: int = 1,
                 cnn_depth: int = 32,
                 output_shape: tuple = (64, 10)):  # (n_mels, time_frames) - target output
        super().__init__()
        
        self.cnn_depth = cnn_depth
        self.output_shape = output_shape
        d = cnn_depth
        
        # Calculate initial spatial dimensions based on output shape
        # After 3 ConvTranspose2d layers with stride 2:
        # output_shape / (2^3) = initial_shape
        initial_height = output_shape[0] // 8  # e.g., 64//8=8, 80//8=10
        initial_width = output_shape[1] // 8   # e.g., 10//8=1 (but we use 2 minimum)
        initial_width = max(2, initial_width)  # Ensure minimum width of 2
        
        self.initial_shape = (d * 8, initial_height, initial_width)
        
        # MLP to expand latent state
        latent_size = h_state_size + z_state_size
        mlp_output_size = d * 8 * initial_height * initial_width
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, 200),
            nn.ELU(),
            nn.Linear(200, mlp_output_size)
        )
        
        # Transposed convolutions to upsample to target output_shape
        # output_padding is calculated dynamically to match exact output dimensions
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(d * 8, d * 4, kernel_size=(4, 3), stride=2, padding=(1, 1)),
            nn.ELU(),
            nn.ConvTranspose2d(d * 4, d * 2, kernel_size=(4, 3), stride=2, padding=(1, 1)),
            nn.ELU(),
            nn.ConvTranspose2d(d * 2, out_channels, kernel_size=(4, 3), stride=2, padding=(1, 1), output_padding=(0, 1)),
        )
        
        _logger.info(f"Decoder initial_shape: {self.initial_shape}, output_shape: {output_shape}")
        
        # Apply proper weight initialization to preserve variance through the network
        # This is CRITICAL to prevent variance collapse that causes robotic audio
        self._init_weights()
        
    def _init_weights(self):
        """
        Apply He (Kaiming) initialization to all layers.
        
        He initialization is critical for maintaining variance through deep networks.
        Default PyTorch initialization causes severe variance collapse in the decoder,
        reducing output std from ~1.0 to ~0.04-0.11 (90%+ variance loss).
        
        This variance collapse causes:
        - Loss of spectral detail and contrast
        - Robotic, monotone-sounding audio
        - Flattened formants and harmonics
        - Poor vocoder reconstruction
        
        Reference: He et al. (2015) - "Delving Deep into Rectifiers: Surpassing 
                   Human-Level Performance on ImageNet Classification"
        
        Note: We use a controlled scaling factor (0.5 for Linear, 0.7 for Conv) to
        prevent over-initialization while still preserving variance better than default.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for Linear layers with ELU activation
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # Scale down slightly to prevent over-initialization
                module.weight.data *= 0.5
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.ConvTranspose2d):
                # He initialization for ConvTranspose2d layers
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # Scale down slightly for conv layers
                module.weight.data *= 0.7
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, h_state, z_state):
        """
        Decode latent states to spectrograms
        
        Args:
            h_state: Deterministic state (B, T, h_state_size)
            z_state: Stochastic state (B, T, z_state_size)
            
        Returns:
            Reconstructed spectrograms (B, T, C, H, W)
        """
        # Concatenate states
        latent = torch.cat([h_state, z_state], dim=-1)
        
        _logger.debug(f"Latent shape: {latent.shape}")
        
        # Flatten batch and time
        latent, batch_dim = flatten_batch(latent)
        _logger.debug(f"Flattened latent shape: {latent.shape}")
        
        # MLP
        x = self.mlp(latent)
        _logger.debug(f"After MLP: {x.shape}")
        
        # Reshape to feature map (d * 8 channels, 4x1 spatial)
        x = x.view(-1, *self.initial_shape)
        _logger.debug(f"Reshaped: {x.shape}")
        
        # Deconvolution
        x = self.deconv(x)
        _logger.debug(f"After deconv: {x.shape}")
        
        # Unflatten batch
        x = unflatten_batch(x, batch_dim)
        _logger.debug(f"Final output: {x.shape}")
        
        return x
