# For this encoder implementation, we will follow the Convolutional Neural Network, following the paper's parameters.
# We took inspiration from Pydreamer: http://github.com/jurgisp/pydreamer/blob/main/pydreamer/models/encoders.py

# Torch imports
import torch
import torch.nn as nn

# internal imports
from ..utils.logger import get_logger
from ..utils.functions import flatten_batch, unflatten_batch

# set up logging
_logger = get_logger("encoder", level="INFO")

class Encoder(nn.Module):
    """Dreamer full encoder implementation, contemplating both the Convolutional Network and the
    Multi-layer Perceptron, that receives the image embedding and the deterministic recurrent state
    """
    def __init__(self, 
                 h_state_size: int = 8, # following dreamer
                 in_channels: int = 1, 
                 cnn_depth: int = 32,
                 embedding_size: int = 256,
                 input_shape: tuple = (64, 10),  # (n_mels, time_frames)
                 ):
        super().__init__()

        # configuring CNN encoder with input shape
        self.cnn_encoder = ConvEncoder(in_channels, cnn_depth, input_shape)
        cnn_output_dim = self.cnn_encoder.out_dim

        # implementing the multilayer perceptron
        self.mlp = MLP(cnn_output_dim + h_state_size, # MLP receives the images and the deter recurrent state
                       embedding_size, 
                       hidden_dim=400, # following pydreamer
                       hidden_layers=2
                       )
        
        # Apply proper weight initialization to preserve variance
        self._init_weights()
        
    def _init_weights(self):
        """
        Apply He (Kaiming) initialization to encoder layers.
        
        Ensures variance is properly preserved from input through the encoding process.
        This prevents variance collapse and maintains spectral information quality.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                module.weight.data *= 0.5  # Controlled scaling
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                module.weight.data *= 0.7  # Controlled scaling for conv
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
      
    def forward(self, observation, deterministic_state):
        """
        Args:
            observation (torch.Tensor): The spectrogram sequence (B, T, C, H, W).
            deterministic_state (torch.Tensor): The deterministic state sequence (B, T, h_state_size).
        
        Returns:
            The final encoded sequence (B, T, embedding_size).
        """
        image_embedding = self.cnn_encoder(observation)

        # concatenate the embedding and the state
        x = torch.cat([image_embedding, deterministic_state], dim=-1)

        return self.mlp(x)


# --- Convolutional Encoder ---
# As the original paper suggests:
# The representation model is implemented as a Convolutional Neural Network (CNN; LeCun et al., 1989)
class ConvEncoder(nn.Module):
    """Implementation of a convolutional encoder
    
    The Convolutional Encoder follows the original implementation and takes inspiration
    from `Pydreamer`. Modified to handle spectrograms with small width (10 frames).
    """
    def __init__(self, in_channels: int = 1, # single audio channel
                 cnn_depth: int = 32,
                 input_shape: tuple = (64, 10)  # (n_mels, time_frames)
                 ):
        super().__init__()
        # Use asymmetric kernels: larger for height (mel-bins), smaller for width (time)
        # This handles spectrograms with various shapes better (64, 80 mels, etc.)
        # Using 3 layers instead of 4 to avoid width collapsing too fast
        d = cnn_depth
        self.model = nn.Sequential(
            # Layer 1: stride 2, padding (1,1)
            nn.Conv2d(in_channels, d * 2, kernel_size=(4, 3), stride=2, padding=(1, 1)),
            nn.ELU(),
            # Layer 2: stride 2, padding (1,1)
            nn.Conv2d(d * 2, d * 4, kernel_size=(4, 3), stride=2, padding=(1, 1)),
            nn.ELU(),
            # Layer 3: stride 2, padding (1,1)
            nn.Conv2d(d * 4, d * 8, kernel_size=(4, 3), stride=2, padding=(1, 1)),
            nn.ELU(),
            nn.Flatten()
        )
        
        # Calculate output dimension dynamically based on input shape
        self.out_dim = self._calculate_conv_output_dim(in_channels, input_shape)
        _logger.info(f"ConvEncoder output dimension: {self.out_dim} (input_shape={input_shape})")
    
    def _calculate_conv_output_dim(self, in_channels: int, input_shape: tuple) -> int:
        """Calculate the output dimension of the convolutional layers dynamically"""
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.zeros(1, in_channels, *input_shape)
            # Forward pass through conv layers (all except Flatten)
            dummy_output = self.model[:-1](dummy_input)  # Exclude Flatten
            # Return flattened size
            return int(torch.flatten(dummy_output, 1).shape[1])

    def forward(self, x):
        _logger.debug(f"x shape before flattening: {x.shape}") # expecting here (B, T, C, n_mels, L)
        
        x, batch_dim = flatten_batch(x, 3)  # Keep last 3 dims (C, H, W) for CNN
        _logger.debug(f"x shape after flattened: {x.shape}") # expecting here (B*T, C, H, W)

        y = self.model(x)
        _logger.debug(f"y shape flattened: {y.shape}")

        y = unflatten_batch(y, batch_dim)
        _logger.debug(f"y shape after unflattened: {y.shape}")

        return y

# --- MLP ---
# The MLP receives the image embedding and the deterministic recurrent state
class MLP(nn.Module):
    """Implementation of a generic multilayer perceptron"""
    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 hidden_layers: int):
        super().__init__()
        self.out_dim = out_dim
        dim = in_dim

        # creating the MLP
        layers = []
        for i in range(hidden_layers):
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.ELU() # dreamer implementation
            ]
            dim = hidden_dim
        
        # adding the output layer
        layers += [
            nn.Linear(dim, out_dim)
        ]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        _logger.debug(f"x shape before flattening: {x.shape}")
        
        x, batch_dim = flatten_batch(x)
        _logger.debug(f"x shape after flattened: {x.shape}")

        y = self.model(x)
        _logger.debug(f"y shape flattened: {y.shape}")

        y = unflatten_batch(y, batch_dim)
        _logger.debug(f"y shape after unflattened: {y.shape}")

        return y

