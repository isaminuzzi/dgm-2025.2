"""This is a collection of relevant and reusable functions that might be reused along this implementation"""

# torch imports
import torch
from torch import Tensor, Size

# general imports
from typing import Tuple, Union

def flatten_batch(x: Tensor, nonbatch_dims: int = 1) -> Tuple[Tensor, Size]:
    """Helper function to flatten a batch of Tensors
       
       (b1,b2,..., X) => (B, X)
   
    Args:
        x (Tensor): tensor that will be flattened
        nonbatch_dims (int): how many dims describing actual features from a single sample
    
    Returns:
        Tuple[Tensor, Size]: flattened tensor and the batch dim
    """
    if nonbatch_dims > 0:
        batch_dim = x.shape[:-nonbatch_dims]
        x = torch.reshape(x, (-1,) + x.shape[-nonbatch_dims:])
        return x, batch_dim
    else:
        batch_dim = x.shape
        x = torch.reshape(x, (-1,))
        return x, batch_dim

def unflatten_batch(x: Tensor, batch_dim: Union[Size, Tuple]) -> Tensor:
    """Helper function to unflatten a flattened tensor
       (B, X) => (b1,b2,..., X)

        Args:
            x (Tensor): tensor that will be flattened
            nonbatch_dims (int): how many dims describing actual features from a single sample
    
        Returns:
            Tuple[Tensor, Size]: flattened tensor and the batch dim
    """
    x = torch.reshape(x, batch_dim + x.shape[1:])
    return x        
