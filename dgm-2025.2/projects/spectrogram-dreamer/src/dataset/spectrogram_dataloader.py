# torch imports
from torch.utils.data import DataLoader, Dataset

def create_dataloader(dataset: Dataset, 
                      batch_size: int = 50, # same batch size used on original Dreamer implementation
                      shuffle: bool = True, 
                      num_workers: int = 0
                      ):
    """
    Creates a DataLoader suitable for the pre-loaded Dreamer dataset.
    num_workers is set to 0 to prevent memory duplication of the large dataset.
    """
    dataloader = DataLoader(
                    dataset, 
                    batch_size = batch_size, 
                    shuffle = shuffle, 
                    # CRITICAL: Must be 0 because the entire dataset is already loaded in memory
                    num_workers = num_workers, 
                    pin_memory = True # Set to True for faster GPU transfer
                 )
    
    return dataloader