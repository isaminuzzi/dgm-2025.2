import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
from tqdm import tqdm

# from tokenizer.tokenizer import SpectrogramPatchTokenizer, SpectrogramVQTokenizer


class SpectogramPatchesDataset(Dataset):
    """Generates the spectogram patches from .pt files .
    
    The dataset is composed by the generated spectograms broken down into 
    sequential time patches. Images will be broken into `sequence_length` time steps.
    The original paper uses 50, so we mainting it.

    This mimics the process of sampling sequences from episodes from the orginal paper.
    """
    def __init__(
        self,
        pt_dir: Path,
        sequence_length: int = 50,
        patch_width: int = 16,
    ):
        
        """
        Args:
            pt_dir (Path): Path to the directory contatining the .pt spectogram files
            sequence_length [L] (int): This is the fixed length 'L' of the sequence of patches. The length of the mini-batch fed into the Recurrent State-Space Model (RSSM)
            patch_width [W] (int): This is the width of the 2D spectogram patch that the CNN will process.
        """
        
        self.sequence_length = sequence_length
        self.patch_width = patch_width
        
        # This is the total number of time steps needed from the original spectogram
        # to produce the `sequence_length` with a hop size of 1.
        # >>> Example: L=50, W=16
        #        Total Window Size = 50 + 16 - 1 = 65 Time Steps

        #        The `unfold` operation efficiently handles this:
        #        - Patch x_1: covers indices 0 to 15.
        #        - Patch x_2: covers indices 1 to 16.
        #        ...
        #        - Patch x_50: covers indices 49 to 64.
        self.total_window_size = self.sequence_length + self.patch_width - 1
        self.pt_files = []
        for f in pt_dir.glob("*.pt"):
            try:
                self.pt_files.append(f)
            except Exception as e:
                print(f"Error reading file {f}, skipping: {e}")
            
            print(f"Found {len(self.pt_files)} spectogram files")
            if len(self.pt_files) == 0:
                raise IOError(f"No .pt files found in {pt_dir}")
        
    def __len__(self) -> int:
        """Returns the total number of spectogram files. Equivalent to the number of episodes"""
        return len(self.pt_files)
            
    def __getitem__(self, index):
        """Loads a full spectogram ("episode") and returns a 
        sequence of random 2D patches from it
        """
        file_path = self.pt_files[index]
        data = torch.load(file_path) # (1, n_mels, T_full)


        full_spec = data['spec'].squeeze(0) # (n_mels, T_full)
        n_mels, T_full = full_spec.shape

        if T_full <= self.total_window_size:
            # if file is too short, we grab another random item
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())

        # Pick a random start index for the "super-patch"
        start_idx = torch.randint(0, T_full - self.total_window_size, (1,)).item()
        
        # Slice the total window from the spectrogram
        # This "super-patch" contains all data for the L patches
        super_patch = full_spec[:, start_idx : start_idx + self.total_window_size] # (n_mels, total_window_size)
        
        # We unfold on dimension 1 (time)
        # Size is our patch_width
        # Step is 1 (to get overlapping patches x_t, x_{t+1}, ...)
        
        # This creates a [n_mels, sequence_length, patch_width] tensor
        patches = super_patch.unfold(dimension=1, size=self.patch_width, step=1)
        
        # Format for the model: [L, C, H, W]
        # We want: [L, 1, n_mels, patch_width]
        
        # [n_mels, L, patch_width] -> [L, n_mels, patch_width]
        patches = patches.permute(1, 0, 2)
        
        # [L, n_mels, patch_width] -> [L, 1, n_mels, patch_width]
        # Add channel dimension
        patches = patches.unsqueeze(1)
        
        # adding actions
        # For now, using dummy actions
        actions = torch.zeros(self.sequence_length, 1)

        # reset
        reset = torch.zeros(self.sequence_length, dtype=torch.bool)
        reset[0] = True

        return {
                "obs": {
                    "spectrogram_patches": patches # Patches shape: [L, 1, H, W]
                }, 
                "actions": actions, # Actions shape: [L, 1]
                "reset": reset # Reset shape: [L]
        }

# class SpectrogramTokenDataset(Dataset):
#     def __init__(
#         self,
#         token_files: List[str],
#         sequence_length: int = 50,
#         use_embeddings: bool = False,
#         use_rewards: bool = False,
#         use_terminals: bool = False
#     ):
#         self.token_files = token_files
#         self.sequence_length = sequence_length
#         self.use_embeddings = use_embeddings
#         self.use_rewards = use_rewards
#         self.use_terminals = use_terminals
        
#         self.num_files = len(token_files)
        
#         sample_size = min(100, self.num_files)
#         total_seqs = 0
#         valid_files = 0
        
#         for i in range(sample_size):
#             try:
#                 data = torch.load(token_files[i])
#                 if self.use_embeddings:
#                     tokens = data.get('embeddings', data.get('patches'))
#                 else:
#                     tokens = data.get('tokens', data.get('indices', data.get('patches')))
                
#                 if tokens is not None:
#                     num_tokens = tokens.shape[0]
#                     # Calcula quantas sequências este arquivo pode gerar
#                     if num_tokens >= sequence_length:
#                         num_seqs = max(0, (num_tokens - sequence_length) // (sequence_length // 2) + 1)
#                         total_seqs += num_seqs
#                         valid_files += 1
#             except Exception as e:
#                 print(f"Warning: Error loading {token_files[i]}: {e}")
#                 continue
        
#         # Estima total baseado na amostra
#         if valid_files > 0:
#             self.avg_seqs_per_file = total_seqs / valid_files
#             self.estimated_valid_files = int(self.num_files * (valid_files / sample_size))
#             self.estimated_total_seqs = int(self.estimated_valid_files * self.avg_seqs_per_file)
#         else:
#             self.avg_seqs_per_file = 0
#             self.estimated_valid_files = 0
#             self.estimated_total_seqs = 0
        
#         print(f"Valid files: {valid_files}/{sample_size} ({100*valid_files/sample_size:.1f}%)")
#         print(f"Estimated {self.estimated_total_seqs:,} sequences from {self.estimated_valid_files:,} valid files")
#         print(f"Average: {self.avg_seqs_per_file:.1f} sequences/file")
#         print(f"Sequences will be created on-demand during training")
        
#         # Cache para armazenar info de arquivos já carregados
#         self._file_cache = {}
    
#     def _get_file_sequences(self, file_path: str):
#         if file_path in self._file_cache:
#             return self._file_cache[file_path]
        
#         data = torch.load(file_path)
        
#         if self.use_embeddings:
#             tokens = data.get('embeddings', data.get('patches'))
#         else:
#             tokens = data.get('tokens', data.get('indices', data.get('patches')))
        
#         if tokens is None:
#             self._file_cache[file_path] = []
#             return []
        
#         # Divide em sequências
#         num_tokens = tokens.shape[0]
#         sequences = []
        
#         for start in range(0, num_tokens - self.sequence_length + 1, self.sequence_length // 2):
#             end = start + self.sequence_length
#             sequences.append({
#                 'tokens': tokens[start:end],
#                 'start': start,
#                 'end': end
#             })
        
#         self._file_cache[file_path] = sequences
#         return sequences
    
#     def __len__(self):
#         # Retorna estimativa de sequências totais
#         return self.estimated_total_seqs
    
#     def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
#         # Mapeia idx para (file_idx, seq_idx) de forma mais inteligente
#         seqs_per_file = int(self.avg_seqs_per_file) or 1
#         file_idx = (idx // seqs_per_file) % self.num_files
#         seq_idx = idx % seqs_per_file
        
#         file_path = self.token_files[file_idx]
#         sequences = self._get_file_sequences(file_path)
        
#         if len(sequences) == 0 or seq_idx >= len(sequences):
#             # Arquivo inválido ou seq_idx fora do range, tenta próximo
#             return self.__getitem__((idx + 1) % len(self))
        
#         tokens = sequences[seq_idx]['tokens']
        
#         T = tokens.shape[0]
        
#         # Observação
#         obs = {
#             'spectrogram_tokens': tokens
#         }
        
#         # Actions (dummy - apenas transição temporal)
#         actions = torch.zeros(T, 1)  # (T, 1)
        
#         # Reset (primeiro timestep é reset)
#         reset = torch.zeros(T, dtype=torch.bool)
#         reset[0] = True
        
#         # Rewards e terminals opcionais
#         if self.use_rewards:
#             obs['reward'] = torch.zeros(T)
        
#         if self.use_terminals:
#             obs['terminal'] = torch.zeros(T, dtype=torch.bool)
#             obs['terminal'][-1] = True  # Último timestep é terminal
        
#         return {
#             'obs': obs,
#             'actions': actions,
#             'reset': reset
#         }


# class AudioSequenceDataset(Dataset):
#     def __init__(
#         self,
#         audio_files: List[str],
#         vq_tokenizer: SpectrogramVQTokenizer,
#         patch_tokenizer: SpectrogramPatchTokenizer,
#         sequence_length: int = 50,
#         use_embeddings: bool = False
#     ):
#         self.audio_files = audio_files
#         self.vq_tokenizer = vq_tokenizer
#         self.patch_tokenizer = patch_tokenizer
#         self.sequence_length = sequence_length
#         self.use_embeddings = use_embeddings
        
#         # Carrega espectrogramas
#         self.spectrograms = []
#         for fpath in audio_files:
#             data = torch.load(fpath)
#             spec = data['spec']
#             self.spectrograms.append(spec)
    
#     def __len__(self):
#         return len(self.spectrograms)
    
#     def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
#         spec = self.spectrograms[idx]
        
#         # Tokeniza
#         with torch.no_grad():
#             patches = self.patch_tokenizer.tokenize(spec)
#             patches = patches.unsqueeze(0)  # (1, N, D)
            
#             if self.use_embeddings:
#                 z_q, _ = self.vq_tokenizer.encode(patches)
#                 tokens = z_q.squeeze(0)  # (N, E)
#             else:
#                 _, indices = self.vq_tokenizer.encode(patches)
#                 tokens = indices.squeeze(0)  # (N,)
        
#         # Trunca ou padding para sequence_length
#         num_tokens = tokens.shape[0]
#         if num_tokens >= self.sequence_length:
#             tokens = tokens[:self.sequence_length]
#         else:
#             # Padding
#             if len(tokens.shape) == 1:
#                 pad = torch.zeros(self.sequence_length - num_tokens, dtype=tokens.dtype)
#             else:
#                 pad = torch.zeros(self.sequence_length - num_tokens, tokens.shape[1])
#             tokens = torch.cat([tokens, pad], dim=0)
        
#         T = self.sequence_length
        
#         obs = {'spectrogram_tokens': tokens}
#         actions = torch.zeros(T, 1)
#         reset = torch.zeros(T, dtype=torch.bool)
#         reset[0] = True
        
#         return {
#             'obs': obs,
#             'actions': actions,
#             'reset': reset
#         }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    
    # Observações
    obs_keys = batch[0]['obs'].keys()
    obs_batch = {}
    
    for key in obs_keys:
        obs_list = [item['obs'][key] for item in batch]
        obs_stacked = torch.stack(obs_list, dim=1)  # (T, B, ...)
        obs_batch[key] = obs_stacked
    
    # Actions e reset
    actions_list = [item['actions'] for item in batch]
    actions_batch = torch.stack(actions_list, dim=1)  # (T, B, A)
    
    reset_list = [item['reset'] for item in batch]
    reset_batch = torch.stack(reset_list, dim=1)  # (T, B)
    
    return {
        'obs': obs_batch,
        'actions': actions_batch,
        'reset': reset_batch
    }

def create_patches_dataloader(
    data_dir: str,
    batch_size: int = 16,
    sequence_length: int = 50,
    patch_width: int = 16,
    shuffle: bool = True,
    num_workers: int = 2,
    max_files: int = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for loading Spectrogram Patch sequences 
    from raw .pt spectrogram files.
    """
    data_path = Path(data_dir)
    
    # search for spectogram files
    spec_files = list(data_path.glob('*.pt'))
    
    if len(spec_files) == 0:
        raise ValueError(f"No spectrogram (.pt) files found in {data_dir}")
    
    if max_files is not None and max_files > 0:
        spec_files = spec_files[:max_files]
        print(f"Found {len(spec_files)} spectrogram files (limited to {max_files})")
    else:
        print(f"Found {len(spec_files)} spectrogram files")
    
    # Create dataset
    dataset = SpectogramPatchesDataset(
        pt_dir=data_path,
        sequence_length=sequence_length,
        patch_width=patch_width
    )
    
    print(f"Created dataset with {len(dataset)} files (episodes)")
    print(f"Estimated batches per epoch: {len(dataset) // batch_size:,}")
    
    # Creates dataloader with optimizations for GPU training
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn, # Use the existing collate_fn
        pin_memory=pin_memory,
        # Persistent workers and prefetch factor are optimizations for multi-worker loading
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True # Ensures all batches are the same size
    )
    
    return dataloader

# def create_dataloader(
#     data_dir: str,
#     batch_size: int = 16,
#     sequence_length: int = 50,
#     use_embeddings: bool = False,
#     shuffle: bool = True,
#     num_workers: int = 2,
#     max_files: int = None,
#     pin_memory: bool = True,
#     persistent_workers: bool = True,
#     prefetch_factor: int = 4
# ) -> DataLoader:

#     data_path = Path(data_dir)
    
#     # Busca arquivos de tokens (tenta _vq_tokens.pt primeiro, depois _tokens.pt)
#     token_files = list(data_path.glob('*_vq_tokens.pt'))
    
#     if len(token_files) == 0:
#         # Fallback para formato antigo
#         token_files = list(data_path.glob('*_tokens.pt'))
    
#     if len(token_files) == 0:
#         raise ValueError(f"No token files found in {data_dir}")
    
#     # Limita número de arquivos se especificado
#     if max_files is not None and max_files > 0:
#         token_files = token_files[:max_files]
#         print(f"Found {len(token_files)} token files (limited to {max_files})")
#     else:
#         print(f"Found {len(token_files)} token files")
    
#     # Cria dataset
#     dataset = SpectrogramTokenDataset(
#         token_files=[str(f) for f in token_files],
#         sequence_length=sequence_length,
#         use_embeddings=use_embeddings
#     )
    
#     print(f"Created dataset with {len(dataset):,} sequences")
#     print(f"Expected batches per epoch: {len(dataset) // batch_size:,}")
    
#     # Cria dataloader with optimizations for GPU training
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         collate_fn=collate_fn,
#         pin_memory=pin_memory,
#         persistent_workers=persistent_workers if num_workers > 0 else False,
#         prefetch_factor=prefetch_factor if num_workers > 0 else None,
#         drop_last=True
#     )
    
#     return dataloader


# def create_audio_dataloader(
#     audio_dir: str,
#     vq_tokenizer_path: str,
#     batch_size: int = 16,
#     sequence_length: int = 50,
#     use_embeddings: bool = False,
#     shuffle: bool = True,
#     num_workers: int = 2
# ) -> DataLoader:

#     # Carrega VQ tokenizer
#     checkpoint = torch.load(vq_tokenizer_path)
#     config = checkpoint.get('config', {})
    
#     vq_tokenizer = SpectrogramVQTokenizer(
#         n_mels=config.get('n_mels', 80),
#         patch_size=config.get('patch_size', 16),
#         embedding_dim=config.get('embedding_dim', 64),
#         num_embeddings=config.get('num_embeddings', 512)
#     )
#     vq_tokenizer.load_state_dict(checkpoint['model_state_dict'])
#     vq_tokenizer.eval()
    
#     # Patch tokenizer
#     patch_tokenizer = SpectrogramPatchTokenizer(
#         patch_size=config.get('patch_size', 16),
#         normalize=True
#     )
    
#     # Busca arquivos de áudio/espectrograma
#     audio_path = Path(audio_dir)
#     spec_files = list(audio_path.glob('*.pt'))
    
#     if len(spec_files) == 0:
#         raise ValueError(f"No spectrogram files found in {audio_dir}")
    
#     print(f"Found {len(spec_files)} spectrogram files")
    
#     # Dataset
#     dataset = AudioSequenceDataset(
#         audio_files=[str(f) for f in spec_files],
#         vq_tokenizer=vq_tokenizer,
#         patch_tokenizer=patch_tokenizer,
#         sequence_length=sequence_length,
#         use_embeddings=use_embeddings
#     )
    
#     # Dataloader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         collate_fn=collate_fn,
#         pin_memory=True
#     )
    
#     return dataloader


if __name__ == "__main__":
    # Testing
    pt_dir = Path("/Users/rcss/Documents/master/ia376/dgm-2025.2/projects/spectrogram-dreamer/data/raw_spectrograms")
    N_MELS = 80
    T_FULL = 100
    NUM_FILES = 5

    # --- Test ---
    try:
        print("-" * 30)
        # Initialize the dataset
        dataset = SpectogramPatchesDataset(pt_dir=pt_dir)   

        # Get the first item (runs __getitem__(0))
        item = dataset[0] 

        # Print the resulting shapes
        patches = item['obs']['spectrogram_patches']
        actions = item['actions']
        reset = item['reset']

        print("Dataset Item Successfully Loaded!")
        print(f"Patch Width (W): {dataset.patch_width}")
        print(f"Sequence Length (L): {dataset.sequence_length}")
        print(f"Total Window Sliced: {dataset.total_window_size} steps")
        print("-" * 30)
        print("Output Tensor Shapes (L, B, C, H, W or L, B, D):")
        print(f"  Patches (obs): {patches.shape} (L, 1, H, W)")
        print(f"  Actions:       {actions.shape} (L, 1)")
        print(f"  Reset:         {reset.shape}   (L)")
        print("-" * 30)
        print(f"Expected Patches Shape: ({dataset.sequence_length}, 1, {N_MELS}, {dataset.patch_width})")

    except IOError as e:
        print(f"Test Failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")