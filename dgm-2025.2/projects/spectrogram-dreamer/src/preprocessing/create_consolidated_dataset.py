#!/usr/bin/env python3
import argparse
import gzip
from pathlib import Path
from tqdm import tqdm
from typing import Optional

import h5py
import numpy as np
import torch
import torchaudio.transforms as T

from src.preprocessing.generate_spectrogram import AudioFile
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_consolidated_dataset(
    input_dir: str,
    output_file: str,
    metadata_file: str,
    segment_duration: float = 0.1,
    overlap: float = 0.5,
    max_segments_per_file: Optional[int] = None,
    n_fft: int = 512,
    win_length: int = 20,
    hop_length: int = 10,
    n_mels: int = 64,
    f_min: int = 50,
    f_max: int = 7600,
    compress: bool = False,
    use_float16: bool = False
):
    """
    Args:
        input_dir: Diretório com áudios validados
        output_file: Caminho para salvar dataset consolidado (.pt)
        metadata_file: TSV com metadados (para vetores de estilo)
        segment_duration: Duração de cada segmento em segundos (padrão: 0.1)
        overlap: Sobreposição entre segmentos 0.0-1.0 (padrão: 0.5)
        max_segments_per_file: Limite de segmentos por áudio (None = ilimitado)
        n_fft: Tamanho da FFT (padrão: 512)
        win_length: Janela em ms (padrão: 20)
        hop_length: Passo em ms (padrão: 10)
        n_mels: Número de bandas mel (padrão: 64)
        f_min: Frequência mínima em Hz (padrão: 50)
        f_max: Frequência máxima em Hz (padrão: 7600)
        compress: Se True, salva com compressão gzip (padrão: False)
        use_float16: Se True, usa float16 - pequena perda, 50% menor (padrão: False)
    
    Returns:
        dict: Configuração do dataset criado
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("CRIANDO DATASET CONSOLIDADO")
    logger.info("=" * 80)
    
    # Buscar arquivos de áudio
    audio_files = list(input_dir.glob('*.mp3')) + list(input_dir.glob('*.wav'))
    logger.info(f"Found {len(audio_files)} audio files in {input_dir}")
    
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {input_dir}")
    
    # Carregar estilos globais do metadata
    logger.info(f"Loading global styles from {metadata_file}...")
    try:
        global_styles = AudioFile.load_global_styles(metadata_file)
        logger.info(f"Loaded {len(global_styles)} style vectors")
    except Exception as e:
        logger.error(f"Failed to load global styles: {e}")
        raise
    
    all_data = []
    skipped_files = []
    total_segments = 0

    # HDF5 incremental writer (created lazily when we have the first segment)
    use_hdf5 = output_file.suffix == '.h5'
    h5f = None
    specs_ds = None
    styles_ds = None
    file_ids_ds = None
    segidx_ds = None

    # track first shapes to build config later (works for both modes)
    first_spec_shape = None
    first_style_dim = None

    # Online statistics (per-mel band)
    total_frames = 0
    sum_per_mel = None
    sumsq_per_mel = None
    
    # Processar cada arquivo de áudio
    logger.info(f"\nProcessing audio files...")
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Criar objeto AudioFile com parâmetros especificados
            # IMPORTANTE: AudioFile.segment_spectrogram() retorna Log-Mel spectrograms
            audio = AudioFile(
                str(audio_path),
                segment_duration=segment_duration,
                overlap=overlap,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max
            )
            
            # Gerar segmentos de espectrograma (Log-Mel, não Power)
            segments = audio.segment_spectrogram()
            
            if len(segments) == 0:
                logger.warning(f"No segments generated for {audio_path.name}")
                skipped_files.append(audio_path.name)
                continue
            
            # Obter vetor de estilo global para este arquivo
            file_id = audio_path.stem
            if file_id not in global_styles:
                logger.warning(f"No global style found for {file_id}, skipping")
                skipped_files.append(audio_path.name)
                continue
            
            global_style = global_styles[file_id]
            
            # Criar vetor de estilo (local + global) para o primeiro segmento
            # Usamos o mesmo vetor para todos os segmentos do mesmo arquivo
            device = audio.waveform.device
            delta_transform = T.ComputeDeltas().to(device)
            
            # Usar primeiro segmento para computar estilo local
            first_seg = segments[0].to(device)
            local_style = AudioFile._compute_local_style(first_seg, delta_transform)
            style_vector = torch.cat([local_style, global_style.to(device)], dim=0)
            
            # Limitar número de segmentos se especificado
            if max_segments_per_file and len(segments) > max_segments_per_file:
                original_len = len(segments)
                segments = segments[:max_segments_per_file]
                logger.debug(f"Limited {file_id}: {original_len} → {len(segments)} segments")
            
            # Converter para float16 se especificado (reduz 50% do tamanho)
            if use_float16:
                segments = [s.half() for s in segments]
                style_vector = style_vector.half()
            
            # Adicionar cada segmento com seu vetor de estilo
            for seg_idx, segment in enumerate(segments):
                # If HDF5 mode, write incrementally to disk to avoid memory blowup
                if use_hdf5:
                    if h5f is None:
                        h5f = h5py.File(str(output_file), 'w')

                    seg_np = segment.cpu().numpy()

                    # create datasets lazily when first segment shape is known
                    if specs_ds is None:
                        n_mels, t = seg_np.shape
                        specs_ds = h5f.create_dataset(
                            'spectrograms',
                            shape=(0, n_mels, t),
                            maxshape=(None, n_mels, t),
                            dtype='f4',
                            chunks=(1, n_mels, t),
                            compression='gzip'
                        )
                        if first_spec_shape is None:
                            first_spec_shape = (n_mels, t)

                    if styles_ds is None:
                        style_dim = style_vector.shape[0]
                        styles_ds = h5f.create_dataset(
                            'styles', shape=(0, style_dim), maxshape=(None, style_dim), dtype='f4', compression='gzip'
                        )
                        if first_style_dim is None:
                            first_style_dim = style_dim

                    if file_ids_ds is None:
                        dt = h5py.string_dtype(encoding='utf-8')
                        file_ids_ds = h5f.create_dataset('file_ids', shape=(0,), maxshape=(None,), dtype=dt, compression='gzip')

                    if segidx_ds is None:
                        segidx_ds = h5f.create_dataset('seg_idx', shape=(0,), maxshape=(None,), dtype='i8', compression='gzip')

                    # append spectrogram
                    cur_len = specs_ds.shape[0]
                    specs_ds.resize(cur_len + 1, axis=0)
                    specs_ds[cur_len] = seg_np.astype('float32')

                    # append style
                    styles_ds.resize((styles_ds.shape[0] + 1), axis=0)
                    styles_ds[-1, :] = style_vector.cpu().numpy().astype('float32')

                    # metadata
                    file_ids_ds.resize((file_ids_ds.shape[0] + 1), axis=0)
                    file_ids_ds[-1] = file_id

                    segidx_ds.resize((segidx_ds.shape[0] + 1), axis=0)
                    segidx_ds[-1] = seg_idx

                    # update online stats per-mel
                    frames = seg_np.shape[1]
                    if sum_per_mel is None:
                        sum_per_mel = np.zeros(seg_np.shape[0], dtype='float64')
                        sumsq_per_mel = np.zeros(seg_np.shape[0], dtype='float64')
                    sum_per_mel += seg_np.sum(axis=1)
                    sumsq_per_mel += (seg_np ** 2).sum(axis=1)
                    total_frames += frames
                else:
                    all_data.append({
                        'spectrogram': segment,           # [n_mels, time_frames]
                        'style_vector': style_vector,     # [style_dim]
                        'file_id': file_id,               # ID do arquivo original
                        'segment_idx': seg_idx            # Índice do segmento
                    })
            
            total_segments += len(segments)
            logger.debug(f"{audio_path.name}: {len(segments)} segments")
            
        except Exception as e:
            logger.warning(f"Failed to process {audio_path.name}: {e}")
            skipped_files.append(audio_path.name)
            continue
    
    if not use_hdf5 and len(all_data) == 0:
        raise ValueError("No data was processed successfully!")

    # Computar estatísticas de normalização
    logger.info(f"\nComputing normalization statistics...")
    if use_hdf5:
        # derive mean/std from online sums
        if total_frames == 0:
            raise ValueError("No data was written to HDF5 file!")

        mean = (sum_per_mel / total_frames).astype('float32')
        var = (sumsq_per_mel / total_frames) - (mean.astype('float64') ** 2)
        std = np.sqrt(np.maximum(var, 1e-12)).astype('float32')

        stats = {
            'mean': torch.from_numpy(mean),
            'std': torch.from_numpy(std),
        }
    else:
        logger.info(f"\nComputing normalization statistics from {len(all_data)} segments...")
        all_specs = torch.stack([d['spectrogram'] for d in all_data])
        
        stats = {
            'mean': all_specs.mean(dim=(0, 2)),  # [n_mels]
            'std': all_specs.std(dim=(0, 2)),    # [n_mels]
        }
    
    logger.info(f"   Mean shape: {stats['mean'].shape}")
    logger.info(f"   Std shape: {stats['std'].shape}")
    
    # Metadados do dataset
    if use_hdf5:
        num_segments = specs_ds.shape[0] if specs_ds is not None else 0
        spectrogram_shape = first_spec_shape
        style_vector_dim = first_style_dim
    else:
        num_segments = len(all_data)
        spectrogram_shape = all_data[0]['spectrogram'].shape
        style_vector_dim = all_data[0]['style_vector'].shape[0]

    config = {
        'num_files': len(audio_files) - len(skipped_files),
        'num_segments': num_segments,
        'segment_duration': segment_duration,
        'overlap': overlap,
        'n_mels': n_mels,
        'n_fft': n_fft,
        'win_length': win_length,
        'hop_length': hop_length,
        'f_min': f_min,
        'f_max': f_max,
        'spectrogram_shape': spectrogram_shape,
        'style_vector_dim': style_vector_dim,
        'dtype': 'float16' if use_float16 else 'float32',
        'compressed': compress,
        'skipped_files': skipped_files
    }
    
    # Preparar dados para salvar
    logger.info(f"\nSaving dataset to {output_file}...")

    if use_hdf5:
        # store computed stats and config as attributes
        if h5f is not None:
            # attach stats and config
            h5f.attrs['num_segments'] = specs_ds.shape[0]
            h5f.attrs['n_mels'] = config['n_mels']
            h5f.attrs['segment_duration'] = config['segment_duration']
            # stats are torch tensors -> convert to numpy
            h5f.attrs['mean'] = stats['mean'].cpu().numpy().astype('float32')
            h5f.attrs['std'] = stats['std'].cpu().numpy().astype('float32')
            # close h5
            h5f.close()

        final_output = output_file
    else:
        data_to_save = {
            'data': all_data,
            'stats': stats,
            'config': config
        }
        # Salvar (com ou sem compressão)
        if compress:
            import gzip
            output_file_gz = Path(str(output_file) + '.gz')
            with gzip.open(str(output_file_gz), 'wb') as f:
                torch.save(data_to_save, f)
            final_output = output_file_gz
        else:
            torch.save(data_to_save, output_file)
            final_output = output_file
    
    # Relatório final
    file_size_mb = final_output.stat().st_size / (1024**2)
    file_size_gb = file_size_mb / 1024
    
    logger.info("\n" + "=" * 80)
    logger.info("DATASET CONSOLIDADO CRIADO COM SUCESSO!")
    logger.info("=" * 80)
    logger.info(f"Arquivo: {final_output}")
    logger.info(f"Tamanho: {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")
    logger.info(f"Arquivos processados: {config['num_files']}")
    logger.info(f"Total de segmentos: {config['num_segments']}")
    logger.info(f"Shape do espectrograma: {config['spectrogram_shape']}")
    logger.info(f"Dimensão do vetor de estilo: {config['style_vector_dim']}")
    logger.info(f"Precisão: {config['dtype']}")
    logger.info(f"Compressão: {'Sim (.gz)' if compress else 'Não'}")
    
    if skipped_files:
        logger.warning(f"\n{len(skipped_files)} arquivos ignorados:")
        for f in skipped_files[:5]:
            logger.warning(f"   - {f}")
        if len(skipped_files) > 5:
            logger.warning(f"   ... e mais {len(skipped_files) - 5}")
    
    # Estimativa de economia de espaço
    estimated_original_kb = total_segments * 16  # KB por arquivo (overhead)
    estimated_original_mb = estimated_original_kb / 1024
    savings_pct = ((estimated_original_mb - file_size_mb) / estimated_original_mb) * 100 if estimated_original_mb > 0 else 0
    
    logger.info(f"\nEconomia estimada de armazenamento:")
    logger.info(f"   Pipeline padrão: ~{estimated_original_mb:.0f} MB ({total_segments} arquivos)")
    logger.info(f"   Dataset consolidado: {file_size_mb:.0f} MB (1 arquivo)")
    logger.info(f"   Economia: ~{savings_pct:.0f}% menos espaço")
    logger.info(f"   Arquivos economizados: {total_segments} → 1")
    
    logger.info("\nComo usar:")
    logger.info("   from src.dataset import create_train_val_dataloaders")
    logger.info(f"   train_loader, val_loader = create_train_val_dataloaders('{final_output}', batch_size=32)")
    logger.info("=" * 80 + "\n")
    
    return config


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cria dataset consolidado a partir de áudios validados',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argumentos obrigatórios
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/1_validated-audio/',
        help='Diretório com áudios validados'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/consolidated_dataset.pt',
        help='Caminho para salvar dataset consolidado'
    )
    parser.add_argument(
        '--metadata-file',
        type=str,
        default='data/data-file/validated.tsv',
        help='Arquivo TSV com metadados'
    )
    
    # Parâmetros de segmentação
    parser.add_argument(
        '--segment-duration',
        type=float,
        default=0.1,
        help='Duração de cada segmento em segundos'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.5,
        help='Sobreposição entre segmentos (0.0-1.0)'
    )
    parser.add_argument(
        '--max-segments-per-file',
        type=int,
        default=None,
        help='Limite de segmentos por arquivo (None = ilimitado)'
    )
    
    # Parâmetros do espectrograma
    parser.add_argument('--n-fft', type=int, default=512, help='Tamanho da FFT')
    parser.add_argument('--win-length', type=int, default=20, help='Janela em ms')
    parser.add_argument('--hop-length', type=int, default=10, help='Passo em ms')
    parser.add_argument('--n-mels', type=int, default=64, help='Número de bandas mel')
    parser.add_argument('--f-min', type=int, default=50, help='Frequência mínima (Hz)')
    parser.add_argument('--f-max', type=int, default=7600, help='Frequência máxima (Hz)')
    
    # Otimizações
    parser.add_argument(
        '--compress',
        action='store_true',
        help='Salvar com compressão gzip (30-40%% menor)'
    )
    parser.add_argument(
        '--float16',
        action='store_true',
        help='Usar float16 em vez de float32 (50%% menor, pequena perda de precisão)'
    )
    
    args = parser.parse_args()
    
    # Criar dataset consolidado
    create_consolidated_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        metadata_file=args.metadata_file,
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        max_segments_per_file=args.max_segments_per_file,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
        compress=args.compress,
        use_float16=args.float16
    )
