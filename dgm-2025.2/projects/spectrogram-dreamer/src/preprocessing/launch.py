"""This file is used to run the complete pipeline with a single command"""

from .pipeline import Pipeline

def launch(use_consolidated: bool = True):
    """
    Launch the preprocessing pipeline.
    
    Args:
        use_consolidated: If True, uses consolidated dataset (RECOMMENDED).
                         If False, uses original pipeline (creates millions of files).
    """
    
    if use_consolidated:
        # RECOMMENDED: Consolidated dataset mode
        p = Pipeline(
            input_dir="data/1_validated-audio/",
            output_dir="data/",  # Not used in consolidated mode
            style_vector_dir="data/",  # Not used in consolidated mode
            file_extension="mp3",
            n_fft=512,
            win_length=20,
            hop_length=10,
            n_mels=64,
            f_min=50,
            f_max=7600,
            segment_duration=0.1,
            overlap=0.5,
            # Consolidated options
            use_consolidated=True,
            consolidated_file="data/dataset_consolidated.h5",
            use_float16=False,  # Recommended: 50% space savings
            compress=False      # Optional: +20-30% savings but slower loading
        )
    else:
        # ORIGINAL: Creates millions of individual files (NOT RECOMMENDED)
        p = Pipeline(
            input_dir="data/1_validated-audio/",
            output_dir="data/2_mel-spectrograms/",
            style_vector_dir="data/3_style-vectors/",
            file_extension="mp3",
            n_fft=512,
            win_length=20,
            hop_length=10,
            n_mels=64,
            f_min=50,
            f_max=7600,
            segment_duration=0.1,
            overlap=0.5,
            use_consolidated=False
        )

    # If you need to run the dataset cleaner first, uncomment and set paths:
    # p.run_dataset_cleaner(metadata_file='data/data-file/validated.tsv',
    #                       clips_dir='data/new-clip/',
    #                       min_votes=2)

    p.process(metadata_file='data/data-file/validated.tsv')


def launch_original():
    """Launch using ORIGINAL pipeline (creates millions of files)"""
    launch(use_consolidated=False)


def launch_consolidated():
    """Launch using CONSOLIDATED dataset (RECOMMENDED - 40-90% space savings)"""
    launch(use_consolidated=True)