"""Following Henrique's work to create DatasetCleaner class"""

# general
import os
from pathlib import Path
import shutil
from tqdm import tqdm

# internal imports
from ..utils.logger import get_logger

# data wrangling
import pandas as pd

_logger = get_logger("dataset_cleaner")

class DatasetCleaner():
    def __init__(self, metadata_file_path: str,
                 clips_dir_path: str,
                 output_dir: str,
                 min_votes: int = 2,    # min votes for audio to be considered valid
        ):
        # initialize other values
        self.min_votes = min_votes
        
        # convert paths
        if os.path.exists(metadata_file_path):
            self.metadata_file_path = Path(metadata_file_path)
        else:
            _logger.error(f"Path '{metadata_file_path}' not found")
            raise FileNotFoundError(f"Path '{metadata_file_path}' not found")

        if os.path.exists(clips_dir_path):
            self.clips_dir_path = Path(clips_dir_path)
        else:
            _logger.error(f"Path '{clips_dir_path}' not found")
            raise FileNotFoundError(f"Path '{clips_dir_path}' not found")

        if not output_dir:
            _logger.info("output_dir not specified. Fallback to default 'data/1_validated-audio/'")
            self.output_dir = Path('data/1_validated-audio/')
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
# --- Main funtion of the cleaner ---
# Applies all necessary changes and saves the validated files to a new dir

    def run(self,):
        """Apply all necessary changes to the dataset"""
        _logger.info("Starting dataset cleaning pipeline...")
        
        # Load metadata
        _logger.info("Loading metadata file...")
        df = self._load_dataframe()
        
        # Filter validated clips
        _logger.info("Filtering validated clips...")
        validated_clips = self._filter_data(df)
        
        # Copy files
        _logger.info("Copying validated audio files...")
        self._copy_validated_files(validated_clips)
        
        _logger.info("Dataset cleaning pipeline completed!")

# --- Helper functions ---
# Necessary functions for cleaning
        
    def _load_dataframe(self) -> pd.DataFrame:
        """Loads the dataframe based on metadatafile
        
        Returns:
            pd.DataFrame: loaded dataframe.
        """
        df = pd.read_csv(self.metadata_file_path, sep='\t', low_memory=False)

        # Normalize columns with mixed types
        for col in ['votes_up', 'votes_down', 'duration']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        _logger.info(f"Total de clips: {len(df)}")
        _logger.info(df.head(5))

        return df
    
    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters the complete DataFrame
        
        Returns:
            pd.DataFrame: validated clips
        """
        
        complete_mask = df.notna().all(axis=1)
        validated_clips = df[
            complete_mask &
            (df['up_votes'] > df['down_votes']) & 
            (df['up_votes'] >= self.min_votes)
        ]

        complete_count = int(complete_mask.sum())
        complete_pct = complete_count / len(df) * 100

        _logger.info(f"Audios with all fields completed: {complete_count} de {len(df)} ({complete_pct:.2f}%)")
        _logger.info(f"Validated clips: {len(validated_clips)}")
        _logger.info(f"Validation ratio: {len(validated_clips)/len(df)*100:.1f}%")

        return validated_clips
    
    def _copy_validated_files(self, validated_clips: pd.DataFrame):
        """Copies validated audio files to output directory and saves metadata
        
        Args:
            validated_clips (pd.DataFrame): DataFrame containing validated clips with 'path' column
        
        Returns:
            None
        """
        
        _logger.info(f"\nCopying files to {self.output_dir}...")
        
        copied = 0
        not_found = 0
        
        for _, row in tqdm(validated_clips.iterrows(), total=len(validated_clips)):
            filename = row['path']
            source = self.clips_dir_path / filename
            destination = self.output_dir / filename

            if source.exists():
                shutil.copy2(source, destination)
                copied += 1
            else:
                not_found += 1
        
        _logger.info(f"\nCopied: {copied} files")
        if not_found > 0:
            _logger.warning(f"Total files not found: {not_found} files")

        # Save filtered metadata
        metadata_output = self.output_dir / 'validated_metadata.tsv'
        validated_clips.to_csv(metadata_output, sep='\t', index=False)
        _logger.info(f"\nMetadata saved to: {metadata_output}")
