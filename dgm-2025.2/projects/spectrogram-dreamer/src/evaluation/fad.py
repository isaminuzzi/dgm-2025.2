# --- Fréchet Audio Distance (FAD) ---
# 
# A metric which is designed to measure how a given audio clip compares
# to another set of audio (studio recorded, amateur, etc)
# 
# The main goal by using this metric is to check how close the generated 
# audio from our spectrogram-dreamer implementation compares to the common-voice 
# dataset. Having a low FAD score here means that our model is coherent and was able to capture
# the realism of the reference dataset. Since common-voice does not have superior quality for most of the
# audios, checking if our model was able to represent this is also interesting
# 
# Original paper: https://arxiv.org/pdf/1812.08466
# For this implementation, we'll be using this library: https://pypi.org/project/frechet-audio-distance/

# general
import os
from typing import Literal, Optional
from pathlib import Path

# frechet audio distance imports
from frechet_audio_distance import FrechetAudioDistance

# internal imports
from base import BaseEvaluator
from ..utils.logger import get_logger

# set up logging
_logger = get_logger("fad_evaluator", level="DEBUG")

class FAD(BaseEvaluator):
    def __init__(self,
                 path_to_background: str, # this is the "grounding" set
                 path_to_eval: str,  # this is the generated audio set 
                 embedding_model: Literal["vggish", "pann", "clap", "encodec"] = "vggish",
                 sample_rate: int = 16000,
                 use_pca: bool = False,
                 use_activation: bool = False,
                 verbose: bool = True,
                 submodel_name: Optional[str] = None,  # for CLAP only
                 enable_fusion: Optional[bool] = None, # for CLAP only
                 channels: Optional[int] = None,       # for EnCodec only
        ):
        """Initializes the Fréchet Audio Distance (FAD) evaluator.

        This class wraps the `frechet_audio_distance` library to compute the distance 
        between the distribution of generated audio and a reference dataset (background).

        Args:
            path_to_background (str): Path to the directory containing reference/ground-truth 
                audio files (e.g., Common Voice validation set).
            path_to_eval (str): Path to the directory containing the generated audio files 
                to be evaluated.
            embedding_model (Literal["vggish", "pann", "clap", "encodec"], optional): 
                The backbone model used to extract audio embeddings. Defaults to "vggish".
            sample_rate (int, optional): The sample rate to which audio will be resampled. 
                Defaults to 16000.
            verbose (bool, optional): Whether to print progress logs from the underlying library. 
                Defaults to True.

            -- VGGish / PANN specific --
            use_pca (bool, optional): Whether to apply PCA to the embeddings. 
                Defaults to False.
            use_activation (bool, optional): Whether to use the activation layer output 
                instead of the penultimate layer. Defaults to False.

            -- CLAP specific --
            submodel_name (Optional[str]): The specific CLAP checkpoint to use 
                (e.g., "630k-audioset-fusion-best"). **Required** if `embedding_model="clap"`.
            enable_fusion (Optional[bool]): Whether to enable the fusion mechanism in CLAP. 
                **Required** if `embedding_model="clap"`.

            -- EnCodec specific --
            channels (Optional[int]): The number of audio channels (e.g., 1 or 2). 
                **Required** if `embedding_model="encodec"`.

        Raises:
            FileNotFoundError: If `path_to_background` or `path_to_eval` does not exist.
            ValueError: If `embedding_model` is not one of the allowed types.
            ValueError: If required model-specific arguments (like `submodel_name` for CLAP 
                or `channels` for EnCodec) are missing.
        """
        # loading sets
        if os.path.exists(path_to_background):
            self.path_to_background = path_to_background
        else:
            _logger.error(f"Path '{path_to_background}' not found")
            raise FileNotFoundError(f"Path '{path_to_background}' not found")

        if os.path.exists(path_to_eval):
            self.path_to_eval = path_to_eval
        else:
            _logger.error(f"Path '{path_to_eval}' not found")
            raise FileNotFoundError(f"Path '{path_to_eval}' not found")

        allowed_models_set = {"vggish", "pann", "clap", "encodec"}
        if embedding_model.lower() not in allowed_models_set:
            _logger.error(f"Selected embedding model '{embedding_model}' is not allowed. Allowed models are: '{"', '".join(m for m in allowed_models_set)}'")
            raise ValueError(f"Selected embedding model '{embedding_model}' is not allowed. Allowed models are: '{"', '".join(m for m in allowed_models_set)}'")
            
        # using CLAP
        if embedding_model.lower() == "clap":
            if not submodel_name:
                _logger.error((f"submodel_name not specified"))
                raise ValueError(f"submodel_name not specified")
            if not enable_fusion:
                _logger.error((f"submodel_name not specified"))
                raise ValueError(f"enable_fusion value not specified")
            
            self.frechet = FrechetAudioDistance(
                            model_name="clap",
                            sample_rate=sample_rate,
                            submodel_name=submodel_name,
                            verbose=verbose,
                            enable_fusion=enable_fusion,
                    )
            
        # using EnCodec
        elif embedding_model.lower() == "encodec":
            if not channels:
                _logger.error(f"channels not specified")
                raise ValueError(f"channels not specified")
            
            self.frechet = FrechetAudioDistance(
                            model_name="encodec",
                            sample_rate=sample_rate,
                            channels=channels,
                            verbose=verbose,
                        )
        
        # using VGGish or PANN
        else:
            self.frechet = FrechetAudioDistance(
                            model_name=embedding_model,
                            sample_rate=sample_rate,
                            use_pca=use_pca, 
                            use_activation=use_activation,
                            verbose=verbose
                        )
        
    def evaluate(self) -> float:
        """Calculates FAD score

        Returns:
            float: FAD score (lower the better)
        """
        _logger.info(f"Calculating FAD score on {self.path_to_eval}...")
        fad_score = self.frechet.score(
            self.path_to_background,
            self.path_to_eval,
            dtype="float32",
        )

        _logger.info(f"FAD score: {fad_score}")

        return fad_score

