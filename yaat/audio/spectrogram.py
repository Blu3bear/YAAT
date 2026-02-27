"""Spectrogram computation for YAAT.

Computes a normalized log-mel spectrogram compatible with the
TensorHero OnsetTransformer input format.
"""

import numpy as np
import librosa

from yaat.config import AudioConfig
from yaat.utils.logging import get_logger, log_array_stats


def compute_spectrogram(
    audio: np.ndarray,
    source_sr: int,
    config: AudioConfig,
) -> np.ndarray:
    """Compute a normalized log-mel spectrogram from audio.

    Pipeline:
        1. Resample to target sample rate (44100 Hz).
        2. Compute power mel spectrogram (512 mel bins, FFT=4096, hop=441).
        3. Convert to dB scale (range [-80, 0]).
        4. Normalize to [0, 1] via (spec + 80) / 80.

    Args:
        audio: 1D float32 audio array (mono).
        source_sr: Sample rate of the input audio.
        config: Audio processing configuration.

    Returns:
        Normalized log-mel spectrogram, shape (n_mels, T) in [0, 1].
    """
    logger = get_logger()

    # Resample if needed
    if source_sr != config.sample_rate:
        logger.info(
            "Resampling from %d Hz to %d Hz",
            source_sr,
            config.sample_rate,
        )
        audio = librosa.resample(
            audio,
            orig_sr=source_sr,
            target_sr=config.sample_rate,
        )

    sr = config.sample_rate

    # Compute power mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=2,
        fmax=sr / 2,
    )

    # Convert to dB (log scale), range approx [-80, 0]
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 1] matching TensorHero convention
    normalized = (log_mel + 80.0) / 80.0
    normalized = np.clip(normalized, 0.0, 1.0)

    total_frames = normalized.shape[1]
    duration_implied = total_frames * config.hop_length / sr

    logger.info(
        "Spectrogram: shape=%s, frames=%d, implied_duration=%.2fs",
        normalized.shape,
        total_frames,
        duration_implied,
    )
    log_array_stats("spectrogram", normalized, logger)

    return normalized
