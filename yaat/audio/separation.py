"""Source separation using Demucs htdemucs_6s model.

Separates audio into 6 stems (drums, bass, other, vocals, guitar, piano)
and extracts the guitar stem for chart generation.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from demucs.api import Separator

from yaat.config import SeparationConfig
from yaat.utils.logging import get_logger


def separate_guitar(
    audio_path: str,
    config: SeparationConfig,
) -> tuple[np.ndarray, int]:
    """Run Demucs source separation and return the guitar stem.

    Args:
        audio_path: Path to the input audio file.
        config: Separation configuration (model name, stem, device).

    Returns:
        Tuple of (guitar_audio_mono, sample_rate) where guitar_audio_mono
        is a 1D numpy float32 array.
    """
    logger = get_logger()
    logger.info(
        "Loading Demucs model '%s' on device '%s'",
        config.model,
        config.device,
    )

    t0 = time.perf_counter()

    # Initialize separator — downloads model weights on first run
    separator = Separator(model=config.model, device=config.device)

    logger.info("Model loaded in %.2fs", time.perf_counter() - t0)

    # Separate the audio file
    t1 = time.perf_counter()
    origin, separated = separator.separate_audio_file(audio_path)
    sep_time = time.perf_counter() - t1

    # Log stem information
    logger.info("Separation completed in %.2fs", sep_time)
    for stem_name, stem_audio in separated.items():
        peak = float(torch.max(torch.abs(stem_audio)))
        logger.info(
            "  Stem %-8s shape=%s peak_amplitude=%.4f",
            stem_name,
            tuple(stem_audio.shape),
            peak,
        )

    # Extract the target stem
    if config.stem not in separated:
        available = list(separated.keys())
        raise ValueError(
            f"Stem '{config.stem}' not found. Available stems: {available}"
        )

    stem_audio = separated[config.stem]  # shape: (channels, samples)

    # Convert stereo to mono
    if stem_audio.shape[0] > 1:
        stem_audio = torch.mean(stem_audio, dim=0)
    else:
        stem_audio = stem_audio.squeeze(0)

    # To numpy float32
    audio_np = stem_audio.cpu().numpy().astype(np.float32)

    logger.info(
        "Extracted '%s' stem: length=%d samples (%.2fs at %dHz)",
        config.stem,
        len(audio_np),
        len(audio_np) / separator.samplerate,
        separator.samplerate,
    )

    return audio_np, separator.samplerate
