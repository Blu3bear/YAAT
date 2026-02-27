"""Input validation for YAAT.

Validates that the input audio file exists, is a supported format,
and meets basic requirements before any processing begins.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from yaat.utils.logging import get_logger

# Supported input audio extensions
SUPPORTED_EXTENSIONS = {".wav", ".ogg", ".mp3"}

# Constraints
MIN_DURATION_S = 1.0
MAX_DURATION_S = 1800.0  # 30 minutes


@dataclass
class AudioMeta:
    """Metadata extracted from a validated audio input."""

    path: Path
    duration_s: float
    sample_rate: int
    channels: int
    file_size_bytes: int


class ValidationError(Exception):
    """Raised when input validation fails."""


def validate_input(input_path: str) -> AudioMeta:
    """Validate an input audio file and extract its metadata.

    Checks:
        - File exists and is readable.
        - Extension is supported (.wav, .ogg, .mp3).
        - File is non-empty.
        - Duration is within acceptable range (1s – 30min).

    Args:
        input_path: Path to the input audio file.

    Returns:
        AudioMeta with extracted file properties.

    Raises:
        ValidationError: If any check fails.
    """
    logger = get_logger()
    path = Path(input_path).resolve()

    # Existence
    if not path.exists():
        raise ValidationError(f"Input file does not exist: {path}")

    if not path.is_file():
        raise ValidationError(f"Input path is not a file: {path}")

    # Extension
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Readable / non-empty
    file_size = os.path.getsize(path)
    if file_size == 0:
        raise ValidationError(f"Input file is empty (0 bytes): {path}")

    # Read audio metadata via soundfile
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise ValidationError(
            f"Cannot read audio metadata from {path}: {exc}"
        ) from exc

    duration_s = info.duration
    sample_rate = info.samplerate
    channels = info.channels

    # Duration bounds
    if duration_s < MIN_DURATION_S:
        raise ValidationError(
            f"Audio too short ({duration_s:.2f}s). Minimum: {MIN_DURATION_S}s"
        )
    if duration_s > MAX_DURATION_S:
        raise ValidationError(
            f"Audio too long ({duration_s:.2f}s). Maximum: {MAX_DURATION_S/60:.0f} minutes"
        )

    meta = AudioMeta(
        path=path,
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
        file_size_bytes=file_size,
    )

    logger.info(
        "Input validated: %s | duration=%.2fs | sr=%d | channels=%d | size=%d bytes",
        path.name,
        duration_s,
        sample_rate,
        channels,
        file_size,
    )

    return meta
