"""Configuration loader for YAAT.

Loads and validates pipeline configuration from a YAML file using Pydantic models.
"""

from pathlib import Path
from typing import Optional

import torch
import yaml
from pydantic import BaseModel, field_validator


class AudioConfig(BaseModel):
    """Audio processing parameters."""

    sample_rate: int = 44100
    n_fft: int = 4096
    hop_length: int = 441
    n_mels: int = 512


class SeparationConfig(BaseModel):
    """Demucs source separation parameters."""

    model: str = "htdemucs_6s"
    stem: str = "guitar"
    device: str = "auto"

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v


class OnsetConfig(BaseModel):
    """Onset detection hyperparameters (NINOS ODF + peak picking)."""

    gamma: float = 0.94
    p: float = 0.99
    w1: int = 10
    w2: int = 1
    w3: int = 1
    w4: int = 8
    w5: int = 10
    delta: float = 1.0


class ModelConfig(BaseModel):
    """OnsetTransformer model parameters."""

    weights_path: str = ""
    device: str = "auto"
    segment_duration_s: float = 4.0
    max_onsets_per_segment: int = 50
    embedding_size: int = 512
    vocab_size: int = 25
    num_heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 2
    forward_expansion: int = 2048
    dropout: float = 0.1

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v


class OutputConfig(BaseModel):
    """Output chart metadata."""

    chart_name: str = ""
    artist: str = "Unknown Artist"
    charter: str = "YAAT"


class PostprocessConfig(BaseModel):
    """Postprocessing constraint parameters."""

    max_notes_per_second: int = 15
    min_note_gap_ms: int = 20
    min_total_notes: int = 10


class YAATConfig(BaseModel):
    """Root configuration for the YAAT pipeline."""

    audio: AudioConfig = AudioConfig()
    separation: SeparationConfig = SeparationConfig()
    onset: OnsetConfig = OnsetConfig()
    model: ModelConfig = ModelConfig()
    output: OutputConfig = OutputConfig()
    postprocess: PostprocessConfig = PostprocessConfig()
    debug: bool = False


def load_config(config_path: Optional[str] = None) -> YAATConfig:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file. If None, uses defaults.

    Returns:
        Validated YAATConfig instance.
    """
    if config_path is None:
        return YAATConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return YAATConfig()

    return YAATConfig(**raw)
