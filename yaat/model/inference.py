"""Model inference for YAAT.

Loads the TensorHero Transformer (model13), splits spectrograms into
4-second segments, runs autoregressive decoding of (time, note) token
pairs, and assembles the final notes array.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from yaat.config import ModelConfig, AudioConfig
from yaat.model.transformer import (
    TensorHeroTransformer,
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    VOCAB_SIZE,
    TIME_OFFSET,
    NOTE_RANGE,
    TIME_RANGE,
)
from yaat.utils.logging import get_logger


def _load_model(config: ModelConfig) -> TensorHeroTransformer:
    """Instantiate and load pre-trained TensorHeroTransformer weights.

    Args:
        config: Model configuration with architecture params and weights path.

    Returns:
        TensorHeroTransformer with loaded weights, in eval mode on target device.

    Raises:
        FileNotFoundError: If weights_path is set but does not exist.
        RuntimeError: If weights cannot be loaded.
    """
    logger = get_logger()

    model = TensorHeroTransformer(
        embedding_size=config.embedding_size,
        trg_vocab_size=config.vocab_size,
        num_heads=config.num_heads,
        num_encoder_layers=config.encoder_layers,
        num_decoder_layers=config.decoder_layers,
        forward_expansion=config.forward_expansion,
        dropout=config.dropout,
        max_len=500,
        device=config.device,
    )

    if config.weights_path:
        weights_path = Path(config.weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}"
            )

        logger.info("Loading model weights from %s", weights_path)
        state_dict = torch.load(
            str(weights_path),
            map_location=config.device,
        )

        # Handle checkpoints that wrap state_dict in a container
        if isinstance(state_dict, dict) and "state" in state_dict:
            state_dict = state_dict["state"]
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
    else:
        logger.warning(
            "No weights_path configured - model will use random initialization"
        )

    model = model.to(config.device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "TensorHeroTransformer ready: %d parameters on device '%s'",
        param_count,
        config.device,
    )

    return model


def _tokens_to_notes_array(tokens: list[int]) -> np.ndarray:
    """Convert model output tokens to a 400-element notes array.

    The model produces interleaved (time_token, note_token) pairs.
    Valid pairs have time_token in [32, 431] and note_token in [0, 31].

    Args:
        tokens: List of predicted token indices (after removing <sos>/<eos>).

    Returns:
        1D notes array of length 400 (one 4-second segment).
    """
    note_vals = set(NOTE_RANGE)
    time_vals = set(TIME_RANGE)

    pairs = []
    for i in range(len(tokens) - 1):
        t_tok = tokens[i]
        n_tok = tokens[i + 1]
        if t_tok in time_vals and n_tok in note_vals:
            pairs.append((t_tok - TIME_OFFSET, n_tok))

    notes_array = np.zeros(400, dtype=np.int32)
    for time_bin, note_val in pairs:
        if 0 <= time_bin < 400:
            notes_array[time_bin] = note_val

    return notes_array


def run_inference(
    spectrogram: np.ndarray,
    onset_bins: list[int],
    model_config: ModelConfig,
    audio_config: AudioConfig,
) -> np.ndarray:
    """Run full inference: load model -> split into 4s segments -> decode -> notes array.

    The model13 checkpoint uses full spectrogram segments (not onset-windowed),
    so onset_bins is accepted for API compatibility but not used.

    Args:
        spectrogram: Normalized log-mel spectrogram, shape (n_mels, T).
        onset_bins: Onset positions (unused by model13, kept for API compat).
        model_config: Model configuration.
        audio_config: Audio configuration.

    Returns:
        1D notes array of length T, with note indices at predicted positions.
    """
    logger = get_logger()
    device = model_config.device

    # Load model
    model = _load_model(model_config)

    n_mels, total_frames = spectrogram.shape
    frames_per_segment = 400  # 4 seconds at 10ms per frame

    # Pad spectrogram so length is divisible by 400
    remainder = total_frames % frames_per_segment
    if remainder != 0:
        pad_width = frames_per_segment - remainder
        spectrogram = np.pad(
            spectrogram,
            ((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=0.0,
        )
    padded_frames = spectrogram.shape[1]
    num_segments = padded_frames // frames_per_segment

    # Full notes array for the entire song
    notes_array = np.zeros(padded_frames, dtype=np.int32)

    t0 = time.perf_counter()

    for seg_idx in range(num_segments):
        seg_start = seg_idx * frames_per_segment
        seg_end = seg_start + frames_per_segment
        seg_spec = spectrogram[:, seg_start:seg_end]  # (512, 400)

        # Pad to max_len (500) as expected by the model
        max_len = 500
        padded_spec = np.pad(
            seg_spec,
            ((0, 0), (0, max_len - seg_spec.shape[1])),
            mode='constant',
            constant_values=0.0,
        )

        # To tensor: shape (1, 512, max_len)
        src_tensor = torch.tensor(
            padded_spec, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Autoregressive decode
        tokens = model.predict(src_tensor, max_len=max_len)

        # Convert tokens to 400-element notes array
        seg_notes = _tokens_to_notes_array(tokens)
        notes_array[seg_start:seg_end] = seg_notes

        seg_note_count = int(np.count_nonzero(seg_notes))
        logger.debug(
            "Segment %d/%d: %d tokens -> %d notes",
            seg_idx + 1,
            num_segments,
            len(tokens),
            seg_note_count,
        )

    inference_time = time.perf_counter() - t0
    total_notes = int(np.count_nonzero(notes_array))
    logger.info(
        "Inference complete: %d segments, %d total notes in %.2fs",
        num_segments,
        total_notes,
        inference_time,
    )

    # Trim back to original length
    notes_array = notes_array[:total_frames]

    return notes_array
