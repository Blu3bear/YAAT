"""Model inference for YAAT.

Loads the OnsetTransformer, splits spectrograms into 4-second segments,
extracts onset-windowed frames, runs autoregressive decoding, and
assembles the final notes array via contour decoding.
"""

import time
from pathlib import Path

import numpy as np
import torch

from yaat.config import ModelConfig, AudioConfig
from yaat.model.transformer import OnsetTransformer
from yaat.model.contour import decode_contour
from yaat.utils.logging import get_logger


def _load_model(config: ModelConfig) -> OnsetTransformer:
    """Instantiate and load pre-trained OnsetTransformer weights.

    Args:
        config: Model configuration with architecture params and weights path.

    Returns:
        OnsetTransformer with loaded weights, in eval mode on the target device.

    Raises:
        FileNotFoundError: If weights_path is set but does not exist.
        RuntimeError: If weights cannot be loaded.
    """
    logger = get_logger()

    model = OnsetTransformer(
        embedding_size=config.embedding_size,
        trg_vocab_size=config.vocab_size,
        num_heads=config.num_heads,
        num_encoder_layers=config.encoder_layers,
        num_decoder_layers=config.decoder_layers,
        forward_expansion=config.forward_expansion,
        dropout=config.dropout,
        max_len=config.max_onsets_per_segment,
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
            weights_only=True,
        )

        # Handle checkpoints that wrap state_dict in a container
        if isinstance(state_dict, dict) and "state" in state_dict:
            state_dict = state_dict["state"]
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model.load_state_dict(state_dict, strict=False)
        logger.info("Model weights loaded successfully")
    else:
        logger.warning(
            "No weights_path configured — model will use random initialization"
        )

    model = model.to(config.device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "OnsetTransformer ready: %d parameters on device '%s'",
        param_count,
        config.device,
    )

    return model


def _extract_onset_windows(
    spectrogram: np.ndarray,
    onset_bins: list[int],
    n_frames: int = 7,
) -> np.ndarray:
    """Extract windowed spectrogram slices around each onset.

    For each onset, extracts ±3 frames (7 total) from the spectrogram,
    padding at boundaries with zeros.

    Args:
        spectrogram: Normalized spectrogram, shape (n_mels, T).
        onset_bins: Onset positions as 10ms time bin indices.
        n_frames: Number of frames per window (default 7 = ±3).

    Returns:
        Array of shape (num_onsets, n_mels * n_frames), with each onset's
        windowed spectrogram flattened.
    """
    n_mels, total_frames = spectrogram.shape
    half = n_frames // 2
    windows = []

    for onset_bin in onset_bins:
        # Clamp window to spectrogram boundaries
        start = onset_bin - half
        end = onset_bin + half + 1

        window = np.zeros((n_mels, n_frames), dtype=np.float32)
        src_start = max(0, start)
        src_end = min(total_frames, end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)

        if src_start < src_end:
            window[:, dst_start:dst_end] = spectrogram[:, src_start:src_end]

        windows.append(window.flatten())

    if not windows:
        return np.zeros((0, n_mels * n_frames), dtype=np.float32)

    return np.stack(windows, axis=0)


def run_inference(
    spectrogram: np.ndarray,
    onset_bins: list[int],
    model_config: ModelConfig,
    audio_config: AudioConfig,
) -> np.ndarray:
    """Run the full inference pipeline: load model → segment → decode → notes array.

    Args:
        spectrogram: Normalized log-mel spectrogram, shape (n_mels, T).
        onset_bins: Onset positions as 10ms time bin indices.
        model_config: Model configuration.
        audio_config: Audio configuration (for n_mels).

    Returns:
        1D notes array of length T, with note indices at onset positions.
    """
    logger = get_logger()
    device = model_config.device

    # Load model
    model = _load_model(model_config)

    n_mels, total_frames = spectrogram.shape
    frames_per_segment = int(model_config.segment_duration_s * 100)  # 400 for 4s

    # Split into segments
    num_segments = max(1, (total_frames + frames_per_segment - 1) // frames_per_segment)

    all_tokens: list[int] = []
    all_segment_onsets: list[list[int]] = []

    t0 = time.perf_counter()

    for seg_idx in range(num_segments):
        seg_start = seg_idx * frames_per_segment
        seg_end = min(seg_start + frames_per_segment, total_frames)

        # Gather onsets within this segment
        seg_onsets = [
            o for o in onset_bins if seg_start <= o < seg_end
        ]
        # Convert to segment-relative indices
        seg_onsets_rel = [o - seg_start for o in seg_onsets]

        if not seg_onsets_rel:
            all_segment_onsets.append([])
            continue

        all_segment_onsets.append(seg_onsets)

        # Extract spectrogram segment (zero-pad if needed)
        seg_spec = np.zeros((n_mels, frames_per_segment), dtype=np.float32)
        actual_len = seg_end - seg_start
        seg_spec[:, :actual_len] = spectrogram[:, seg_start:seg_end]

        # Extract onset windows from the segment
        windows = _extract_onset_windows(seg_spec, seg_onsets_rel)

        # Cap at max onsets per segment
        if len(windows) > model_config.max_onsets_per_segment:
            windows = windows[: model_config.max_onsets_per_segment]
            seg_onsets = seg_onsets[: model_config.max_onsets_per_segment]
            all_segment_onsets[-1] = seg_onsets

        # To tensor: shape (1, num_onsets, n_mels * 7)
        src_tensor = torch.tensor(
            windows, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Max output length: 2 tokens per onset (plurality + motion) + 2 for sos/eos
        max_decode_len = 2 * len(windows) + 2

        # Autoregressive decode
        tokens = model.predict(src_tensor, max_len=max_decode_len)
        all_tokens.extend(tokens)

        logger.debug(
            "Segment %d/%d: %d onsets, %d tokens generated",
            seg_idx + 1,
            num_segments,
            len(seg_onsets_rel),
            len(tokens),
        )

    inference_time = time.perf_counter() - t0
    logger.info(
        "Inference complete: %d segments, %d total tokens in %.2fs",
        num_segments,
        len(all_tokens),
        inference_time,
    )

    # Flatten all segment onsets back to global onset list
    global_onsets = []
    for seg_onsets in all_segment_onsets:
        global_onsets.extend(seg_onsets)

    # Decode contour tokens into notes array
    notes_array = decode_contour(all_tokens, global_onsets, total_frames)

    return notes_array
