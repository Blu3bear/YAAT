"""Onset detection using NINOS ODF and peak picking.

Implements the onset detection pipeline from TensorHero:
    1. Filter spectrogram by amplitude (keep top p%).
    2. Compute NINOS (Normalized Inverse Note Onset Sparsity) ODF.
    3. Peak picking (Böck et al. 2012) to select onset frames.
    4. Convert to 10ms time bins.
"""

import math

import numpy as np
import librosa

from yaat.config import AudioConfig, OnsetConfig
from yaat.utils.logging import get_logger


def filter_spec_by_amplitude(
    spec: np.ndarray,
    p: float = 0.99,
) -> np.ndarray:
    """Zero out frequency bins below the p-th percentile of energy.

    Args:
        spec: Magnitude spectrogram, shape (freq, time).
        p: Percentile threshold (0–1). Bins below this percentile are zeroed.

    Returns:
        Filtered spectrogram (same shape).
    """
    threshold = np.percentile(np.abs(spec), p * 100)
    filtered = spec.copy()
    filtered[np.abs(filtered) < threshold] = 0.0
    return filtered


def ninos(
    audio: np.ndarray,
    sr: int,
    spec: np.ndarray | None = None,
    gamma: float = 0.94,
    n_fft: int = 2048,
    hop_length: int = 205,
) -> tuple[np.ndarray, int, int]:
    """Compute the NINOS onset detection function.

    NINOS is a normalized, inverse-sparsity measure that uses the l2 and l4
    norms of sorted spectrogram magnitudes to detect transients.

    Args:
        audio: 1D audio signal.
        sr: Sample rate.
        spec: Pre-computed STFT magnitude spectrogram. If None, computed here.
        gamma: Fraction of frequency bins to retain (bottom gamma%).
        n_fft: FFT size for STFT computation.
        hop_length: Hop length for STFT computation.

    Returns:
        Tuple of (ninos_odf, J, hop_length) where ninos_odf is the ODF array,
        J is the number of retained bins, and hop_length is the STFT hop.
    """
    if spec is None:
        spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

    # Sort magnitudes per frame (ascending) and keep bottom gamma fraction
    sorted_spec = np.sort(spec, axis=0)
    J = math.floor(sorted_spec.shape[0] * gamma)
    sorted_spec = sorted_spec[:J, :]

    # Compute l2-norm squared and l4-norm per frame
    l2_squared = np.square(np.linalg.norm(sorted_spec, ord=2, axis=0))
    l4 = np.linalg.norm(sorted_spec, ord=4, axis=0)

    # NINOS = l2^2 / (J^(1/4) * l4), avoid division by zero
    denominator = (J ** 0.25) * l4
    denominator[denominator == 0] = 1e-10

    odf = l2_squared / denominator

    return odf, J, hop_length


def onset_select(
    odf_arr: np.ndarray,
    w1: int = 10,
    w2: int = 1,
    w3: int = 1,
    w4: int = 8,
    w5: int = 10,
    delta: float = 1.0,
) -> list[int]:
    """Peak picking to select onset frames from an ODF.

    Three conditions must hold for a frame to be selected as an onset:
        1. Local maximum: value >= all values in [frame-w1, frame+w2].
        2. Threshold: value >= local mean in [frame-w3, frame+w4] + delta.
        3. Minimum distance: at least w5 frames since last selected onset.

    Based on Böck et al. (2012).

    Args:
        odf_arr: 1D onset detection function values.
        w1: Left window for local maximum check.
        w2: Right window for local maximum check.
        w3: Left window for local threshold check.
        w4: Right window for local threshold check.
        w5: Minimum inter-onset distance in frames.
        delta: Threshold offset.

    Returns:
        List of frame indices selected as onsets.
    """
    onsets = []
    last_onset = -w5  # Allow first onset immediately

    for i in range(len(odf_arr)):
        # Condition 1: Local maximum
        left = max(0, i - w1)
        right = min(len(odf_arr), i + w2 + 1)
        if odf_arr[i] < np.max(odf_arr[left:right]):
            continue

        # Condition 2: Exceeds local mean + delta
        left_t = max(0, i - w3)
        right_t = min(len(odf_arr), i + w4 + 1)
        local_mean = np.mean(odf_arr[left_t:right_t])
        if odf_arr[i] < local_mean + delta:
            continue

        # Condition 3: Minimum distance from last onset
        if i - last_onset < w5:
            continue

        onsets.append(i)
        last_onset = i

    return onsets


def detect_onsets(
    audio: np.ndarray,
    sr: int,
    audio_config: AudioConfig,
    onset_config: OnsetConfig,
) -> list[int]:
    """Full onset detection pipeline: spectrogram → NINOS → peak picking → 10ms bins.

    Args:
        audio: 1D mono audio signal.
        sr: Sample rate.
        audio_config: Audio config with hop_length for final conversion.
        onset_config: Onset detection hyperparameters.

    Returns:
        List of onset positions as 10ms time bin indices.
    """
    logger = get_logger()

    # Compute STFT magnitude spectrogram for onset detection
    # Use a different FFT/hop than the mel spectrogram — TensorHero onset.py
    # uses n_fft=2048, hop_length=205 for 44100 Hz
    n_fft_onset = 2048
    hop_onset = 205

    spec = np.abs(librosa.stft(audio, n_fft=n_fft_onset, hop_length=hop_onset))

    # Filter by amplitude
    spec_filtered = filter_spec_by_amplitude(spec, p=onset_config.p)

    # NINOS ODF
    odf, J, _ = ninos(
        audio,
        sr,
        spec=spec_filtered,
        gamma=onset_config.gamma,
        n_fft=n_fft_onset,
        hop_length=hop_onset,
    )

    # Normalize and invert ODF (higher = more likely onset)
    odf_min = np.min(odf)
    odf_range = np.max(odf) - odf_min
    if odf_range > 0:
        odf = -(odf - odf_min) / odf_range + 1.0
    else:
        odf = np.zeros_like(odf)

    # Peak picking
    onset_frames = onset_select(
        odf,
        w1=onset_config.w1,
        w2=onset_config.w2,
        w3=onset_config.w3,
        w4=onset_config.w4,
        w5=onset_config.w5,
        delta=onset_config.delta,
    )

    # Convert ODF frames to seconds, then to 10ms bins
    onset_times_s = [f * hop_onset / sr for f in onset_frames]
    onset_bins = [round(t * 100) for t in onset_times_s]  # 100 bins/sec = 10ms

    # Remove duplicates and sort
    onset_bins = sorted(set(onset_bins))

    logger.info(
        "Onset detection: %d onsets found (%.2f onsets/sec over %.2fs)",
        len(onset_bins),
        len(onset_bins) / (len(audio) / sr) if len(audio) > 0 else 0,
        len(audio) / sr,
    )

    return onset_bins
