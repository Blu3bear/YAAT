"""Postprocessing validation for YAAT.

Enforces chart constraints on the generated notes array to ensure
the output is a valid, playable Clone Hero / YARG chart:
    - Button validity: all note indices in [1, 31].
    - Temporal bounds: no notes beyond audio duration.
    - Density ceiling: max notes per second.
    - Minimum gap: no two notes closer than min_note_gap_ms.
    - Empty chart guard: abort if too few notes.
"""

import numpy as np

from yaat.config import PostprocessConfig
from yaat.utils.logging import get_logger


class ChartValidationError(Exception):
    """Raised when the generated chart fails critical validation."""


def validate_notes(
    notes_array: np.ndarray,
    audio_duration_s: float,
    config: PostprocessConfig,
) -> np.ndarray:
    """Validate and enforce constraints on a notes array.

    Args:
        notes_array: 1D array, index=10ms tick, value=note index (0=no note, 1–31=note).
        audio_duration_s: Duration of the source audio in seconds.
        config: Postprocessing constraint parameters.

    Returns:
        Cleaned notes array with invalid notes removed.

    Raises:
        ChartValidationError: If the chart has too few notes after cleaning.
    """
    logger = get_logger()
    original_count = int(np.count_nonzero(notes_array))
    removals = {
        "invalid_index": 0,
        "beyond_duration": 0,
        "density_excess": 0,
        "min_gap": 0,
    }

    # ─── 1. Button validity: note indices must be in [1, 31] ────────────────
    invalid_mask = (notes_array < 0) | (notes_array > 31)
    removals["invalid_index"] = int(np.count_nonzero(notes_array[invalid_mask]))
    notes_array[invalid_mask] = 0

    # ─── 2. Temporal bounds: truncate beyond audio duration ─────────────────
    max_tick = int(audio_duration_s * 100)  # 10ms ticks
    if len(notes_array) > max_tick:
        beyond = notes_array[max_tick:]
        removals["beyond_duration"] = int(np.count_nonzero(beyond))
        notes_array = notes_array[:max_tick]

    # ─── 3. Minimum gap enforcement ────────────────────────────────────────
    min_gap_ticks = max(1, config.min_note_gap_ms // 10)
    onset_indices = np.where(notes_array > 0)[0]
    for i in range(1, len(onset_indices)):
        gap = onset_indices[i] - onset_indices[i - 1]
        if gap < min_gap_ticks:
            notes_array[onset_indices[i]] = 0
            removals["min_gap"] += 1

    # ─── 4. Density ceiling: max notes per second via sliding window ───────
    max_nps = config.max_notes_per_second
    if max_nps > 0:
        window_ticks = 100  # 1 second = 100 × 10ms ticks
        onset_indices = np.where(notes_array > 0)[0]

        # Sliding window check
        i = 0
        while i < len(onset_indices):
            window_start = onset_indices[i]
            window_end = window_start + window_ticks

            # Count notes in this 1-second window
            in_window = onset_indices[
                (onset_indices >= window_start) & (onset_indices < window_end)
            ]

            if len(in_window) > max_nps:
                # Remove excess notes by dropping every other note from the end
                excess = in_window[max_nps:]
                for idx in excess:
                    notes_array[idx] = 0
                    removals["density_excess"] += 1

                # Recompute onset_indices after removal
                onset_indices = np.where(notes_array > 0)[0]
            else:
                i += 1

    # ─── 5. Empty chart guard ──────────────────────────────────────────────
    final_count = int(np.count_nonzero(notes_array))
    total_removed = original_count - final_count

    logger.info("Validation — original notes: %d, final: %d", original_count, final_count)
    for rule, count in removals.items():
        if count > 0:
            logger.info("  Removed by %-20s: %d notes", rule, count)

    # Density stats
    if audio_duration_s > 0 and final_count > 0:
        avg_nps = final_count / audio_duration_s
        logger.info(
            "  Average density: %.2f notes/sec over %.2fs",
            avg_nps,
            audio_duration_s,
        )

    if final_count < config.min_total_notes:
        raise ChartValidationError(
            f"Chart has only {final_count} notes after validation "
            f"(minimum: {config.min_total_notes}). "
            "The model may not have produced meaningful output."
        )

    return notes_array
