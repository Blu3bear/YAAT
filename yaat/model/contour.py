"""Note contour encoding and decoding for YAAT.

Handles the conversion between the OnsetTransformer's token vocabulary
(plurality + motion pairs) and concrete Guitar Hero notes (1–31).

Vocabulary:
    Tokens 0–2:  <sos>, <eos>, <pad>
    Tokens 3–15: Note pluralities (13 types)
    Tokens 16–24: Motions (-4 to +4)

Note Plurality Table (from TensorHero paper):
    Each plurality defines a group of notes sharing the same button count
    and general shape. The anchor index selects a specific note within
    the group.
"""

import numpy as np

from yaat.model.transformer import (
    PLURALITY_OFFSET,
    MOTION_OFFSET,
    NUM_PLURALITIES,
    NUM_MOTIONS,
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
)
from yaat.utils.logging import get_logger

# ─── Note-to-index mapping (1–31) ───────────────────────────────────────────
# These map the one-hot note index used in the notes array to the set of
# Guitar Hero buttons held. The ordering matches TensorHero's encoding.

# fmt: off
NOTE_INDEX_TO_BUTTONS: dict[int, str] = {
    1:  "G",      2:  "R",      3:  "Y",      4:  "B",      5:  "O",      # singles
    6:  "GR",     7:  "RY",     8:  "YB",     9:  "BO",                   # double-0
    10: "GY",    11: "RB",     12: "YO",                                   # double-1
    13: "GB",    14: "RO",                                                  # double-2
    15: "GO",                                                               # double-3
    16: "GRY",   17: "RYB",    18: "YBO",                                  # triple-0
    19: "GRB",   20: "RYO",                                                # triple-1
    21: "GYB",   22: "RBO",                                                # triple-2
    23: "GRO",   24: "GYO",    25: "GBO",                                  # triple-3
    26: "GRYB",  27: "RYBO",                                               # quad-0
    28: "GRYO",  29: "GRBO",   30: "GYBO",                                 # quad-1
    31: "GRYBO",                                                            # pent (all 5)
    # Note: open note is handled as a special case (index can vary)
}
# fmt: on

# ─── Plurality table ────────────────────────────────────────────────────────
# Each plurality is an ordered list of note indices. The anchor selects
# which note to use within the plurality.

PLURALITY_TABLE: list[list[int]] = [
    # 0: single (s)
    [1, 2, 3, 4, 5],
    # 1: double-0 (d0) — adjacent pairs
    [6, 7, 8, 9],
    # 2: double-1 (d1) — 1 gap
    [10, 11, 12],
    # 3: double-2 (d2) — 2 gap
    [13, 14],
    # 4: double-3 (d3) — 3 gap
    [15],
    # 5: triple-0 (t0) — adjacent triples
    [16, 17, 18],
    # 6: triple-1 (t1) — 1 gap between 2nd and 3rd
    [19, 20],
    # 7: triple-2 (t2) — 1 gap between 1st and 2nd
    [21, 22],
    # 8: triple-3 (t3) — 2 gaps between all
    [23, 24, 25],
    # 9: quad-0 (q0) — adjacent quads
    [26, 27],
    # 10: quad-1 (q1) — 1 gap
    [28, 29, 30],
    # 11: pent (p) — all five
    [31],
    # 12: open (o) — open note (no buttons pressed, strum only)
    # We use index 0 as a sentinel for open in the notes array;
    # however open notes are rarely predicted. We encode as note 0.
    [0],
]

PLURALITY_NAMES = [
    "single", "double-0", "double-1", "double-2", "double-3",
    "triple-0", "triple-1", "triple-2", "triple-3",
    "quad-0", "quad-1", "pent", "open",
]

# Build reverse lookup: note_index → (plurality_idx, anchor)
_NOTE_TO_PLURALITY: dict[int, tuple[int, int]] = {}
for _p_idx, _notes in enumerate(PLURALITY_TABLE):
    for _a_idx, _n_idx in enumerate(_notes):
        _NOTE_TO_PLURALITY[_n_idx] = (_p_idx, _a_idx)


def encode_contour(notes_array: np.ndarray) -> list[tuple[int, int]]:
    """Encode a notes array into a list of (plurality_token, motion_token) pairs.

    Args:
        notes_array: 1D array where each index is a 10ms tick and values are
            note indices (0 = no note, 1–31 = note).

    Returns:
        List of (plurality_token, motion_token) pairs for each onset.
    """
    pairs = []
    prev_anchor = 0

    for note_idx in notes_array:
        if note_idx == 0:
            continue

        note_idx = int(note_idx)
        if note_idx not in _NOTE_TO_PLURALITY:
            continue

        p_idx, anchor = _NOTE_TO_PLURALITY[note_idx]
        motion = anchor - prev_anchor

        # Clamp motion to [-4, 4]
        motion = max(-4, min(4, motion))

        plurality_token = p_idx + PLURALITY_OFFSET
        motion_token = motion + MOTION_OFFSET + 4  # -4→16, 0→20, +4→24

        pairs.append((plurality_token, motion_token))
        prev_anchor = anchor

    return pairs


def decode_contour(
    tokens: list[int],
    onset_bins: list[int],
    total_frames: int,
) -> np.ndarray:
    """Decode model output tokens into a notes array.

    Args:
        tokens: List of output tokens from the OnsetTransformer. Expected
            format: [plurality, motion, plurality, motion, ...] pairs.
        onset_bins: List of onset positions as 10ms time bin indices.
        total_frames: Total number of 10ms frames in the song.

    Returns:
        1D notes array of length total_frames. Index = 10ms tick,
        value = note index (0 = no note, 1–31 = note).
    """
    logger = get_logger()

    notes_array = np.zeros(total_frames, dtype=np.int32)

    # Filter out special tokens
    filtered = [t for t in tokens if t not in (SOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]

    # Parse pairwise: (plurality_token, motion_token)
    pairs = []
    i = 0
    while i + 1 < len(filtered):
        p_tok = filtered[i]
        m_tok = filtered[i + 1]

        # Validate token ranges
        if PLURALITY_OFFSET <= p_tok < PLURALITY_OFFSET + NUM_PLURALITIES:
            if MOTION_OFFSET <= m_tok < MOTION_OFFSET + NUM_MOTIONS:
                pairs.append((p_tok, m_tok))
        i += 2

    # Decode pairs to notes
    anchor = 0
    notes_placed = 0

    for pair_idx, (p_tok, m_tok) in enumerate(pairs):
        if pair_idx >= len(onset_bins):
            break

        onset_bin = onset_bins[pair_idx]
        if onset_bin >= total_frames:
            continue

        # Decode plurality and motion
        p_idx = p_tok - PLURALITY_OFFSET
        motion = (m_tok - MOTION_OFFSET) - 4  # Token 16 → -4, Token 20 → 0, etc.

        plurality = PLURALITY_TABLE[p_idx]
        plurality_size = len(plurality)

        # Update anchor with wrapping
        anchor = anchor + motion
        if anchor >= plurality_size:
            anchor = 0
        elif anchor < 0:
            anchor = plurality_size - 1

        note_idx = plurality[anchor]
        notes_array[onset_bin] = note_idx
        notes_placed += 1

    # Log distribution stats
    unique, counts = np.unique(notes_array[notes_array > 0], return_counts=True)
    logger.info(
        "Contour decoded: %d notes placed from %d token pairs",
        notes_placed,
        len(pairs),
    )
    for note_val, count in zip(unique, counts):
        name = NOTE_INDEX_TO_BUTTONS.get(int(note_val), f"unknown({note_val})")
        logger.debug("  Note %2d (%s): %d occurrences", note_val, name, count)

    return notes_array
