"""Chart file writer for YAAT.

Converts a notes array into a valid .chart file and assembles the
complete Clone Hero / YARG song directory with:
    - notes.chart  (the chart file)
    - song.ogg     (audio, converted from .wav if needed)
    - song.ini     (metadata for in-game display)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from yaat.config import OutputConfig
from yaat.utils.logging import get_logger

# TensorHero tick encoding: Resolution=192, BPM=31.25, TS=1
# This means each beat = 192 ticks, and at 31.25 BPM:
#   192 ticks/beat ÷ 31.25 beats/min = 6.144 ticks/sec ... wait
# Actually: each 10ms frame maps to exactly 1 tick at R=192 with the
# combined B=31.25 and TS=1 encoding the paper uses, BUT the .chart
# format writes ticks, not frames. Since each frame IS one "resolution unit"
# in their encoding, tick = frame_index (we just scale by the resolution).
#
# Per TensorHero inference.py: tick = frame_index (i.e. ticks ARE frame indices)
# The chart header sets R=192 and B=31250 (31.25 BPM * 1000) with TS=1.
# This maps 1 resolution-unit = 10ms because:
#   at 31.25 BPM: 1 beat = 60/31.25 = 1.92 seconds
#   1.92 seconds / 192 ticks-per-beat = 0.01 second/tick = 10ms per tick
CHART_RESOLUTION = 192
CHART_BPM_ENCODED = 31250  # 31.25 BPM × 1000 (as encoded in .chart)
CHART_TIME_SIG = 1  # 1 beat per measure

# Mapping from note index (1–31) to list of .chart note types.
# In .chart format: 0=Green, 1=Red, 2=Yellow, 3=Blue, 4=Orange, 7=Open
# Mapping matches the tensor-hero simplified notes encoding (model13).
# fmt: off
NOTES_TO_CHART_TYPES: dict[int, list[int]] = {
    1:  [0],                  # Green
    2:  [1],                  # Red
    3:  [2],                  # Yellow
    4:  [3],                  # Blue
    5:  [4],                  # Orange
    6:  [0, 1],              # G+R
    7:  [0, 2],              # G+Y
    8:  [0, 3],              # G+B
    9:  [0, 4],              # G+O
    10: [1, 2],              # R+Y
    11: [1, 3],              # R+B
    12: [1, 4],              # R+O
    13: [2, 3],              # Y+B
    14: [2, 4],              # Y+O
    15: [3, 4],              # B+O
    16: [0, 1, 2],          # G+R+Y
    17: [0, 1, 3],          # G+R+B
    18: [0, 1, 4],          # G+R+O
    19: [0, 2, 3],          # G+Y+B
    20: [0, 2, 4],          # G+Y+O
    21: [0, 3, 4],          # G+B+O
    22: [1, 2, 3],          # R+Y+B
    23: [1, 2, 4],          # R+Y+O
    24: [1, 3, 4],          # R+B+O
    25: [2, 3, 4],          # Y+B+O
    26: [0, 1, 2, 3],       # G+R+Y+B
    27: [0, 1, 2, 4],       # G+R+Y+O
    28: [0, 1, 3, 4],       # G+R+B+O
    29: [0, 2, 3, 4],       # G+Y+B+O
    30: [1, 2, 3, 4],       # R+Y+B+O
    31: [0, 1, 2, 3, 4],    # G+R+Y+B+O (all)
    32: [7],                 # Open note
}
# fmt: on


def _notes_array_to_chart_events(notes_array: np.ndarray) -> list[str]:
    """Convert a notes array into .chart note event lines.

    Args:
        notes_array: 1D array, index=10ms tick, value=note index (0=no note).

    Returns:
        List of strings like "  768 = N 0 0" for each note event.
    """
    events = []
    onset_indices = np.where(notes_array > 0)[0]

    for frame_idx in onset_indices:
        note_idx = int(notes_array[frame_idx])

        if note_idx in NOTES_TO_CHART_TYPES:
            chart_types = NOTES_TO_CHART_TYPES[note_idx]
        else:
            # Fallback: treat as open note
            chart_types = [7]

        # Each tick = frame_index (10ms per tick at our resolution/BPM)
        tick = int(frame_idx)

        for note_type in chart_types:
            events.append(f"  {tick} = N {note_type} 0")

    return events


def write_chart_file(
    notes_array: np.ndarray,
    output_path: Path,
    config: OutputConfig,
) -> None:
    """Write a .chart file from a notes array.

    Args:
        notes_array: 1D validated notes array.
        output_path: Path to write the .chart file.
        config: Output metadata configuration.
    """
    logger = get_logger()

    song_name = config.chart_name or "Untitled"
    artist = config.artist
    charter = config.charter

    # Build note events
    note_events = _notes_array_to_chart_events(notes_array)

    # Compose the .chart file
    lines = []

    # [Song] section
    lines.append("[Song]")
    lines.append("{")
    lines.append(f'  Name = "{song_name}"')
    lines.append(f'  Artist = "{artist}"')
    lines.append(f'  Charter = "{charter}"')
    lines.append(f"  Resolution = {CHART_RESOLUTION}")
    lines.append("  Offset = 0")
    lines.append(f"  Player2 = bass")
    lines.append(f"  Difficulty = 0")
    lines.append(f"  PreviewStart = 0")
    lines.append(f"  PreviewEnd = 0")
    lines.append(f'  Genre = "rock"')
    lines.append(f'  MediaType = "cd"')
    lines.append(f'  MusicStream = "song.ogg"')
    lines.append("}")

    # [SyncTrack] section
    lines.append("[SyncTrack]")
    lines.append("{")
    lines.append(f"  0 = TS {CHART_TIME_SIG}")
    lines.append(f"  0 = B {CHART_BPM_ENCODED}")
    lines.append("}")

    # [Events] section
    lines.append("[Events]")
    lines.append("{")
    lines.append("}")

    # [ExpertSingle] section
    lines.append("[ExpertSingle]")
    lines.append("{")
    lines.extend(note_events)
    lines.append("}")

    # Write file
    chart_text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(chart_text, encoding="utf-8")

    logger.info(
        "Chart written: %s (%d note events, %d bytes)",
        output_path,
        len(note_events),
        len(chart_text),
    )


def write_song_ini(
    output_dir: Path,
    config: OutputConfig,
    song_length_ms: int,
) -> None:
    """Write a song.ini metadata file.

    Args:
        output_dir: Directory to write song.ini into.
        config: Output metadata configuration.
        song_length_ms: Song length in milliseconds.
    """
    song_name = config.chart_name or "Untitled"
    ini_lines = [
        "[song]",
        f"name = {song_name}",
        f"artist = {config.artist}",
        f"charter = {config.charter}",
        f"album = ",
        f"genre = rock",
        f"year = ",
        f"diff_guitar = -1",
        f"preview_start_time = 0",
        f"song_length = {song_length_ms}",
        f"loading_phrase = Generated by YAAT",
        f"delay = 0",
    ]

    ini_path = output_dir / "song.ini"
    ini_path.write_text("\n".join(ini_lines) + "\n", encoding="utf-8")

    get_logger().info("song.ini written: %s", ini_path)


def convert_audio_to_ogg(input_path: Path, output_path: Path) -> None:
    """Convert an audio file to .ogg format for Clone Hero.

    Tries ffmpeg first; falls back to copying the file if conversion fails
    or if the input is already .ogg.

    Args:
        input_path: Source audio file.
        output_path: Destination .ogg file path.
    """
    logger = get_logger()

    if input_path.suffix.lower() == ".ogg":
        shutil.copy2(str(input_path), str(output_path))
        logger.info("Audio copied (already .ogg): %s", output_path)
        return

    # Try ffmpeg
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-acodec",
                "libvorbis",
                "-q:a",
                "8",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("Audio converted to .ogg via ffmpeg: %s", output_path)
            return
        else:
            logger.warning("ffmpeg conversion failed: %s", result.stderr[:200])
    except FileNotFoundError:
        logger.warning("ffmpeg not found on PATH")
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg conversion timed out")

    # Fallback: try soundfile (writes WAV-in-OGG or raw copy)
    try:
        audio_data, sr = sf.read(str(input_path))
        sf.write(str(output_path), audio_data, sr, format="OGG", subtype="VORBIS")
        logger.info("Audio converted to .ogg via soundfile: %s", output_path)
    except Exception as exc:
        # Last resort: just copy the original file with .ogg extension
        # Clone Hero / YARG will still load it
        logger.warning("Could not convert to .ogg (%s). Copying original file.", exc)
        shutil.copy2(str(input_path), str(output_path))
        logger.info("Audio copied as: %s", output_path)


def assemble_chart_directory(
    notes_array: np.ndarray,
    input_audio_path: Path,
    output_dir: Path,
    output_config: OutputConfig,
    audio_duration_s: float,
) -> Path:
    """Assemble a complete Clone Hero / YARG chart directory.

    Creates:
        output_dir/
        ├── notes.chart
        ├── song.ogg
        └── song.ini

    Args:
        notes_array: Validated notes array.
        input_audio_path: Path to the original input audio file.
        output_dir: Directory to create the chart package in.
        output_config: Output metadata configuration.
        audio_duration_s: Audio duration in seconds.

    Returns:
        Path to the output directory.
    """
    logger = get_logger()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect chart name from filename if not set
    if not output_config.chart_name:
        output_config.chart_name = input_audio_path.stem

    # Write notes.chart
    chart_path = output_dir / "notes.chart"
    write_chart_file(notes_array, chart_path, output_config)

    # Convert and write audio
    audio_path = output_dir / "song.ogg"
    convert_audio_to_ogg(input_audio_path, audio_path)

    # Write song.ini
    song_length_ms = int(audio_duration_s * 1000)
    write_song_ini(output_dir, output_config, song_length_ms)

    logger.info("Chart directory assembled: %s", output_dir)
    for f in sorted(output_dir.iterdir()):
        logger.info("  %s (%d bytes)", f.name, f.stat().st_size)

    return output_dir
