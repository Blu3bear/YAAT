"""Pipeline orchestrator for YAAT.

Wires together all pipeline stages:
    1. Load config
    2. Validate input
    3. Source separation (Demucs htdemucs_6s → guitar stem)
    4. Spectrogram computation
    5. Onset detection (NINOS + peak picking)
    6. Model inference (OnsetTransformer)
    7. Contour decoding
    8. Postprocess validation
    9. Chart writing + directory assembly
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np

from yaat.config import YAATConfig, load_config
from yaat.schema import validate_input
from yaat.audio.separation import separate_guitar
from yaat.audio.spectrogram import compute_spectrogram
from yaat.audio.onset import detect_onsets
from yaat.model.inference import run_inference
from yaat.postprocess.validate import validate_notes
from yaat.postprocess.chart_writer import assemble_chart_directory
from yaat.utils.logging import setup_logging, get_logger, log_stage

import logging


def run(
    input_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
) -> Path:
    """Run the full YAAT pipeline: audio → chart directory.

    Args:
        input_path: Path to the input audio file (.wav).
        output_dir: Path to the output directory for the chart package.
        config_path: Optional path to a YAML config file. Uses defaults if None.

    Returns:
        Path to the assembled chart directory.
    """
    pipeline_start = time.perf_counter()

    # ─── 1. Load config ────────────────────────────────────────────────────
    config = load_config(config_path)
    logger = setup_logging(level=logging.DEBUG if config.debug else logging.INFO)

    log_stage("Configuration", logger)
    logger.info("Config loaded from: %s", config_path or "(defaults)")
    logger.info(
        "  Separation model: %s (stem: %s)",
        config.separation.model,
        config.separation.stem,
    )
    logger.info("  Model weights: %s", config.model.weights_path or "(random init)")
    logger.info(
        "  Device: sep=%s, model=%s", config.separation.device, config.model.device
    )
    logger.info("  Debug mode: %s", config.debug)

    debug_dir = None
    if config.debug:
        debug_dir = Path(output_dir) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info("  Debug output: %s", debug_dir)

    # ─── 2. Validate input ─────────────────────────────────────────────────
    log_stage("Input Validation", logger)
    meta = validate_input(input_path)

    # ─── 3. Source separation ──────────────────────────────────────────────
    log_stage("Source Separation", logger)
    t0 = time.perf_counter()
    guitar_audio, guitar_sr = separate_guitar(str(meta.path), config.separation)
    logger.info("Source separation took %.2fs", time.perf_counter() - t0)

    if debug_dir:
        np.save(str(debug_dir / "guitar_stem.npy"), guitar_audio)

    # ─── 4. Spectrogram computation ────────────────────────────────────────
    log_stage("Spectrogram Computation", logger)
    t0 = time.perf_counter()
    spectrogram = compute_spectrogram(guitar_audio, guitar_sr, config.audio)
    logger.info("Spectrogram computation took %.2fs", time.perf_counter() - t0)

    if debug_dir:
        np.save(str(debug_dir / "spectrogram.npy"), spectrogram)

    # ─── 5. Onset detection ────────────────────────────────────────────────
    log_stage("Onset Detection", logger)
    logger.info("Skipped — model13 uses full spectrogram segments, not onset windows")
    onset_bins: list = []

    if debug_dir:
        np.save(str(debug_dir / "onset_bins.npy"), np.array(onset_bins))

    # ─── 6. Model inference + contour decoding ─────────────────────────────
    log_stage("Model Inference", logger)
    t0 = time.perf_counter()
    notes_array = run_inference(spectrogram, onset_bins, config.model, config.audio)
    logger.info("Model inference took %.2fs", time.perf_counter() - t0)

    note_count = int(np.count_nonzero(notes_array))
    logger.info("Raw notes generated: %d", note_count)

    if debug_dir:
        np.save(str(debug_dir / "notes_raw.npy"), notes_array)

    # ─── 7. Postprocess validation ─────────────────────────────────────────
    log_stage("Postprocess Validation", logger)
    t0 = time.perf_counter()
    notes_array = validate_notes(notes_array, meta.duration_s, config.postprocess)
    logger.info("Postprocessing took %.2fs", time.perf_counter() - t0)

    if debug_dir:
        np.save(str(debug_dir / "notes_validated.npy"), notes_array)

    # ─── 8. Chart writing + directory assembly ─────────────────────────────
    log_stage("Chart Assembly", logger)
    t0 = time.perf_counter()
    result_dir = assemble_chart_directory(
        notes_array=notes_array,
        input_audio_path=meta.path,
        output_dir=Path(output_dir),
        output_config=config.output,
        audio_duration_s=meta.duration_s,
    )
    logger.info("Chart assembly took %.2fs", time.perf_counter() - t0)

    # ─── Summary ───────────────────────────────────────────────────────────
    total_time = time.perf_counter() - pipeline_start
    final_notes = int(np.count_nonzero(notes_array))

    log_stage("Pipeline Complete", logger)
    logger.info("Total pipeline time: %.2fs", total_time)
    logger.info("Input:  %s (%.2fs)", meta.path.name, meta.duration_s)
    logger.info("Output: %s", result_dir)
    logger.info(
        "Notes:  %d (%.2f notes/sec)", final_notes, final_notes / meta.duration_s
    )

    return result_dir
