# YAAT — Development Context Document

> **Purpose:** This document captures the full project plan, architecture decisions, implementation status, and key technical details for the YAAT (Yet Another Auto-charter Tool) project. Use it to resume development on another machine or in a new session.

---

## 1. Project Overview

YAAT is an AI system that generates playable Guitar Hero / Clone Hero / YARG chart files from raw audio input. It accepts a `.wav` file, processes it through a multi-stage pipeline, and outputs a directory ready to import into Clone Hero or YARG.

**Target users:** PC gamers who play Clone Hero and YARG and want charts for songs that don't have them.

**Core constraint:** Must run on a high-end gaming PC (≤32 GB RAM, ≤16 GB VRAM). Processing a single song should take ≤10 minutes.

---

## 2. Architecture — Pipeline Stages

The system is a sequential pipeline with 9 stages:

```
.wav input → Validate → Demucs Separation → Spectrogram → Onset Detection → Model Inference → Contour Decode → Postprocess → Chart Assembly → output directory
```

| # | Stage | Module | Description |
|---|-------|--------|-------------|
| 1 | Config | `yaat/config.py` | Load YAML config, validate via Pydantic |
| 2 | Input Validation | `yaat/schema.py` | File exists, is .wav, 1s–30min duration |
| 3 | Source Separation | `yaat/audio/separation.py` | Demucs `htdemucs_6s` → extract `guitar` stem |
| 4 | Spectrogram | `yaat/audio/spectrogram.py` | Log-mel spectrogram (512 bins, FFT=4096, hop=441) |
| 5 | Onset Detection | `yaat/audio/onset.py` | NINOS ODF + peak picking → 10ms onset bins |
| 6 | Model Inference | `yaat/model/inference.py` | OnsetTransformer: 4s segments, 7-frame onset windows |
| 7 | Contour Decoding | `yaat/model/contour.py` | (plurality, motion) token pairs → notes array (0–31) |
| 8 | Postprocessing | `yaat/postprocess/validate.py` | Button validity, temporal bounds, density, min gap |
| 9 | Chart Assembly | `yaat/postprocess/chart_writer.py` | Write `notes.chart` + `song.ogg` + `song.ini` |

**Orchestrator:** `yaat/pipeline.py` — chains all stages, logs timing + intermediate stats at each step.

---

## 3. Project Structure

```
YAAT/
├── config.yaml                        # Default pipeline configuration (YAML)
├── requirements.txt                   # Python dependencies
├── tensorhero_capstone.md             # Reference paper on TensorHero model
├── context.md                         # This document
└── yaat/
    ├── __init__.py                    # Library API: generate_chart() (lazy import)
    ├── __main__.py                    # CLI: python -m yaat --input song.wav --output ./out/
    ├── config.py                      # Pydantic config loader from YAML
    ├── schema.py                      # Input validation (exists, format, duration bounds)
    ├── pipeline.py                    # Orchestrator: chains all 8 stages with logging
    ├── audio/
    │   ├── __init__.py
    │   ├── separation.py              # Demucs htdemucs_6s → guitar stem
    │   ├── spectrogram.py             # Log-mel spectrogram (512 bins, FFT=4096, hop=441)
    │   └── onset.py                   # NINOS ODF + peak picking → 10ms onset bins
    ├── model/
    │   ├── __init__.py
    │   ├── transformer.py             # OnsetTransformer architecture (512d, 8-head, 2 layers)
    │   ├── contour.py                 # Note contour tables + decode (plurality,motion)→notes
    │   └── inference.py               # Segment → onset windows → autoregressive decode
    ├── postprocess/
    │   ├── __init__.py
    │   ├── validate.py                # Button validity, temporal bounds, density, min gap
    │   └── chart_writer.py            # notes array → .chart + song.ogg + song.ini directory
    └── utils/
        ├── __init__.py
        └── logging.py                 # Python logging setup + array stats helpers
```

---

## 4. Key Technical Decisions

### 4.1 Demucs Source Separation
- **Model:** `htdemucs_6s` from `adefossez/demucs` (NOT via HuggingFace `transformers` — Demucs is incompatible with that library).
- **6 stems:** drums, bass, other, vocals, **guitar**, piano.
- **Stem used:** `guitar` (explicit guitar stem, not `other` as in original TensorHero paper).
- **Library:** `demucs` package installed from pip. Uses `demucs.api.Separator` for inference. Auto-downloads model weights on first run from Facebook's CDN.
- The `iBoostAI/Demucs-v4` HuggingFace repo is just a mirror of checkpoint files — not compatible with `transformers.AutoModel`.

### 4.2 OnsetTransformer (TensorHero Model)
- **Chosen over** the full Transformer variant — the paper found it performed better.
- **Architecture:** d_model=512, 8-head attention, 2 encoder + 2 decoder layers, ff=2048, dropout=0.1.
- **Input:** 7-frame spectrogram windows (±3 frames around each onset), flattened from (512, 7) → 3584 dims, projected to 512 via dense + sigmoid.
- **Output:** 25-token vocabulary — `[<sos>=0, <eos>=1, <pad>=2, pluralities 3–15, motions 16–24]`.
- **Decoding:** Autoregressive, greedy argmax, outputs (plurality, motion) pairs until `<eos>`.
- **Weights:** Pre-trained weights from tensor-hero GitHub repo (`saved_models_backup/model13/`). Path set in `config.yaml` → `model.weights_path`.
- **Source repo:** https://github.com/elliottwaissbluth/tensor-hero

### 4.3 Note Contour System
- 13 **note pluralities**: single, double-0 through double-3, triple-0 through triple-3, quad-0, quad-1, pent, open.
- 9 **motions**: -4 to +4 (shift within the plurality's anchor positions).
- **Anchor wrapping:** If anchor goes out of bounds, wraps to 0 or |plurality|-1.
- **31 possible notes** from 5-button Guitar Hero controller (all combinations + open).
- Full plurality table and note-to-button mappings are in `yaat/model/contour.py`.

### 4.4 .chart File Format
- **Encoding:** TensorHero's normalized format — Resolution=192, BPM=31.25 (encoded as `B 31250`), TS=1.
- This means **1 tick = 10ms**, aligning spectrogram frames directly to chart ticks.
- Notes: `<tick> = N <type> 0` where type ∈ {0=Green, 1=Red, 2=Yellow, 3=Blue, 4=Orange, 7=Open}.
- Chords = multiple `N` events at the same tick.
- **Output difficulty:** ExpertSingle only (MVP). Difficulty conversion deferred to later.

### 4.5 Chart Directory Structure (Clone Hero / YARG compatible)
```
output_dir/
├── notes.chart    # The chart file
├── song.ogg       # Audio (converted from .wav via ffmpeg, fallback to soundfile)
└── song.ini       # Metadata for in-game display
```

### 4.6 Postprocessing Constraints
- **Button validity:** Note indices must be in [1, 31].
- **Temporal bounds:** No notes beyond audio duration (truncate, never extend).
- **Density ceiling:** Max 15 notes/second (sliding 1s window); excess notes removed.
- **Minimum gap:** No two notes closer than 20ms (2 ticks).
- **Empty chart guard:** Abort if < 10 total notes after validation.

### 4.7 Configuration
- **YAML file** (`config.yaml`) loaded via **Pydantic** `BaseModel` classes for runtime type validation.
- Pydantic was chosen over dataclasses because the config comes from an external YAML file (untrusted input) and needs schema enforcement + `field_validator` for computed defaults (e.g., `"auto"` → `"cuda"/"cpu"`).

### 4.8 Interface
- **CLI:** `python -m yaat --input song.wav --output ./chart_dir/ [--config config.yaml]`
- **Library:** `from yaat import generate_chart; generate_chart("song.wav", "./out/")`
- `__init__.py` uses lazy imports so `--help` works without loading heavy torch/demucs dependencies.

### 4.9 Audio Preprocessing Details
- **Resample** to 44100 Hz.
- **Log-mel spectrogram:** 512 mel bins, FFT=4096, hop_length=441 (= 10ms per frame at 44100 Hz).
- **Normalization:** `librosa.power_to_db(spec, ref=np.max)` → range [-80, 0] → `(spec + 80) / 80` → [0, 1].
- **Onset detection (NINOS):** Compute STFT with n_fft=2048, hop=205. Filter bottom 1% amplitude. NINOS ODF uses sorted magnitudes, l2/l4 norms. Peak picking with 5 hyperparams (w1=10, w2=1, w3=1, w4=8, w5=10, delta=1.0). Convert frames to 10ms bins.

---

## 5. Dependencies

```
torch>=2.0
torchaudio>=2.0
librosa>=0.10
demucs>=4.0        # From adefossez/demucs, NOT via transformers
numpy
pyyaml
pydantic>=2.0
soundfile
```

**Note:** `demucs` needs to be installed separately — `pip install demucs`. It was not fully installed in the initial session's venv.

**Python version:** 3.13.5 (system install at `C:/Users/jayse/AppData/Local/Programs/Python/Python313/python.exe`). A `.venv` was also set up at `c:\Users\jayse\GitClones\YAAT\.venv`.

---

## 6. Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Project scaffold | ✅ Complete | All dirs, `__init__.py` files |
| `config.yaml` | ✅ Complete | All parameters with defaults |
| `requirements.txt` | ✅ Complete | Note: `demucs` not listed (pip install separately) |
| `yaat/config.py` | ✅ Complete | Pydantic models, YAML loader, auto device detection |
| `yaat/schema.py` | ✅ Complete | File validation, AudioMeta dataclass |
| `yaat/utils/logging.py` | ✅ Complete | Logger setup, array stats, stage banners |
| `yaat/audio/separation.py` | ✅ Complete | Demucs htdemucs_6s, guitar stem extraction |
| `yaat/audio/spectrogram.py` | ✅ Complete | Log-mel, normalization |
| `yaat/audio/onset.py` | ✅ Complete | NINOS ODF, peak picking, 10ms bin conversion |
| `yaat/model/transformer.py` | ✅ Complete | OnsetTransformer, OnsetInputEmbedding, autoregressive predict() |
| `yaat/model/contour.py` | ✅ Complete | Full plurality table, encode/decode, note mappings |
| `yaat/model/inference.py` | ✅ Complete | Segment splitting, onset windowing, model loading |
| `yaat/postprocess/validate.py` | ✅ Complete | All 5 constraint checks |
| `yaat/postprocess/chart_writer.py` | ✅ Complete | .chart writer, song.ini, audio conversion, directory assembly |
| `yaat/pipeline.py` | ✅ Complete | 9-stage orchestrator with timing + debug dumps |
| `yaat/__main__.py` | ✅ Complete | argparse CLI |
| `yaat/__init__.py` | ✅ Complete | Lazy import `generate_chart()` |
| Syntax check (all files) | ✅ Passed | `py_compile` on all 14 .py files |
| CLI `--help` | ✅ Working | Tested successfully |
| Dependencies installed | ⚠️ Partial | torch, torchaudio, librosa, pydantic, etc. installed. **demucs not yet installed.** |
| Pre-trained weights | ❌ Not yet | Need to download from tensor-hero repo |
| End-to-end test | ❌ Not yet | No test audio file run through pipeline yet |
| Unit tests | ❌ Not yet | No test files created |

---

## 7. Next Steps

1. **Install demucs** in the venv: `pip install demucs`
2. **Download TensorHero weights** from https://github.com/elliottwaissbluth/tensor-hero (`saved_models_backup/model13/`) and set `model.weights_path` in `config.yaml`.
3. **End-to-end test:** Run `python -m yaat --input test_song.wav --output ./test_chart/` on a short audio file.
4. **Validate output:** Import the generated chart directory into Clone Hero or YARG.
5. **Add `demucs` to `requirements.txt`** once install method is finalized.
6. **Unit tests:** Test individual stages (spectrogram shape, onset detection on synthetic audio, contour round-trip, chart writer parse-back).
7. **Optimization:** Profile memory/time on longer songs; consider chunked Demucs processing.
8. **Future features:**
   - Difficulty conversion (easy/medium/hard) via transition matrices.
   - Multi-instrument charts (bass, drums, vocals) using other Demucs stems.
   - Held notes and modifier support (force, tap).
   - Genre-specific model training/fine-tuning.

---

## 8. Reference Material

- **TensorHero paper:** `tensorhero_capstone.md` in repo root — full details on model architecture, note contour system, training, and evaluation.
- **TensorHero source:** https://github.com/elliottwaissbluth/tensor-hero
- **.chart format spec:** https://github.com/TheNathannator/GuitarGame_ChartFormats
- **Demucs:** https://github.com/adefossez/demucs (archived fork of facebookresearch/demucs)
- **HuggingFace mirror:** https://huggingface.co/iBoostAI/Demucs-v4 (checkpoint mirror only, not `transformers`-compatible)
