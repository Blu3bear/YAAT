# YAAT (Yet Another Ai Tool)

YAAT is a pipeline system for generating rhythm game charts (Clone Hero / YARG) from audio files.

## Installation


### Setup
1. Clone the repository (including submodules):
   ```bash
   git clone --recursive https://github.com/yourusername/YAAT.git
   cd YAAT
   ```

2. Install dependencies:
   First, install `demucs` from the local source (submodule):
   ```bash
   pip install -e external/demucs
   ```

   Then install the remaining requirements:
   ```bash
   pip install -r requirements.txt
   ```

   This includes `pytubefix` for YouTube retrieval and `imageio-ffmpeg` so m4a/webm downloads can be converted to WAV even if `ffmpeg` is not installed globally.

   *Note: `demucs` is included as a submodule in `external/demucs` to ensure compatibility and ease of installation.*

   **Alternatively, use Conda:**
   ```bash
   conda env create -f environment_yaat.yml
   conda activate yaat
   ```

## Usage

### Command Line Interface (CLI)
Run the tool directly from your terminal:

```bash
python -m yaat -f path/to/song.wav --output ./output_chart/
```

Or use agentic retrieval from YouTube search:

```bash
python -m yaat -s "artist song title" --output ./output_chart/
```

| Argument | Short | Description | Required |
|----------|-------|-------------|----------|
| `--file` | `-f` | Path to a local input audio file. | Yes (if `--search` is not used) |
| `--search` | `-s` | Search term for retrieval agent (YouTube + `pytubefix`). | Yes (if `--file` is not used) |
| `--output` | `-o` | Directory where the chart will be saved. | Yes |
| `--config` | `-c` | Path to a custom `config.yaml`. | No |

### Python Library
You can also use YAAT within your own Python scripts:

```python
from yaat.pipeline import run

# Generate a chart
run(
    input_path="song.wav",
    output_dir="./my_chart",
    config_path="config.yaml"  # Optional
)
```

## Output
The tool generates a directory containing:
- `notes.chart` / `song.ini` (Rhythm game files)
- Separated audio stems (Guitar, Song)



