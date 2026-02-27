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
python -m yaat --input path/to/song.wav --output ./output_chart/
```

| Argument | Short | Description | Required |
|----------|-------|-------------|----------|
| `--input` | `-i` | Path to the input `.wav` audio file. | Yes |
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



