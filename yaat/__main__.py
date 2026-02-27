"""CLI entry point for YAAT.

Usage:
    python -m yaat --input song.wav --output ./chart_dir/ [--config config.yaml]
"""

import argparse
import sys


def main() -> int:
    """Parse arguments and run the YAAT pipeline."""
    parser = argparse.ArgumentParser(
        prog="yaat",
        description="YAAT - Yet Another Auto-charter Tool. "
        "Generate Clone Hero / YARG charts from audio files.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input audio file (.wav)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output chart directory",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file (optional, uses defaults if omitted)",
    )

    args = parser.parse_args()

    # Import here to avoid slow imports when just checking --help
    from yaat.pipeline import run

    try:
        result_dir = run(
            input_path=args.input,
            output_dir=args.output,
            config_path=args.config,
        )
        print(f"\nChart generated successfully: {result_dir}")
        return 0
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
