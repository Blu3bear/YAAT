"""CLI entry point for YAAT.

Usage:
    python -m yaat -f song.wav --output ./chart_dir/ [--config config.yaml]
    python -m yaat -s "artist song title" --output ./chart_dir/ [--config config.yaml]
"""

import argparse
import sys


def main() -> int:
    """Parse arguments and run the YAAT agentic orchestration flow."""
    parser = argparse.ArgumentParser(
        prog="yaat",
        description="YAAT - Yet Another Auto-charter Tool. "
        "Generate Clone Hero / YARG charts from local files or YouTube search.",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--file", "-f",
        dest="file_path",
        help="Path to local input audio file",
    )
    source_group.add_argument(
        "--search", "-s",
        dest="search_term",
        help="Search term for YouTube retrieval agent (example: 'radiohead karma police')",
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

    # Import here to avoid heavy imports when just checking --help
    from yaat.agentic import run_agentic

    try:
        result = run_agentic(
            file_path=args.file_path,
            search_term=args.search_term,
            output_dir=args.output,
            config_path=args.config,
        )

        if result.source_type == "search":
            print(
                "\nRetrieved from YouTube:" 
                f" {result.selected_title} (channel: {result.selected_channel})"
            )
            print(f"Resolved audio path: {result.input_audio_path}")

        print(f"\nChart generated successfully: {result.chart_dir}")
        return 0
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
