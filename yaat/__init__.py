"""YAAT - Yet Another Auto-charter Tool.

Generates playable Clone Hero / YARG chart files from audio.
"""


def generate_chart(*args, **kwargs):
    """Run the full YAAT pipeline. See yaat.pipeline.run for details."""
    from yaat.pipeline import run
    return run(*args, **kwargs)


def generate_chart_agentic(*args, **kwargs):
    """Run the agentic YAAT orchestration flow."""
    from yaat.agentic import run_agentic
    return run_agentic(*args, **kwargs)


__all__ = ["generate_chart", "generate_chart_agentic"]
