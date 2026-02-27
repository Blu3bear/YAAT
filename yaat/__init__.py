"""YAAT - Yet Another Auto-charter Tool.

Generates playable Clone Hero / YARG chart files from audio.
"""


def generate_chart(*args, **kwargs):
    """Run the full YAAT pipeline. See yaat.pipeline.run for details."""
    from yaat.pipeline import run
    return run(*args, **kwargs)


__all__ = ["generate_chart"]
