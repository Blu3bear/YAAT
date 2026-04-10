"""Agentic orchestration layer for YAAT.

This module defines a two-agent flow:
1. InputResolutionAgent:
   - Accept local file input, or
   - Use YouTubeRetrievalAgent to search + download from a query.
2. ChartGenerationAgent:
   - Runs the existing classical YAAT pipeline on resolved audio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from yaat.agentic.retrieval import RetrievedAudio, YouTubeRetrievalAgent
from yaat.utils.logging import get_logger, log_stage, setup_logging


class AgenticInputError(ValueError):
    """Raised when CLI-provided inputs are invalid for agentic flow."""


@dataclass
class AgenticRunResult:
    """Output of an agentic YAAT run."""

    chart_dir: Path
    input_audio_path: Path
    source_type: str
    selected_title: Optional[str] = None
    selected_channel: Optional[str] = None


class InputResolutionAgent:
    """Resolve user input into a local audio file path for chart generation."""

    def __init__(self, retrieval_agent: Optional[YouTubeRetrievalAgent] = None):
        self.retrieval_agent = retrieval_agent or YouTubeRetrievalAgent()
        self.logger = get_logger()

    def resolve(
        self,
        *,
        file_path: Optional[str],
        search_term: Optional[str],
        working_dir: Path,
    ) -> tuple[Path, str, Optional[RetrievedAudio]]:
        """Resolve local-file or search-term input to an audio path.

        Returns:
            (resolved_audio_path, source_type, retrieved_audio_details)
        """
        has_file = bool(file_path)
        has_search = bool(search_term)

        if has_file == has_search:
            raise AgenticInputError(
                "Provide exactly one input source: --file/-f or --search/-s."
            )

        if has_file:
            resolved_path = Path(file_path).expanduser().resolve()
            if not resolved_path.exists():
                raise FileNotFoundError(f"Local input file does not exist: {resolved_path}")
            return resolved_path, "file", None

        retrieval_dir = working_dir / "_retrieval_audio"
        retrieved = self.retrieval_agent.retrieve(search_term or "", retrieval_dir)
        return retrieved.audio_path, "search", retrieved


class ChartGenerationAgent:
    """Run the existing classical YAAT pipeline on resolved audio."""

    def generate(self, *, audio_path: Path, output_dir: str, config_path: Optional[str]) -> Path:
        from yaat.pipeline import run as run_classical_pipeline

        return run_classical_pipeline(
            input_path=str(audio_path),
            output_dir=output_dir,
            config_path=config_path,
        )


class YAATAgenticOrchestrator:
    """High-level orchestrator coordinating retrieval and chart generation agents."""

    def __init__(
        self,
        input_agent: Optional[InputResolutionAgent] = None,
        chart_agent: Optional[ChartGenerationAgent] = None,
    ):
        self.input_agent = input_agent or InputResolutionAgent()
        self.chart_agent = chart_agent or ChartGenerationAgent()

    def run(
        self,
        *,
        file_path: Optional[str],
        search_term: Optional[str],
        output_dir: str,
        config_path: Optional[str] = None,
    ) -> AgenticRunResult:
        """Run the full agentic flow and return orchestration metadata."""
        logger = setup_logging(level=logging.INFO)

        log_stage("Agentic Input Resolution", logger)
        resolved_audio_path, source_type, retrieved = self.input_agent.resolve(
            file_path=file_path,
            search_term=search_term,
            working_dir=Path(output_dir),
        )

        if source_type == "file":
            logger.info("InputResolutionAgent selected local file: %s", resolved_audio_path)
        else:
            assert retrieved is not None
            logger.info("InputResolutionAgent retrieved audio: %s", resolved_audio_path)
            logger.info(
                "Selected YouTube result: '%s' by %s",
                retrieved.candidate.title,
                retrieved.candidate.channel,
            )

        log_stage("Classical Chart Generation Agent", logger)
        chart_dir = self.chart_agent.generate(
            audio_path=resolved_audio_path,
            output_dir=output_dir,
            config_path=config_path,
        )

        return AgenticRunResult(
            chart_dir=chart_dir,
            input_audio_path=resolved_audio_path,
            source_type=source_type,
            selected_title=(retrieved.candidate.title if retrieved else None),
            selected_channel=(retrieved.candidate.channel if retrieved else None),
        )


def run_agentic(
    *,
    file_path: Optional[str],
    search_term: Optional[str],
    output_dir: str,
    config_path: Optional[str] = None,
) -> AgenticRunResult:
    """Convenience wrapper for running agentic YAAT orchestration."""
    orchestrator = YAATAgenticOrchestrator()
    return orchestrator.run(
        file_path=file_path,
        search_term=search_term,
        output_dir=output_dir,
        config_path=config_path,
    )
