"""YouTube retrieval agent for YAAT.

This agent:
1. Searches YouTube from a free-text query.
2. Scores and ranks likely matches.
3. Prompts for disambiguation if top results are similarly confident.
4. Downloads audio via pytubefix.
5. Converts to a YAAT-compatible audio format when needed.
"""

from __future__ import annotations

import shutil
import subprocess
import re
import sys
import warnings
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    from pytubefix import Search, YouTube
except Exception:  # pragma: no cover - handled at runtime with a clear error
    Search = None  # type: ignore[assignment]
    YouTube = None  # type: ignore[assignment]

from yaat.utils.logging import get_logger

SUPPORTED_PIPELINE_EXTENSIONS = {".wav", ".ogg", ".mp3"}


class RetrievalError(RuntimeError):
    """Raised when search, selection, or download fails."""


@dataclass
class SearchCandidate:
    """Candidate song result returned by the retrieval agent."""

    title: str
    url: str
    channel: str
    length_s: int
    score: float


@dataclass
class RetrievedAudio:
    """Final resolved audio for downstream chart generation."""

    audio_path: Path
    candidate: SearchCandidate


def _normalize_text(text: str) -> str:
    """Normalize text for lightweight string matching."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(cleaned.split())


def _safe_filename(text: str) -> str:
    """Create a filesystem-safe filename stem."""
    text = re.sub(r"[\\/:*?\"<>|]", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:140] if text else "youtube_audio"


def _format_duration(seconds: int) -> str:
    """Format seconds as mm:ss or hh:mm:ss."""
    if seconds <= 0:
        return "unknown"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _score_candidate(query: str, title: str) -> float:
    """Compute a confidence score for how well a title matches the query."""
    q_norm = _normalize_text(query)
    t_norm = _normalize_text(title)

    if not q_norm or not t_norm:
        return 0.0

    ratio = SequenceMatcher(None, q_norm, t_norm).ratio()

    q_tokens = set(q_norm.split())
    t_tokens = set(t_norm.split())
    token_overlap = (len(q_tokens & t_tokens) / len(q_tokens)) if q_tokens else 0.0

    contains_bonus = 0.1 if q_norm in t_norm else 0.0
    cover_penalty = 0.08 if ("cover" in t_norm and "cover" not in q_norm) else 0.0

    score = 0.55 * ratio + 0.45 * token_overlap + contains_bonus - cover_penalty
    return max(0.0, min(1.0, score))


class YouTubeRetrievalAgent:
    """Retrieval agent that resolves a search term to downloadable audio."""

    def __init__(
        self,
        max_results: int = 8,
        ambiguity_delta: float = 0.08,
        max_presented: int = 5,
        interactive: bool = True,
    ):
        self.max_results = max_results
        self.ambiguity_delta = ambiguity_delta
        self.max_presented = max_presented
        self.interactive = interactive
        self.logger = get_logger()

    @staticmethod
    def _ensure_pytubefix_available() -> tuple[Any, Any]:
        """Return pytubefix classes or raise with install guidance."""
        if Search is None or YouTube is None:
            raise RetrievalError(
                "pytubefix is not available. Install it with: pip install pytubefix"
            )
        return Search, YouTube

    def search(self, query: str) -> list[SearchCandidate]:
        """Search YouTube and return ranked candidates."""
        search_cls, _ = self._ensure_pytubefix_available()

        try:
            search = search_cls(query)
            raw_results = list(search.results or [])[: self.max_results]
        except Exception as exc:
            raise RetrievalError(f"YouTube search failed: {exc}") from exc

        if not raw_results:
            raise RetrievalError(f"No YouTube results found for query: '{query}'")

        candidates: list[SearchCandidate] = []
        for item in raw_results:
            title = str(getattr(item, "title", "") or "").strip()
            channel = str(getattr(item, "author", "Unknown channel") or "Unknown channel")
            length_s = int(getattr(item, "length", 0) or 0)

            url = str(getattr(item, "watch_url", "") or "").strip()
            if not url:
                video_id = str(getattr(item, "video_id", "") or "").strip()
                if video_id:
                    url = f"https://www.youtube.com/watch?v={video_id}"

            if not title or not url:
                continue

            candidates.append(
                SearchCandidate(
                    title=title,
                    url=url,
                    channel=channel,
                    length_s=length_s,
                    score=_score_candidate(query, title),
                )
            )

        if not candidates:
            raise RetrievalError("Search returned results but none could be parsed safely.")

        candidates.sort(key=lambda c: c.score, reverse=True)
        self.logger.info("RetrievalAgent found %d candidate(s) for '%s'", len(candidates), query)
        return candidates

    def _select_candidate(self, query: str, ranked: list[SearchCandidate]) -> SearchCandidate:
        """Select the best candidate, prompting the user if confidence is ambiguous."""
        top = ranked[0]
        ambiguous = [c for c in ranked if (top.score - c.score) <= self.ambiguity_delta]
        ambiguous = ambiguous[: self.max_presented]

        if len(ambiguous) < 2:
            self.logger.info("RetrievalAgent selected top result with confidence %.2f", top.score)
            return top

        if not self.interactive or not sys.stdin.isatty():
            self.logger.info(
                "Multiple similarly scored results found for '%s'; non-interactive mode selected top result.",
                query,
            )
            return top

        print("\nMultiple likely matches found. Select the intended song:")
        for idx, candidate in enumerate(ambiguous, start=1):
            print(
                f"  {idx}. {candidate.title} | {candidate.channel} | "
                f"{_format_duration(candidate.length_s)} | confidence={candidate.score:.2f}"
            )

        while True:
            choice = input(f"Enter 1-{len(ambiguous)} (default 1): ").strip()
            if not choice:
                return ambiguous[0]
            if choice.isdigit():
                selected = int(choice)
                if 1 <= selected <= len(ambiguous):
                    return ambiguous[selected - 1]
            print("Invalid selection. Please try again.")

    def _download_audio_stream(self, candidate: SearchCandidate, download_dir: Path) -> tuple[Path, str]:
        """Download the highest quality audio stream for a selected candidate."""
        _, youtube_cls = self._ensure_pytubefix_available()

        try:
            yt = youtube_cls(candidate.url)
        except Exception as exc:
            raise RetrievalError(f"Failed to initialize YouTube object: {exc}") from exc

        stream = yt.streams.get_audio_only()

        if stream is None:
            raise RetrievalError("No downloadable audio stream found for selected YouTube result.")

        download_dir.mkdir(parents=True, exist_ok=True)
        base_name = _safe_filename(f"{yt.title}_{yt.video_id}")

        try:
            downloaded = Path(
                stream.download(
                    output_path=str(download_dir),
                    filename=base_name+".m4a",
                )
            )
        except Exception as exc:
            raise RetrievalError(f"Audio download failed: {exc}") from exc

        self.logger.info("Downloaded audio stream to %s", downloaded)
        return downloaded, base_name

    def _resolve_ffmpeg_executable(self) -> str | None:
        """Resolve ffmpeg executable from PATH or imageio-ffmpeg."""
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path

        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    def _convert_with_ffmpeg(self, source_path: Path, target_path: Path) -> str | None:
        """Try ffmpeg conversion and return None on success, or error text on failure."""
        ffmpeg_exe = self._resolve_ffmpeg_executable()
        if not ffmpeg_exe:
            return "ffmpeg executable not found (PATH and imageio-ffmpeg unavailable)"

        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(target_path),
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except Exception as exc:
            return f"ffmpeg execution failed: {exc}"

        if result.returncode == 0 and target_path.exists():
            return None

        stderr_tail = (result.stderr or "").strip()
        if len(stderr_tail) > 800:
            stderr_tail = stderr_tail[-800:]
        return f"ffmpeg returned code {result.returncode}: {stderr_tail}"

    def _convert_to_wav(self, source_path: Path, target_path: Path) -> None:
        """Convert a downloaded file to WAV.

        Conversion strategy:
            1) ffmpeg (PATH or imageio-ffmpeg binary)
            2) torchaudio
            3) librosa
        """
        errors: list[str] = []

        ffmpeg_error = self._convert_with_ffmpeg(source_path, target_path)
        if ffmpeg_error is None:
            return
        errors.append(ffmpeg_error)

        try:
            import soundfile as sf
        except Exception as soundfile_exc:
            errors.append(f"soundfile unavailable for fallback writers: {soundfile_exc}")
            raise RetrievalError(
                "Downloaded audio could not be converted to WAV. "
                + " | ".join(errors)
            ) from soundfile_exc

        torchaudio_error: Exception | None = None

        try:
            import torchaudio

            backends = []
            try:
                backends = list(torchaudio.list_audio_backends())
            except Exception:
                backends = []

            if "ffmpeg" in backends:
                waveform, sample_rate = torchaudio.load(str(source_path), backend="ffmpeg")
            else:
                waveform, sample_rate = torchaudio.load(str(source_path))

            sf.write(str(target_path), waveform.detach().cpu().numpy().T, sample_rate)
            return
        except Exception as exc:
            torchaudio_error = exc
            errors.append(f"torchaudio error: {exc}")

        try:
            import librosa

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                audio, sample_rate = librosa.load(str(source_path), sr=None, mono=False)
            if getattr(audio, "ndim", 1) == 1:
                sf.write(str(target_path), audio, sample_rate)
            else:
                sf.write(str(target_path), audio.T, sample_rate)
            return
        except Exception as librosa_exc:
            errors.append(f"librosa error: {librosa_exc}")
            raise RetrievalError(
                "Downloaded audio could not be converted to WAV. "
                + " | ".join(errors)
            ) from librosa_exc

    def _prepare_pipeline_audio(self, source_path: Path, base_name: str, download_dir: Path) -> Path:
        """Return a YAAT-compatible file path for the classical pipeline."""
        if source_path.suffix.lower() in SUPPORTED_PIPELINE_EXTENSIONS:
            return source_path

        wav_path = download_dir / f"{base_name}.wav"
        self._convert_to_wav(source_path, wav_path)
        self.logger.info("Converted downloaded audio to %s", wav_path)
        return wav_path

    def retrieve(self, query: str, download_dir: Path) -> RetrievedAudio:
        """Resolve a search query to a local YAAT-compatible audio file."""
        ranked = self.search(query)
        selected = self._select_candidate(query, ranked)

        self.logger.info(
            "RetrievalAgent selected: '%s' by %s (confidence %.2f)",
            selected.title,
            selected.channel,
            selected.score,
        )

        downloaded_path, base_name = self._download_audio_stream(selected, download_dir)
        prepared_path = self._prepare_pipeline_audio(downloaded_path, base_name, download_dir)

        return RetrievedAudio(audio_path=prepared_path, candidate=selected)
