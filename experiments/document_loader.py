"""
Document loading utilities for earnings call transcripts.

This module focuses on the NVDA transcripts included with the repo, but the
helpers are designed so that additional tickers / quarters can be added with
minimal friction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

from consensus_core import Document, normalize_transcript_text


def _strip_site_chrome(text: str) -> str:
    """
    Remove lightweight site chrome / disclaimers from a transcript.

    We aim to preserve speaker names and section headings while trimming obvious
    wrapper artifacts (e.g., ads, copyright statements).
    """

    lines = text.split("\n")
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue

        lower = stripped.lower()
        if any(
            token in lower
            for token in ("advertisement", "privacy policy", "terms of use")
        ):
            continue
        if stripped.startswith("<<<") and stripped.endswith(">>>"):
            # Defensive: drop stray delimiters if present.
            continue

        cleaned.append(stripped)

    collapsed = "\n".join(cleaned)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def load_transcript(path: Path) -> Document:
    """Load and normalize a single transcript JSON file."""

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    symbol = payload.get("symbol", "UNKNOWN")
    fiscal_quarter = payload.get("fiscal_quarter", "?")
    fiscal_year = payload.get("fiscal_year", "?")
    doc_id = f"{symbol}_{fiscal_quarter}_FY{fiscal_year}"

    raw_text = payload.get("transcript", "")
    normalized = normalize_transcript_text(raw_text)
    cleaned = _strip_site_chrome(normalized)

    metadata = {
        "symbol": symbol,
        "fiscal_quarter": fiscal_quarter,
        "fiscal_year": fiscal_year,
        "call_date": payload.get("call_date", "Unknown"),
        "source": payload.get("source", "Unknown"),
        "path": str(path),
    }

    return Document(doc_id=doc_id, text=cleaned, metadata=metadata)


def load_transcripts(paths: Iterable[Path]) -> List[Document]:
    """Load multiple transcripts."""

    documents: List[Document] = []
    for path in paths:
        documents.append(load_transcript(path))
    return documents
