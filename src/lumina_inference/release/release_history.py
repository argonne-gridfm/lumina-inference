"""Maintain a running release-history table inside the HF model card.

The table is rendered into the ``{release_history}`` placeholder of
``docs/huggingface_model_card.md``. Each release prepends a row; the
function will (best-effort) fetch the existing ``README.md`` from the
target HF repo and parse out previous rows so historical entries are
preserved.

Network access is optional: when the Hub is unreachable or the repo
does not yet exist, history starts fresh.

Copyright 2026 UChicago Argonne, LLC.
All rights reserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence


HISTORY_HEADER = (
    "| Version | Date | Epochs | Val Loss | Training Case | Commit |\n"
    "|---------|------|--------|----------|---------------|--------|"
)

# Match the entire "## Release History" section: header + blank lines +
# optional Markdown table rows, stopping at the next "## " heading or EOF.
_HISTORY_SECTION_RE = re.compile(
    r"##\s+Release History\s*\n(?P<body>.*?)(?=\n##\s|\Z)",
    re.DOTALL,
)

# Match a single data row of the history table.
_HISTORY_ROW_RE = re.compile(
    r"^\|\s*(?P<version>[^|]+?)\s*"
    r"\|\s*(?P<date>[^|]+?)\s*"
    r"\|\s*(?P<epochs>[^|]+?)\s*"
    r"\|\s*(?P<val_loss>[^|]+?)\s*"
    r"\|\s*(?P<training_case>[^|]+?)\s*"
    r"\|\s*(?P<commit>[^|]+?)\s*\|\s*$",
    re.MULTILINE,
)


@dataclass
class ReleaseHistoryEntry:
    """A single row in the release-history table."""

    version: str
    date: str
    epochs: str
    val_loss: str
    training_case: str
    commit: str

    def as_row(self) -> str:
        return (
            f"| {self.version} | {self.date} | {self.epochs} "
            f"| {self.val_loss} | {self.training_case} | {self.commit} |"
        )


def _is_header_or_separator(row: re.Match) -> bool:
    """Skip the header row and the |---| separator row when parsing."""
    version = row.group("version").strip().lower()
    if version in {"version", ""}:
        return True
    if set(version) <= {"-", ":"}:
        return True
    return False


def parse_history(markdown: str) -> List[ReleaseHistoryEntry]:
    """Extract release-history rows from an existing model card.

    Args:
        markdown: Full text of an existing ``README.md``.

    Returns:
        List of entries, in the order they appear in the source (newest
        first by convention; this function does not re-sort).
    """
    section = _HISTORY_SECTION_RE.search(markdown)
    if not section:
        return []

    entries: List[ReleaseHistoryEntry] = []
    for row in _HISTORY_ROW_RE.finditer(section.group("body")):
        if _is_header_or_separator(row):
            continue
        entries.append(
            ReleaseHistoryEntry(
                version=row.group("version").strip(),
                date=row.group("date").strip(),
                epochs=row.group("epochs").strip(),
                val_loss=row.group("val_loss").strip(),
                training_case=row.group("training_case").strip(),
                commit=row.group("commit").strip(),
            )
        )
    return entries


def render_history(entries: Sequence[ReleaseHistoryEntry]) -> str:
    """Render an entries list as a complete Markdown table."""
    if not entries:
        return "_No prior releases recorded._"

    lines = [HISTORY_HEADER]
    lines.extend(e.as_row() for e in entries)
    return "\n".join(lines)


def fetch_existing_readme(
    repo_id: str, *, token: Optional[str] = None
) -> Optional[str]:
    """Download the current ``README.md`` for a repo, or ``None`` on failure.

    Returns ``None`` if the repo does not exist, has no README, or the
    Hub is unreachable — callers should treat this as "no history yet".
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import (
            EntryNotFoundError,
            RepositoryNotFoundError,
        )
    except ImportError:
        return None

    try:
        path = hf_hub_download(
            repo_id=repo_id, filename="README.md", token=token
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except Exception:
        # Network errors, auth problems — degrade gracefully.
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def build_history(
    new_entry: ReleaseHistoryEntry,
    *,
    existing_readme: Optional[str] = None,
) -> str:
    """Prepend ``new_entry`` to history parsed from ``existing_readme``.

    Entries with a version matching ``new_entry.version`` in the
    existing card are dropped (the new entry supersedes them) — this
    keeps re-uploads idempotent.

    Args:
        new_entry: The release being published.
        existing_readme: Prior ``README.md`` content from the Hub, or
            ``None`` if no prior release exists.

    Returns:
        Rendered Markdown table for the ``{release_history}``
        placeholder.
    """
    prior: List[ReleaseHistoryEntry] = (
        parse_history(existing_readme) if existing_readme else []
    )
    prior = [e for e in prior if e.version != new_entry.version]
    return render_history([new_entry, *prior])
