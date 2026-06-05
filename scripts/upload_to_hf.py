#!/usr/bin/env python
"""Thin CLI wrapper around lumina_inference.release.uploader.

Allows running the uploader directly from a source checkout without
installing the package. The ``--version`` flag is required for every
invocation (including ``--dry-run``) so the rendered model card and
``Release History`` table reflect the intended release tag.

Examples
--------

Dry-run (stage artifacts locally, skip repo creation, upload, tag, and
collection join — useful for previewing ``README.md`` and ``LICENSE``
before publishing):

    python scripts/upload_to_hf.py \
        --checkpoint /path/to/checkpoint.pt \
        --repo argonne/LUMINA-2M \
        --version v0.1.0 \
        --staging-dir ./hf_release \
        --dry-run

Real publish (push to HF, create immutable git tag, add repo to
Collection):

    python scripts/upload_to_hf.py \
        --checkpoint /path/to/checkpoint.pt \
        --repo argonne/LUMINA-2M \
        --version v0.2.0 \
        --collection-slug argonne/lumina-models-xxxx

Version tags must match ``vMAJOR.MINOR.PATCH`` with an optional
pre-release suffix (e.g. ``v0.2.0-rc1``); see Option B in
``docs/hf_release_strategy.md``.

For an installed package, prefer the ``lumina-upload`` console script
provided by ``pyproject.toml`` — it takes the same flags.

Copyright (c) 2026, UChicago Argonne, LLC
All rights reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a source checkout without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lumina_inference.release.uploader import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
