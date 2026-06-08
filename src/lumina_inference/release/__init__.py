"""Release tooling for LUMINA checkpoints.

Maintainer-only helpers for packaging trained LUMINA checkpoints and
uploading them to the Hugging Face Hub. Not required for inference; the
upload-side dependencies (huggingface_hub upload extras) are pulled in
via the optional ``[release]`` extra.

Copyright 2026 UChicago Argonne, LLC.
All rights reserved.
"""

from lumina_inference.release.model_card import (
    format_parameter_count,
    generate_model_card,
)
from lumina_inference.release.release_history import (
    ReleaseHistoryEntry,
    build_history,
    fetch_existing_readme,
    parse_history,
    render_history,
)

__all__ = [
    "ReleaseHistoryEntry",
    "build_history",
    "fetch_existing_readme",
    "format_parameter_count",
    "generate_model_card",
    "parse_history",
    "render_history",
]
