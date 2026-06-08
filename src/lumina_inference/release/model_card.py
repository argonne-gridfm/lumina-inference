"""Model card generation for LUMINA Hugging Face releases.

Produces a populated Markdown model card from a loaded :class:`Modeler`
plus the originating training checkpoint. Architecture, parameter
counts, and input-channel schema are derived from the live model and
``config_data``; training provenance (epoch, val_loss, timestamp,
case_name) is read from the raw checkpoint dict.

The Markdown template ships as package data at
``lumina_inference.release._data/model_card_template.md`` (also visible
as ``docs/huggingface_model_card.md`` via a symlink in source checkouts)
and uses :py:meth:`str.format` placeholders.

Copyright 2026 UChicago Argonne, LLC.
All rights reserved.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch

from lumina_inference.modeler import Modeler


def _default_template_text() -> str:
    """Read the bundled model-card template via importlib.resources.

    Works in both source checkouts and installed wheels (where
    ``__file__`` based path walks would land outside the package).
    """
    return (
        resources.files("lumina_inference.release._data")
        .joinpath("model_card_template.md")
        .read_text(encoding="utf-8")
    )


def format_parameter_count(num_params: int) -> str:
    """Format a parameter count as a human-readable string.

    Args:
        num_params: Total number of parameters.

    Returns:
        Compact string such as ``"671.0M"`` or ``"1.0B"``.
    """
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.1f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    return str(num_params)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return ``(total_params, trainable_params)`` for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _format_input_channels(input_channels: Mapping[str, Any]) -> str:
    """Render the input_channels mapping as a Markdown bullet list.

    Accepts either ``{node_type: int}`` (feature count) or
    ``{node_type: list[str]}`` (named feature columns).
    """
    if not input_channels:
        return "- _No input channel metadata available_"

    lines = []
    for node_type, value in input_channels.items():
        label = node_type.replace("_", " ").title()
        if isinstance(value, int):
            lines.append(f"- **{label}**: {value} features")
        elif isinstance(value, (list, tuple)):
            lines.append(f"- **{label}**: {len(value)} features")
        elif isinstance(value, Mapping):
            lines.append(f"- **{label}**: {len(value)} features")
        else:
            lines.append(f"- **{label}**: {value}")
    return "\n".join(lines)


def _resolve_architecture(modeler: Modeler, source: Mapping[str, Any]) -> str:
    """Best-effort architecture label (e.g. ``"HGT"``).

    Accepts either a normalized provenance dict (with ``model_class``)
    or a raw checkpoint (with ``model_class`` / ``model_name`` /
    ``model_kwargs``).
    """
    if modeler.config_data is not None:
        models_cfg = modeler.config_data.get("config", {}).get("models", {})
        for key in ("HGT", "HeteroGNN", "RGAT", "HEAT"):
            if key in models_cfg:
                return key.upper()

    arch = source.get("model_class") or source.get("model_name")
    if isinstance(arch, str):
        return arch.split(".")[-1].upper()

    if modeler.model is not None:
        return type(modeler.model).__name__.upper()
    return "UNKNOWN"


def _resolve_input_channels(
    modeler: Modeler, source: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Prefer the loaded config; fall back to source-provided values."""
    if modeler.config_data is not None:
        ic = modeler.config_data.get("input_channels")
        if ic:
            return ic
    ic = source.get("input_channels")
    if ic:
        return ic
    return source.get("model_kwargs", {}).get("input_channels", {}) or {}


def _resolve_training_case(source: Mapping[str, Any]) -> str:
    """Render training case(s) for the card.

    Handles the lumina trainer's ``case_names`` (plural list) and the
    older ``case_name`` (singular string) seen in test fixtures.
    """
    case_names = source.get("case_names")
    if case_names:
        if isinstance(case_names, str):
            return case_names
        return ", ".join(str(c) for c in case_names)
    case_name = source.get("case_name")
    if case_name:
        return str(case_name)
    return "Unknown"


def _resolve_val_loss(source: Mapping[str, Any]) -> str:
    val_loss = source.get("val_loss")
    if val_loss is None:
        val_loss = source.get("best_val_loss")
    if val_loss is None:
        return "Unknown"
    return f"{float(val_loss):.6f}"


def generate_model_card(
    modeler: Modeler,
    source: Mapping[str, Any],
    model_name: str,
    hf_repo_id: str,
    *,
    model_version: str = "unreleased",
    commit_sha: str = "pending",
    release_history: str = "_No prior releases recorded._",
    training_data_size: str = "15,000 samples per case",
    template_path: Optional[Path] = None,
) -> str:
    """Render a populated Hugging Face model card.

    Args:
        modeler: A :class:`Modeler` that has already had ``load_model``
            invoked. Used to derive architecture, parameter counts, and
            input channel schema from the live model.
        source: Either a *normalized provenance* dict (as produced by
            :meth:`HFUploader._normalize_checkpoint`) or a raw training
            checkpoint. Used for training provenance fields. Recognized
            keys (first non-empty wins for each field):

            * case(s): ``case_names`` (preferred, list) or ``case_name``
            * val loss: ``val_loss`` or ``best_val_loss``
            * epoch: ``epoch``
            * date:  ``timestamp``
            * arch:  ``model_class`` or ``model_name``

        model_name: Public release name, e.g. ``"LUMINA-2M"``.
        hf_repo_id: Hugging Face repo identifier, e.g.
            ``"argonne/LUMINA-2M"``.
        model_version: Release version tag, e.g. ``"v0.2.0"``.
        commit_sha: Short commit SHA for the upload (filled in after the
            push for the next release's table; placeholder otherwise).
        release_history: Pre-rendered Markdown table for the
            ``Release History`` section. Use
            :func:`lumina_inference.release.release_history.build_history`.
        training_data_size: Free-text description of the training set
            size. Defaults to the historical LUMINA-1B value.
        template_path: Override path to the model-card Markdown
            template. Defaults to the bundled template at
            ``lumina_inference.release._data/model_card_template.md``.

    Returns:
        The fully populated Markdown model card as a string.

    Raises:
        RuntimeError: If ``modeler.model`` is ``None`` (i.e.
            ``load_model`` has not been called).
        FileNotFoundError: If the template path does not exist.
    """
    if modeler.model is None:
        raise RuntimeError(
            "Modeler.model is None; call Modeler.load_model() before "
            "generating a model card."
        )

    if template_path is None:
        template = _default_template_text()
    else:
        template = Path(template_path).read_text(encoding="utf-8")

    total_params, trainable_params = count_parameters(modeler.model)
    arch = _resolve_architecture(modeler, source)
    input_channels = _resolve_input_channels(modeler, source)

    return template.format(
        model_name=model_name,
        hf_repo_id=hf_repo_id,
        model_version=model_version,
        commit_sha=commit_sha,
        release_history=release_history,
        model_architecture=arch,
        total_parameters=f"{total_params:,}",
        total_parameters_formatted=format_parameter_count(total_params),
        trainable_parameters=f"{trainable_params:,}",
        trainable_parameters_formatted=format_parameter_count(trainable_params),
        training_case=_resolve_training_case(source),
        training_data_size=training_data_size,
        training_date=str(source.get("timestamp", "Unknown")),
        final_validation_loss=_resolve_val_loss(source),
        training_epochs=str(source.get("epoch", "Unknown")),
        input_channels=_format_input_channels(input_channels),
    )
