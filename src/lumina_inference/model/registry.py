"""Model registry and factory for dynamic model loading.

Vendored from lumina-core trainer/opf/utils — model resolution and
spec-building logic only. Training-specific utilities are excluded.

Copyright (c) 2026, Argonne National Laboratory
All rights reserved.
"""

from typing import Dict, Optional, Tuple, Type

import torch

from lumina_inference.model.hetero_model import HEAT, HGT, RGAT, OPFHeteroGNN

# ---------------------------------------------------------------------------
# Registry: canonical name → model class
# ---------------------------------------------------------------------------

HETERO_MODEL_CLASSES: Dict[str, Type[torch.nn.Module]] = {
    "HeteroGNN": OPFHeteroGNN,
    "RGAT": RGAT,
    "HEAT": HEAT,
    "HGT": HGT,
}

# Case-insensitive lookup table mapping various aliases to canonical names
_HETERO_MODEL_TYPE_LOOKUP: Dict[str, str] = {
    "heterognn": "HeteroGNN",
    "opfheterognn": "HeteroGNN",
    "rgat": "RGAT",
    "heat": "HEAT",
    "hgt": "HGT",
}


def _canonical_hetero_model_type(model_type: object) -> Optional[str]:
    """Normalize a model type string to its canonical form.

    Args:
        model_type: A string model type identifier (case-insensitive).

    Returns:
        The canonical model type string, or None if not recognized.
    """
    if not isinstance(model_type, str):
        return None
    return _HETERO_MODEL_TYPE_LOOKUP.get(model_type.strip().lower())


def resolve_hetero_model_type(
    model_type: Optional[str] = None,
    model_class_path: Optional[str] = None,
    default: str = "HeteroGNN",
) -> str:
    """Resolve a canonical model type from config fields.

    Checks ``model_class_path`` first (fully-qualified class name like
    ``"lumina.model.opf.hetero_model.HGT"``), then ``model_type``
    (short name like ``"HGT"``), then falls back to ``default``.

    Args:
        model_type: Short model type name from config (e.g. ``"HGT"``).
        model_class_path: Fully-qualified class path from config
            (e.g. ``"lumina.model.opf.hetero_model.HGT"``).
        default: Fallback model type if neither field resolves.

    Returns:
        Canonical model type string (e.g. ``"HGT"``, ``"HeteroGNN"``).

    Raises:
        ValueError: If the provided model type or class path is not
            recognized and no valid default exists.
    """
    if isinstance(model_class_path, str) and model_class_path.strip():
        class_name = model_class_path.rsplit(".", 1)[-1]
        normalized = _canonical_hetero_model_type(class_name)
        if normalized is not None:
            return normalized
        raise ValueError(
            f"Unsupported hetero model class path '{model_class_path}'. "
            f"Supported classes: {sorted(HETERO_MODEL_CLASSES.keys()) + ['OPFHeteroGNN']}"
        )

    if isinstance(model_type, str) and model_type.strip():
        normalized = _canonical_hetero_model_type(model_type)
        if normalized is not None:
            return normalized
        raise ValueError(
            f"Unsupported hetero model type '{model_type}'. "
            f"Supported types: {sorted(HETERO_MODEL_CLASSES.keys())}"
        )

    normalized_default = _canonical_hetero_model_type(default)
    if normalized_default is not None:
        return normalized_default

    supported = ", ".join(sorted(HETERO_MODEL_CLASSES.keys()))
    raise ValueError(
        f"Unable to resolve hetero model type from model_type='{model_type}' "
        f"and model_class_path='{model_class_path}'. Supported types: {supported}"
    )


def build_hetero_model_spec(
    model_type: str,
    metadata: dict,
    input_channels: dict,
    models_config: dict,
    out_channels: int = 2,
) -> Tuple[Type[torch.nn.Module], dict, dict, bool]:
    """Build a model class reference and constructor kwargs from config.

    Given a resolved model type, looks up the corresponding class from
    the registry and assembles the keyword arguments needed to
    instantiate it.

    Args:
        model_type: Canonical model type (e.g. ``"HGT"``). Should be
            the output of :func:`resolve_hetero_model_type`.
        metadata: Graph metadata dict with ``nodes`` and ``edges`` keys.
        input_channels: Dict mapping node types to input feature counts.
        models_config: The ``config.models`` section from the checkpoint
            config, containing per-model-type hyperparameters.
        out_channels: Number of output channels per target node type.

    Returns:
        A 4-tuple of:
        - ``model_class``: The model class to instantiate.
        - ``model_kwargs``: Dict of keyword arguments for the constructor.
        - ``model_config``: The raw config dict used (for diagnostics).
        - ``used_fallback``: True if the model-specific config section
          was missing and ``HeteroGNN`` config was used instead.

    Raises:
        ValueError: If ``model_type`` is not in the registry.
    """
    normalized_type = resolve_hetero_model_type(
        model_type=model_type, default=None
    )
    if normalized_type not in HETERO_MODEL_CLASSES:
        supported = ", ".join(sorted(HETERO_MODEL_CLASSES.keys()))
        raise ValueError(
            f"Unsupported hetero model type '{normalized_type}'. "
            f"Supported types: {supported}"
        )

    if not isinstance(models_config, dict):
        models_config = {}

    model_config = models_config.get(normalized_type)
    used_fallback = False
    if not isinstance(model_config, dict):
        fallback = models_config.get("HeteroGNN")
        if isinstance(fallback, dict):
            model_config = fallback
            used_fallback = normalized_type != "HeteroGNN"
        else:
            model_config = {}

    model_kwargs = {
        "metadata": metadata,
        "input_channels": input_channels,
    }
    model_kwargs.update(model_config)
    model_kwargs["out_channels"] = int(out_channels)

    # Set defaults for common parameters
    model_kwargs.setdefault("hidden_channels", 64)
    model_kwargs.setdefault("num_layers", 3)
    model_kwargs.setdefault("backend", "sage")

    # Architecture-specific defaults
    if normalized_type in {"RGAT", "HGT"}:
        model_kwargs.setdefault("num_heads", 1)
    if normalized_type == "HEAT":
        model_kwargs.setdefault("attention_heads", 1)

    return (
        HETERO_MODEL_CLASSES[normalized_type],
        model_kwargs,
        model_config,
        used_fallback,
    )
