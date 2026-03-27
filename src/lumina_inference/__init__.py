"""
lumina-inference: Lightweight inference package for LUMINA trained models.

Load models from Hugging Face and run predictions without lumina-core.

Copyright (c) 2025, UChicago Argonne, LLC
All rights reserved.
"""

__version__ = "0.1.0"

from lumina_inference.dataset.ingestion import (
    build_hetero_data,
    load_from_dict,
    load_from_hdf5,
    load_from_json_file,
    load_from_json_string,
    load_from_matpower,
    process_opf_dict,
)
from lumina_inference.dataset.opf_dataset import process_json_file
from lumina_inference.modeler import Modeler


def __getattr__(name: str):
    """Lazy import for optional-dependency symbols (e.g. h5py)."""
    if name == "process_hdf5_file":
        from lumina_inference.dataset.opf_dataset import process_hdf5_file

        return process_hdf5_file
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Modeler",
    "__version__",
    # Ingestion functions (top-level convenience)
    "build_hetero_data",
    "load_from_dict",
    "load_from_hdf5",
    "load_from_json_file",
    "load_from_json_string",
    "load_from_matpower",
    "process_hdf5_file",
    "process_json_file",
    "process_opf_dict",
]
