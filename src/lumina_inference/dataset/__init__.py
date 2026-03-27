"""Dataset utilities for LUMINA inference."""

from lumina_inference.dataset.ingestion import (
    build_hetero_data,
    load_from_dict,
    load_from_hdf5,
    load_from_json_file,
    load_from_json_string,
    load_from_matpower,
    process_opf_dict,
)
from lumina_inference.dataset.opf_dataset import OPFDataset, process_json_file
from lumina_inference.dataset.validation import (
    FeatureDimensionError,
    LuminaIngestionError,
    MissingDataError,
    SchemaValidationError,
    detect_schema,
    validate_hetero_data,
    validate_opf_schema,
)


def __getattr__(name: str):
    """Lazy import for optional-dependency symbols."""
    if name == "process_hdf5_file":
        from lumina_inference.dataset.opf_dataset import process_hdf5_file

        return process_hdf5_file
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Dataset classes
    "OPFDataset",
    # Ingestion functions
    "build_hetero_data",
    "load_from_dict",
    "load_from_hdf5",
    "load_from_json_file",
    "load_from_json_string",
    "load_from_matpower",
    "process_hdf5_file",
    "process_json_file",
    "process_opf_dict",
    # Validation
    "detect_schema",
    "validate_hetero_data",
    "validate_opf_schema",
    # Exceptions
    "FeatureDimensionError",
    "LuminaIngestionError",
    "MissingDataError",
    "SchemaValidationError",
]
