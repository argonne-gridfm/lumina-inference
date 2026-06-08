"""Dataset utilities for LUMINA inference."""

from lumina_inference.dataset.ingestion import (
    build_hetero_data,
    load_from_dict,
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

__all__ = [
    # Dataset classes
    "OPFDataset",
    # Ingestion functions
    "build_hetero_data",
    "load_from_dict",
    "load_from_json_file",
    "load_from_json_string",
    "load_from_matpower",
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
