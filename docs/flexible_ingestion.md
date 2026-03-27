# Flexible Ingestion Guide

This document describes the flexible ingestion capabilities added to `lumina-inference`, enabling users to interact with LUMINA models in multiple ways beyond the standard `OPFDataset` pipeline.

## Overview

Previously, running inference required the full `OPFDataset` → `DataLoader` pipeline, which downloads, extracts, and processes OPF JSON files from the GridOpt dataset. The flexible ingestion layer adds:

1. **Multiple OPF input formats** — Python dicts, JSON strings, JSON files, HDF5 files, and MATPOWER `.m` files
2. **Single-sample prediction** — `Modeler.predict_single()` bypasses the DataLoader entirely
3. **Generic heterogeneous graph support** — `build_hetero_data()` for non-OPF use cases
4. **Schema detection and validation** — automatic OPF vs. generic routing with informative errors

## Quick Start: End-to-End Prediction

```python
import torch
from lumina_inference import Modeler, load_from_dict

# 1. Initialize the modeler (once)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modeler = Modeler(device)
modeler.from_pretrained("argonne/LUMINA-OPF-HGNN")

# 2. Load data from any source
data = load_from_dict(opf_dict)

# 3. Predict (single sample — no DataLoader needed)
predictions = modeler.predict_single(data)
bus_va_vm = predictions["bus"]           # [num_buses, 2]  (va, vm)
gen_pg_qg = predictions["generator"]    # [num_gens, 2]   (pg, qg)
```

## Ingestion Functions

All ingestion functions are available at the top level:

```python
from lumina_inference import (
    load_from_dict,
    load_from_hdf5,
    load_from_json_file,
    load_from_json_string,
    load_from_matpower,
    process_json_file,
    process_hdf5_file,
    process_opf_dict,
    build_hetero_data,
)
```

### `load_from_dict(data_dict, require_solution=False)`

Convert a Python dictionary in OPF format to a PyTorch Geometric `HeteroData` object.

```python
opf_dict = {
    "grid": {
        "context": [[[100.0]]],  # baseMVA
        "nodes": {
            "bus": [[69.0, 1, 0.94, 1.06], [69.0, 2, 0.94, 1.06]],
            "generator": [[100, 0.4, 0.0, 3.32, 0.0, -0.1, 0.1, 1.06, 0.0, 20.0, 0.0]],
            "load": [[0.217, 0.127]],
            "shunt": [[0.19, 0.0]],
        },
        "edges": {
            "ac_line": {
                "senders": [0],
                "receivers": [1],
                "features": [[-0.52, 0.52, 0.026, 0.026, 0.019, 0.059, 0.0, 0.0, 0.0]],
            },
            "transformer": {"senders": [], "receivers": [], "features": []},
            "generator_link": {"senders": [0], "receivers": [0]},
            "load_link": {"senders": [0], "receivers": [0]},
            "shunt_link": {"senders": [1], "receivers": [0]},
        },
    },
    "metadata": {"objective": 0.0},
}

data = load_from_dict(opf_dict)
predictions = modeler.predict_single(data)
# predictions["bus"].shape → [2, 2]  (va, vm per bus)
# predictions["generator"].shape → [1, 2]  (pg, qg per generator)
```

**Parameters:**
- `data_dict` — Python dict in OPF format (must contain `grid` and `metadata` keys)
- `require_solution` — If `True`, raises `MissingDataError` when the `solution` key is absent

### `load_from_json_file(path, require_solution=False)`

Load an OPF JSON file and convert to `HeteroData`.

```python
data = load_from_json_file("path/to/case14_sample.json")
predictions = modeler.predict_single(data)
```

### `load_from_json_string(json_str, require_solution=False)`

Parse a JSON string and convert to `HeteroData`.

```python
import json
data = load_from_json_string(json.dumps(opf_dict))
predictions = modeler.predict_single(data)
```

### `load_from_hdf5(path, n_jobs=1)`

Load OPF data from an HDF5 file and convert to a list of `HeteroData` objects. Each scenario in the HDF5 file becomes a separate `HeteroData`. Requires `h5py` and `pydantic` (install with `pip install lumina-inference[hdf5]`).

```python
from lumina_inference import load_from_hdf5

# Load all scenarios from an HDF5 file
data_list = load_from_hdf5("path/to/scenarios.h5")

# Predict on the first scenario
predictions = modeler.predict_single(data_list[0])
print(predictions["bus"].shape)       # [num_buses, 2]
print(predictions["generator"].shape) # [num_gens, 2]

# Predict on all scenarios
for i, data in enumerate(data_list):
    preds = modeler.predict_single(data)
    print(f"Scenario {i}: bus={preds['bus'].shape}, gen={preds['generator'].shape}")
```

**Parameters:**
- `path` — Path to the HDF5 (`.h5`) file
- `n_jobs` — Number of parallel jobs for processing scenarios. Use `1` for sequential (default), `-1` for all available cores.

**Parallel processing for large files:**

```python
# Use all CPU cores for faster processing
data_list = load_from_hdf5("large_dataset.h5", n_jobs=-1)
```

**Batch prediction with DataLoader:**

```python
from torch_geometric.loader import DataLoader

data_list = load_from_hdf5("scenarios.h5")
loader = DataLoader(data_list, batch_size=32)
all_preds = modeler.run_predictions(loader)
```

### `process_hdf5_file(h5_file, n_jobs=1)`

Lower-level function that reads an HDF5 file and converts each scenario to `HeteroData`. This is the function that `load_from_hdf5` delegates to. Available via lazy import (requires `h5py`).

```python
from lumina_inference import process_hdf5_file

data_list = process_hdf5_file("path/to/scenarios.h5", n_jobs=4)
```

**HDF5 vs JSON schema alignment:**

HDF5 files may store features in a different column order than JSON files. The pydantic-based schema models in `lumina_inference.dataset.schema` handle automatic alignment:

| Component | JSON Schema | HDF5 Schema |
|---|---|---|
| Bus | `base_kv, bus_type, vmin, vmax` | `vmin, vmax, zone, area, bus_type` |
| Generator | `mbase, pg, pmin, pmax, qg, qmin, qmax, vg, ...` | `pg, qg, pmax, pmin, qmax, qmin, vg, mbase, ...` |

The `_align_features()` function uses `get_alignment_map()` from the schema models to remap indices automatically.

### `load_from_matpower(path)`

Load a MATPOWER `.m` case file and convert to `HeteroData`. Requires `pandapower` to be installed.

```python
from lumina_inference import load_from_matpower

data = load_from_matpower("path/to/case14.m")
predictions = modeler.predict_single(data)
```

This function:
1. Parses the `.m` file using regex-based matrix extraction
2. Converts to pandapower format for bus ID mapping and per-unit conversion
3. Assembles the OPF dict structure
4. Calls `process_opf_dict()` to produce `HeteroData`

### `process_opf_dict(obj, require_solution=False)`

Core converter from OPF dict to `HeteroData`. All `load_from_*` functions delegate to this.

```python
data = process_opf_dict(opf_dict, require_solution=False)
predictions = modeler.predict_single(data)
```

**Key behaviors:**
- Bus type column (index 1) is one-hot encoded: `[1=PQ, 2=PV, 3=ref, 4=isolated]` → 4 columns
- Final bus features: `[base_kv, vmin, vmax, pq, pv, ref, isolated]` (7 features)
- All tensors are created with `dtype=torch.float32`
- When `require_solution=False` and no solution is present, `.y` and `.edge_label` are not set
- `baseMVA` is stored as a scalar attribute on the `HeteroData` object

### `process_json_file(json_file, require_solution=True)`

Backward-compatible wrapper. Reads a JSON file and delegates to `process_opf_dict()`.

```python
from lumina_inference.dataset.opf_dataset import process_json_file

# Training mode (default) — requires solution
data = process_json_file("path/to/case.json")

# Inference mode — solution optional
data = process_json_file("path/to/case.json", require_solution=False)
predictions = modeler.predict_single(data)
```

## Single-Sample Prediction

### `Modeler.predict_single(data, minmax_scaling=True, validate=True)`

Run prediction on a single `HeteroData` object without a DataLoader.

```python
from lumina_inference import Modeler, load_from_dict

device = torch.device("cpu")
modeler = Modeler(device)
modeler.from_pretrained("argonne/LUMINA-OPF-HGNN")

data = load_from_dict(opf_dict)
predictions = modeler.predict_single(data)

print(predictions["bus"].shape)       # [n_bus, 2]
print(predictions["generator"].shape) # [n_gen, 2]
```

**Parameters:**
- `data` — A `HeteroData` object (from any ingestion function)
- `minmax_scaling` — Apply OPF-specific sigmoid-bounded scaling (default `True`)
- `validate` — Validate data against the loaded model's schema (default `True`)

Internally, this wraps the data in `Batch.from_data_list([data])` and delegates to `predict_batch()`.

## Batch Prediction

For multiple samples, use a `DataLoader` with `run_predictions()`:

```python
from torch_geometric.loader import DataLoader
from lumina_inference import load_from_hdf5

# Load many scenarios at once
data_list = load_from_hdf5("scenarios.h5", n_jobs=-1)

# Batch predict
loader = DataLoader(data_list, batch_size=32)
all_preds = modeler.run_predictions(loader)
```

## Generic Heterogeneous Graphs

### `build_hetero_data(nodes, edges, graph_attrs=None)`

Build a `HeteroData` object from dictionaries of node features and edge indices. This supports any heterogeneous graph schema, not just OPF.

```python
import torch
from lumina_inference import build_hetero_data

data = build_hetero_data(
    nodes={
        "protein": {
            "x": torch.randn(100, 64),
            "y": torch.randn(100, 1),  # optional targets
        },
        "drug": {
            "x": torch.randn(50, 32),
        },
    },
    edges={
        ("protein", "interacts", "drug"): {
            "edge_index": torch.randint(0, 50, (2, 200)),
            "edge_attr": torch.randn(200, 8),  # optional
        },
        ("drug", "treats", "protein"): {
            "edge_index": torch.randint(0, 100, (2, 150)),
        },
    },
    graph_attrs={
        "experiment_id": torch.tensor([42]),
    },
)
```

**Parameters:**
- `nodes` — Dict mapping node type names to dicts with `"x"` (required), `"y"` (optional)
- `edges` — Dict mapping `(src, rel, dst)` tuples to dicts with `"edge_index"` (required), `"edge_attr"` (optional), `"edge_label"` (optional)
- `graph_attrs` — Optional dict of graph-level attributes

NumPy arrays are automatically converted to PyTorch tensors.

## Schema Detection and Validation

### `detect_schema(data)`

Returns `"opf"` if the data contains all OPF node types (`bus`, `generator`, `load`, `shunt`) and at least one OPF edge type. Returns `"generic"` otherwise.

```python
from lumina_inference.dataset import detect_schema

schema = detect_schema(data)  # "opf" or "generic"
```

### `validate_hetero_data(data, model_metadata=None, model_input_channels=None)`

Validates that a `HeteroData` object is well-formed. Optionally checks against model expectations.

```python
from lumina_inference.dataset import validate_hetero_data

# Basic validation
validate_hetero_data(data)

# Validate against model config
validate_hetero_data(
    data,
    model_metadata=config_data.get("metadata"),
    model_input_channels=config_data.get("input_channels"),
)
```

**Raises:**
- `SchemaValidationError` — when node types don't match model expectations
- `FeatureDimensionError` — when feature dimensions don't match model input channels
- `MissingDataError` — when required data (e.g., node features) is missing

### `validate_opf_schema(data)`

Validates OPF-specific schema requirements: all 4 node types present with correct feature dimensions.

```python
from lumina_inference.dataset import validate_opf_schema

validate_opf_schema(data)  # raises SchemaValidationError or FeatureDimensionError
```

**Expected feature dimensions:**

| Node Type | Features | Dimension |
|---|---|---|
| `bus` | `base_kv, vmin, vmax, pq, pv, ref, isolated` | 7 |
| `generator` | `mbase, pg, pmin, pmax, qg, qmin, qmax, vg, cost_sq, cost_lin, cost_off` | 11 |
| `load` | `pd, qd` | 2 |
| `shunt` | `bs, gs` | 2 |

## Conditional Model Routing

The `OPFHeteroGNN` model automatically detects whether input data follows the OPF schema or is a generic heterogeneous graph:

- **OPF data**: Applies sigmoid-bounded min-max scaling using voltage limits (`vmin`, `vmax`) and generator power limits (`pmin`, `pmax`, `qmin`, `qmax`) extracted from input features
- **Generic data**: Returns raw model outputs without scaling

This detection happens at runtime in the `forward()` method — no configuration needed.

## Exception Hierarchy

All ingestion errors inherit from `LuminaIngestionError`:

```
LuminaIngestionError
├── SchemaValidationError    (expected, actual)
├── FeatureDimensionError    (node_type, expected_dim, actual_dim)
└── MissingDataError         (missing_field, context)
```

```python
from lumina_inference.dataset import (
    LuminaIngestionError,
    SchemaValidationError,
    FeatureDimensionError,
    MissingDataError,
)

try:
    data = load_from_dict(bad_dict)
except SchemaValidationError as e:
    print(f"Expected: {e.expected}, Got: {e.actual}")
except LuminaIngestionError as e:
    print(f"Ingestion error: {e}")
```

## Installation Extras

| Extra | Install Command | Enables |
|---|---|---|
| `hdf5` | `pip install lumina-inference[hdf5]` | `load_from_hdf5`, `process_hdf5_file` |
| (pandapower) | `pip install pandapower` | `load_from_matpower` |

## Migration from Standard Pipeline

### Before (DataLoader required)

```python
dataset = OPFDataset(root="./data", case_name="pglib_opf_case14_ieee")
loader = DataLoader(dataset, batch_size=1)
preds = modeler.run_predictions(loader, max_batches=1)
```

### After (single-sample, no DataLoader)

```python
data = load_from_json_file("path/to/case14_sample.json", require_solution=False)
preds = modeler.predict_single(data)
```

### After (from HDF5 file)

```python
data_list = load_from_hdf5("scenarios.h5")
preds = modeler.predict_single(data_list[0])
```

### After (from MATPOWER file)

```python
data = load_from_matpower("case14.m")
preds = modeler.predict_single(data)
```

### After (from in-memory dict)

```python
data = load_from_dict(my_opf_dict)
preds = modeler.predict_single(data)
```

## API Reference Summary

| Function | Input | Output | Optional Deps |
|---|---|---|---|
| `load_from_dict()` | Python dict | `HeteroData` | — |
| `load_from_json_string()` | JSON string | `HeteroData` | — |
| `load_from_json_file()` | `.json` path | `HeteroData` | — |
| `load_from_hdf5()` | `.h5` path | `List[HeteroData]` | `h5py`, `pydantic` |
| `load_from_matpower()` | `.m` path | `HeteroData` | `pandapower` |
| `process_json_file()` | `.json` path | `HeteroData` | — |
| `process_hdf5_file()` | `.h5` path | `List[HeteroData]` | `h5py`, `pydantic` |
| `process_opf_dict()` | Python dict | `HeteroData` | — |
| `build_hetero_data()` | nodes/edges dicts | `HeteroData` | — |
| `Modeler.predict_single()` | `HeteroData` | `Dict[str, Tensor]` | — |
| `Modeler.run_predictions()` | `DataLoader` | `List[Dict]` | — |

## Examples

See [`examples/flexible_ingestion.py`](../examples/flexible_ingestion.py) for 7 runnable examples covering all ingestion methods (dict, JSON, HDF5, MATPOWER, generic), schema validation, and end-to-end prediction patterns.
