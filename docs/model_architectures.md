# Multi-Architecture Model Loading

`lumina-inference` supports dynamic loading of multiple GNN architectures from
checkpoint configs. The model type is resolved automatically from the
checkpoint's configuration data, so the same `Modeler` API works for all
supported architectures without any code changes.

## Supported Architectures

| Canonical Name | Class | Description | Key Parameters |
|---|---|---|---|
| `HeteroGNN` | `OPFHeteroGNN` | Heterogeneous message-passing GNN with configurable backend convolution (SAGEConv, GCNConv, GINConv, GATConv) | `backend`, `hidden_channels`, `num_layers` |
| `RGAT` | `RGAT` | Relational Graph Attention Network with per-relation attention | `num_heads`, `hidden_channels`, `num_layers` |
| `HEAT` | `HEAT` | Heterogeneous Edge-Attributed Transformer — operates on a homogeneous view internally via `HEATConv` | `attention_heads`, `edge_type_emb_dim`, `edge_attr_emb_dim` |
| `HGT` | `HGT` | Heterogeneous Graph Transformer using PyG's `HGTConv` | `num_heads`, `hidden_channels`, `num_layers` |

All architectures share the same forward signature:

```python
forward(x_dict, edge_index_dict, edge_attr_dict=None, minmax_scaling=False, **kwargs)
→ dict  # {"bus": Tensor, "generator": Tensor}
```

## How Model Type Is Resolved

When `Modeler.load_model()` is called, the model architecture is determined
from the checkpoint config in the following priority order:

1. **`model_class`** field (fully-qualified class path) — e.g.
   `"lumina.model.opf.hetero_model.HGT"`. The class name is extracted from the
   last segment of the dotted path.

2. **`model`** field (short name) — e.g. `"HGT"`, `"RGAT"`, `"HeteroGNN"`.
   Case-insensitive lookup is used, so `"hgt"` and `"HGT"` both resolve to the
   `HGT` architecture.

3. **Default** — If neither field is present (e.g. legacy configs), defaults to
   `HeteroGNN` (`OPFHeteroGNN`), preserving backward compatibility.

### Recognized Aliases

The following aliases are recognized (case-insensitive):

| Input | Resolves To |
|---|---|
| `HeteroGNN`, `heterognn` | `HeteroGNN` |
| `OPFHeteroGNN`, `opfheterognn` | `HeteroGNN` |
| `RGAT`, `rgat` | `RGAT` |
| `HEAT`, `heat` | `HEAT` |
| `HGT`, `hgt` | `HGT` |

## Checkpoint Config Format

A typical HuggingFace checkpoint config (`config.json`) contains:

```json
{
  "model": "HGT",
  "model_class": "lumina.model.opf.hetero_model.HGT",
  "case_name": "pglib_opf_case14_ieee",
  "metadata": {
    "nodes": {
      "bus": 4,
      "generator": 7,
      "load": 2,
      "shunt": 2
    },
    "edges": {
      "('bus', 'ac_line', 'bus')": 9,
      "('generator', 'generator_link', 'bus')": 0,
      "('bus', 'generator_link', 'generator')": 0,
      "('load', 'load_link', 'bus')": 0,
      "('bus', 'load_link', 'load')": 0,
      "('shunt', 'shunt_link', 'bus')": 0,
      "('bus', 'shunt_link', 'shunt')": 0
    }
  },
  "input_channels": {
    "bus": 4,
    "generator": 7,
    "load": 2,
    "shunt": 2
  },
  "out_channels": 2,
  "config": {
    "models": {
      "HGT": {
        "hidden_channels": 2048,
        "num_layers": 8,
        "num_heads": 4,
        "dropout": 0.0
      }
    }
  }
}
```

### Config Fallback Behavior

If the `config.models` section does not contain a key matching the resolved
model type (e.g. config says `"model": "HGT"` but `config.models` only has
`"HeteroGNN"`), the system falls back to the `HeteroGNN` config section and
logs a diagnostic message. This handles cases where older configs were saved
with only `HeteroGNN` hyperparameters.

If no config section is found at all, sensible defaults are applied:

- `hidden_channels`: 64
- `num_layers`: 3
- `backend`: `"sage"` (for `HeteroGNN`)
- `num_heads`: 1 (for `RGAT`, `HGT`)
- `attention_heads`: 1 (for `HEAT`)

## Usage

### Basic Usage (Automatic Resolution)

```python
import json
import torch
from safetensors.torch import load_file
from lumina_inference import Modeler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modeler = Modeler(device)

# Config and weights from a HuggingFace checkpoint
config_data = json.load(open("config.json"))
state_dict = load_file("model.safetensors")

# Model type is resolved automatically from config_data
model, config = modeler.load_model(config_data, state_dict)
print(f"Loaded model: {type(model).__name__}")
# e.g. "Loaded model: HGT"
```

### Using the Registry Directly

For advanced use cases, the registry functions can be used directly:

```python
from lumina_inference.model.registry import (
    HETERO_MODEL_CLASSES,
    resolve_hetero_model_type,
    build_hetero_model_spec,
)

# Resolve model type from config fields
model_type = resolve_hetero_model_type(
    model_type=config_data.get("model"),
    model_class_path=config_data.get("model_class"),
    default="HeteroGNN",
)
print(f"Resolved model type: {model_type}")

# Build model class and constructor kwargs
model_class, model_kwargs, model_config, used_fallback = build_hetero_model_spec(
    model_type=model_type,
    metadata=config_data["metadata"],
    input_channels=config_data["input_channels"],
    models_config=config_data.get("config", {}).get("models", {}),
    out_channels=config_data.get("out_channels", 2),
)

# Instantiate
model = model_class(**model_kwargs)
```

### Listing Available Architectures

```python
from lumina_inference.model.registry import HETERO_MODEL_CLASSES

print("Supported architectures:", list(HETERO_MODEL_CLASSES.keys()))
# ['HeteroGNN', 'RGAT', 'HEAT', 'HGT']
```

## Architecture Details

### OPFHeteroGNN (`HeteroGNN`)

The default heterogeneous GNN architecture. Uses `HeteroConv` wrappers around
homogeneous convolution layers. The `backend` parameter selects the underlying
convolution type:

- `"sage"` — `SAGEConv` (default)
- `"gcn"` — `GCNConv`
- `"gin"` — `GINConv`
- `"gat"` — `GATConv`

Each layer applies the selected convolution independently per edge type, then
aggregates messages at each node.

### RGAT

Relational Graph Attention Network. Uses `RGATConv` from PyG, which applies
multi-head attention with relation-specific parameters. Supports edge
attributes via optional `edge_attr_dim`.

### HEAT

Heterogeneous Edge-Attributed Transformer. Internally converts the
heterogeneous graph to a homogeneous representation via
`HeteroData.to_homogeneous()` and applies `HEATConv` layers. This architecture
uses lazy parameter initialization (`in_channels=-1`), so parameters are
materialized during the first forward pass or when loading a checkpoint.

**Note:** When loading HEAT checkpoints, `lumina-inference` automatically
detects the uninitialized (lazy) parameters and uses `assign=True` in
`load_state_dict` to materialize them from the checkpoint tensors.

### HGT

Heterogeneous Graph Transformer. Uses PyG's `HGTConv` which applies
type-specific linear transformations and multi-head attention across different
node and edge types. This is the architecture used for LUMINA release models.

## Lazy Parameter Handling

The `HEAT` architecture uses `HEATConv` with `in_channels=-1`, which creates
`UninitializedParameter` objects. These lazy parameters cannot be inspected via
`state_dict()` until they are materialized (either by a forward pass or by
loading a checkpoint).

`Modeler.load_checkpoint_into_model()` handles this automatically:

1. Checks if the model has any `UninitializedParameter` instances
2. If so, loads the checkpoint with `assign=True` to replace the uninitialized
   parameters directly with checkpoint tensors
3. If not (normal case), uses the standard key-remapping approach

This is transparent to the user — all architectures load through the same
`Modeler.load_model()` API.

## Backward Compatibility

- **Legacy configs** without `"model"` or `"model_class"` fields default to
  `OPFHeteroGNN`, matching the previous hard-coded behavior.
- **String edge keys** in metadata (from JSON serialization) are automatically
  converted to tuple keys.
- **Existing code** that uses `Modeler.load_model()` with `OPFHeteroGNN`
  checkpoints continues to work without changes.

## Design Rationale

### Why vendor instead of importing from `lumina-core`?

`lumina-inference` is designed as a standalone inference package. Adding
`lumina-core` as a dependency would pull in training frameworks (`torch.distributed`,
W&B, Hydra/OmegaConf) that are unnecessary for inference.

### Why a registry instead of dynamic `importlib` loading?

The registry approach is safer and self-contained — only known, vendored model
classes can be instantiated. Dynamic `importlib` loading would require
`lumina-core` to be installed and could allow arbitrary code execution from
checkpoint metadata.

### Why not include `HEAT_v2`?

`HEAT_v2` is not registered in `lumina-core`'s `HETERO_MODEL_CLASSES` and is
not used for any released model checkpoints. It can be added to the registry
later if needed.
