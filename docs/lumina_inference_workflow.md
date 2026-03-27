## Step-by-Step Inference Workflow: Object Inputs and Outputs

### Overall Attributes

- **Model size**: Several versions (1B-parameter, 671M-parameter, 673M-parameter, and 681M-parameter)
- **Number of samples**: Usually at 15k which is the OPFData sample sizes but this size depends on the data input
Based on the provided scripts, here is a summary of the model's input and output data structures.
- **Input Data Structure**

  The model consumes **PyG `HeteroData` objects**, where each object represents a power grid as a heterogeneous graph. These objects can be created via multiple pathways:
  - The `OPFDataset` class (batch pipeline from raw JSON files)
  - `load_from_dict()`, `load_from_json_file()`, `load_from_json_string()` (single-sample OPF ingestion)
  - `load_from_matpower()` (MATPOWER `.m` file ingestion)
  - `build_hetero_data()` (generic heterogeneous graph construction)

  *   **Node Types**: The graph contains four types of nodes, each with a feature matrix (`x`):
      *   `bus`: A tensor with 7 features per bus, including `base_kv`, voltage limits (`vmin`, `vmax`), and a one-hot encoding of the bus type.
      *   `generator`: A tensor with 11 features per generator, including power limits (`pmin`, `pmax`, `qmin`, `qmax`) and cost coefficients.
      *   `load`: A tensor with 2 features per load: active power demand (`pd`) and reactive power demand (`qd`).
      *   `shunt`: A tensor with 2 features per shunt: susceptance (`bs`) and conductance (`gs`).
  *   **Edge Types**: The graph connects nodes using several edge types. Physical connections carry attributes, while virtual links do not:
      *   `('bus', 'ac_line', 'bus')`: Edges representing AC lines, with 9 attributes (`edge_attr`) like resistance, reactance, and thermal ratings (`rate_a`).
      *   `('bus', 'transformer', 'bus')`: Edges representing transformers, with 11 attributes (`edge_attr`).
      *   `('generator', 'generator_link', 'bus')`, `('load', 'load_link', 'bus')`, etc.: Virtual edges that connect components to the busbar without carrying features.

- **Output Data Structure**

  The model returns a **dictionary of tensors** where each key corresponds to a node type and the value contains the predictions for all nodes of that type.

  *   `predictions['bus']`: A tensor of shape `[number_of_buses, 2]`. The two columns represent the predicted **voltage angle (`va`)** and **voltage magnitude (`vm`)** for each bus.
  *   `predictions['generator']`: A tensor of shape `[number_of_generators, 2]`. The two columns represent the predicted **active power (`pg`)** and **reactive power (`qg`)** output for each generator.

- **Latency**: Mainly from processing through sample batch iterations

- **Throughputs**: TBD but likely proportional to graph structure and data size


---

### Step 1: Download Model Artifacts

```python
config_path = hf_hub_download(repo_id="argonne/LUMINA-1B", filename="config.json")
safetensors_path = hf_hub_download(repo_id="argonne/LUMINA-1B", filename="model.safetensors")
```

- **Input**: Hugging Face `repo_id` string and `filename` strings
- **Output**: Two local file paths (`str`)
  - `config_path` → path to `config.json` on disk
  - `safetensors_path` → path to `model.safetensors` on disk

---

### Step 2: Load Configuration

```python
with open(config_path, "r") as f:
    config_data = json.load(f)
```

- **Input**: `config_path` (`str`)
- **Output**: `config_data` (`dict`) containing at minimum [2]:
  - `config_data["metadata"]` → node type and edge type definitions, including an `"edges"` sub-dict where keys may be string-serialized tuples
  - `config_data["input_channels"]` → per-node-type feature dimensions
  - `config_data["config"]["models"]["HeteroGNN"]["hidden_channels"]` → model hidden dimension
  - `config_data["config"]["models"]["HeteroGNN"]["num_layers"]` → number of GNN layers
  - `config_data["config"]["models"]["HeteroGNN"]["backend"]` → GNN backend type
  - `config_data["case_name"]` → string identifying the power grid case (e.g., `"pglib_opf_case14_ieee"`)

---

### Step 3: Instantiate the Modeler

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modeler = Modeler(device)
```

- **Input**: `device` (`torch.device`)
- **Output**: `modeler` (`Modeler`) — a stateful wrapper with `model` attribute initially set to `None` [2]

---

### Step 4: Load Model Weights

```python
state_dict = load_file(safetensors_path)
modeler.load_model(config_data, state_dict)
```

This step has two sub-operations:

**4a.** `load_file(safetensors_path)`
- **Input**: `safetensors_path` (`str`)
- **Output**: `state_dict` (`dict[str, torch.Tensor]`) — raw weight tensors with checkpoint-format keys using angle-bracket/triple-underscore notation (e.g., `<bus___ac_line___weight>`) [2]

**4b.** `modeler.load_model(config_data, state_dict)`
- **Input**: `config_data` (`dict`), `state_dict` (`dict[str, torch.Tensor]`)
- **Internal operations** [2]:
  1. Converts string-serialized edge keys in `config_data["metadata"]["edges"]` to Python tuples via `ast.literal_eval`
  2. Constructs an `OPFHeteroGNN` model using metadata, input channels, hidden channels, num layers, and backend from config
  3. Remaps checkpoint keys from `<a___b___c>` format to `('a', 'b', 'c')` tuple format via `convert_checkpoint_key_to_model_key`
  4. Loads remapped weights into model via `load_state_dict(strict=False)`
  5. Sets model to `eval()` mode
  6. Stores `config_data` on the modeler instance for later validation
- **Output**: Tuple of (`torch.nn.Module`, `dict`), and internally sets `modeler.model` and `modeler.config_data` [2]

---

### Step 5: Load Data

At this point, there are **two pathways** for loading data into the model:

#### Pathway A: Batch Pipeline (OPFDataset + DataLoader)

The original pipeline for processing many samples through a DataLoader.

**Step 5A: Load Dataset**

```python
case_name = config_data.get("case_name", "pglib_opf_case14_ieee")
dataset = OPFDataset(root="./opf_data", case_name=case_name)
```

- **Input**: `root` (`str`), `case_name` (`str`)
- **Internal operations** [4]:
  1. Checks for processed `.pt` file on disk
  2. If not found, downloads `.tar.gz` from Google Cloud Storage, extracts JSON files
  3. Processes each JSON file via `process_json_file(json_file, require_solution=True)`, which delegates to `process_opf_dict()` to build a `HeteroData` object per sample containing:
     - **`bus.x`**: `[n_bus, 7]` — `base_kv, vmin, vmax, pq, pv, ref, isolated` (bus type one-hot encoded) [4]
     - **`bus.y`**: `[n_bus, 2]` — `va, vm` (ground truth labels) [4]
     - **`generator.x`**: `[n_gen, 11]` — `mbase, pg, pmin, pmax, qg, qmin, qmax, vg, cost_squared, cost_linear, cost_offset` [4]
     - **`generator.y`**: `[n_gen, 2]` — `pg, qg` [4]
     - **`load.x`**: `[n_load, 2]` — `pd, qd` [4]
     - **`shunt.x`**: `[n_shunt, 2]` — `bs, gs` [4]
     - **`(bus, ac_line, bus).edge_index`**: `[2, n_lines]` [4]
     - **`(bus, ac_line, bus).edge_attr`**: `[n_lines, 9]` — `angmin, angmax, b_fr, b_to, br_r, br_x, rate_a, rate_b, rate_c` [4]
     - **`(bus, transformer, bus).edge_index`**: `[2, n_transformers]` [4]
     - **`(bus, transformer, bus).edge_attr`**: `[n_transformers, 11]` — `angmin, angmax, br_r, br_x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to` [4]
     - **Six virtual link `edge_index` tensors** (generator, load, shunt — forward and reverse), each `[2, n_links]` with no attributes [4]
  4. Collates all `HeteroData` objects and saves to disk as `.pt`
- **Output**: `dataset` (`OPFDataset` / `InMemoryDataset`) — indexable collection of `HeteroData` objects [4]

**Step 6A: Create DataLoader**

```python
loader = DataLoader(dataset, batch_size=1, shuffle=False)
```

- **Input**: `dataset` (`OPFDataset`), `batch_size` (`int`), `shuffle` (`bool`)
- **Internal operation**: Wraps the dataset with a custom `Collater` that calls `Batch.from_data_list` on lists of `HeteroData` objects to produce batched heterogeneous graphs [3]
- **Output**: `loader` (`DataLoader`) — iterable yielding `Batch` objects, where each `Batch` exposes:
  - `batch.x_dict` → `dict[str, Tensor]` mapping node type names to stacked feature matrices
  - `batch.edge_index_dict` → `dict[tuple, Tensor]` mapping edge type tuples to edge index tensors
  - `batch.edge_attr_dict` → `dict[tuple, Tensor]` mapping edge type tuples to edge attribute tensors (where attributes exist)
  - `batch.node_types` → list of node type strings
  - `batch.edge_types` → list of edge type tuples

**Step 7A: Run Batch Predictions**

```python
preds = modeler.run_predictions(loader, max_batches=50)
```

- **Input**: `loader` (iterable of `Batch`), `max_batches` (`int`)
- **Internal operation per batch** [2]:
  1. `to_float32(batch)` — casts all `.x`, `.y`, and `.edge_attr` tensors to `float32`
  2. `batch.to(device)` — moves batch to target device
  3. Forward pass: `model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, minmax_scaling=True)`
  4. Detaches prediction tensors and moves to CPU
  5. Moves batch back to CPU
  6. Appends `(predictions_cpu, batch_cpu)` tuple to results list
- **Output**: `preds` (`List[Tuple[dict, Batch]]`) — list of 50 tuples, each containing:
  - `predictions_cpu` (`dict[str, torch.Tensor]`):
    - `predictions_cpu["bus"]`: `[n_bus, 2]` — predicted `va, vm` [2]
    - `predictions_cpu["generator"]`: `[n_gen, 2]` — predicted `pg, qg` [2]
  - `batch_cpu` (`Batch`) — the original input batch moved to CPU, retaining all input features, edge indices, and ground truth labels [2]

#### Pathway B: Flexible Single-Sample Ingestion (New)

A streamlined pathway for inference on individual samples without the full dataset pipeline.

**Step 5B: Load Single Sample**

```python
from lumina_inference import load_from_dict, load_from_json_file, load_from_matpower

# Option 1: From a Python dict
data = load_from_dict(opf_dict)

# Option 2: From a JSON file
data = load_from_json_file("path/to/case.json", require_solution=False)

# Option 3: From a JSON string
data = load_from_json_string(json_str, require_solution=False)

# Option 4: From a MATPOWER .m file (requires pandapower)
data = load_from_matpower("path/to/case14.m")

# Option 5: Build generic HeteroData (non-OPF)
data = build_hetero_data(
    nodes={"node_a": {"x": torch.randn(10, 8)}},
    edges={("node_a", "connects", "node_a"): {"edge_index": torch.randint(0, 10, (2, 20))}},
)
```

- **Input**: Varies by function — Python dict, file path, JSON string, or node/edge dicts
- **Internal operations**:
  - All OPF functions delegate to `process_opf_dict(obj, require_solution=False)` which:
    1. Validates presence of `grid`, `grid.nodes`, and `grid.edges` keys
    2. One-hot encodes bus type column (index 1): `[1=PQ, 2=PV, 3=ref, 4=isolated]` → 4 columns
    3. Creates all node feature tensors with `dtype=torch.float32`
    4. Extracts edge indices and attributes
    5. Optionally sets `.y` and `.edge_label` if `solution` key is present
  - `load_from_matpower()` additionally parses the `.m` file via regex, converts through pandapower for bus ID mapping and per-unit conversion, then assembles the OPF dict
  - `build_hetero_data()` accepts raw tensors/arrays and constructs `HeteroData` directly
- **Output**: `data` (`HeteroData`) — a single graph sample ready for prediction

**Step 6B–7B: Run Single-Sample Prediction**

```python
predictions = modeler.predict_single(data)
```

- **Input**: `data` (`HeteroData`), optional `minmax_scaling` (`bool`, default `True`), optional `validate` (`bool`, default `True`)
- **Internal operations**:
  1. If `validate=True`, calls `_validate_batch(data)` which checks data against the loaded model's `config_data` metadata and input channels using `validate_hetero_data()`
  2. Wraps the single sample in `Batch.from_data_list([data])`
  3. Delegates to `predict_batch()` for the forward pass
- **Output**: `predictions` (`dict[str, torch.Tensor]`):
  - `predictions["bus"]`: `[n_bus, 2]` — predicted `va, vm`
  - `predictions["generator"]`: `[n_gen, 2]` — predicted `pg, qg`

---

### Step 8: Inspect Results

**From Pathway A (batch):**

```python
predictions_cpu, batch_cpu = preds[0]
```

- **Input**: First element of `preds` list
- **Output**: Two objects unpacked from the tuple:
  - `predictions_cpu` — dictionary with string keys `"bus"` and `"generator"`, each mapping to a 2-column `float32` CPU tensor [2]
  - `batch_cpu` — full `Batch` object on CPU containing all original input data and ground truth labels from the dataset [2][4]

**From Pathway B (single-sample):**

```python
print(predictions["bus"].shape)       # [n_bus, 2]
print(predictions["generator"].shape) # [n_gen, 2]
```

- **Output**: Dictionary with string keys `"bus"` and `"generator"`, each mapping to a 2-column `float32` CPU tensor


---

Here are the main objects in this inference workflow, with their types and primary methods—the "nouns and verbs" of the modeling story.

- **Hugging Face artifact handles**
  - **Type:** file path strings (`str`)
  - **Created by:** `hf_hub_download(...)`
  - **Main methods / actions:** download `config.json` and `model.safetensors` from the HF repo [1]

- **`config_data`**
  - **Type:** Python dictionary (`dict`)
  - **Created by:** `json.load(...)`
  - **Main uses / actions:** provides model metadata, input channel sizes, model architecture settings, and case name used to reconstruct the model and select dataset defaults [1][2]

- **`state_dict`**
  - **Type:** dictionary of tensors (`dict[str, torch.Tensor]`)
  - **Created by:** `load_file(safetensors_path)`
  - **Main uses / actions:** supplies checkpoint weights to `Modeler.load_model(...)`; keys are remapped before loading into the model [1][2]

- **`device`**
  - **Type:** `torch.device`
  - **Created by:** `torch.device("cuda" if ... else "cpu")`
  - **Main uses / actions:** controls where the model and batch tensors are placed for inference [1][2]

- **`Modeler`**
  - **Type:** inference wrapper class
  - **Constructor:** `Modeler(device, *, fail_on_missing=False, verbose=True)`
  - **Main methods:**
    - `load_model(config_data, state_dict)` — constructs `OPFHeteroGNN`, remaps checkpoint keys, loads weights, switches model to eval mode, stores `config_data` for validation [2]
    - `convert_checkpoint_key_to_model_key(key)` — converts checkpoint key format to model key format [2]
    - `load_checkpoint_into_model(model, checkpoint_dict, ...)` — loads remapped weights with `strict=False` [2]
    - `to_float32(batch)` — casts node features, targets, and edge attributes to `float32` [2]
    - `predict_batch(batch, minmax_scaling=True)` — runs one forward pass and returns `(predictions_cpu, batch_cpu)` [2]
    - `predict_single(data, minmax_scaling=True, validate=True)` — wraps a single `HeteroData` in a batch and runs prediction; optionally validates against model schema [2]
    - `run_predictions(loader, max_batches=None, minmax_scaling=True)` — iterates over batches and collects prediction/batch pairs [1][2]
    - `_validate_batch(data)` — validates `HeteroData` against the loaded model's metadata and input channels [2]

- **`OPFHeteroGNN`**
  - **Type:** `torch.nn.Module`
  - **Created inside:** `Modeler.load_model(...)`
  - **Constructor parameters:** `metadata`, `input_channels`, `hidden_channels`, `num_layers`, `backend`, `output_node_types=None`, `scaling_config=None` [2]
  - **Main methods / actions:**
    - `forward(x_dict, edge_index_dict, edge_attr_dict=None, minmax_scaling=False)` — generates predictions with conditional routing [2]:
      - Detects OPF vs. generic schema via `_is_opf_schema(x_dict)`
      - For OPF data: extracts voltage/power bounds before input transformation, applies sigmoid-bounded min-max scaling after output
      - For generic data: returns raw outputs without scaling
    - `_is_opf_schema(x_dict)` — checks if `bus` and `generator` node types are present with sufficient feature dimensions [2]
    - `_extract_bounds_opf(x_dict)` — extracts `vmin`, `vmax`, `pmin`, `pmax`, `qmin`, `qmax` from input features [2]
    - `_apply_scaling_opf(outputs, bounds)` — applies sigmoid-bounded scaling for bus voltage and generator power [2]
  - **Output heads:** configurable via `output_node_types` parameter; defaults to `{"bus": out_channels, "generator": out_channels}` [2]

- **`OPFDataset`**
  - **Type:** `torch_geometric.data.InMemoryDataset`
  - **Constructor:** `OPFDataset(root, case_name, group_id=0, ...)`
  - **Main methods / actions:**
    - `download()` — downloads raw dataset archive if needed [4]
    - `download_and_extract(name)` — fetches and extracts `.tar.gz` raw data [4]
    - `process()` — processes raw files into cached PyG data [4]
    - `process_json_group(group_id)` — processes a group of JSON samples [4]
    - `metadata()` — reports node and edge feature dimensions [4]
    - dataset indexing / iteration inherited from `InMemoryDataset`
  - **Main role:** stores many per-sample `HeteroData` graph objects for batch inference [4]

- **`HeteroData` sample**
  - **Type:** `torch_geometric.data.HeteroData`
  - **Created by:** Multiple pathways:
    - `process_json_file(json_file, require_solution=True)` inside dataset processing [4]
    - `process_opf_dict(obj, require_solution=False)` — core OPF dict converter [5]
    - `load_from_dict(data_dict, require_solution=False)` — from Python dict [5]
    - `load_from_json_file(path, require_solution=False)` — from JSON file [5]
    - `load_from_json_string(json_str, require_solution=False)` — from JSON string [5]
    - `load_from_matpower(path)` — from MATPOWER `.m` file [5]
    - `build_hetero_data(nodes, edges, graph_attrs=None)` — generic construction [5]
  - **Main contents / actions:**
    - node stores like `hdata['bus'].x`, `hdata['generator'].x`, `hdata['load'].x`, `hdata['shunt'].x` [4]
    - target stores like `hdata['bus'].y`, `hdata['generator'].y` (present only when solution data is available) [4][5]
    - edge stores like `hdata['bus', 'ac_line', 'bus'].edge_index` and `.edge_attr` [4]
    - graph-level attributes like `hdata.baseMVA` [4][5]
  - **Main role:** one graph sample representing one OPF case instance (or a generic heterogeneous graph)

- **`process_json_file`**
  - **Type:** dataset preprocessing function
  - **Signature:** `process_json_file(json_file, require_solution=True)`
  - **Main actions:** reads JSON, delegates to `process_opf_dict()` to build and return a `HeteroData` object with node features, targets, edge indices, and edge attributes [4][5]
  - **Note:** The `require_solution` parameter (default `True` for backward compatibility) controls whether the `solution` key is required. Set to `False` for inference-only use.

- **`process_opf_dict`**
  - **Type:** core ingestion function
  - **Signature:** `process_opf_dict(obj, require_solution=False)`
  - **Main actions:** validates dict structure, one-hot encodes bus type, creates `float32` tensors for all node features and edge attributes, optionally sets `.y` and `.edge_label` from solution data [5]
  - **Main role:** single source of truth for OPF dict → `HeteroData` conversion; all `load_from_*` functions and `process_json_file` delegate to this [5]

- **Ingestion functions** (new)
  - **`load_from_dict(data_dict, require_solution=False)`** — converts a Python dict to `HeteroData` [5]
  - **`load_from_json_file(path, require_solution=False)`** — loads and converts a JSON file [5]
  - **`load_from_json_string(json_str, require_solution=False)`** — parses and converts a JSON string [5]
  - **`load_from_matpower(path)`** — parses a MATPOWER `.m` file via regex, converts through pandapower, produces `HeteroData` [5]
  - **`build_hetero_data(nodes, edges, graph_attrs=None)`** — constructs generic `HeteroData` from dicts of tensors/arrays [5]

- **Validation utilities** (new)
  - **`detect_schema(data)`** — returns `"opf"` or `"generic"` based on node types present [5]
  - **`validate_hetero_data(data, model_metadata=None, model_input_channels=None)`** — validates data structure and optionally checks against model expectations [5]
  - **`validate_opf_schema(data)`** — validates OPF-specific node types and feature dimensions [5]
  - **Exception hierarchy:** `LuminaIngestionError` → `SchemaValidationError`, `FeatureDimensionError`, `MissingDataError` [5]

- **`DataLoader`**
  - **Type:** custom loader class extending `torch.utils.data.DataLoader`
  - **Constructor:** `DataLoader(dataset, batch_size=1, shuffle=False, ...)`
  - **Main role / actions:** iterates over the dataset and batches graph samples into PyG batch objects using a custom collate function [1][3]

- **`Collater`**
  - **Type:** callable helper class
  - **Main method:** `__call__(batch)`
  - **Main action:** merges a list of `HeteroData`/`BaseData` items into a `Batch` via `Batch.from_data_list(...)` [3]

- **`Batch`**
  - **Type:** `torch_geometric.data.Batch`
  - **Created by:** `Collater.__call__` / `Batch.from_data_list(...)` [3], or internally by `predict_single()` [2]
  - **Main properties / actions:**
    - `x_dict` — node feature tensors keyed by node type [2]
    - `edge_index_dict` — edge index tensors keyed by edge type [2]
    - `edge_attr_dict` — edge attribute tensors keyed by edge type when present [2]
    - `node_types`, `edge_types` — schema descriptors used by `to_float32(...)` [2]
    - `.to(device)` — moves the batch to CPU/GPU [2]

- **`predictions`**
  - **Type:** dictionary (`dict[str, torch.Tensor]`)
  - **Created by:** model forward pass in `predict_batch(...)` or `predict_single(...)` [2]
  - **Main contents:** per-output tensors such as `predictions['bus']` and `predictions['generator']` [2]
  - **Main role:** raw model outputs for the current batch or sample

- **`predictions_cpu`**
  - **Type:** dictionary (`dict[str, torch.Tensor]`)
  - **Created by:** detaching and moving model outputs to CPU in `predict_batch(...)` [2]
  - **Main role:** portable/storable inference results, safe to inspect outside GPU context [2]

- **`batch_cpu`**
  - **Type:** `Batch`
  - **Created by:** moving the input batch back to CPU in `predict_batch(...)` [2]
  - **Main role:** retains the original input graph and labels alongside stored predictions [2]

- **`preds` / `pred_batch_pairs`**
  - **Type:** `list[tuple[dict, Batch]]`
  - **Created by:** `Modeler.run_predictions(...)`
  - **Main role:** collected inference results across batches, where each item is `(predictions_cpu, batch_cpu)` [2]

---

### References

- [1] `examples/huggingface_inference.py`, `examples/flexible_ingestion.py`
- [2] `src/lumina_inference/modeler.py`, `src/lumina_inference/model/hetero_model.py`
- [3] `src/lumina_inference/loader/opf_loader.py`
- [4] `src/lumina_inference/dataset/opf_dataset.py`
- [5] `src/lumina_inference/dataset/ingestion.py`, `src/lumina_inference/dataset/validation.py`
