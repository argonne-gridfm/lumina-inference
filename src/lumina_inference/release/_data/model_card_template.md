---
tags:
- power-systems
- optimal-power-flow
- graph-neural-network
- pytorch
- heterogeneous-graph
library_name: pytorch
pipeline_tag: other
license: apache-2.0
---

# {model_name}

## Model Description

**{model_name}** is a {total_parameters_formatted}-parameter heterogeneous graph neural network ({model_architecture}) designed for **fast, high-fidelity prediction of AC Optimal Power Flow (ACOPF) solutions**.
The model encodes power system components—buses, generators, loads, and transmission lines—as distinct node and edge types, capturing structural and physical heterogeneity in electric grids.

Trained on ACOPF datasets spanning **multiple network topologies**, {model_name} generalizes across diverse grid configurations and operating conditions. It provides efficient approximations of OPF variables such as voltages, power injections, and line flows, serving as a scalable surrogate for traditional optimization solvers.

## Model Details

- **Version**: {model_version}
- **Release Commit**: {commit_sha}
- **Model Architecture**: {model_architecture}
- **Total Parameters**: {total_parameters} ({total_parameters_formatted})
- **Trainable Parameters**: {trainable_parameters} ({trainable_parameters_formatted})
- **Training Case**: {training_case}
- **Training Data Size**: {training_data_size}
- **Training Date**: {training_date}
- **Final Validation Loss**: {final_validation_loss}
- **Training Epochs**: {training_epochs}

## Input Channels
{input_channels}

## Model Files

This repository contains the model in multiple formats:
- `config.json` - Model configuration and metadata
- `model.pt` - Complete PyTorch checkpoint with metadata and config
- `model.safetensors` - Model weights in SafeTensors format (recommended)
- `requirements.txt` - Required Python libraries

## Installation

1. Clone or pull the latest `lumina-inference` repository:

   ```bash
   git clone git@github.com:argonne-gridfm/lumina-inference.git
   ```

2. Install the package in editable mode:

   ```bash
   cd lumina-inference
   pip install -e .
   ```

## Basic Usage

### Model Setup

```python
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from lumina_inference import Modeler
from lumina_inference.dataset import OPFDataset
from lumina_inference.loader import DataLoader

# Download model artifacts from Hugging Face
config_path = hf_hub_download(repo_id="{hf_repo_id}", filename="config.json")

# Load config
with open(config_path, "r") as f:
    config_data = json.load(f)

# Set up device and modeler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modeler = Modeler(device)
```

### Load Weights Option 1: Using SafeTensors (Recommended)

```python
# Load weights from SafeTensors
safetensors_path = hf_hub_download(
    repo_id="{hf_repo_id}", filename="model.safetensors"
)
state_dict = load_file(safetensors_path)
modeler.load_model(config_data, state_dict)
```

### Load Weights Option 2: Using PyTorch Checkpoint

```python
# Load weights from PyTorch Checkpoint on CPU
model_path = hf_hub_download(repo_id="{hf_repo_id}", filename="model.pt")
checkpoint = torch.load(model_path, map_location="cpu")
state_dict = checkpoint.get("model_state_dict")
modeler.load_model(config_data, state_dict)
```

### Load Data and Run Model (Batch)

```python
# Simple argument defaults
case_name = config_data.get("case_name", "pglib_opf_case14_ieee")
batch_size = 1
max_batches = 50

# Loading OPF dataset
dataset = OPFDataset(root="./opf_data", case_name=case_name)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Run model across data batches
preds = modeler.run_predictions(loader, max_batches=max_batches)

# Inspect first prediction
if preds:
    predictions_cpu, batch_cpu = preds[0]
    print(f"Prediction keys: {{list(predictions_cpu.keys())}}")
    for key, tensor in predictions_cpu.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {{key}}: shape={{tensor.shape}}, dtype={{tensor.dtype}}")
```

### Single-Sample Prediction (No DataLoader)

```python
from lumina_inference import load_from_json_file, load_from_dict

# From a JSON file
data = load_from_json_file("path/to/opf_sample.json")
result = modeler.predict_single(data)

# From a Python dictionary
data = load_from_dict(opf_dict)
result = modeler.predict_single(data)
```


## Data Source

Google DeepMind. (2024). OPFData: Large-scale datasets for AC optimal power flow. Google Cloud Storage. Retrieved November 12, 2025, from storage.googleapis.com.


## Release History

{release_history}


## Acknowledgements

This work was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research, and Laboratory Directed Research and Development (LDRD) funding from Argonne National Laboratory, under Contract DE-AC02-06CH11357. This research used resources of the Argonne Leadership Computing Facility at Argonne National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under contract DE-AC02-06CH11357. This research used resources of the Argonne Leadership Computing Facility and the National Energy Research Scientific Computing Center (NERSC), which are U.S. Department of Energy Office of Science User Facilities.


## License

Released under the Apache License, Version 2.0. See the bundled `LICENSE` and `NOTICE` files for the full terms and attribution.

Copyright 2026 UChicago Argonne, LLC.
