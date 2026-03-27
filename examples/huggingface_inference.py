"""Example: Load a LUMINA model from Hugging Face and run predictions.

This script demonstrates how to use lumina-inference to:
1. Download model artifacts from Hugging Face
2. Load the model configuration and weights
3. Load an OPF dataset
4. Run predictions on the dataset

Usage:
    python examples/huggingface_inference.py
"""

import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from lumina_inference.modeler import Modeler
from lumina_inference.dataset.opf_dataset import OPFDataset
from lumina_inference.loader.opf_loader import DataLoader

# Download model artifacts from Hugging Face
config_path = hf_hub_download(
    repo_id="argonne/LUMINA-1B", filename="config.json"
)
safetensors_path = hf_hub_download(
    repo_id="argonne/LUMINA-1B", filename="model.safetensors"
)

# Load config
with open(config_path, "r") as f:
    config_data = json.load(f)

# Set up device and modeler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modeler = Modeler(device)

# Load model
state_dict = load_file(safetensors_path)
modeler.load_model(config_data, state_dict)

# Load dataset and create loader
case_name = config_data.get("case_name", "pglib_opf_case14_ieee")
dataset = OPFDataset(root="./opf_data", case_name=case_name)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Run predictions (no evaluation)
preds = modeler.run_predictions(loader, max_batches=50)
print(f"Collected predictions for {len(preds)} batches.")

# Inspect first prediction
if preds:
    predictions_cpu, batch_cpu = preds[0]
    print(f"Prediction keys: {list(predictions_cpu.keys())}")
    for key, tensor in predictions_cpu.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
